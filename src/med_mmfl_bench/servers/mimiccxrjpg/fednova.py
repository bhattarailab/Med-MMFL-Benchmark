"""FedNova server for MIMIC-CXR multimodal classification.

Implements Federated Normalised Averaging (FedNova) which normalises
local updates by the number of local steps, removing the objective
inconsistency introduced by heterogeneous local training.
"""

import copy
import gc
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from med_mmfl_bench.servers.mimiccxrjpg.fedavg import FedavgServer
from med_mmfl_bench.trainers.mimiccxr import ClassificationTrainer as MimicClientTrainer
from med_mmfl_bench.utils import get_lr_scheduler
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)


class MimicNovaTrainer(MimicClientTrainer):
    """MIMIC-CXR trainer with FedNova normalised gradient tracking."""

    def __init__(
        self, args: Any, config: Any, wandb: Any = None, client_id: int = -1,
    ) -> None:
        super().__init__(args, config, wandb, client_id=client_id)
        self.ai: float = 0.0
        self.di: Dict[str, torch.Tensor] = {}
        self.rho: float = float(self.config.optimizer.momentum)

    def run(self, comms: int) -> None:
        """Run FedNova local training with gradient normalisation tracking."""
        self.model.cuda()
        self.global_weights = copy.deepcopy(self.model.state_dict())

        self.tau = 0
        for i in range(self.config.train.local_epoch):
            if self.client_id == -1:
                logger.info(
                    "Server — round %d, local_epoch %d, round_epoch %d",
                    comms, self.local_epoch, i,
                )
            else:
                logger.info(
                    "Client %d — local_epoch %d, round %d, round_epoch %d",
                    self.client_id, self.local_epoch, comms, i,
                )
            self.train_epoch()
            self.local_epoch += 1

        self.model.cpu()
        gc.collect()

    def val_wo_log(self) -> float:
        """Validate without WandB logging (for LR scheduling)."""
        self.model.cuda()
        self.model.eval()
        with tqdm(self.val_loader, unit="batch", desc="Validating") as tepoch:
            for frames, label, text, _ in tepoch:
                images = frames.cuda()
                label = label.cuda()
                with torch.no_grad():
                    output = self.model(self.tokenizer, images, text)
                self.evaluator.update(output["logits"], label.long())

        metrics = self.evaluator.compute()
        val_auc = metrics["AUC"].item()
        logger.info("Val AUC: %.4f", val_auc)
        self.evaluator.reset()
        self.model.cpu()
        return val_auc

    def train_epoch(self) -> None:
        """One training epoch with FedNova step counting."""
        self.model.cuda()
        self.model.train()

        with tqdm(self.train_loader, unit="batch", desc="Training (FedNova)") as tepoch:
            for frames, label, text, _ in tepoch:
                self.optimizer.zero_grad()
                images = frames.cuda()
                label = label.cuda()

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = self.model(self.tokenizer, images, text)
                    loss = self.criterion(output["logits"], label)

                self.grad_scaler.scale(loss).backward()

                if self.config.train.grad_clip > 0:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.train.grad_clip,
                    )

                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                tepoch.set_postfix(Loss=loss.item())

                self.tau += 1

        # Compute the normalised direction di
        self.ai = (
            self.tau - self.rho * (1 - pow(self.rho, self.tau)) / (1 - self.rho)
        ) / (1 - self.rho)

        state_dict = self.model.state_dict()
        self.di = copy.deepcopy(self.global_weights)
        for key in self.di:
            self.di[key] = torch.div(self.global_weights[key] - state_dict[key], self.ai)


class FednovaServer(FedavgServer):
    """FedNova server — inherits FedAvg with normalised aggregation."""

    def __init__(self, args: Any, config: Any, wandb: Any = False) -> None:
        super().__init__(args, config, wandb)
        self.trainer = MimicNovaTrainer(self.args, self.config, self.wandb)
        self.trainer.lr_scheduler = get_lr_scheduler(
            self.config.lr_scheduler.name,
            self.trainer.optimizer,
            self.config.lr_scheduler,
        )

    def _aggregate(self) -> Dict[str, torch.Tensor]:
        """Aggregate using FedNova normalised directions."""
        logger.info(
            "[%s] [Round: %s] Aggregate updated signals!",
            self.args.algorithm.upper(), str(self.round).zfill(4),
        )
        omega: List[int] = []
        ais: List[float] = []
        dis: List[Dict[str, torch.Tensor]] = []

        for client in self.clients:
            omega.append(len(client))
            ai, di = client.upload()
            ais.append(ai)
            dis.append(di)

        pis = np.array(omega) / sum(omega)
        return self.server_optimizer.aggregate(pis=pis, dis=dis, ais=ais)

    def _update_global_model(self, d_avg: Dict[str, torch.Tensor]) -> None:
        """Apply the aggregated normalised direction to the global model."""
        model = self.model.state_dict()
        for key in model:
            if "weight" in key or "bias" in key:
                model[key] = model[key] - d_avg[key].to("cpu")
        self.model.load_state_dict(model, strict=True)

    def update(self) -> None:
        """Execute one FedNova communication round."""
        self.dispatch()

        for client in self.clients:
            client.update(self.round)

        d_avg = self._aggregate()
        self._update_global_model(d_avg=d_avg)
        self.trainer.model.load_state_dict(self.model.state_dict())

        logger.info(":::: Validating Model :::: Round : %d", self.round)
        val_auc = self.trainer.val()

        if self.wandb:
            self.wandb.log(
                {"Server LR": self.trainer.optimizer.param_groups[0]["lr"]},
                step=self.round,
            )

        if val_auc > self.best_val_auc:
            self.best_val_auc = val_auc
            self.best_epoch = self.round
            self.trainer.save_best(self.round)
            self.trainer.save_log()

        logger.info(":::: Testing Model :::: Round : %d", self.round)
        self.trainer.test()
        self.round += 1
        self.trainer.cur_epoch = self.round