"""FedNova client for MIMIC-CXR multimodal classification."""

import copy
import gc
from typing import Any, Dict, Tuple

import torch
from tqdm import tqdm

from med_mmfl_bench.clients.mimiccxrjpg.fedavg import FedavgClient
from med_mmfl_bench.trainers.mimiccxr import ClassificationTrainer as MimicClientTrainer
from med_mmfl_bench.utils import get_lr_scheduler
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)


class MimicNovaTrainer(MimicClientTrainer):
    """MIMIC-CXR trainer with FedNova normalised gradient tracking (client-side)."""

    def __init__(
        self, args: Any, config: Any, wandb: Any = None, client_id: int = -1,
    ) -> None:
        super().__init__(args, config, wandb, client_id=client_id)
        self.ai: float = 0.0
        self.di: Dict[str, torch.Tensor] = {}
        self.rho: float = float(self.config.optimizer.momentum)

    def run(self, comms: int) -> None:
        """Run FedNova local training with step normalisation."""
        self.model.cuda()
        self.global_weights = copy.deepcopy(self.model.state_dict())
        self.tau = 0

        for i in range(self.config.train.local_epoch):
            if self.client_id == -1:
                logger.info(
                    "Server — round %d, local_epoch %d, round_epoch %d",
                    comms, self.cur_epoch, i,
                )
            else:
                logger.info(
                    "Client %d — local_epoch %d, round %d, round_epoch %d",
                    self.client_id, self.local_epoch, comms, i,
                )

            self.train_epoch()
            val_auc = self.val()
            self.lr_scheduler.step(val_auc)

            if self.client_id == -1:
                self.cur_epoch += 1
            else:
                self.local_epoch += 1

        self.model.cpu()
        gc.collect()

    def val(self) -> float:
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


class FednovaClient(FedavgClient):
    """FedNova client — uploads normalised gradient direction."""

    def __init__(
        self, args: Any, config: Any, client_id: int, wandb: Any = None,
    ) -> None:
        super().__init__(args, config, client_id, wandb)

    def build_trainer(self) -> None:
        """Instantiate the FedNova trainer with LR scheduler."""
        self.trainer = MimicNovaTrainer(
            self.args, self.config, self.wandb, client_id=self.client_id,
        )
        self.trainer.lr_scheduler = get_lr_scheduler(
            self.config.lr_scheduler.name,
            self.trainer.optimizer,
            self.config.lr_scheduler,
        )

    def upload(self) -> Tuple[float, Dict[str, torch.Tensor]]:
        """Return the normalisation coefficient and gradient direction."""
        return self.trainer.ai, self.trainer.di