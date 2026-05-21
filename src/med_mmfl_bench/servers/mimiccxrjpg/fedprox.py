"""FedProx server for MIMIC-CXR multimodal classification.

Extends FedAvg with a proximal regularisation term that penalises
deviation of the local model from the global model during training.
"""

import copy
from typing import Any

import torch
from tqdm import tqdm

from med_mmfl_bench.models import get_model
from med_mmfl_bench.servers.mimiccxrjpg.fedavg import FedavgServer
from med_mmfl_bench.trainers.mimiccxr import ClassificationTrainer as MimicClientTrainer
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)


class MimicProxTrainer(MimicClientTrainer):
    """MIMIC-CXR trainer with FedProx proximal loss."""

    def __init__(
        self, args: Any, config: Any, wandb: Any = None, client_id: int = -1,
    ) -> None:
        super().__init__(args, config, wandb, client_id=client_id)
        self._mu = float(config.train.mu)
        self.global_model = get_model(self.config.model.name, self.config.model)

    def prox_loss(self) -> torch.Tensor:
        """Compute the proximal term ||w - w_global||^2."""
        self.global_model.cuda()
        self.model.cuda()
        prox_term = torch.tensor(0.0, device="cuda")

        for k, k_t in zip(
            self.model.state_dict().keys(),
            self.global_model.state_dict().keys(),
        ):
            if "weight" in k or "bias" in k:
                prox_term += torch.norm(
                    self.model.state_dict()[k] - self.global_model.state_dict()[k_t],
                    p=2,
                ) ** 2
        self.global_model.cpu()
        return (self._mu * 0.5) * prox_term

    def train_epoch(self) -> None:
        """One training epoch with FedProx proximal regularisation."""
        self.model.train()
        self.model.cuda()
        with tqdm(self.train_loader, unit="batch", desc="Training (FedProx)") as tepoch:
            for frames, label, text, _ in tepoch:
                self.optimizer.zero_grad()
                images = frames.cuda()
                label = label.cuda()

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = self.model(self.tokenizer, images, text)
                    loss = self.criterion(output["logits"], label)
                    loss += self.prox_loss()

                self.grad_scaler.scale(loss).backward()

                if self.config.train.grad_clip > 0:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad.clip_grad_norm_(
                        self.model.parameters(), self.config.train.grad_clip,
                    )

                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                tepoch.set_postfix(Loss=loss.item())


class FedproxServer(FedavgServer):
    """FedProx server — inherits FedAvg and swaps in a proximal trainer."""

    def __init__(self, args: Any, config: Any, wandb: Any = False) -> None:
        super().__init__(args, config, wandb)
        self.trainer = MimicProxTrainer(self.args, self.config, self.wandb)

    def update(self) -> None:
        """Execute one FedProx communication round."""
        self.dispatch()

        for client in self.clients:
            client.trainer.global_model.load_state_dict(
                self.trainer.global_model.state_dict()
            )
            client.update(self.round)

        self.model.load_state_dict(self._aggregate())
        self.trainer.model.load_state_dict(self.model.state_dict())

        logger.info(":::: Validating Model :::: Round : %d", self.round)
        val_auc = self.trainer.val()
        if val_auc > self.best_val_auc:
            self.best_val_auc = val_auc
            self.best_epoch = self.round
            self.trainer.save_best(self.round)
            self.trainer.save_log()

        logger.info(":::: Testing Model :::: Round : %d", self.round)
        self.trainer.test()
        self.round += 1
        self.trainer.cur_epoch = self.round