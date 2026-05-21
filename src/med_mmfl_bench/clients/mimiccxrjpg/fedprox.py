"""FedProx client for MIMIC-CXR multimodal classification."""

from typing import Any

import torch
from tqdm import tqdm

from med_mmfl_bench.clients.mimiccxrjpg.fedavg import FedavgClient
from med_mmfl_bench.models import get_model
from med_mmfl_bench.trainers.mimiccxr import ClassificationTrainer as MimicClientTrainer
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)


class MimicProxTrainer(MimicClientTrainer):
    """MIMIC-CXR trainer with FedProx proximal regularisation (client-side)."""

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
        device = next(self.model.parameters()).device
        global_params = [p.to(device) for p in self.global_model.parameters()]

        prox_term = torch.tensor(0.0, device=device)
        with torch.no_grad():
            for local_w, global_w in zip(self.model.parameters(), global_params):
                prox_term += torch.square((local_w - global_w).norm(2))

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
                    loss = self.criterion(output["logits"], label) + self.prox_loss()

                self.grad_scaler.scale(loss).backward()

                if self.config.train.grad_clip > 0:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad.clip_grad_norm_(
                        self.model.parameters(), self.config.train.grad_clip,
                    )

                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                tepoch.set_postfix(Loss=loss.item())


class FedproxClient(FedavgClient):
    """FedProx client — swaps in a proximal trainer."""

    def __init__(
        self, args: Any, config: Any, client_id: int, wandb: Any = None,
    ) -> None:
        super().__init__(args, config, client_id, wandb)

    def build_trainer(self) -> None:
        """Instantiate the FedProx trainer."""
        self.trainer = MimicProxTrainer(
            self.args, self.config, self.wandb, client_id=self.client_id,
        )