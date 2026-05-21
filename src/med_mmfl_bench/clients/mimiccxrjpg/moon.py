"""MOON client for MIMIC-CXR multimodal classification."""

import copy
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from med_mmfl_bench.clients.mimiccxrjpg.fedavg import FedavgClient
from med_mmfl_bench.models import get_model
from med_mmfl_bench.trainers.mimiccxr import ClassificationTrainer as MimicClientTrainer
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)


class MimicMoonTrainer(MimicClientTrainer):
    """MIMIC-CXR trainer with MOON contrastive loss (client-side)."""

    def __init__(
        self, args: Any, config: Any, wandb: Any = None, client_id: int = -1,
    ) -> None:
        super().__init__(args, config, wandb, client_id=client_id)
        self.mu = float(config.train.moon_mu)
        self.tempr = float(self.config.train.moon_tempr)

        self.global_model = get_model(self.config.model.name, self.config.model)
        self.prev_net = copy.deepcopy(self.global_model).cpu()

        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.crossentropy = nn.CrossEntropyLoss()

    def moon_loss(
        self, images: torch.Tensor, text: Any,
        output: dict, bs: int,
    ) -> torch.Tensor:
        """Compute MOON contrastive loss."""
        self.global_model.cuda()
        with torch.no_grad():
            output_g = self.global_model(self.tokenizer, images, text)
        self.global_model.cpu()

        self.prev_net.cuda()
        with torch.no_grad():
            output_p = self.prev_net(self.tokenizer, images, text)
        self.prev_net.cpu()

        op = F.normalize(
            torch.cat((output["image_features"], output["caption_features"]), dim=1),
            dim=-1,
        )
        op_g = F.normalize(
            torch.cat((output_g["image_features"], output_g["caption_features"]), dim=1),
            dim=-1,
        )
        op_prev = F.normalize(
            torch.cat((output_p["image_features"], output_p["caption_features"]), dim=1),
            dim=-1,
        )

        logits = self.cosine_similarity(op, op_g).view(-1, 1)
        logits_prev = self.cosine_similarity(op, op_prev).view(-1, 1)
        logits = torch.cat((logits, logits_prev), dim=1) / self.tempr

        dummy_labels = torch.zeros(bs, dtype=torch.long, device="cuda")
        return self.crossentropy(logits, dummy_labels)

    def train_epoch(self) -> None:
        """One training epoch with MOON contrastive regularisation."""
        self.model.train()
        self.model.cuda()
        with tqdm(self.train_loader, unit="batch", desc="Training (MOON)") as tepoch:
            for frames, label, text, _ in tepoch:
                self.optimizer.zero_grad()
                images = frames.cuda()
                label = label.cuda()

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = self.model(self.tokenizer, images, text)
                    loss = self.criterion(output["logits"], label)
                    loss += self.moon_loss(images, text, output, label.size(0)) * self.mu

                self.grad_scaler.scale(loss).backward()

                if self.config.train.grad_clip > 0:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad.clip_grad_norm_(
                        self.model.parameters(), self.config.train.grad_clip,
                    )

                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                tepoch.set_postfix(Loss=loss.item())

    def run(self, comms: int) -> None:
        """Train locally and update the previous model snapshot."""
        super().run(comms)
        self.prev_net.load_state_dict(self.model.state_dict())


class MoonClient(FedavgClient):
    """MOON client — swaps in a contrastive trainer."""

    def __init__(
        self, args: Any, config: Any, client_id: int, wandb: Any = None,
    ) -> None:
        super().__init__(args, config, client_id, wandb)

    def build_trainer(self) -> None:
        """Instantiate the MOON trainer."""
        self.trainer = MimicMoonTrainer(
            self.args, self.config, self.wandb, client_id=self.client_id,
        )