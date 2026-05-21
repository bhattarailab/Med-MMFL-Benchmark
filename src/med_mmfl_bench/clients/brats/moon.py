"""MOON client for BraTS multimodal segmentation."""

import copy
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from med_mmfl_bench.clients.brats.fedavg import FedavgClient
from med_mmfl_bench.models import get_model
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)


class MoonClient(FedavgClient):
    """MOON client — per-modality contrastive loss across 4 MRI modalities."""

    def __init__(
        self, args: Any, config: Any, client_id: int,
        training_set: Any, test_set: Any, wandb: Any = None,
    ) -> None:
        super().__init__(args, config, client_id, training_set, test_set, wandb)
        self.mu: float = float(self.config.train.moon_mu)
        self.tempr: float = float(self.config.train.moon_tempr)

        self.global_model = get_model(self.config.model.name, self.config.model)
        self.prev_net = copy.deepcopy(self.global_model).cpu()
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.crossentropy = nn.CrossEntropyLoss()

    def _flatten_and_normalise(
        self, feats: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Flatten spatial dims and L2-normalise each modality representation."""
        return [F.normalize(f.view(f.size(0), -1), dim=-1) for f in feats]

    def _compute_moon_contrastive(
        self, local_reprs: List[torch.Tensor],
        global_reprs: List[torch.Tensor],
        prev_reprs: List[torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        """Compute per-modality MOON contrastive loss."""
        local_n = self._flatten_and_normalise(local_reprs)
        global_n = self._flatten_and_normalise(global_reprs)
        prev_n = self._flatten_and_normalise(prev_reprs)

        dummy_labels = torch.zeros(batch_size, dtype=torch.long, device="cuda")
        l_con = torch.tensor(0.0, device="cuda")

        for loc, glob, prev in zip(local_n, global_n, prev_n):
            pos = self.cosine_similarity(loc, glob).view(-1, 1)
            neg = self.cosine_similarity(loc, prev).view(-1, 1)
            logits = torch.cat([pos, neg], dim=1) / self.tempr
            l_con += self.crossentropy(logits, dummy_labels)

        return l_con

    def update(self, comm_round: int = 0) -> Dict[str, float]:
        """Run MOON local training with per-modality contrastive loss."""
        self.model.train()
        self.model.to(self.device)
        self._create_optimizer_and_scaler()

        logger.info("Client %s (MOON) training, round %d", self.client_id, comm_round)
        for i in range(self.config.train.local_epoch):
            epoch_loss: List[float] = []
            with tqdm(self.train_loader, unit="batch", desc=f"Client {self.client_id} (MOON)") as tepoch:
                for x, _, target, mask, _ in tepoch:
                    self.optimizer.zero_grad()
                    x = x.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                    mask = mask.cuda(non_blocking=True)

                    self.model.is_training = True
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        fuse_pred, representations, sep_preds, prm_preds = self.model(x, mask)
                        loss = self._compute_brats_loss(
                            fuse_pred, sep_preds, prm_preds, target, comm_round,
                        )

                    # Global and previous model representations (no grad)
                    with torch.no_grad():
                        self.global_model.cuda()
                        self.global_model.is_training = True
                        _, repr_g, _, _ = self.global_model(x, mask)
                        self.global_model.cpu()

                        self.prev_net.cuda()
                        self.prev_net.is_training = True
                        _, repr_prev, _, _ = self.prev_net(x, mask)
                        self.prev_net.cpu()

                    # MOON contrastive loss across all 4 modalities
                    l_con = self._compute_moon_contrastive(
                        representations, repr_g, repr_prev, x.size(0),
                    )
                    loss += self.mu * l_con

                    self.grad_scaler.scale(loss).backward()
                    if self.config.train.grad_clip > 0:
                        self.grad_scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.train.grad_clip,
                        )
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()

                    #cleanup to save memory
                    del fuse_pred, sep_preds, prm_preds, x, target, mask
                    torch.cuda.empty_cache()

                    tepoch.set_postfix(loss=loss.item())
                    if torch.isfinite(loss):
                        epoch_loss.append(loss.item())

            avg_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0.0
            logger.info(
                "[CLIENT %s] round %d, epoch %d/%d, avg_loss=%.4f",
                self.client_id, comm_round, i + 1, self.config.train.local_epoch, avg_loss,
            )
            if self.wandb:
                self.wandb.log({f"client/{self.client_id}/train_loss": avg_loss})

        self.model.cpu()
        self.prev_net.load_state_dict(self.model.state_dict())
        return {"loss": loss.item()}
