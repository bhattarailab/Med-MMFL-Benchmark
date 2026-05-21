"""FedNova client for BraTS multimodal segmentation."""

import copy
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from tqdm import tqdm

from med_mmfl_bench.clients.brats.fedavg import FedavgClient
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)


class FednovaClient(FedavgClient):
    """FedNova client — tracks normalised gradient directions for aggregation."""

    def __init__(
        self, args: Any, config: Any, client_id: int,
        training_set: Any, test_set: Any, wandb: Any = None,
    ) -> None:
        super().__init__(args, config, client_id, training_set, test_set, wandb)
        self.ai: float = 0.0
        self.di: Dict[str, torch.Tensor] = {}
        self.rho: float = float(self.config.optimizer.momentum)

    def update(self, comm_round: int = 0) -> Dict[str, float]:
        """Run FedNova local training with step-normalised gradient tracking."""
        self.model.train()
        self.model.to(self.device)
        global_weights = copy.deepcopy(self.model.state_dict())
        self._create_optimizer_and_scaler()

        logger.info("Client %s (FedNova) training, round %d", self.client_id, comm_round)
        tau = 0
        for i in range(self.config.train.local_epoch):
            epoch_loss: List[float] = []
            with tqdm(self.train_loader, unit="batch", desc=f"Client {self.client_id} (FedNova)") as tepoch:
                for x, _, target, mask, _ in tepoch:
                    self.optimizer.zero_grad()
                    x = x.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                    mask = mask.cuda(non_blocking=True)

                    self.model.is_training = True
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        fuse_pred, _, sep_preds, prm_preds = self.model(x, mask)
                        loss = self._compute_brats_loss(
                            fuse_pred, sep_preds, prm_preds, target, comm_round,
                        )

                    if torch.isfinite(loss):
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
                        epoch_loss.append(loss.item())
                        tau += 1
                    else:
                        logger.warning("Skipping update due to non-finite loss.")

            # FedNova normalisation coefficient
            self.ai = (
                tau - self.rho * (1 - pow(self.rho, tau)) / (1 - self.rho)
            ) / (1 - self.rho)
            state_dict = self.model.state_dict()
            self.di = copy.deepcopy(global_weights)
            for key in self.di:
                self.di[key] = torch.div(global_weights[key] - state_dict[key], self.ai)

            avg_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0.0
            logger.info(
                "[CLIENT %s] round %d, epoch %d/%d, avg_loss=%.4f",
                self.client_id, comm_round, i + 1, self.config.train.local_epoch, avg_loss,
            )
            if self.wandb:
                self.wandb.log({f"client/{self.client_id}/train_loss": avg_loss})

        self.model.cpu()
        return {"loss": loss.item()}

    def upload(self) -> Tuple[float, Dict[str, torch.Tensor]]:
        """Upload normalisation coefficient and gradient direction."""
        return self.ai, self.di
