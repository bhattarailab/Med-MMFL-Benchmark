"""FedProx client for BraTS multimodal segmentation."""

import copy
from typing import Any, Dict, List

import torch
from torch import nn
from tqdm import tqdm

from med_mmfl_bench.clients.brats.fedavg import FedavgClient
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)


class FedproxClient(FedavgClient):
    """FedProx client — adds a proximal regularisation term to the BraTS loss."""

    def __init__(
        self, args: Any, config: Any, client_id: int,
        training_set: Any, test_set: Any, wandb: Any = None,
    ) -> None:
        super().__init__(args, config, client_id, training_set, test_set, wandb)
        self.prox_mu: float = float(self.config.train.proximal_mu)

    def update(self, comm_round: int = 0) -> Dict[str, float]:
        """Run FedProx local training with proximal loss."""
        global_model = copy.deepcopy(self.model)
        global_model.eval()
        global_model.to(self.device)

        self.model.train()
        self.model.to(self.device)
        self._create_optimizer_and_scaler()

        logger.info("Client %s (FedProx) training, round %d", self.client_id, comm_round)
        for i in range(self.config.train.local_epoch):
            epoch_loss: List[float] = []
            with tqdm(self.train_loader, unit="batch", desc=f"Client {self.client_id} (FedProx)") as tepoch:
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

                        # Proximal term: mu/2 * ||w - w_t||^2
                        prox_term = sum(
                            torch.norm(
                                self.model.state_dict()[k] - global_model.state_dict()[k], p=2,
                            ) ** 2
                            for k in self.model.state_dict()
                            if "weight" in k or "bias" in k
                        )
                        loss += (self.prox_mu / 2) * prox_term



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
        global_model.cpu()
        return {"loss": loss.item()}
