"""FedAvg client for BraTS multimodal segmentation."""

import copy
import gc
from typing import Any, Dict, List

import torch
from torch import nn
from torchmetrics import MetricCollection
from tqdm import tqdm

from med_mmfl_bench.clients import BaseClient
from med_mmfl_bench.losses import get_criterion, criterions
from med_mmfl_bench.metrics import get_metrics
from med_mmfl_bench.models import get_model
from med_mmfl_bench.utils.logging import get_logger
from med_mmfl_bench.utils.optimizers import get_optimizer

logger = get_logger(__name__)


class FedavgClient(BaseClient):
    """FedAvg client for BraTS multimodal segmentation.

    Each client trains a local BraTS segmentation model using the
    fused + separable + prm loss scheme.

    Args:
        args: CLI arguments.
        config: YAML configuration object.
        client_id: Integer identifier for this client.
        training_set: Client-specific training dataset.
        test_set: Client-specific validation/test dataset.
        wandb: Optional WandB run, or ``None``.
    """

    def __init__(
        self, args: Any, config: Any, client_id: int,
        training_set: Any, test_set: Any, wandb: Any = None,
    ) -> None:
        super().__init__()
        self.args = args
        self.config = config
        self.client_id = client_id
        self.training_set = training_set
        self.test_set = test_set
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wandb = wandb

        self.train_loader = self._create_dataloader(self.training_set, shuffle=True)
        self.test_loader = self._create_dataloader(self.test_set, shuffle=False)

        self._build_model()
        self.evaluator = self._build_evaluator()

    def set_id(self, client_id: Any) -> None:
        """Set the client identifier."""
        self.id = client_id

    def _build_model(self) -> None:
        """Build the local model, criterion, optimizer, and grad scaler."""
        self.model = get_model(self.config.model.name, self.config.model).to(self.device)
        self.criterion = get_criterion(self.config.criterion.name)
        self.optimizer = get_optimizer(
            self.config.optimizer.name, self.model.parameters(), self.config.optimizer,
        )
        self.grad_scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    def _build_evaluator(self) -> MetricCollection:
        """Build metrics from config."""
        if not isinstance(self.config.metrics, list):
            self.config.metrics = [self.config.metrics]
        logger.info("Metrics: %s", self.config.metrics)
        return MetricCollection(get_metrics(self.config.metrics)).to(self.device)

    def _create_dataloader(self, dataset: Any, shuffle: bool) -> torch.utils.data.DataLoader:
        """Create a DataLoader for the given dataset."""
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.config.dataloader.batch_size,
            shuffle=shuffle,
            num_workers=self.config.dataloader.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def _create_optimizer_and_scaler(self) -> None:
        """Re-create optimizer and grad scaler (called each round)."""
        del self.optimizer
        del self.grad_scaler
        gc.collect()
        torch.cuda.empty_cache()
        self.optimizer = get_optimizer(
            self.config.optimizer.name, self.model.parameters(), self.config.optimizer,
        )
        self.grad_scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    def _compute_brats_loss(
        self, fuse_pred: torch.Tensor, sep_preds: List[torch.Tensor],
        prm_preds: List[torch.Tensor], target: torch.Tensor, comm_round: int,
    ) -> torch.Tensor:
        """Compute the fused + separable + prm BraTS loss."""
        fuse_loss = (
            criterions.softmax_weighted_loss(fuse_pred, target, num_cls=4)
            + criterions.dice_loss(fuse_pred, target, num_cls=4)
        )
        sep_loss = sum(
            criterions.softmax_weighted_loss(sp, target, num_cls=4)
            + criterions.dice_loss(sp, target, num_cls=4)
            for sp in sep_preds
        )
        prm_loss = sum(
            criterions.softmax_weighted_loss(pp, target, num_cls=4)
            + criterions.dice_loss(pp, target, num_cls=4)
            for pp in prm_preds
        )
        if comm_round < self.config.model.region_fusion_epochs:
            return fuse_loss * 0.0 + sep_loss + prm_loss
        return fuse_loss + sep_loss + prm_loss

    def update(self, comm_round: int = 0) -> Dict[str, float]:
        """Run one local training round."""
        self.model.train()
        self.model.to(self.device)
        self._create_optimizer_and_scaler()

        logger.info("Client %s training, round %d", self.client_id, comm_round)
        for i in range(self.config.train.local_epoch):
            epoch_loss: List[float] = []
            with tqdm(self.train_loader, unit="batch", desc=f"Client {self.client_id}") as tepoch:
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
        
        return {"loss": loss.item()}

    def download(self, model: torch.nn.Module) -> None:
        """Receive the global model."""
        self.model = copy.deepcopy(model)

    def upload(self) -> Dict[str, torch.Tensor]:
        """Return a deep copy of local model weights."""
        return copy.deepcopy(self.model.state_dict())

    def __len__(self) -> int:
        return len(self.training_set)

    def __repr__(self) -> str:
        return f"CLIENT <{self.id}>"