"""FedAvg client for EHRXQA generative visual question answering.

Handles local training with BLIP-based VQA models, gradient scaling,
and model upload/download for federated learning.
"""

import copy
import gc
from typing import Any, Dict, Optional

import torch
from torch import nn
from torchmetrics import MetricCollection
from tqdm import tqdm

from med_mmfl_bench.clients.base import BaseClient
from med_mmfl_bench.losses import get_criterion
from med_mmfl_bench.metrics import get_metrics
from med_mmfl_bench.models import get_model
from med_mmfl_bench.utils.logging import get_logger
from med_mmfl_bench.utils.optimizers import get_optimizer

logger = get_logger(__name__)

__all__ = ["FedavgClient"]


class FedavgClient(BaseClient):
    """FedAvg client for EHRXQA federated VQA training.

    Each client trains a local copy of the global BLIP model on its
    private partition and uploads the updated state dict.

    Args:
        args: Experiment arguments.
        config: Full configuration object.
        client_id: Unique client identifier.
        training_set: Client's training dataset (or Subset).
        test_set: Client's validation dataset (or Subset).
        wandb: WandB run object or None.
    """

    def __init__(
        self,
        args: Any,
        config: Any,
        client_id: Any,
        training_set: Any,
        test_set: Any,
        wandb: Any = None,
    ) -> None:
        super().__init__()
        self.args = args
        self.config = config
        self.client_id = client_id
        self.training_set = training_set
        self.test_set = test_set
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_loader = self._create_dataloader(self.training_set, shuffle=True)
        self.test_loader = self._create_dataloader(self.test_set, shuffle=False)

        self._build_model()
        self.evaluator = self._build_evaluator()
        self.wandb = wandb

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> None:
        """Initialize model, criterion, optimizer, and gradient scaler."""
        self.model = get_model(self.config.model.name, self.config.model)
        self.criterion = get_criterion(self.config.criterion.name)
        self.optimizer = get_optimizer(
            self.config.optimizer.name,
            self.model.parameters(),
            self.config.optimizer,
        )
        self.grad_scaler = torch.cuda.amp.GradScaler()

    def _build_evaluator(self) -> MetricCollection:
        """Build a MetricCollection from the configured metric names."""
        if not isinstance(self.config.metrics, list):
            self.config.metrics = [self.config.metrics]
        metric_map = get_metrics(self.config.metrics)
        return MetricCollection(metric_map).to(self.device)

    def _create_dataloader(
        self, dataset: Any, shuffle: bool
    ) -> torch.utils.data.DataLoader:
        """Create a DataLoader for the given dataset.

        Args:
            dataset: PyTorch Dataset or Subset.
            shuffle: Whether to shuffle the data.

        Returns:
            Configured DataLoader.
        """
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.config.dataloader.batch_size,
            shuffle=shuffle,
            num_workers=self.config.dataloader.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def _create_optimizer_and_scaler(self) -> None:
        """Re-initialize optimizer and gradient scaler.

        Called at the start of each communication round to ensure a
        fresh optimizer state.
        """
        del self.optimizer
        del self.grad_scaler
        gc.collect()
        torch.cuda.empty_cache()
        self.optimizer = get_optimizer(
            self.config.optimizer.name,
            self.model.parameters(),
            self.config.optimizer,
        )
        self.grad_scaler = torch.cuda.amp.GradScaler()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def update(self, comm_round: int = 0) -> Dict[str, float]:
        """Perform local training for one communication round.

        Args:
            comm_round: Current communication round number.

        Returns:
            Dictionary with ``"loss"`` key containing the final batch loss.
        """
        self.model.train()
        self.model.to(self.device)
        self._create_optimizer_and_scaler()

        loss_value = 0.0
        logger.info("[CLIENT %s] Starting local training...", self.client_id)

        for epoch in range(self.config.train.local_epoch):
            with tqdm(self.train_loader, unit="batch") as tepoch:
                for batch in tepoch:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    self.optimizer.zero_grad()

                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self.model(**batch)
                        loss = outputs.loss

                    if torch.isfinite(loss):
                        self.grad_scaler.scale(loss).backward()

                        if self.config.train.grad_clip > 0:
                            self.grad_scaler.unscale_(self.optimizer)
                            nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.train.grad_clip,
                            )

                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()

                        loss_value = loss.item()
                        tepoch.set_postfix(loss=loss_value)
                    else:
                        logger.warning("Skipping update due to non-finite loss")

            logger.info(
                "[CLIENT %s] [COMM: %d] Epoch %d/%d completed.",
                self.client_id,
                comm_round,
                epoch + 1,
                self.config.train.local_epoch,
            )

        self.model.to("cpu")
        return {"loss": loss_value}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the local model on the client's validation set.

        Returns:
            Dictionary with ``"metrics"`` key.
        """
        self.model.eval()
        self.model.to(self.device)

        logger.info("[CLIENT %s] Evaluating...", self.client_id)

        for batch in self.test_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                generated_ids = self.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    max_length=int(self.config.model.max_length),
                )

            preds = self.model.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            refs = self.model.tokenizer.batch_decode(
                batch["labels"], skip_special_tokens=True
            )
            self.evaluator.update(preds, refs)

        metrics = self.evaluator.compute()
        for name, value in metrics.items():
            logger.info("  %s: %s", name, value)

        if self.wandb:
            self.wandb.log({
                f"{self.client_id}/val/{k}": v.item() if hasattr(v, "item") else v
                for k, v in metrics.items()
            })

        self.evaluator.reset()
        self.model.to("cpu")
        return {"metrics": metrics}

    # ------------------------------------------------------------------
    # Model transfer
    # ------------------------------------------------------------------

    def download(self, model: nn.Module) -> None:
        """Download (deep-copy) the global model.

        Args:
            model: Global model to copy.
        """
        self.model = copy.deepcopy(model)

    def upload(self) -> Dict[str, torch.Tensor]:
        """Upload the local model's state dictionary.

        Returns:
            Deep copy of the local model's state dict.
        """
        return copy.deepcopy(self.model.state_dict())

    def __len__(self) -> int:
        return len(self.training_set)

    def __repr__(self) -> str:
        return f"CLIENT <{self.client_id}>"