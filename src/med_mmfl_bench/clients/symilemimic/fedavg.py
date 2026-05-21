"""FedAvg client for SymileMIMIC multimodal retrieval."""

import copy
import os
from typing import Any, Dict

import torch
from torchmetrics import MetricCollection

from med_mmfl_bench.clients import BaseClient
from med_mmfl_bench.losses import get_criterion
from med_mmfl_bench.metrics import get_metrics
from med_mmfl_bench.models import get_model
from med_mmfl_bench.trainers.symilemimictrainer import ClientTrainer as SymileClientTrainer
from med_mmfl_bench.utils.optimizers import get_optimizer
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)


class FedavgClient(BaseClient):
    """FedAvg client for SymileMIMIC multimodal retrieval.

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
        self.args = args
        self.config = config
        self.client_id = client_id
        self.training_set = training_set
        self.test_set = test_set
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wandb = wandb

        self.save_dir = os.path.join(self.args.exp_dir, f"client_{self.client_id}")
        os.makedirs(self.save_dir, exist_ok=True)

        self.train_loader = self._create_dataloader(self.training_set, shuffle=True)
        self.test_loader = self._create_dataloader(self.test_set, shuffle=False)

        self.evaluator = self._build_evaluator()
        self.build_trainer()
        logger.info("Client %d initialised at %s", self.client_id, self.save_dir)

    def _create_dataloader(self, dataset: Any, shuffle: bool) -> torch.utils.data.DataLoader:
        """Create a DataLoader for the given dataset."""
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.config.dataloader.batch_size_train,
            shuffle=shuffle,
            num_workers=self.config.dataloader.num_workers,
            pin_memory=False,
            drop_last=False,
        )

    def _build_evaluator(self) -> MetricCollection:
        """Build metrics from config."""
        if not isinstance(self.config.metrics, list):
            self.config.metrics = [self.config.metrics]
        return MetricCollection(get_metrics(self.config.metrics)).to(self.device)

    def build_trainer(self) -> None:
        """Instantiate the underlying trainer."""
        self.trainer = SymileClientTrainer(self.args, self.config, self.wandb)
        self.trainer.model = get_model(self.config.model.name, self.config.model)
        self.trainer.model.cpu()
        self.trainer.optimizer = get_optimizer(
            self.config.optimizer.name, self.trainer.model.parameters(), self.config.optimizer,
        )
        self.trainer.criterion = get_criterion(self.config.criterion.name)
        self.trainer.train_loader = self.train_loader
        self.trainer.test_loader = self.test_loader
        self.trainer.client_id = self.client_id

    def update(self, comm_round: int = 0) -> None:
        """Run one local training round."""
        logger.info("[CLIENT %d] [COMM: %d] started.", self.client_id, comm_round)
        self.trainer.run_train()
        logger.info("[CLIENT %d] [COMM: %d] completed.", self.client_id, comm_round)

    def download(self, model: torch.nn.Module) -> None:
        """Load global model weights."""
        self.trainer.model.load_state_dict(model.state_dict())

    def upload(self) -> Dict[str, torch.Tensor]:
        """Return a deep copy of local model weights."""
        return copy.deepcopy(self.trainer.model.state_dict())

    def __len__(self) -> int:
        return len(self.training_set)

    def __repr__(self) -> str:
        return f"CLIENT <{self.client_id}>"