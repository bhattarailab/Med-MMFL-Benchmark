"""FedAvg server for SymileMIMIC multimodal retrieval.

Orchestrates federated averaging across SymileMIMIC clients.  Training,
validation, and zero-shot retrieval evaluation are delegated to the
:class:`~med_mmfl_bench.trainers.symilemimictrainer.ClientTrainer`.
"""

import gc
import os
import pickle
from importlib import import_module
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torchmetrics import MetricCollection

from med_mmfl_bench.losses import get_criterion
from med_mmfl_bench.metrics import get_metrics
from med_mmfl_bench.models import get_model
from med_mmfl_bench.trainers.symilemimictrainer import ClientTrainer as SymileClientTrainer
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)


class FedavgServer:
    """FedAvg server for SymileMIMIC multimodal retrieval.

    Args:
        args: CLI arguments (must include ``exp_dir``, ``name``, ``algorithm``).
        config: YAML configuration object.
        val_dataset: Validation dataset.
        test_dataset: Test dataset.
        client_datasets: List of per-client dataset dicts.
        server_trainset: Optional server-side training dataset.
        wandb: Optional WandB run instance, or ``False``.
    """

    def __init__(
        self, args: Any, config: Any,
        val_dataset: Any, test_dataset: Any,
        client_datasets: List[Dict[str, Any]],
        server_trainset: Any = None,
        wandb: Any = False,
    ) -> None:
        self.args = args
        self.config = config
        self.wandb = wandb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.round: int = 0
        self.val_track: List[float] = []
        self.best_epoch: int = 0
        self.best_val_loss: float = float("inf")

        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_dataset = server_trainset

        self._init_model()
        self.evaluator = self._build_evaluator().to(self.device)
        self.trainer = SymileClientTrainer(args, config, wandb)

        self.curr_lr = self.config.optimizer.learning_rate
        self.clients = self._create_clients(client_datasets)
        self.val_loader, self.test_loader = self._get_dataloader()
        self.num_of_clients = len(self.clients)

        self._get_algorithm()
        self.set_trainer()
        self.save_dir = os.path.join(self.args.exp_dir, self.args.name, "server")
        os.makedirs(self.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def set_trainer(self) -> None:
        """Configure the server-side trainer with model/criterion/loaders."""
        self.trainer = SymileClientTrainer(self.args, self.config, self.wandb)
        self.trainer.model = get_model(self.config.model.name, self.config.model)
        self.trainer.criterion = get_criterion(self.config.criterion.name)
        self.trainer.optimizer = self.server_optimizer
        self.trainer.val_loader = self.val_loader
        self.trainer.test_loader = self.test_loader
        self.trainer.train_loader = (
            self._create_dataloader(self.train_dataset, shuffle=True)
            if self.train_dataset is not None else None
        )

    def _init_model(self) -> None:
        """Build the global model and criterion."""
        self.model = get_model(self.config.model.name, self.config.model)
        self.model.cpu()
        self.criterion = get_criterion(self.config.criterion.name)

    def _get_dataloader(self) -> tuple:
        """Create validation and test DataLoaders."""
        val_loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.dataloader.batch_size_val,
            shuffle=False,
            num_workers=self.config.dataloader.num_workers,
            pin_memory=False,
            drop_last=False,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config.dataloader.batch_size_test,
            shuffle=False,
            num_workers=self.config.dataloader.num_workers,
            pin_memory=False,
            drop_last=False,
        )
        return val_loader, test_loader

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
        """Build a MetricCollection from the config."""
        if not isinstance(self.config.metrics, list):
            self.config.metrics = [self.config.metrics]
        logger.info("Metrics: %s", self.config.metrics)
        return MetricCollection(get_metrics(self.config.metrics)).to(self.device)

    def _get_algorithm(self, **kwargs: Any) -> Any:
        """Dynamically load the FL algorithm optimizer."""
        algo_module = import_module(
            f"med_mmfl_bench.algorithms.{self.args.algorithm}",
            package=__package__,
        )
        cls_name = f"{self.args.algorithm.title()}Optimizer"
        self.server_optimizer = algo_module.__dict__[cls_name](model=self.model, **kwargs)
        return self.server_optimizer

    def _create_clients(self, client_datasets: List[Dict[str, Any]]) -> List[Any]:
        """Instantiate per-client objects."""
        client_module = import_module(
            f"med_mmfl_bench.clients.symilemimic.{self.args.algorithm}",
            package=__package__,
        )
        cls_name = f"{self.args.algorithm.title()}Client"
        client_cls = client_module.__dict__[cls_name]

        clients: List[Any] = []
        logger.info("Creating %d clients.", len(client_datasets))
        for datasets in client_datasets:
            client = client_cls(
                args=self.args, config=self.config,
                client_id=datasets["client_id"],
                training_set=datasets["train_set"],
                test_set=datasets["val_set"],
                wandb=self.wandb,
            )
            client.id = datasets["client_id"]
            clients.append(client)
        return clients

    # ------------------------------------------------------------------
    # Federation protocol
    # ------------------------------------------------------------------

    def _aggregate(self) -> Dict[str, torch.Tensor]:
        """Collect client models and compute a weighted average."""
        logger.info(
            "[%s] [Round: %s] Aggregate updated signals!",
            self.args.algorithm.upper(), str(self.round).zfill(4),
        )
        omega: List[int] = []
        w: List[Dict[str, torch.Tensor]] = []

        for client in self.clients:
            omega.append(len(client))
            w.append(client.upload())

        return self.server_optimizer.aggregate(
            client_models=w, omega=np.array(omega),
        )

    def dispatch(self) -> None:
        """Broadcast the current global model to all clients."""
        for client in self.clients:
            client.download(self.model)

    def update(self) -> None:
        """Execute one communication round of federated learning."""
        self.dispatch()
        for client in self.clients:
            client.update(self.round)

        self.model.load_state_dict(self._aggregate())
        self.trainer.model.load_state_dict(self.model.state_dict())

        logger.info(":::: Validating Model :::: Round : %d", self.round)
        val_loss = self.trainer.val()
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = self.round
            self.trainer.save(self.round)

        logger.info(":::: Testing Model :::: Round : %d", self.round)
        self.trainer.test()
        self.round += 1

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_best(self, round_num: int) -> None:
        """Save the best global model checkpoint."""
        ckpt_path = os.path.join(self.save_dir, "model_best.pth")
        torch.save({"net": self.model.state_dict(), "comms": round_num}, ckpt_path)
        logger.info(
            "[%s] [%s] [Round: %s] Saved best checkpoint.",
            self.args.algorithm.upper(),
            self.config.dataset.dset_name.upper(),
            str(round_num).zfill(4),
        )

    def load_best(self) -> None:
        """Load the best model checkpoint."""
        ckpt_path = os.path.join(self.save_dir, "model_best.pth")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["net"])
        logger.info("Best model at comms round: %s", checkpoint["comms"])

    def save_log(self) -> None:
        """Persist validation metrics to disk."""
        log_path = os.path.join(self.save_dir, "val_metrics.pkl")
        with open(log_path, "wb") as f:
            pickle.dump(self.val_track, f)

    def evaluate(self) -> None:
        """Evaluate the global model (placeholder)."""
        return None

    def finalize(self) -> None:
        """Finalize training (placeholder)."""
        return None