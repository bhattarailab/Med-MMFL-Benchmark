"""FedAvg server for MIMIC-CXR multimodal classification.

Orchestrates federated averaging across MIMIC-CXR clients.  The server
delegates data loading and model management to a
:class:`~med_mmfl_bench.trainers.mimiccxr.ClassificationTrainer`, keeping
this module focused purely on the federation protocol.
"""

import os
import pickle
from importlib import import_module
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from med_mmfl_bench.models import get_model
from med_mmfl_bench.losses import get_criterion
from med_mmfl_bench.trainers.mimiccxr import ClassificationTrainer as MimicClientTrainer
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)


class FedavgServer:
    """FedAvg server for MIMIC-CXR multimodal classification.

    Args:
        args: Parsed CLI arguments (must include ``exp_dir``, ``name``,
            ``num_clients``, ``algorithm``).
        config: YAML configuration object.
        wandb: Optional WandB run instance, or ``False`` to disable.
    """

    def __init__(self, args: Any, config: Any, wandb: Any = False) -> None:
        self.args = args
        self.config = config
        self.wandb = wandb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.round: int = 0
        self.val_track: List[float] = []
        self.best_epoch: int = 0
        self.best_val_auc: float = 0.0

        self._init_model()
        self.trainer = MimicClientTrainer(args, config, wandb, client_id=-1)
        self.clients = self._create_clients()
        self._get_algorithm()

        self.save_dir = os.path.join(self.args.exp_dir, self.args.name, "server")
        os.makedirs(self.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_model(self) -> None:
        """Build the global model and criterion."""
        self.model = get_model("mimic_mmclf", self.config.model)
        self.model.cpu()
        self.criterion = get_criterion(self.config.criterion.name)

    def _get_algorithm(self, **kwargs: Any) -> Any:
        """Dynamically load the FL algorithm optimizer."""
        algo_module = import_module(
            f"med_mmfl_bench.algorithms.{self.args.algorithm}",
            package=__package__,
        )
        cls_name = f"{self.args.algorithm.title()}Optimizer"
        algorithm_cls = algo_module.__dict__[cls_name]
        self.server_optimizer = algorithm_cls(model=self.model, **kwargs)
        return self.server_optimizer

    def _create_clients(self) -> List[Any]:
        """Instantiate per-client objects based on the chosen algorithm."""
        client_module = import_module(
            f"med_mmfl_bench.clients.mimiccxrjpg.{self.args.algorithm}",
            package=__package__,
        )
        cls_name = f"{self.args.algorithm.title()}Client"
        client_cls = client_module.__dict__[cls_name]

        clients: List[Any] = []
        logger.info("Creating %d clients.", self.args.num_clients)
        for idx in range(int(self.args.num_clients)):
            client = client_cls(
                args=self.args, config=self.config,
                client_id=idx, wandb=self.wandb,
            )
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
            client.download(self.trainer.model)

    def update(self) -> None:
        """Execute one communication round of federated learning."""
        self.dispatch()

        for client in self.clients:
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

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_best(self, round_num: int) -> None:
        """Save the best global model checkpoint."""
        ckpt_path = os.path.join(self.save_dir, "model_best.pth")
        torch.save({"net": self.model.state_dict(), "comms": round_num}, ckpt_path)
        logger.info(
            "[%s] [%s] [Round: %s] Saved best global checkpoint.",
            self.args.algorithm.upper(),
            self.config.dataset.dset_name.upper(),
            str(round_num).zfill(4),
        )

    def load_best(self) -> None:
        """Load the best global model checkpoint."""
        ckpt_path = os.path.join(self.save_dir, "model_best.pth")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["net"])
        logger.info("Best model at comms round: %s", checkpoint["comms"])

    def save_log(self) -> None:
        """Persist validation metrics to disk."""
        log_path = os.path.join(self.save_dir, "val_metrics.pkl")
        with open(log_path, "wb") as f:
            pickle.dump(self.val_track, f)

    # ------------------------------------------------------------------
    # Evaluation & finalisation
    # ------------------------------------------------------------------

    def evaluate(self) -> None:
        """Evaluate the best global model on the test set."""
        self.trainer.load_best()
        self.trainer.test()