"""FedAvg client for MIMIC-CXR multimodal classification."""

import copy
import os
from typing import Any, Dict

import torch

from med_mmfl_bench.clients import BaseClient
from med_mmfl_bench.trainers.mimiccxr import ClassificationTrainer as MimicClientTrainer
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)


class FedavgClient(BaseClient):
    """FedAvg client — delegates training to a ClassificationTrainer.

    Args:
        args: CLI arguments (must include ``exp_dir``).
        config: YAML configuration object.
        client_id: Integer identifier for this client.
        wandb: Optional WandB run, or ``None``.
    """

    def __init__(
        self, args: Any, config: Any, client_id: int, wandb: Any = None,
    ) -> None:
        self.args = args
        self.config = config
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wandb = wandb

        self.save_dir = os.path.join(self.args.exp_dir, f"client_{self.client_id}")
        os.makedirs(self.save_dir, exist_ok=True)

        self.build_trainer()
        logger.info("Client %d initialised at %s", self.client_id, self.save_dir)

    def build_trainer(self) -> None:
        """Instantiate the underlying trainer."""
        self.trainer = MimicClientTrainer(
            self.args, self.config, self.wandb, client_id=self.client_id,
        )

    def update(self, comm_round: int = 0) -> None:
        """Run one local training round."""
        logger.info("[CLIENT %d] [COMM: %d] started.", self.client_id, comm_round)
        self.trainer.run(comm_round)
        logger.info("[CLIENT %d] [COMM: %d] completed.", self.client_id, comm_round)

    def download(self, model: torch.nn.Module) -> None:
        """Load global model weights into the local trainer."""
        self.trainer.model.load_state_dict(model.state_dict())

    def upload(self) -> Dict[str, torch.Tensor]:
        """Return a deep copy of local model weights."""
        return copy.deepcopy(self.trainer.model.state_dict())

    def __len__(self) -> int:
        return len(self.trainer.train_set)

    def __repr__(self) -> str:
        return f"CLIENT <{self.client_id}>"