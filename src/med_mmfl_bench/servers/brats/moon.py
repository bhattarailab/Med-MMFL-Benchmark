"""MOON server for BraTS multimodal segmentation."""

import gc
from typing import Any, Dict, List

import torch

from med_mmfl_bench.servers.brats.fedavg import FedavgServer
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)


class MoonServer(FedavgServer):
    """MOON server — syncs global model to all clients before local updates."""

    def __init__(
        self, args: Any, config: Any,
        val_dataset: Any, test_dataset: Any,
        client_datasets: List[Dict[str, Any]],
        wandb: Any = False,
    ) -> None:
        super().__init__(args, config, val_dataset, test_dataset, client_datasets, wandb)

    def update(self) -> None:
        """Execute one MOON communication round."""
        self.dispatch()

        # Sync global model weights to each client's reference model
        for client in self.clients:
            client.global_model.load_state_dict(self.global_model.state_dict())

        self._request(eval=False)
        gc.collect()
        torch.cuda.empty_cache()

        self.global_model.load_state_dict(self._aggregate())
        self.round += 1