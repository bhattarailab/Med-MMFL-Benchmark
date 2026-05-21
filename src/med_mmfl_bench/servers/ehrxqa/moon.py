"""MOON server for EHRXQA federated VQA.

MOON (Model-Contrastive Federated Learning) modifies the client-side
training with contrastive loss. The server additionally synchronises
the global model copy held by each client before each round.
"""

import gc
from typing import Any, Dict, List

import torch

from med_mmfl_bench.servers.ehrxqa.fedavg import FedavgServer
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["MoonServer"]


class MoonServer(FedavgServer):
    """MOON server for EHRXQA.

    Extends FedAvg by pushing the current global model to each
    client's ``global_model`` attribute before local training.
    """

    def __init__(
        self,
        args: Any,
        config: Any,
        val_dataset: Any,
        test_dataset: Any,
        client_datasets: List[Dict[str, Any]],
        wandb: Any = False,
    ) -> None:
        super().__init__(args, config, val_dataset, test_dataset, client_datasets, wandb)

    def update(self) -> None:
        """Execute one MOON communication round.

        In addition to the standard FedAvg flow, synchronises each
        client's ``global_model`` attribute with the current global
        model before training.
        """
        self.dispatch()

        # Synchronise the reference global model on each client
        for client in self.clients:
            client.global_model.load_state_dict(self.global_model.state_dict())
            client.global_model.cpu()
            client.prev_net.cpu()
            client.model.cpu()

        self._request(eval=False, retain_model=True)

        gc.collect()
        torch.cuda.empty_cache()

        self.global_model.load_state_dict(self._aggregate())
        self.evaluate()
        self.round += 1