"""FedProx server for SymileMIMIC multimodal retrieval."""

from typing import Any, Dict, List

from med_mmfl_bench.servers.symilemimic.fedavg import FedavgServer
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)


class FedproxServer(FedavgServer):
    """FedProx server — inherits FedAvg and adds proximal global model sync."""

    def __init__(
        self, args: Any, config: Any,
        val_dataset: Any, test_dataset: Any,
        client_datasets: List[Dict[str, Any]],
        server_trainset: Any = None,
        wandb: Any = False,
    ) -> None:
        super().__init__(
            args, config, val_dataset, test_dataset,
            client_datasets, server_trainset, wandb,
        )

    def update(self) -> None:
        """Execute one FedProx communication round."""
        self.dispatch()
        for client in self.clients:
            client.trainer.global_model.load_state_dict(
                self.trainer.model.state_dict()
            )
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