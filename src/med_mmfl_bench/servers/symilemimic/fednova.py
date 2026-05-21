"""FedNova server for SymileMIMIC multimodal retrieval."""

import copy
from typing import Any, Dict, List

import numpy as np

from med_mmfl_bench.servers.symilemimic.fedavg import FedavgServer
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)


class FednovaServer(FedavgServer):
    """FedNova server with normalised aggregation for SymileMIMIC."""

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

    def _aggregate(self) -> Dict[str, Any]:
        """Aggregate using FedNova normalised directions."""
        logger.info(
            "[%s] [Round: %s] Aggregate updated signals!",
            self.args.algorithm.upper(), str(self.round).zfill(4),
        )
        omega: List[int] = []
        ais: List[float] = []
        dis: List[dict] = []

        for client in self.clients:
            logger.info("[Round %d] Aggregating client %s", self.round, client.id)
            omega.append(len(client))
            ai, di = client.upload()
            ais.append(ai)
            dis.append(di)

        pis = np.array(omega) / sum(omega)
        return self.server_optimizer.aggregate(pis=pis, dis=dis, ais=ais)

    def _update_global_model(self, d_avg: dict) -> None:
        """Apply the aggregated normalised direction to the global model."""
        model = copy.deepcopy(self.model.state_dict())
        for key in model:
            if "weight" in key or "bias" in key:
                model[key] = model[key] - d_avg[key].to("cpu")
        self.model.load_state_dict(model, strict=True)

    def update(self) -> None:
        """Execute one FedNova communication round."""
        self.dispatch()
        for client in self.clients:
            client.update(self.round)

        d_avg = self._aggregate()
        self._update_global_model(copy.deepcopy(d_avg))
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