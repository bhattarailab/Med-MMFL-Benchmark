"""FedNova server for PathVQA federated VQA.

FedNova normalises local model updates by the effective number of
SGD steps, correcting for heterogeneous local training across clients.
"""

import copy
import gc
from typing import Any, Dict, List

import numpy as np
import torch

from med_mmfl_bench.servers.pathvqa.fedavg import FedavgServer
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["FednovaServer"]


class FednovaServer(FedavgServer):
    """FedNova server for PathVQA.

    Overrides aggregation to use normalized stochastic gradients
    and the global update step to apply the averaged direction.
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

    def _aggregate(self) -> Dict[str, torch.Tensor]:
        """Aggregate using FedNova normalized gradients.

        Returns:
            Averaged gradient direction dictionary.
        """
        logger.info(
            "[%s] [Round: %04d] Aggregate updated signals!",
            self.args.algorithm.upper(),
            self.round,
        )

        omega, ais, dis = [], [], []
        for client in self.clients:
            omega.append(len(client))
            ai, di = client.upload()
            ais.append(ai)
            dis.append(di)

        omega_arr = np.array(omega)
        pis = omega_arr / omega_arr.sum()

        return self.server_optimizer.aggregate(pis=pis, dis=dis, ais=ais)

    def update(self) -> None:
        """Execute one FedNova communication round."""
        self.dispatch()

        self.sampled_client_ids = self._sample_clients()
        logger.info("[Round %d] Sampled clients: %s", self.round, self.sampled_client_ids)

        self._request(eval=False, retain_model=True)

        gc.collect()
        torch.cuda.empty_cache()

        d_avg = self._aggregate()
        self._update_global_model(copy.deepcopy(d_avg))
        self.evaluate()
        self.round += 1

    def _update_global_model(self, d_avg: Dict[str, torch.Tensor]) -> None:
        """Apply the averaged gradient direction to the global model.

        Args:
            d_avg: Averaged normalised gradient direction.
        """
        model_sd = self.global_model.state_dict()
        for key in model_sd.keys():
            if "weight" in key or "bias" in key:
                model_sd[key] = model_sd[key] - d_avg[key].to("cpu")
        self.global_model.load_state_dict(model_sd, strict=True)