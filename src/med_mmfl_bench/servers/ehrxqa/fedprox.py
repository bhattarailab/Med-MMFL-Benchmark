"""FedProx server for EHRXQA — identical to FedAvg server.

FedProx only modifies the client-side training (proximal term);
the server aggregation and evaluation logic is unchanged.
"""

from typing import Any, Dict, List

from med_mmfl_bench.servers.ehrxqa.fedavg import FedavgServer

__all__ = ["FedproxServer"]


class FedproxServer(FedavgServer):
    """FedProx server (same as FedAvg; proximal term is client-side only)."""

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