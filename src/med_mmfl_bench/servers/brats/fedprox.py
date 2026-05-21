"""FedProx server for BraTS multimodal segmentation."""

from typing import Any, Dict, List

from med_mmfl_bench.servers.brats.fedavg import FedavgServer


class FedproxServer(FedavgServer):
    """FedProx server — identical to FedAvg at the server level."""

    def __init__(
        self, args: Any, config: Any,
        val_dataset: Any, test_dataset: Any,
        client_datasets: List[Dict[str, Any]],
        wandb: Any = False,
    ) -> None:
        super().__init__(args, config, val_dataset, test_dataset, client_datasets, wandb)