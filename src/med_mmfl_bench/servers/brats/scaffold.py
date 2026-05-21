"""SCAFFOLD server for BraTS multimodal segmentation."""

import copy
import gc
from typing import Any, Dict, List, Optional

import torch

from med_mmfl_bench.servers.brats.fedavg import FedavgServer
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)


class ScaffoldServer(FedavgServer):
    """SCAFFOLD server with server/client control variates for BraTS."""

    def __init__(
        self, args: Any, config: Any,
        val_dataset: Any, test_dataset: Any,
        client_datasets: List[Dict[str, Any]],
        wandb: Any = False,
    ) -> None:
        super().__init__(args, config, val_dataset, test_dataset, client_datasets, wandb)
        self.server_control = self._init_control(self.global_model)
        self._set_control_device(self.server_control, to_cuda=True)
        self.client_controls: Dict[Any, Dict[str, torch.Tensor]] = {
            cid: self._init_control(self.global_model) for cid in self.client_ids
        }

    def _init_control(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """Create a zero-initialised control variate dict."""
        return {
            name: torch.zeros_like(p).cpu()
            for name, p in model.state_dict().items()
        }

    def _set_control_device(
        self, control: Dict[str, torch.Tensor], to_cuda: bool = True,
    ) -> None:
        """Move control variates to cuda or cpu."""
        for name in control:
            control[name] = control[name].to(self.device) if to_cuda else control[name].cpu()

    def _set_client_control(
        self, client_id: Any, client_control: Dict[str, torch.Tensor],
    ) -> None:
        """Store the updated client control variate."""
        self.client_controls[client_id] = client_control

    def _request(self, eval: bool = False) -> Optional[Dict[int, int]]:
        """Sequential client updates with SCAFFOLD control variate passing."""
        if eval:
            return super()._request(eval=True)

        update_sizes: Dict[int, int] = {}
        for client in self.clients:
            if client.model is None:
                client.download(self.global_model)
            client.set_control(
                server_control=copy.deepcopy(self.server_control),
                client_control=copy.deepcopy(self.client_controls[client.id]),
            )
            client.update(self.round)
            self.delta_controls.append(copy.deepcopy(client.delta_control))
            self._set_client_control(
                client.id, copy.deepcopy(client.client_control),
            )
            update_sizes[client.id] = len(client.training_set)
        return update_sizes

    def _update_server_control(self) -> None:
        """Update the server control variate from aggregated client deltas."""
        delta_avg = copy.deepcopy(self.delta_controls[0])
        new_control = copy.deepcopy(self.server_control)

        for key in delta_avg:
            delta_avg[key] = torch.div(self.delta_controls[0][key], self.num_of_clients)

        for key in delta_avg:
            for idx in range(1, len(self.delta_controls)):
                delta_avg[key] += torch.div(self.delta_controls[idx][key], self.num_of_clients)
            new_control[key] = new_control[key] - delta_avg[key]

        self.server_control = copy.deepcopy(new_control)

    def update(self) -> None:
        """Execute one SCAFFOLD communication round."""
        self.delta_controls: List[Dict[str, torch.Tensor]] = []

        self.dispatch()
        self._request(eval=False)
        gc.collect()
        torch.cuda.empty_cache()

        self.global_model.load_state_dict(self._aggregate())

        logger.info("Updating server control after round: %d", self.round)
        self._update_server_control()
        logger.info("Server control successfully updated.")

        self.round += 1