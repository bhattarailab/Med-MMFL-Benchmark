"""SCAFFOLD server for MIMIC-CXR multimodal classification.

Implements the SCAFFOLD algorithm which uses control variates to correct
for client drift in heterogeneous federated learning settings.
"""

import copy
import gc
from typing import Any, Dict, List

import torch

from med_mmfl_bench.servers.mimiccxrjpg.fedavg import FedavgServer
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)


class ScaffoldServer(FedavgServer):
    """SCAFFOLD server with server/client control variates.

    Args:
        args: Parsed CLI arguments.
        config: YAML configuration object.
        wandb: Optional WandB run instance.
    """

    def __init__(self, args: Any, config: Any, wandb: Any = False) -> None:
        super().__init__(args, config, wandb)

        self.server_control = self._init_control(self.trainer.model)
        self._set_control_device(self.server_control, to_cuda=True)
        self.client_controls: Dict[int, Dict[str, torch.Tensor]] = {
            idx: self._init_control(self.trainer.model)
            for idx in range(len(self.clients))
        }

    # ------------------------------------------------------------------
    # Control variate helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _init_control(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """Create a zero-initialised control variate dict."""
        return {
            name: torch.zeros_like(p.data).cpu()
            for name, p in model.state_dict().items()
        }

    def _set_control_device(
        self, control: Dict[str, torch.Tensor], to_cuda: bool = True,
    ) -> None:
        """Move all control variate tensors to cuda or cpu."""
        for name in control:
            control[name] = control[name].to(self.device) if to_cuda else control[name].cpu()

    def _set_client_control(
        self, client_id: int, client_control: Dict[str, torch.Tensor],
    ) -> None:
        """Store the updated client control variate."""
        self.client_controls[client_id] = client_control

    def _update_server_control(self) -> None:
        """Update the server control variate from aggregated client deltas."""
        delta_avg = copy.deepcopy(self.delta_controls[0])
        new_control = copy.deepcopy(self.server_control)
        n_clients = len(self.clients)

        for key in delta_avg:
            delta_avg[key] = torch.div(self.delta_controls[0][key], n_clients)

        for key in delta_avg:
            for idx in range(1, len(self.delta_controls)):
                delta_avg[key] += torch.div(self.delta_controls[idx][key], n_clients)
            new_control[key] = new_control[key] - delta_avg[key]

        self.server_control = copy.deepcopy(new_control)

    # ------------------------------------------------------------------
    # Federation protocol
    # ------------------------------------------------------------------

    def _request(self, **kwargs: Any) -> None:
        """Update all clients with SCAFFOLD control variates."""
        for client in self.clients:
            if client.trainer.model is None:
                client.download(self.trainer.model)

            client.set_control(
                server_control=copy.deepcopy(self.server_control),
                client_control=copy.deepcopy(self.client_controls[client.client_id]),
            )

            client.update(self.round)

            self.delta_controls.append(copy.deepcopy(client.trainer.delta_control))
            self._set_client_control(
                client.client_id, copy.deepcopy(client.trainer.client_control),
            )

    def update(self) -> None:
        """Execute one SCAFFOLD communication round."""
        self.delta_controls: List[Dict[str, torch.Tensor]] = []

        self.dispatch()
        self._request()

        gc.collect()
        torch.cuda.empty_cache()

        self.trainer.model.load_state_dict(self._aggregate())

        logger.info("Updating server control after round: %d", self.round)
        self._update_server_control()
        logger.info("Server control successfully updated.")

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