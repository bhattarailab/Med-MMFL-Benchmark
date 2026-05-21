"""SCAFFOLD server for SymileMIMIC multimodal retrieval."""

import copy
from typing import Any, Dict, List

import torch

from med_mmfl_bench.servers.symilemimic.fedavg import FedavgServer
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)


class ScaffoldServer(FedavgServer):
    """SCAFFOLD server with server/client control variates for SymileMIMIC."""

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
        self.server_control = self._init_control(self.trainer.model)
        self._set_control_device(self.server_control, to_cuda=True)
        self.client_controls: Dict[Any, Dict[str, torch.Tensor]] = {
            client.client_id: self._init_control(self.trainer.model)
            for client in self.clients
        }

    def _init_control(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """Create a zero-initialised control variate dict."""
        return {
            name: torch.zeros_like(p.data).cpu()
            for name, p in self.trainer.model.named_parameters()
            if ("weight" not in name) or ("bias" not in name)
        }

    def _set_control_device(
        self, control: Dict[str, torch.Tensor], to_cuda: bool = True,
    ) -> None:
        """Move control variates to cuda or cpu."""
        for name in control:
            control[name] = control[name].to("cuda") if to_cuda else control[name].cpu()

    def _set_client_control(
        self, client_id: Any, client_control: Dict[str, torch.Tensor],
    ) -> None:
        """Store the updated client control variate."""
        self.client_controls[client_id] = client_control

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
        for client in self.clients:
            client.trainer.set_control(
                server_control=copy.deepcopy(self.server_control),
                client_control=copy.deepcopy(self.client_controls[client.client_id]),
            )
            client.update(self.round)
            self.delta_controls.append(copy.deepcopy(client.trainer.delta_control))
            self._set_client_control(
                client.client_id, copy.deepcopy(client.trainer.client_control),
            )

        self.model.load_state_dict(self._aggregate())
        self.trainer.model.load_state_dict(self.model.state_dict())

        logger.info("Updating server control after round: %d", self.round)
        self._update_server_control()
        logger.info("Server control successfully updated.")

        logger.info(":::: Validating Model :::: Round : %d", self.round)
        val_loss = self.trainer.val()
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = self.round
            self.trainer.save(self.round)

        logger.info(":::: Testing Model :::: Round : %d", self.round)
        self.trainer.test()
        self.round += 1