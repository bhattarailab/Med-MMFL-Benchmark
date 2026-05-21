"""SCAFFOLD server for PathVQA federated VQA.

SCAFFOLD (Stochastic Controlled Averaging) maintains server and
per-client control variates to correct for client drift.
"""

import copy
import gc
import os
from collections import ChainMap
from typing import Any, Dict, List, Optional

import concurrent.futures
import torch

from med_mmfl_bench.servers.pathvqa.fedavg import FedavgServer
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["ScaffoldServer"]


class ScaffoldServer(FedavgServer):
    """SCAFFOLD server for PathVQA.

    Extends FedAvg with server and client control variates that are
    updated after each communication round to correct gradient drift.
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

        self.server_control = self._init_control(self.global_model)
        self._set_control_device(self.server_control, to_device=True)
        self.client_controls: Dict[Any, Dict[str, torch.Tensor]] = {
            client_id: self._init_control(self.global_model)
            for client_id in self.client_ids
        }

    # ------------------------------------------------------------------
    # Control variate helpers
    # ------------------------------------------------------------------

    def _init_control(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """Initialize zero-valued control variates matching model parameters."""
        return {
            name: torch.zeros_like(p.data).cpu()
            for name, p in model.state_dict().items()
        }

    def _set_control_device(
        self,
        control: Dict[str, torch.Tensor],
        to_device: bool = True,
    ) -> None:
        """Move control variates to device or CPU."""
        for name in control:
            control[name] = (
                control[name].to(self.device) if to_device else control[name].cpu()
            )

    def _set_client_control(
        self, client_id: Any, client_control: Dict[str, torch.Tensor]
    ) -> None:
        """Store an updated client control variate."""
        self.client_controls[client_id] = client_control

    # ------------------------------------------------------------------
    # Overridden FL methods
    # ------------------------------------------------------------------

    def _request(
        self,
        eval: bool = False,
        retain_model: bool = False,
        save_raw: bool = False,
    ) -> Optional[Dict[str, int]]:
        """Dispatch update/eval requests with SCAFFOLD control variates."""
        def _update_client(client: Any) -> tuple:
            if client.model is None:
                client.download(self.global_model)

            client.set_control(
                server_control=copy.deepcopy(self.server_control),
                client_control=copy.deepcopy(self.client_controls[client.id]),
            )
            update_result = client.update(self.round)

            self.delta_controls.append(copy.deepcopy(client.delta_control))
            self._set_client_control(client.id, copy.deepcopy(client.client_control))

            return (
                {client.id: len(client.training_set)},
                {client.id: update_result},
            )

        def _evaluate_client(client: Any) -> tuple:
            if client.model is None:
                client.download(self.global_model)
            eval_result = client.evaluate()
            if not retain_model:
                client.model = None
            return (
                {client.id: len(client.test_set)},
                {client.id: eval_result},
            )

        fn = _evaluate_client if eval else _update_client
        max_workers = max(
            1,
            min(
                len(self.clients) // (1 if eval else 2),
                os.cpu_count() - 1,
            ),
        )

        jobs, results = [], []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for client in self.clients:
                jobs.append(pool.submit(fn, client))
            for job in concurrent.futures.as_completed(jobs):
                results.append(job.result())

        sizes, _ = list(map(list, zip(*results)))
        sizes = dict(ChainMap(*sizes))
        return None if eval else sizes

    def update(self) -> None:
        """Execute one SCAFFOLD communication round."""
        self.delta_controls: List[Dict[str, torch.Tensor]] = []

        self.dispatch()
        self._request(eval=False, retain_model=True)

        gc.collect()
        torch.cuda.empty_cache()

        self.global_model.load_state_dict(self._aggregate())

        logger.info("Updating server control after round %d...", self.round)
        self._update_server_control()
        logger.info("Server control updated.")

        self.evaluate()
        self.round += 1

    def _update_server_control(self) -> None:
        """Update server control variate using averaged client deltas."""
        delta_avg = copy.deepcopy(self.delta_controls[0])
        new_control = copy.deepcopy(self.server_control)

        for key in delta_avg:
            delta_avg[key] = torch.div(
                self.delta_controls[0][key], self.num_of_clients
            )

        for key in delta_avg:
            for idx in range(1, len(self.delta_controls)):
                delta_avg[key] += torch.div(
                    self.delta_controls[idx][key], self.num_of_clients
                )
            new_control[key] = new_control[key] - delta_avg[key]

        self.server_control = copy.deepcopy(new_control)