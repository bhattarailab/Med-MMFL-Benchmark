"""SCAFFOLD client for EHRXQA federated VQA.

Applies control variate corrections to gradients during local
training to compensate for client drift.
"""

import copy
from typing import Any, Dict, Optional

import torch
from torch import nn
from tqdm import tqdm

from med_mmfl_bench.clients.ehrxqa.fedavg import FedavgClient
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["ScaffoldClient"]


class ScaffoldClient(FedavgClient):
    """SCAFFOLD client for EHRXQA.

    After the standard forward/backward pass, corrects gradients
    using ``grad += c_server - c_client`` before the optimizer step.
    After training, computes delta controls for the server.
    """

    def __init__(
        self,
        args: Any,
        config: Any,
        client_id: Any,
        training_set: Any,
        test_set: Any,
        wandb: Any = None,
    ) -> None:
        super().__init__(args, config, client_id, training_set, test_set, wandb)

    def set_control(
        self,
        server_control: Dict[str, torch.Tensor],
        client_control: Dict[str, torch.Tensor],
    ) -> None:
        """Set the server and client control variates for this round.

        Args:
            server_control: Server-side control variate.
            client_control: This client's control variate.
        """
        self.server_control = server_control
        self.client_control = client_control
        self._set_control_device(self.client_control, to_device=True)

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

    def update(self, comm_round: int = 0) -> Dict[str, float]:
        """Perform SCAFFOLD local training with control variate correction.

        Args:
            comm_round: Current communication round number.

        Returns:
            Dictionary with ``"loss"`` key.
        """
        last_global_model = copy.deepcopy(self.model)
        self.model.train()
        self.model.to(self.device)
        last_global_model.to(self.device)
        self._create_optimizer_and_scaler()

        n_total_bs = int(self.config.train.local_epoch * len(self.train_loader))
        loss_value = 0.0

        logger.info("[CLIENT %s] Starting SCAFFOLD training...", self.client_id)
        for epoch in range(self.config.train.local_epoch):
            with tqdm(self.train_loader, unit="batch") as tepoch:
                for batch in tepoch:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    self.optimizer.zero_grad()

                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self.model(**batch)
                        loss = outputs.loss

                    if torch.isfinite(loss):
                        self.grad_scaler.scale(loss).backward()

                        # Apply control variate correction (read-only)
                        self._apply_control_variates(
                            self.model,
                            server_control=self.server_control,
                            client_control=self.client_control,
                        )

                        if self.config.train.grad_clip > 0:
                            self.grad_scaler.unscale_(self.optimizer)
                            nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.train.grad_clip,
                            )

                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()

                        loss_value = loss.item()
                        tepoch.set_postfix(Loss=loss_value)
                    else:
                        logger.warning("Skipping update due to non-finite loss")

            logger.info(
                "[CLIENT %s] [COMM: %d] Epoch %d/%d completed.",
                self.client_id,
                comm_round,
                epoch + 1,
                self.config.train.local_epoch,
            )

        # Compute delta model and update local control variates
        delta_model = self._get_delta_model(last_global_model, self.model)

        client_control, delta_control = self._update_local_control(
            delta_model=delta_model,
            server_control=self.server_control,
            client_control=self.client_control,
            steps=n_total_bs,
            lr=float(self.config.optimizer.learning_rate),
        )

        self.client_control = copy.deepcopy(client_control)
        self.delta_control = copy.deepcopy(delta_control)

        logger.info("[CLIENT %s] SCAFFOLD update completed.", self.client_id)
        self._set_control_device(self.client_control, to_device=False)
        self.model.to("cpu")

        return {"loss": loss_value}

    # ------------------------------------------------------------------
    # SCAFFOLD-specific helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_control_variates(
        model: nn.Module,
        server_control: Dict[str, torch.Tensor],
        client_control: Dict[str, torch.Tensor],
    ) -> None:
        """Apply control variate correction to gradients in-place.

        Adds ``c_server - c_client`` to each parameter's gradient.

        Args:
            model: The model being trained.
            server_control: Server control variate.
            client_control: Client control variate.
        """
        for name, param in model.named_parameters():
            if param.grad is not None and "running" not in name and "num_batch" not in name:
                param.grad.data += server_control[name].data - client_control[name].data

    @staticmethod
    def _get_delta_model(
        model0: nn.Module, model1: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """Compute the parameter difference between two models.

        Returns:
            Dictionary of ``model0 - model1`` for each parameter.
        """
        state_dict = {}
        for name, param0 in model0.state_dict().items():
            param1 = model1.state_dict()[name]
            state_dict[name] = param0.detach() - param1.detach()
        return state_dict

    @staticmethod
    def _update_local_control(
        delta_model: Dict[str, torch.Tensor],
        server_control: Dict[str, torch.Tensor],
        client_control: Dict[str, torch.Tensor],
        steps: int,
        lr: float,
    ) -> tuple:
        """Update the local control variate in-place.

        Args:
            delta_model: Parameter difference (global - local).
            server_control: Server control variate.
            client_control: Client control variate.
            steps: Total number of local SGD steps.
            lr: Learning rate.

        Returns:
            Tuple of (updated_client_control, delta_control).
        """
        delta_control: Dict[str, torch.Tensor] = {}

        for name in delta_model:
            c = server_control[name]
            ci = client_control[name]
            delta = delta_model[name]

            old_ci = ci.clone()
            ci.data = ci.data - c.data + delta.data / (steps * lr)
            delta_control[name] = old_ci - ci

        return client_control, delta_control