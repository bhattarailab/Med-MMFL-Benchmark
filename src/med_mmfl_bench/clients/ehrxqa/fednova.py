"""FedNova client for EHRXQA federated VQA.

Extends FedAvg by tracking the effective number of local SGD steps
and computing normalised gradient directions for aggregation.
"""

import copy
from typing import Any, Dict, Tuple

import torch
from torch import nn
from tqdm import tqdm

from med_mmfl_bench.clients.ehrxqa.fedavg import FedavgClient
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["FednovaClient"]


class FednovaClient(FedavgClient):
    """FedNova client for EHRXQA.

    After local training, computes a normalisation coefficient ``a_i``
    and a normalised gradient direction ``d_i`` for the server to
    aggregate.

    Args:
        args: Experiment arguments.
        config: Configuration with ``optimizer.momentum`` (rho).
        client_id: Unique client identifier.
        training_set: Client's training dataset.
        test_set: Client's validation dataset.
        wandb: WandB run object or None.
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
        self.ai: float = 0.0
        self.di: Dict[str, torch.Tensor] = {}
        self.rho: float = self.config.optimizer.momentum

    def update(self, comm_round: int = 0) -> Dict[str, float]:
        """Perform local training and compute FedNova coefficients.

        Args:
            comm_round: Current communication round number.

        Returns:
            Dictionary with ``"loss"`` key.
        """
        self.model.train()
        self.model.to(self.device)
        global_weights = copy.deepcopy(self.model.state_dict())
        self._create_optimizer_and_scaler()

        loss_value = 0.0
        tau = 0

        logger.info("[CLIENT %s] Starting FedNova training...", self.client_id)
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

                        if self.config.train.grad_clip > 0:
                            self.grad_scaler.unscale_(self.optimizer)
                            nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.train.grad_clip,
                            )

                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()

                        loss_value = loss.item()
                        tepoch.set_postfix(loss=loss_value)
                        tau += 1
                    else:
                        logger.warning("Skipping update due to non-finite loss")

        # Compute FedNova normalisation coefficient a_i
        self.ai = (
            tau - self.rho * (1 - pow(self.rho, tau)) / (1 - self.rho)
        ) / (1 - self.rho)

        # Compute normalised gradient direction d_i
        state_dict = self.model.state_dict()
        self.di = copy.deepcopy(global_weights)
        for key in self.di:
            self.di[key] = torch.true_divide(
                global_weights[key] - state_dict[key], self.ai
            )

        logger.info(
            "[CLIENT %s] [COMM: %d] Training completed (%d steps).",
            self.client_id,
            comm_round,
            tau,
        )
        self.model.to("cpu")
        return {"loss": loss_value}

    def upload(self) -> Tuple[float, Dict[str, torch.Tensor]]:
        """Upload normalisation coefficient and gradient direction.

        Returns:
            Tuple of (a_i, d_i).
        """
        return self.ai, self.di