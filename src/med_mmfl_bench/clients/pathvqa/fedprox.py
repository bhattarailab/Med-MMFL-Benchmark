"""FedProx client for PathVQA federated VQA.

Extends FedAvg with a proximal regularization term that penalizes
deviation from the global model during local training.
"""

import copy
from typing import Any, Dict

import torch
from torch import nn
from tqdm import tqdm

from med_mmfl_bench.clients.pathvqa.fedavg import FedavgClient
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["FedproxClient"]


class FedproxClient(FedavgClient):
    """FedProx client for PathVQA.

    Adds a proximal penalty ``(mu/2) * ||w - w_global||^2`` to the
    local loss to limit client drift.
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
        self.prox_mu: float = self.config.train.proximal_mu

    def update(self, comm_round: int = 0) -> Dict[str, float]:
        """Perform local training with proximal regularization.

        Args:
            comm_round: Current communication round number.

        Returns:
            Dictionary with ``"loss"`` key.
        """
        global_model = copy.deepcopy(self.model)
        self.model.train()
        self.model.to(self.device)
        global_model.to(self.device)
        self._create_optimizer_and_scaler()

        loss_value = 0.0
        logger.info("[CLIENT %s] Starting FedProx training...", self.client_id)

        for epoch in range(self.config.train.local_epoch):
            with tqdm(self.train_loader, unit="batch") as tepoch:
                for batch in tepoch:
                    self.optimizer.zero_grad()
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    label = batch["labels"]

                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        output = self.model(**batch)["logits"]
                        loss = self.criterion(output, label)

                    # Proximal term: (mu/2) * sum(||w - w_t||^2)
                    prox_term = 0.0
                    for k, k_t in zip(
                        self.model.state_dict().keys(),
                        global_model.state_dict().keys(),
                    ):
                        if ("weight" in k and "weight" in k_t) or (
                            "bias" in k and "bias" in k_t
                        ):
                            prox_term += torch.norm(
                                self.model.state_dict()[k]
                                - global_model.state_dict()[k_t],
                                p=2,
                            ) ** 2

                    loss += (self.prox_mu / 2) * prox_term

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

        self.model.to("cpu")
        return {"loss": loss_value}