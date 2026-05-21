"""FedProx client for EHRXQA federated VQA.

Extends FedAvg with a proximal regularization term that penalizes
deviation from the global model during local training.
"""

import copy
from typing import Any, Dict

import torch
from tqdm import tqdm

from med_mmfl_bench.clients.ehrxqa.fedavg import FedavgClient
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["FedproxClient"]


class FedproxClient(FedavgClient):
    """FedProx client for EHRXQA.

    Adds a proximal penalty ``(mu/2) * ||w - w_global||^2`` to the
    local loss to limit client drift.

    Args:
        args: Experiment arguments.
        config: Configuration with ``train.proximal_mu`` for the penalty weight.
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

        last_loss = 0.0

        for epoch in range(self.config.train.local_epoch):
            epoch_loss = 0.0
            num_batches = 0

            with tqdm(self.train_loader, unit="batch") as tepoch:
                for batch in tepoch:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    self.optimizer.zero_grad(set_to_none=True)

                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self.model(**batch)
                        loss = outputs.loss

                        # Proximal term: (mu/2) * sum(||w - w_t||^2)
                        prox_term = 0.0
                        local_sd = self.model.state_dict()
                        global_sd = global_model.state_dict()
                        for (k, v), (k_t, v_t) in zip(
                            local_sd.items(), global_sd.items()
                        ):
                            if ("weight" in k and "weight" in k_t) or (
                                "bias" in k and "bias" in k_t
                            ):
                                prox_term += torch.norm(v - v_t, p=2) ** 2

                        loss = loss + (self.prox_mu / 2) * prox_term

                    if torch.isfinite(loss):
                        self.grad_scaler.scale(loss).backward()

                        if self.config.train.grad_clip > 0:
                            self.grad_scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.train.grad_clip,
                            )

                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()

                        last_loss = loss.item()
                        epoch_loss += last_loss
                        num_batches += 1
                        tepoch.set_postfix(Loss=last_loss)
                    else:
                        logger.warning("Skipping update due to non-finite loss")

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            logger.info(
                "[CLIENT %s] [COMM: %d] Epoch %d/%d completed. Avg Loss: %.4f",
                self.client_id,
                comm_round,
                epoch + 1,
                self.config.train.local_epoch,
                avg_epoch_loss,
            )

        self.model.to("cpu")
        del global_model
        torch.cuda.empty_cache()

        return {"loss": last_loss}
