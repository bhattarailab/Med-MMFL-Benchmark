"""MOON client for PathVQA federated VQA.

Extends FedAvg with model-contrastive learning that encourages the
local model to stay close to the global model while moving away from
the previous local model.
"""

import copy
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from med_mmfl_bench.clients.pathvqa.fedavg import FedavgClient
from med_mmfl_bench.models import get_model
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["MoonClient"]


class MoonClient(FedavgClient):
    """MOON client for PathVQA.

    Uses contrastive learning at the representation level: the
    current model's CLS representation is the anchor, the global
    model's representation is the positive, and the previous round's
    local model is the negative.
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
        self.mu: float = self.config.train.moon_mu
        self.tempr: float = self.config.train.moon_tempr

        self.global_model = get_model(self.config.model.name, self.config.model)
        self.prev_net = copy.deepcopy(self.global_model).cpu()

        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.crossentropy = nn.CrossEntropyLoss()

    def update(self, comm_round: int = 0) -> Dict[str, float]:
        """Perform MOON local training with contrastive loss.

        Args:
            comm_round: Current communication round number.

        Returns:
            Dictionary with ``"loss"`` key.
        """
        self.model.train()
        self.model.to(self.device)
        self._create_optimizer_and_scaler()

        loss_value = 0.0
        logger.info(
            "[CLIENT %s] Starting MOON training (round %d)...",
            self.client_id,
            comm_round,
        )

        for epoch in range(self.config.train.local_epoch):
            epoch_loss = []
            with tqdm(self.train_loader, unit="batch") as tepoch:
                for batch in tepoch:
                    self.optimizer.zero_grad()
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    label = batch["labels"]

                    # Forward with representation extraction
                    self.model.return_repr = True
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        representations, output = self.model(**batch)
                        output = output["logits"]
                        loss = self.criterion(output, label)
                    self.model.return_repr = False

                    # Extract global and previous representations
                    with torch.no_grad():
                        self.global_model.to(self.device)
                        self.global_model.return_repr = True
                        repr_g, _ = self.global_model(**batch)
                        self.global_model.return_repr = False
                        self.global_model.cpu()

                        self.prev_net.to(self.device)
                        self.prev_net.return_repr = True
                        repr_prev, _ = self.prev_net(**batch)
                        self.prev_net.return_repr = False
                        self.prev_net.cpu()

                    # Contrastive loss
                    rep, rep_g, rep_prev = map(
                        lambda t: F.normalize(t, dim=-1),
                        [representations, repr_g, repr_prev],
                    )

                    logits_pos = self.cosine_similarity(rep, rep_g).view(-1, 1)
                    logits_neg = self.cosine_similarity(rep, rep_prev).view(-1, 1)
                    logits = torch.cat((logits_pos, logits_neg), dim=1) / self.tempr

                    dummy_labels = torch.zeros(
                        label.size(0), dtype=torch.long, device=self.device
                    )
                    l_con = self.crossentropy(logits, dummy_labels)

                    loss += self.mu * l_con

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
                    if torch.isfinite(loss):
                        epoch_loss.append(loss_value)

            avg_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0.0
            logger.info(
                "[CLIENT %s] [COMM: %d] Epoch %d/%d completed. Avg Loss: %.4f",
                self.client_id,
                comm_round,
                epoch + 1,
                self.config.train.local_epoch,
                avg_loss,
            )
            if self.wandb:
                self.wandb.log(
                    {f"client/{self.client_id}/train_loss": avg_loss},
                    step=comm_round,
                )

        self.model.to("cpu")
        # Save current model as previous for next round
        self.prev_net.load_state_dict(self.model.state_dict())
        return {"loss": loss_value}
