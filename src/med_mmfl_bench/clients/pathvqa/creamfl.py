"""CreamFL client for PathVQA federated VQA.

Implements client-side CREAM with intra-modal contrastive learning
between the current, global, and previous model representations.
"""

import copy
import gc
from typing import Any, Dict, Optional, Tuple

import torch
from tqdm import tqdm

from med_mmfl_bench.clients.pathvqa.fedavg import FedavgClient
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["CreamflClient"]


class CreamflClient(FedavgClient):
    """CreamFL client for PathVQA.

    After standard local training, performs CREAM distillation using
    intra-modal contrastive loss.
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
        self.old_model: Optional[torch.nn.Module] = None
        self.global_text_feat: Optional[torch.Tensor] = None
        self.global_vision_feat: Optional[torch.Tensor] = None
        self.eval_loader: Optional[torch.utils.data.DataLoader] = None

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_features(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract text and vision features from a model.

        Returns:
            Tuple of (text_features, vision_features).
        """
        model.eval()
        model.cuda()

        text_feats, vision_feats = [], []

        with torch.no_grad():
            for inputs in tqdm(dataloader, unit="batch", desc="Extracting features"):
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

                model.return_repr = True
                text_repr, vision_repr, _ = model(**inputs)
                model.return_repr = False

                text_feats.append(text_repr)
                vision_feats.append(vision_repr)

        model.cpu()
        return torch.cat(text_feats, dim=0), torch.cat(vision_feats, dim=0)

    # ------------------------------------------------------------------
    # Contrastive loss
    # ------------------------------------------------------------------

    @staticmethod
    def compute_contrastive_loss_intra(
        student_feat: torch.Tensor,
        target_feat: torch.Tensor,
        old_feat: torch.Tensor,
        temperature: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute intra-modal contrastive loss.

        Returns:
            Tuple of (logits ``(B, 2)``, labels ``(B,)``).
        """
        pos = torch.sum(student_feat * target_feat, dim=-1, keepdim=True)
        neg = torch.sum(student_feat * old_feat, dim=-1, keepdim=True)

        logits = torch.cat([pos, neg], dim=1) / temperature
        labels = torch.zeros(
            student_feat.size(0), dtype=torch.long, device=student_feat.device
        )
        return logits, labels

    # ------------------------------------------------------------------
    # CREAM distillation
    # ------------------------------------------------------------------

    def cream_loss_update(self) -> None:
        """Compute CREAM loss and update the model batch-by-batch."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        criterion = torch.nn.CrossEntropyLoss().to(device)

        global_text = self.global_text_feat.to(device)
        global_vision = self.global_vision_feat.to(device)

        logger.debug("Extracting previous model features...")
        prev_text_feat, prev_vision_feat = self.extract_features(
            self.old_model, self.eval_loader
        )
        prev_text_feat = prev_text_feat.to(device)
        prev_vision_feat = prev_vision_feat.to(device)

        self.model.train()
        self.model.cuda()

        for b_idx, inputs in enumerate(
            tqdm(self.eval_loader, unit="batch", desc="CREAM loss")
        ):
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            self.model.return_repr = True
            text_repr, vision_repr, _ = self.model(**inputs)
            self.model.return_repr = False

            bs = inputs["labels"].size(0)
            start = b_idx * bs
            end = start + bs

            logits_text, labels_text = self.compute_contrastive_loss_intra(
                text_repr,
                global_text[start:end],
                prev_text_feat[start:end],
                temperature=0.2,
            )
            logits_vision, labels_vision = self.compute_contrastive_loss_intra(
                vision_repr,
                global_vision[start:end],
                prev_vision_feat[start:end],
                temperature=0.2,
            )

            logits = torch.cat([logits_text, logits_vision], dim=0)
            labels = torch.cat([labels_text, labels_vision], dim=0)
            loss_intra = criterion(logits, labels)

            if not torch.isnan(loss_intra):
                batch_loss = loss_intra * float(
                    self.config.train.get("interintra_weight", 1.0)
                )

                self.optimizer.zero_grad()
                batch_loss.backward()

                if self.config.train.get("grad_clip", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.train.grad_clip,
                    )

                self.optimizer.step()
            else:
                logger.warning("NaN loss at batch %d, skipping", b_idx)

            del loss_intra
            gc.collect()

        self.model.eval()
        self.model.cpu()

    # ------------------------------------------------------------------
    # CREAM update
    # ------------------------------------------------------------------

    def update(
        self,
        global_text_feat: torch.Tensor,
        global_vision_feat: torch.Tensor,
        global_eval_loader: torch.utils.data.DataLoader,
    ) -> Dict[str, str]:
        """Update the local model using the CREAM algorithm.

        Args:
            global_text_feat: Global text representations.
            global_vision_feat: Global vision representations.
            global_eval_loader: DataLoader for the public/eval dataset.

        Returns:
            Status dictionary.
        """
        self.global_text_feat = global_text_feat
        self.global_vision_feat = global_vision_feat
        self.eval_loader = global_eval_loader

        self.old_model = copy.deepcopy(self.model)
        self.old_model.cpu()
        self.old_model.eval()

        # Standard local training
        logger.info("[CLIENT %s] Standard training...", self.client_id)
        for epoch in range(self.config.train.local_epoch):
            self.model.train()
            self.model.cuda()

            for inputs in tqdm(self.train_loader, unit="batch"):
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

                self.optimizer.zero_grad()
                outputs = self.model(**inputs)
                loss = self.criterion(outputs["logits"], inputs["labels"])
                loss.backward()

                if self.config.train.get("grad_clip", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.train.grad_clip,
                    )

                self.optimizer.step()

            self.model.eval()
            self.model.cpu()

        # CREAM distillation
        logger.info("[CLIENT %s] CREAM distillation...", self.client_id)
        self.cream_loss_update()

        return {"status": "updated"}

    def __len__(self) -> int:
        return len(self.training_set)

    def __repr__(self) -> str:
        return f"CREAMFL_CLIENT <{self.client_id}>"