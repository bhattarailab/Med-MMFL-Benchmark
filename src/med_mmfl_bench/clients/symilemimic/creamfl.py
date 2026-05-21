"""CreamFL client for SymileMIMIC multimodal retrieval."""

import copy
import gc
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from med_mmfl_bench.clients.symilemimic.fedavg import FedavgClient
from med_mmfl_bench.losses import get_criterion
from med_mmfl_bench.models import get_model
from med_mmfl_bench.trainers.symilemimictrainer import ClientTrainer as SymileClientTrainer
from med_mmfl_bench.utils.logging import get_logger
from med_mmfl_bench.utils.optimizers import get_optimizer

logger = get_logger(__name__)


class CreamflClientTrainer(SymileClientTrainer):
    """SymileMIMIC trainer with CreamFL contrastive distillation.

    Uses intra-modal (positive=global, negative=previous model) and
    inter-modal (6-way pairwise InfoNCE) contrastive losses for
    knowledge distillation across 3 modalities (CXR, ECG, Labs).
    """

    def __init__(self, args: Any, config: Any, wandb: Any = None) -> None:
        super().__init__(args, config, wandb)
        if not hasattr(self, "scaler"):
            self.scaler = torch.cuda.amp.GradScaler()

    def set_global_features(
        self,
        global_cxr_feat: torch.Tensor,
        global_ecg_feat: torch.Tensor,
        global_labs_feat: torch.Tensor,
    ) -> None:
        """Store the global feature vectors for contrastive alignment."""
        self.global_cxr_feat = global_cxr_feat
        self.global_ecg_feat = global_ecg_feat
        self.global_labs_feat = global_labs_feat

    @staticmethod
    def compute_contrastive_loss_intra(
        student_feat: torch.Tensor,
        target_feat: torch.Tensor,
        old_feat: torch.Tensor,
        temperature: float = 0.5,
    ) -> tuple:
        """Compute intra-modal contrastive loss.

        Args:
            student_feat: Current model features ``[B, D]``.
            target_feat: Global model features ``[B, D]`` (positive).
            old_feat: Previous model features ``[B, D]`` (negative).
            temperature: Temperature scaling factor.

        Returns:
            Tuple of ``(logits, labels)`` for cross-entropy.
        """
        pos = torch.sum(student_feat * target_feat, dim=-1, keepdim=True).view(-1, 1)
        neg = torch.sum(student_feat * old_feat, dim=-1, keepdim=True).view(-1, 1)
        logits = torch.cat([pos, neg], dim=1) / temperature
        labels = torch.zeros(student_feat.size(0), dtype=torch.long, device=student_feat.device)
        return logits, labels

    # ------------------------------------------------------------------
    # CreamFL loss (memory-efficient batch-by-batch version)
    # ------------------------------------------------------------------

    def cream_loss(self) -> None:
        """Compute and backpropagate the CreamFL loss batch-by-batch."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        criterion = torch.nn.CrossEntropyLoss().to(device)

        global_cxr = self.global_cxr_feat.to(device)
        global_ecg = self.global_ecg_feat.to(device)
        global_labs = self.global_labs_feat.to(device)

        # Extract previous model representations
        prev_cxr_list, prev_ecg_list, prev_labs_list = [], [], []
        with torch.no_grad():
            with tqdm(self.eval_loader, unit="batch", desc="Prev model feats") as tepoch:
                for inputs in tepoch:
                    cxr, ecg, labs_p, labs_m, hadm_id, _ = inputs
                    cxr, ecg, labs_p, labs_m, hadm_id = (
                        cxr.to(device), ecg.to(device), labs_p.to(device),
                        labs_m.to(device), hadm_id.to(device),
                    )
                    batch = [cxr, ecg, labs_p, labs_m, hadm_id]
                    r_c, r_e, r_l, _ = self.old_model(batch)
                    prev_cxr_list.append(r_c.cpu())
                    prev_ecg_list.append(r_e.cpu())
                    prev_labs_list.append(r_l.cpu())

        prev_cxr = torch.cat(prev_cxr_list, dim=0).to(device)
        prev_ecg = torch.cat(prev_ecg_list, dim=0).to(device)
        prev_labs = torch.cat(prev_labs_list, dim=0).to(device)

        def contrastive_loss_pair(
            local_feats: torch.Tensor, global_feats: torch.Tensor,
            temperature: float = 0.5,
        ) -> torch.Tensor:
            """InfoNCE loss between two modalities."""
            local_feats = F.normalize(local_feats, dim=-1)
            global_feats = F.normalize(global_feats, dim=-1)
            logits = torch.matmul(local_feats, global_feats.T) / temperature
            labels = torch.arange(local_feats.size(0), device=local_feats.device)
            return F.cross_entropy(logits, labels)

        def multi_modal_contrastive_loss(
            f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor,
            g1: torch.Tensor, g2: torch.Tensor, g3: torch.Tensor,
            temperature: float = 0.5,
        ) -> torch.Tensor:
            """Average of all 6 pairwise inter-modal contrastive losses."""
            return (
                contrastive_loss_pair(f1, g2, temperature)
                + contrastive_loss_pair(f1, g3, temperature)
                + contrastive_loss_pair(f2, g1, temperature)
                + contrastive_loss_pair(f2, g3, temperature)
                + contrastive_loss_pair(f3, g1, temperature)
                + contrastive_loss_pair(f3, g2, temperature)
            ) / 6.0

        with tqdm(self.eval_loader, unit="batch", desc="CreamFL loss") as tepoch:
            for b_idx, inputs in enumerate(tepoch):
                cxr, ecg, labs_p, labs_m, hadm_id, _ = inputs
                cxr, ecg, labs_p, labs_m, hadm_id = (
                    cxr.to(device), ecg.to(device), labs_p.to(device),
                    labs_m.to(device), hadm_id.to(device),
                )
                batch = [cxr, ecg, labs_p, labs_m, hadm_id]
                r_c, r_e, r_l, _ = self.model(batch)

                bs = cxr.size(0)
                start = b_idx * bs
                end = start + bs

                # Intra-modal contrastive loss
                logits_cxr, labels_cxr = self.compute_contrastive_loss_intra(
                    r_c, global_cxr[start:end], prev_cxr[start:end], temperature=0.5,
                )
                logits_ecg, labels_ecg = self.compute_contrastive_loss_intra(
                    r_e, global_ecg[start:end], prev_ecg[start:end], temperature=0.5,
                )
                logits_labs, labels_labs = self.compute_contrastive_loss_intra(
                    r_l, global_labs[start:end], prev_labs[start:end], temperature=0.5,
                )

                loss_intra = (
                    criterion(logits_cxr, labels_cxr)
                    + criterion(logits_ecg, labels_ecg)
                    + criterion(logits_labs, labels_labs)
                ) / 3.0

                # Inter-modal contrastive loss
                loss_inter = multi_modal_contrastive_loss(
                    r_c, r_e, r_l,
                    global_cxr[start:end], global_ecg[start:end], global_labs[start:end],
                    temperature=0.5,
                )

                if not (torch.isnan(loss_intra) or torch.isnan(loss_inter)):
                    batch_loss = (loss_intra + loss_inter) * float(
                        self.config.train.interintra_weight
                    )
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    self.optimizer.step()
                    del batch_loss

                del logits_cxr, labels_cxr, logits_ecg, labels_ecg, logits_labs, labels_labs
                gc.collect()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Standard local training (task loss only)."""
        self.model.cuda()
        self.model.train()
        with tqdm(self.train_loader, unit="batch", desc="Training (CreamFL)") as tepoch:
            for inputs in tepoch:
                cxr, ecg, labs_p, labs_m, hadm_id, _ = inputs
                cxr, ecg, labs_p, labs_m, hadm_id = (
                    cxr.cuda(), ecg.cuda(), labs_p.cuda(), labs_m.cuda(), hadm_id.cuda(),
                )
                batch = [cxr, ecg, labs_p, labs_m, hadm_id]

                self.optimizer.zero_grad()
                r_c, r_e, r_l, logit_scale_exp = self.model(batch)
                loss = self.loss_fn(r_c, r_e, r_l, logit_scale_exp, self.config.train.negative_sampling)
                loss.backward()
                self.optimizer.step()
                tepoch.set_postfix(Loss=loss.item())

        self.model.eval()
        self.model.cpu()

    def run_cream(self) -> None:
        """Run the CreamFL contrastive distillation step."""
        self.model.cuda()
        self.model.train()
        self.old_model.cuda()
        self.old_model.eval()
        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast(
            enabled=self.config.train.get("use_fp16", False)
        ):
            self.cream_loss()

        self.model.eval()
        self.model.cpu()
        self.old_model.cpu()

    def run_train(self) -> None:
        """Execute local training + CreamFL distillation."""
        for _ in range(self.config.train.local_epoch):
            self.train()
            self.run_cream()


class CreamflClient(FedavgClient):
    """CreamFL client for SymileMIMIC."""

    def __init__(
        self, args: Any, config: Any, client_id: int,
        training_set: Any, test_set: Any, wandb: Any = None,
    ) -> None:
        super().__init__(args, config, client_id, training_set, test_set, wandb)

    def build_trainer(self) -> None:
        """Instantiate the CreamFL trainer."""
        self.trainer = CreamflClientTrainer(self.args, self.config, self.wandb)
        self.trainer.model = get_model(self.config.model.name, self.config.model)
        self.trainer.model.cpu()
        self.trainer.optimizer = get_optimizer(
            self.config.optimizer.name, self.trainer.model.parameters(), self.config.optimizer,
        )
        self.trainer.criterion = get_criterion(self.config.criterion.name)
        self.trainer.train_loader = self.train_loader
        self.trainer.test_loader = self.test_loader
        self.trainer.client_id = self.client_id

    def update(
        self,
        global_cxr_feat: torch.Tensor,
        global_ecg_feat: torch.Tensor,
        global_labs_feat: torch.Tensor,
        global_eval_loader: Any,
    ) -> None:
        """Update using CreamFL contrastive distillation."""
        self.trainer.set_global_features(global_cxr_feat, global_ecg_feat, global_labs_feat)
        self.trainer.eval_loader = global_eval_loader

        self.trainer.old_model = copy.deepcopy(self.trainer.model)
        self.trainer.old_model.cpu()
        self.trainer.old_model.eval()

        self.trainer.run_train()