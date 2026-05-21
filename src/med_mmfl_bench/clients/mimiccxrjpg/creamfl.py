"""CreamFL client for MIMIC-CXR multimodal classification."""

import copy
import gc
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from med_mmfl_bench.clients.mimiccxrjpg.fedavg import FedavgClient
from med_mmfl_bench.trainers.mimiccxr import ClassificationTrainer as ClientTrainer
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)


class CreamflClientTrainer(ClientTrainer):
    """MIMIC-CXR trainer with CreamFL contrastive distillation (client-side)."""

    def __init__(
        self, args: Any, config: Any, wandb: Any = None, client_id: int = -1,
    ) -> None:
        super().__init__(args, config, wandb, client_id=client_id)
        if not hasattr(self, "scaler"):
            self.scaler = torch.cuda.amp.GradScaler()

    def set_global_features(
        self, global_img_feat: torch.Tensor, global_txt_feat: torch.Tensor,
    ) -> None:
        """Store the global feature vectors for contrastive alignment."""
        self.global_img_feat = global_img_feat
        self.global_txt_feat = global_txt_feat

    def compute_contrastive_loss_intra(
        self,
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
        logits = torch.cat([pos, neg], dim=1)
        labels = torch.zeros(student_feat.size(0), dtype=torch.long, device=student_feat.device)
        return logits, labels

    def cream_loss(self) -> None:
        """Compute and backpropagate the CreamFL loss (intra + inter modal)."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        criterion = torch.nn.CrossEntropyLoss().to(device)

        global_img = self.global_txt_feat.to(device)
        global_txt = self.global_img_feat.to(device)

        # Extract previous model representations
        prev_img_list, prev_txt_list = [], []
        with torch.no_grad():
            with tqdm(self.eval_loader, unit="batch", desc="Prev model feats") as tepoch:
                for frames, labels, text, _ in tepoch:
                    images = frames.cuda()
                    output = self.old_model(self.tokenizer, images, text)
                    prev_img_list.append(output["image_features"].cpu())
                    prev_txt_list.append(output["caption_features"].cpu())

        prev_img_feat = torch.cat(prev_img_list, dim=0).to(device)
        prev_txt_feat = torch.cat(prev_txt_list, dim=0).to(device)

        def contrastive_loss_pair(
            local_feats: torch.Tensor, global_feats: torch.Tensor,
            temperature: float = 0.5,
        ) -> torch.Tensor:
            local_feats = F.normalize(local_feats, dim=-1)
            global_feats = F.normalize(global_feats, dim=-1)
            logits = torch.matmul(local_feats, global_feats.T) / temperature
            labels = torch.arange(local_feats.size(0), device=local_feats.device)
            return F.cross_entropy(logits, labels)

        with tqdm(self.eval_loader, unit="batch", desc="CreamFL loss") as tepoch:
            for b_idx, inputs in enumerate(tepoch):
                frames, label, text, _ = inputs
                images = frames.cuda()
                label = label.cuda()

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = self.model(self.tokenizer, images, text)

                bs = images.size(0)
                start = b_idx * bs
                end = start + bs

                # Intra-modal contrastive loss
                logits_img, labels_img = self.compute_contrastive_loss_intra(
                    output["image_features"],
                    global_img[start:end],
                    prev_img_feat[start:end],
                )
                logits_txt, labels_txt = self.compute_contrastive_loss_intra(
                    output["caption_features"],
                    global_txt[start:end],
                    prev_txt_feat[start:end],
                )
                logits = torch.cat([logits_img, logits_txt], dim=0) / 0.5
                labels = torch.cat([labels_img, labels_txt], dim=0)
                loss_intra = criterion(logits, labels)

                # Inter-modal contrastive loss
                loss_inter_1 = contrastive_loss_pair(
                    output["image_features"], global_txt[start:end],
                )
                loss_inter_2 = contrastive_loss_pair(
                    output["caption_features"], global_img[start:end],
                )
                loss_inter = loss_inter_1 + loss_inter_2

                if not (torch.isnan(loss_intra) or torch.isnan(loss_inter)):
                    logger.debug(
                        "Inter: %.4f, Intra: %.4f", loss_inter.item(), loss_intra.item(),
                    )
                    batch_loss = (loss_intra + loss_inter) * float(
                        self.config.train.interintra_weight
                    )
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    self.optimizer.step()
                    del batch_loss

                del loss_inter_1, loss_inter_2, loss_inter
                gc.collect()

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

    def run(self, comms: int) -> None:
        """Run local training + CreamFL distillation."""
        self.model.cuda()
        for i in range(self.config.train.local_epoch):
            if self.client_id == -1:
                logger.info(
                    "Server — round %d, local_epoch %d, round_epoch %d",
                    comms, self.cur_epoch, i,
                )
            else:
                logger.info(
                    "Client %d — local_epoch %d, round %d, round_epoch %d",
                    self.client_id, self.local_epoch, comms, i,
                )
            self.train_epoch()
            self.run_cream()
            if self.client_id == -1:
                self.cur_epoch += 1
            else:
                self.local_epoch += 1

        self.model.cpu()
        gc.collect()


class CreamflClient(FedavgClient):
    """CreamFL client — uses contrastive representation distillation."""

    def __init__(
        self, args: Any, config: Any, client_id: int, wandb: Any = None,
    ) -> None:
        super().__init__(args, config, client_id, wandb)

    def build_trainer(self) -> None:
        """Instantiate the CreamFL trainer."""
        self.trainer = CreamflClientTrainer(
            self.args, self.config, self.wandb, client_id=self.client_id,
        )

    def update(
        self,
        global_img_feat: torch.Tensor,
        global_txt_feat: torch.Tensor,
        comm_round: int = 0,
    ) -> None:
        """Update using CreamFL contrastive distillation."""
        self.trainer.set_global_features(global_img_feat, global_txt_feat)
        self.trainer.old_model = copy.deepcopy(self.trainer.model)
        self.trainer.old_model.cpu()
        self.trainer.old_model.eval()
        self.trainer.run(comms=comm_round)