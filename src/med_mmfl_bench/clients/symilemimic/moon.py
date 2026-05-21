"""MOON client for SymileMIMIC multimodal retrieval."""

import copy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from med_mmfl_bench.clients.symilemimic.fedavg import FedavgClient
from med_mmfl_bench.losses import get_criterion
from med_mmfl_bench.models import get_model
from med_mmfl_bench.trainers.symilemimictrainer import ClientTrainer as SymileClientTrainer
from med_mmfl_bench.utils.logging import get_logger
from med_mmfl_bench.utils.optimizers import get_optimizer

logger = get_logger(__name__)


class MoonTrainer(SymileClientTrainer):
    """SymileMIMIC trainer with MOON contrastive loss."""

    def __init__(self, args: Any, config: Any, wandb: Any = None) -> None:
        super().__init__(args, config, wandb)
        self._mu = float(config.train.moon_mu)
        self.tempr = float(config.train.moon_tempr)
        self.global_model = get_model(self.config.model.name, self.config.model).cpu()
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.prev_net = copy.deepcopy(self.global_model).cpu()
        self.crossentropy = nn.CrossEntropyLoss()

    def loss_fn(
        self, r_c: torch.Tensor, r_e: torch.Tensor, r_l: torch.Tensor,
        logit_scale_exp: torch.Tensor, negative_sampling: int,
        batch: list,
    ) -> torch.Tensor:
        """Task loss + MOON contrastive loss across all 3 modalities."""
        task_loss = self.criterion(r_c, r_e, r_l, logit_scale_exp, negative_sampling)

        with torch.no_grad():
            self.global_model.cuda()
            r_c_g, r_e_g, r_l_g, _ = self.global_model(batch)
            self.global_model.cpu()
            self.prev_net.cuda()
            r_c_prev, r_e_prev, r_l_prev, _ = self.prev_net(batch)
            self.prev_net.cpu()

        # Flatten and normalise
        r_c_g = F.normalize(r_c_g.view(r_c_g.size(0), -1), dim=-1)
        r_e_g = F.normalize(r_e_g.view(r_e_g.size(0), -1), dim=-1)
        r_l_g = F.normalize(r_l_g.view(r_l_g.size(0), -1), dim=-1)

        r_c_prev = F.normalize(r_c_prev.view(r_c_prev.size(0), -1), dim=-1)
        r_e_prev = F.normalize(r_e_prev.view(r_e_prev.size(0), -1), dim=-1)
        r_l_prev = F.normalize(r_l_prev.view(r_l_prev.size(0), -1), dim=-1)

        r_c = F.normalize(r_c, dim=-1)
        r_e = F.normalize(r_e, dim=-1)
        r_l = F.normalize(r_l, dim=-1)

        # Contrastive logits: positive = global, negative = previous
        logits_cxr = torch.cat([
            self.cosine_similarity(r_c, r_c_g).view(-1, 1),
            self.cosine_similarity(r_c, r_c_prev).view(-1, 1),
        ], dim=1) / self.tempr

        logits_ecg = torch.cat([
            self.cosine_similarity(r_e, r_e_g).view(-1, 1),
            self.cosine_similarity(r_e, r_e_prev).view(-1, 1),
        ], dim=1) / self.tempr

        logits_labs = torch.cat([
            self.cosine_similarity(r_l, r_l_g).view(-1, 1),
            self.cosine_similarity(r_l, r_l_prev).view(-1, 1),
        ], dim=1) / self.tempr

        dummy_labels = torch.zeros(batch[0].size(0), dtype=torch.long, device="cuda")
        l_con = (
            self.crossentropy(logits_cxr, dummy_labels)
            + self.crossentropy(logits_ecg, dummy_labels)
            + self.crossentropy(logits_labs, dummy_labels)
        )
        return task_loss + self._mu * l_con

    def train(self) -> None:
        """One training epoch with MOON contrastive loss."""
        self.model.cuda()
        self.model.train()
        with tqdm(self.train_loader, unit="batch", desc="Training (MOON)") as tepoch:
            for inputs in tepoch:
                cxr, ecg, labs_p, labs_m, hadm_id, _ = inputs
                cxr, ecg, labs_p, labs_m, hadm_id = (
                    cxr.cuda(), ecg.cuda(), labs_p.cuda(), labs_m.cuda(), hadm_id.cuda(),
                )
                batch = [cxr, ecg, labs_p, labs_m, hadm_id]

                self.optimizer.zero_grad()
                r_c, r_e, r_l, logit_scale_exp = self.model(batch)
                loss = self.loss_fn(
                    r_c, r_e, r_l, logit_scale_exp,
                    self.config.train.negative_sampling, batch,
                )

                if self.wandb:
                    log_n = np.log(len(batch[0]))
                    self.wandb.log({
                        f"[CLIENT {self.client_id}] train_loss": loss.item(),
                        f"[CLIENT {self.client_id}] logit_scale_exp": logit_scale_exp,
                        f"[CLIENT {self.client_id}] log_n": log_n,
                    })

                loss.backward()
                self.optimizer.step()
                tepoch.set_postfix(Loss=loss.item())

        self.model.eval()
        self.model.cpu()

    def run_train(self) -> None:
        """Run local epochs and update previous model snapshot."""
        for _ in range(self.config.train.local_epoch):
            self.train()
        self.prev_net.load_state_dict(self.model.state_dict())


class MoonClient(FedavgClient):
    """MOON client for SymileMIMIC."""

    def __init__(
        self, args: Any, config: Any, client_id: int,
        training_set: Any, test_set: Any, wandb: Any = None,
    ) -> None:
        super().__init__(args, config, client_id, training_set, test_set, wandb)

    def build_trainer(self) -> None:
        """Instantiate the MOON trainer."""
        logger.info("Building MOON trainer for client %d.", self.client_id)
        self.trainer = MoonTrainer(self.args, self.config, self.wandb)
        self.trainer.model = get_model(self.config.model.name, self.config.model)
        self.trainer.model.cpu()
        self.trainer.optimizer = get_optimizer(
            self.config.optimizer.name, self.trainer.model.parameters(), self.config.optimizer,
        )
        self.trainer.criterion = get_criterion(self.config.criterion.name)
        self.trainer.train_loader = self.train_loader
        self.trainer.test_loader = self.test_loader
        self.trainer.client_id = self.client_id