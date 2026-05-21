"""FedNova client for SymileMIMIC multimodal retrieval."""

import copy
from typing import Any, Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm

from med_mmfl_bench.clients.symilemimic.fedavg import FedavgClient
from med_mmfl_bench.losses import get_criterion
from med_mmfl_bench.models import get_model
from med_mmfl_bench.trainers.symilemimictrainer import ClientTrainer as SymileClientTrainer
from med_mmfl_bench.utils.logging import get_logger
from med_mmfl_bench.utils.optimizers import get_optimizer

logger = get_logger(__name__)


class SymileNovaTrainer(SymileClientTrainer):
    """SymileMIMIC trainer with FedNova normalised gradient tracking."""

    def __init__(self, args: Any, config: Any, wandb: Any = None) -> None:
        super().__init__(args, config, wandb)
        self.ai: float = 0.0
        self.di: Dict[str, torch.Tensor] = {}
        self.rho: float = float(config.optimizer.momentum)

    def run_train(self) -> None:
        """Run FedNova training with step-normalised gradients."""
        global_weights = copy.deepcopy(self.model.state_dict())
        tau = 0
        for _ in range(self.config.train.local_epoch):
            tau = self.train(tau)
            self.ai = (
                tau - self.rho * (1 - pow(self.rho, tau)) / (1 - self.rho)
            ) / (1 - self.rho)
            state_dict = self.model.state_dict()
            self.di = copy.deepcopy(global_weights)
            for key in self.di:
                self.di[key] = torch.div(global_weights[key] - state_dict[key], self.ai)

    def train(self, tau: int) -> int:
        """One training epoch with step counting."""
        self.model.cuda()
        self.model.train()
        with tqdm(self.train_loader, unit="batch", desc="Training (FedNova)") as tepoch:
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

                if self.wandb:
                    log_n = np.log(len(batch[0]))
                    self.wandb.log({
                        f"[CLIENT {self.client_id}] train_loss": loss.item(),
                        f"[CLIENT {self.client_id}] logit_scale_exp": logit_scale_exp,
                        f"[CLIENT {self.client_id}] log_n": log_n,
                    })
                tau += 1
                tepoch.set_postfix(Loss=loss.item())

        self.model.eval()
        self.model.cpu()
        return tau


class FednovaClient(FedavgClient):
    """FedNova client for SymileMIMIC — uploads normalised gradient directions."""

    def __init__(
        self, args: Any, config: Any, client_id: int,
        training_set: Any, test_set: Any, wandb: Any = None,
    ) -> None:
        super().__init__(args, config, client_id, training_set, test_set, wandb)

    def build_trainer(self) -> None:
        """Instantiate the FedNova trainer."""
        self.trainer = SymileNovaTrainer(self.args, self.config, self.wandb)
        self.trainer.model = get_model(self.config.model.name, self.config.model)
        self.trainer.model.cpu()
        self.trainer.optimizer = get_optimizer(
            self.config.optimizer.name, self.trainer.model.parameters(), self.config.optimizer,
        )
        self.trainer.criterion = get_criterion(self.config.criterion.name)
        self.trainer.train_loader = self.train_loader
        self.trainer.test_loader = self.test_loader
        self.trainer.client_id = self.client_id

    def upload(self) -> Tuple[float, Dict[str, torch.Tensor]]:
        """Upload normalisation coefficient and gradient direction."""
        logger.info("Client %d uploading FedNova updates.", self.client_id)
        return copy.deepcopy(self.trainer.ai), copy.deepcopy(self.trainer.di)