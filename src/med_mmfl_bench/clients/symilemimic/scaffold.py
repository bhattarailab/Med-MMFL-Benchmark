"""SCAFFOLD client for SymileMIMIC multimodal retrieval."""

import copy
import gc
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm

from med_mmfl_bench.clients.symilemimic.fedavg import FedavgClient
from med_mmfl_bench.losses import get_criterion
from med_mmfl_bench.models import get_model
from med_mmfl_bench.utils.logging import get_logger
from med_mmfl_bench.utils.optimizers import get_optimizer

logger = get_logger(__name__)


class ScaffoldTrainer:
    """SymileMIMIC trainer with SCAFFOLD control variate correction.

    This is a standalone trainer (not inheriting from
    :class:`ClientTrainer`) because SCAFFOLD requires direct control over
    gradient modification before the optimizer step.
    """

    def __init__(self, args: Any, config: Any, wandb: Any = None) -> None:
        self.args = args
        self.config = config
        self.wandb = wandb

        self.save_dir = self.args.exp_dir
        self.best_val_loss = float("inf")
        os.makedirs(self.save_dir, exist_ok=True)
        self.track_val_data: list = []

    def loss_fn(
        self, r_c: torch.Tensor, r_e: torch.Tensor, r_l: torch.Tensor,
        logit_scale_exp: torch.Tensor, negative_sampling: int,
    ) -> torch.Tensor:
        """Compute the Symile loss."""
        return self.criterion(r_c, r_e, r_l, logit_scale_exp, negative_sampling)

    def set_control(
        self,
        server_control: Dict[str, torch.Tensor],
        client_control: Dict[str, torch.Tensor],
    ) -> None:
        """Set server and client control variates."""
        self.server_control = server_control
        self.client_control = client_control
        self._set_control_device(self.client_control, to_cuda=True)

    def _set_control_device(
        self, control: Dict[str, torch.Tensor], to_cuda: bool = True,
    ) -> None:
        """Move control variates to cuda or cpu."""
        for name in control:
            control[name] = control[name].to("cuda") if to_cuda else control[name].cpu()

    def _apply_control_variates(
        self, model: torch.nn.Module,
        server_control: Dict[str, torch.Tensor],
        client_control: Dict[str, torch.Tensor],
    ) -> None:
        _scale = 1e-3  # SCAFFOLD correction scaling factor for symile multimodal retrieval (tuned for stability)
        """Apply control variate correction to gradients before optimizer step."""
        for name, param in model.named_parameters():
            if param.grad is not None and ("weight" in name or "bias" in name):
                param.grad.data += (
                    server_control[name].data - client_control[name].data
                ) * _scale

    def train(self) -> None:
        """One training epoch with SCAFFOLD control variate correction."""
        self.model.cuda()
        self.model.train()

        with tqdm(self.train_loader, unit="batch", desc="Training (SCAFFOLD)") as tepoch:
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

                self._apply_control_variates(
                    self.model,
                    server_control=copy.deepcopy(self.server_control),
                    client_control=copy.deepcopy(self.client_control),
                )
                self.optimizer.step()

                if self.wandb:
                    log_n = np.log(len(batch[0]))
                    self.wandb.log({
                        f"[CLIENT {self.client_id}] train_loss": loss.item(),
                        f"[CLIENT {self.client_id}] logit_scale_exp": logit_scale_exp,
                        f"[CLIENT {self.client_id}] log_n": log_n,
                    })
                tepoch.set_postfix(Loss=loss.item())

                del r_c, r_e, r_l, loss, batch
                torch.cuda.empty_cache()
                gc.collect()

        self.model.eval()
        self.model.cpu()

    def run_train(self) -> None:
        """Run SCAFFOLD local training with control variate updates."""
        last_global_model = copy.deepcopy(self.model)
        last_global_model.cpu()

        n_total_bs = int(self.config.train.local_epoch * len(self.train_loader))
        for _ in range(self.config.train.local_epoch):
            self.train()

        delta_model = self._get_delta_model(last_global_model, copy.deepcopy(self.model))

        if delta_model:
            logger.info("Delta model successfully computed.")

        client_control, delta_control = self._update_local_control(
            delta_model=delta_model,
            server_control=self.server_control,
            client_control=self.client_control,
            steps=n_total_bs,
            lr=float(self.config.optimizer.learning_rate),
        )

        self.client_control = copy.deepcopy(client_control)
        self.delta_control = copy.deepcopy(delta_control)

        logger.info("Client SCAFFOLD update completed.")
        self._set_control_device(self.client_control, to_cuda=False)

    @staticmethod
    def _get_delta_model(
        model0: torch.nn.Module, model1: torch.nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """Compute the parameter delta between two models."""
        state_dict: Dict[str, torch.Tensor] = {}
        for name, param0 in model0.state_dict().items():
            if "weight" in name or "bias" in name:
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
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Update local control variate using the SCAFFOLD update rule."""
        new_control = copy.deepcopy(client_control)
        delta_control = copy.deepcopy(client_control)

        for name in delta_model:
            ci = client_control[name]
            device = ci.device
            c = server_control[name].to(device)
            delta = delta_model[name].to(device)

            new_ci = ci.data - c.data + delta.data / (steps * lr)
            new_control[name].data = new_ci
            delta_control[name].data = ci.data - new_ci

        logger.info("Local control successfully updated.")
        return new_control, delta_control


class ScaffoldClient(FedavgClient):
    """SCAFFOLD client for SymileMIMIC."""

    def __init__(
        self, args: Any, config: Any, client_id: int,
        training_set: Any, test_set: Any, wandb: Any = None,
    ) -> None:
        super().__init__(args, config, client_id, training_set, test_set, wandb)
        self.config = config

    def build_trainer(self) -> None:
        """Instantiate the SCAFFOLD trainer."""
        self.trainer = ScaffoldTrainer(self.args, self.config, self.wandb)
        self.trainer.model = get_model(self.config.model.name, self.config.model)
        self.trainer.model.cpu()
        self.trainer.optimizer = get_optimizer(
            self.config.optimizer.name, self.trainer.model.parameters(), self.config.optimizer,
        )
        self.trainer.criterion = get_criterion(self.config.criterion.name)
        self.trainer.train_loader = self.train_loader
        self.trainer.test_loader = self.test_loader
        self.trainer.client_id = self.client_id