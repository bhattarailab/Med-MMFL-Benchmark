"""SCAFFOLD client for BraTS multimodal segmentation."""

import copy
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from tqdm import tqdm

from med_mmfl_bench.clients.brats.fedavg import FedavgClient
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)


class ScaffoldClient(FedavgClient):
    """SCAFFOLD client with control variate correction for BraTS."""

    def __init__(
        self, args: Any, config: Any, client_id: int,
        training_set: Any, test_set: Any, wandb: Any = None,
    ) -> None:
        super().__init__(args, config, client_id, training_set, test_set, wandb)

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
            control[name] = control[name].to(self.device) if to_cuda else control[name].cpu()

    def _apply_control_variates(
        self, model: torch.nn.Module,
        server_control: Dict[str, torch.Tensor],
        client_control: Dict[str, torch.Tensor],
    ) -> None:
        """Apply SCAFFOLD gradient correction before the optimizer step."""
        for name, param in model.named_parameters():
            if param.grad is not None and "running" not in name and "num_batch" not in name:
                param.grad.data += server_control[name].data - client_control[name].data

    @staticmethod
    def _get_delta_model(
        model0: torch.nn.Module, model1: torch.nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """Compute the parameter delta between two models."""
        return {
            name: param0.detach() - model1.state_dict()[name].detach()
            for name, param0 in model0.state_dict().items()
        }

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
            c = server_control[name]
            ci = client_control[name]
            delta = delta_model[name]

            new_ci = ci.data - c.data + delta.data / (steps * lr)
            new_control[name].data = new_ci
            delta_control[name].data = ci.data - new_ci

        logger.debug("Local control updated.")
        return new_control, delta_control

    def update(self, comm_round: int = 0) -> Dict[str, float]:
        """Run SCAFFOLD local training with control variate correction."""
        last_global_model = copy.deepcopy(self.model)
        self.model.train()
        self.model.to(self.device)
        last_global_model.to(self.device)
        self._create_optimizer_and_scaler()

        n_total_bs = int(self.config.train.local_epoch * len(self.train_loader))

        logger.info("Client %s (SCAFFOLD) training, round %d", self.client_id, comm_round)
        for i in range(self.config.train.local_epoch):
            epoch_loss: List[float] = []
            with tqdm(self.train_loader, unit="batch", desc=f"Client {self.client_id} (SCAFFOLD)") as tepoch:
                for x, _, target, mask, _ in tepoch:
                    self.optimizer.zero_grad()
                    x = x.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                    mask = mask.cuda(non_blocking=True)

                    self.model.is_training = True
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        fuse_pred, _, sep_preds, prm_preds = self.model(x, mask)
                        loss = self._compute_brats_loss(
                            fuse_pred, sep_preds, prm_preds, target, comm_round,
                        )

                    self.grad_scaler.scale(loss).backward()

                    # SCAFFOLD gradient correction
                    self._apply_control_variates(
                        self.model,
                        server_control=copy.deepcopy(self.server_control),
                        client_control=copy.deepcopy(self.client_control),
                    )

                    if self.config.train.grad_clip > 0:
                        self.grad_scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.train.grad_clip,
                        )
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()

                    #cleanup to save memory
                    del fuse_pred, sep_preds, prm_preds, x, target, mask
                    torch.cuda.empty_cache()

                    tepoch.set_postfix(loss=loss.item())
                    if torch.isfinite(loss):
                        epoch_loss.append(loss.item())

            avg_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0.0
            logger.info(
                "[CLIENT %s] round %d, epoch %d/%d, avg_loss=%.4f",
                self.client_id, comm_round, i + 1, self.config.train.local_epoch, avg_loss,
            )
            if self.wandb:
                self.wandb.log({f"client/{self.client_id}/train_loss": avg_loss})

        # Update control variates
        delta_model = self._get_delta_model(last_global_model, self.model)
        client_control, delta_control = self._update_local_control(
            delta_model=delta_model,
            server_control=self.server_control,
            client_control=self.client_control,
            steps=n_total_bs,
            lr=float(self.config.optimizer.learning_rate),
        )
        self.client_control = copy.deepcopy(client_control)
        self.delta_control = copy.deepcopy(delta_control)

        logger.info("Client %s SCAFFOLD update completed.", self.client_id)
        self._set_control_device(self.client_control, to_cuda=False)
        self.model.cpu()
        return {"loss": loss.item()}