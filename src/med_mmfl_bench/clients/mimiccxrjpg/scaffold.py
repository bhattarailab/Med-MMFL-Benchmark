"""SCAFFOLD client for MIMIC-CXR multimodal classification."""

import copy
from collections import defaultdict
from typing import Any, Dict, Tuple

import torch
from tqdm import tqdm

from med_mmfl_bench.clients.mimiccxrjpg.fedavg import FedavgClient
from med_mmfl_bench.trainers.mimiccxr import ClassificationTrainer as MimicClientTrainer
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)


class MimicScaffoldTrainer(MimicClientTrainer):
    """MIMIC-CXR trainer with SCAFFOLD control variate correction."""

    def __init__(
        self, args: Any, config: Any, wandb: Any = None, client_id: int = -1,
    ) -> None:
        super().__init__(args, config, wandb, client_id=client_id)

    def _apply_control_variates(self) -> None:
        """Apply control variate correction to gradients before optimizer step."""
        for name, param in self.model.named_parameters():
            if param.grad is not None and "running" not in name and "num_batch" not in name:
                param.grad.data += (
                    self.server_control[name].data - self.client_control[name].data
                )

    def train_epoch(self) -> None:
        """One training epoch with SCAFFOLD control variate correction."""
        self.model.train()
        self.model.cuda()
        with tqdm(self.train_loader, unit="batch", desc="Training (SCAFFOLD)") as tepoch:
            for frames, label, text, _ in tepoch:
                self.optimizer.zero_grad()
                images = frames.cuda()
                label = label.cuda()

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = self.model(self.tokenizer, images, text)
                    loss = self.criterion(output["logits"], label)

                self.grad_scaler.scale(loss).backward()
                self._apply_control_variates()

                if self.config.train.grad_clip > 0:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad.clip_grad_norm_(
                        self.model.parameters(), self.config.train.grad_clip,
                    )

                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                tepoch.set_postfix(Loss=loss.item())


class ScaffoldClient(FedavgClient):
    """SCAFFOLD client with control variate tracking."""

    def __init__(
        self, args: Any, config: Any, client_id: int, wandb: Any = None,
    ) -> None:
        super().__init__(args, config, client_id, wandb)
        self.control_logger = ControlLogger(verbose=True)

    def build_trainer(self) -> None:
        """Instantiate the SCAFFOLD trainer."""
        self.trainer = MimicScaffoldTrainer(
            self.args, self.config, self.wandb, client_id=self.client_id,
        )

    def set_control(
        self,
        server_control: Dict[str, torch.Tensor],
        client_control: Dict[str, torch.Tensor],
    ) -> None:
        """Set server and client control variates on the trainer."""
        self.trainer.server_control = server_control
        self.trainer.client_control = client_control
        self._set_control_device(self.trainer.client_control, to_cuda=True)

    def _set_control_device(
        self, control: Dict[str, torch.Tensor], to_cuda: bool = True,
    ) -> None:
        """Move control variates to cuda or cpu."""
        for name in control:
            control[name] = control[name].to(self.device) if to_cuda else control[name].cpu()

    def update(self, comm_round: int = 0) -> None:
        """Run one SCAFFOLD local round with control variate updates."""
        last_global_model = copy.deepcopy(self.trainer.model)
        last_global_model.cpu()

        n_total_bs = int(self.config.train.local_epoch * len(self.trainer.train_loader))
        self.trainer.run(comm_round)

        delta_model = self._get_delta_model(last_global_model, self.trainer.model)

        client_control, delta_control = self._update_local_control(
            delta_model=delta_model,
            server_control=self.trainer.server_control,
            client_control=self.trainer.client_control,
            steps=n_total_bs,
            lr=float(self.trainer.optimizer.param_groups[0]["lr"]),
        )

        self.trainer.client_control = copy.deepcopy(client_control)
        self.trainer.delta_control = copy.deepcopy(delta_control)

        self.control_logger.log_round(self.trainer.client_control, round_num=comm_round)

        logger.info("Client %d SCAFFOLD update completed.", self.client_id)
        self._set_control_device(self.trainer.client_control, to_cuda=False)
        self.trainer.model.cpu()

    @staticmethod
    def _get_delta_model(
        model0: torch.nn.Module, model1: torch.nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """Compute the parameter delta between two models."""
        state_dict: Dict[str, torch.Tensor] = {}
        model0.cuda()
        model1.cuda()
        for name, param0 in model0.state_dict().items():
            param1 = model1.state_dict()[name]
            state_dict[name] = param0.detach() - param1.detach()
        model0.cpu()
        model1.cpu()
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
            c = server_control[name]
            ci = client_control[name]
            delta = delta_model[name]

            new_ci = ci.data - c.data + delta.data / (steps * lr)
            new_control[name].data = new_ci
            delta_control[name].data = ci.data - new_ci

        logger.info("Local control successfully updated.")
        return new_control, delta_control


# ---------------------------------------------------------------------------
# Debugging utility — kept for development diagnostics
# ---------------------------------------------------------------------------

class ControlLogger:
    """Logs SCAFFOLD control variate statistics between rounds."""

    def __init__(self, verbose: bool = True) -> None:
        self.prev_client_control: Dict[str, torch.Tensor] = {}
        self.verbose = verbose

    def summarize_grouped(
        self, control_dict: Dict[str, torch.Tensor], name: str = "control",
    ) -> Dict[str, Dict[str, list]]:
        """Aggregate mean/std/norm per layer (grouped)."""
        grouped: Dict[str, Dict[str, list]] = defaultdict(
            lambda: {"mean": [], "std": [], "norm": []}
        )
        for k, v in control_dict.items():
            if isinstance(v, torch.Tensor):
                layer = k.split(".")[0]
                grouped[layer]["mean"].append(v.mean().item())
                grouped[layer]["std"].append(v.std().item())
                grouped[layer]["norm"].append(v.norm().item())

        if self.verbose:
            logger.debug("--- %s (Grouped Summary) ---", name.upper())
            for layer, stats in grouped.items():
                mean_m = sum(stats["mean"]) / len(stats["mean"])
                mean_s = sum(stats["std"]) / len(stats["std"])
                mean_n = sum(stats["norm"]) / len(stats["norm"])
                logger.debug(
                    "%25s | mean=%+.6f, std=%.6f, norm=%.6f",
                    layer, mean_m, mean_s, mean_n,
                )
        return dict(grouped)

    def compute_differences(
        self,
        prev_control: Dict[str, torch.Tensor],
        new_control: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Compute average absolute difference per layer."""
        diffs: Dict[str, list] = defaultdict(list)
        for k, new_val in new_control.items():
            if isinstance(new_val, torch.Tensor) and k in prev_control:
                prev_val = prev_control[k].to(new_val.device)
                diff = (new_val - prev_val).abs().mean().item()
                layer = k.split(".")[0]
                diffs[layer].append(diff)

        avg_diffs = {layer: sum(vals) / len(vals) for layer, vals in diffs.items()}

        if self.verbose:
            logger.debug("--- CONTROL Δ (Average Abs Diff per Layer) ---")
            for layer, val in avg_diffs.items():
                logger.debug("%25s | Δ=%.8f", layer, val)
        return avg_diffs

    def log_round(
        self, client_control: Dict[str, torch.Tensor], round_num: int = None,
    ) -> None:
        """Log control summary and per-round deltas."""
        tag = f"Round {round_num}" if round_num is not None else "Current"
        logger.debug("================= %s =================", tag)

        self.summarize_grouped(client_control, name="Client Control")

        if self.prev_client_control:
            self.compute_differences(self.prev_client_control, client_control)
        else:
            logger.debug("(First round — no previous control to compare)")

        self.prev_client_control = {
            k: v.clone().detach().cpu()
            for k, v in client_control.items()
            if isinstance(v, torch.Tensor)
        }