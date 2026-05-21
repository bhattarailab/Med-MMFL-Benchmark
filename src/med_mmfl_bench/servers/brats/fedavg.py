"""FedAvg server for BraTS multimodal segmentation.

Orchestrates federated averaging across BraTS clients with Dice-based
evaluation for Whole Tumour (WT), Tumour Core (TC), and Enhancing Tumour
(ET) composite regions.
"""

import gc
import os
import pickle
from importlib import import_module
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F_nn
from tqdm import tqdm
from torchmetrics import MetricCollection

from med_mmfl_bench.metrics import get_metrics
from med_mmfl_bench.models import get_model
from med_mmfl_bench.servers import BaseServer
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)


class FedavgServer(BaseServer):
    """FedAvg server for BraTS multimodal segmentation.

    Args:
        args: CLI arguments.
        config: YAML configuration object.
        val_dataset: Validation dataset.
        test_dataset: Test dataset.
        client_datasets: List of per-client dataset dicts.
        wandb: Optional WandB run instance, or ``False``.
    """

    def __init__(
        self, args: Any, config: Any,
        val_dataset: Any, test_dataset: Any,
        client_datasets: List[Dict[str, Any]],
        wandb: Any = False,
    ) -> None:
        super().__init__()
        self.args = args
        self.config = config
        self.wandb = wandb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.round: int = 0
        self.val_track: List[float] = []
        self.val_metric: float = 0.0

        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self._init_model()
        self.evaluator = self._build_evaluator().to(self.device)

        self.curr_lr = self.config.optimizer.learning_rate
        self.clients, self.client_ids = self._create_clients(client_datasets)
        self.num_of_clients = len(self.client_ids)

        self.val_dataloader, self.test_dataloader = self._get_dataloader()
        self._get_algorithm()

        self.save_dir = os.path.join(self.args.exp_dir, self.args.name, "server")
        os.makedirs(self.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_model(self) -> None:
        """Build the global segmentation model."""
        self.global_model = get_model(self.config.model.name, self.config.model)

    def _get_dataloader(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Create validation and test DataLoaders."""
        kwargs = dict(
            shuffle=False,
            num_workers=self.config.dataloader.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.config.dataloader.eval_batch_size, **kwargs,
        )
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.config.dataloader.eval_batch_size, **kwargs,
        )
        return val_loader, test_loader

    def _build_evaluator(self) -> MetricCollection:
        """Build a MetricCollection from the config."""
        if not isinstance(self.config.metrics, list):
            self.config.metrics = [self.config.metrics]
        logger.info("Metrics: %s", self.config.metrics)
        return MetricCollection(get_metrics(self.config.metrics)).to(self.device)

    def _get_algorithm(self, **kwargs: Any) -> Any:
        """Dynamically load the FL algorithm optimizer."""
        algo_module = import_module(
            f"med_mmfl_bench.algorithms.{self.args.algorithm}", package=__package__,
        )
        cls_name = f"{self.args.algorithm.title()}Optimizer"
        self.server_optimizer = algo_module.__dict__[cls_name](model=self.global_model, **kwargs)
        return self.server_optimizer

    def _create_clients(self, client_datasets: List[Dict[str, Any]]) -> Tuple[List[Any], List[Any]]:
        """Instantiate per-client objects."""
        client_module = import_module(
            f"med_mmfl_bench.clients.brats.{self.args.algorithm}", package=__package__,
        )
        cls_name = f"{self.args.algorithm.title()}Client"
        client_cls = client_module.__dict__[cls_name]

        clients: List[Any] = []
        client_ids: List[Any] = []
        for datasets in client_datasets:
            client = client_cls(
                args=self.args, config=self.config,
                client_id=datasets["client_id"],
                training_set=datasets["train_set"],
                test_set=datasets["val_set"],
                wandb=self.wandb,
            )
            client.set_id(datasets["client_id"])
            clients.append(client)
            client_ids.append(client.id)

        return clients, client_ids

    # ------------------------------------------------------------------
    # BraTS Dice evaluation (shared logic)
    # ------------------------------------------------------------------

    @staticmethod
    def _to_brats_composite(
        predictions: torch.Tensor, targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert 4-class predictions/targets to 3-region BraTS composites.

        Returns ``(combined_pred, combined_target)`` each of shape ``[B, 3, H, W, (D)]``
        for Whole Tumour (WT), Tumour Core (TC), and Enhancing Tumour (ET).
        """
        # predictions: softmax output [B, C, H, W, (D)]
        pred_classes = torch.argmax(predictions, dim=1)
        num_classes = predictions.shape[1]
        pred_oh = F_nn.one_hot(pred_classes, num_classes)
        pred_oh = pred_oh.permute(0, -1, *range(1, pred_oh.ndim - 1)).float()

        # Composite regions from one-hot channels
        net_p, ed_p, et_p = pred_oh[:, 1], pred_oh[:, 2], pred_oh[:, 3]
        wt_p = torch.clamp(net_p + ed_p + et_p, 0, 1)
        tc_p = torch.clamp(net_p + et_p, 0, 1)
        combined_pred = torch.stack([wt_p, tc_p, et_p], dim=1)

        net_t, ed_t, et_t = targets[:, 1], targets[:, 2], targets[:, 3]
        wt_t = torch.clamp(net_t + ed_t + et_t, 0, 1)
        tc_t = torch.clamp(net_t + et_t, 0, 1)
        combined_target = torch.stack([wt_t, tc_t, et_t], dim=1)

        return combined_pred, combined_target

    def _run_eval(
        self, dataloader: torch.utils.data.DataLoader, phase: str = "val",
    ) -> Dict[str, Any]:
        """Run evaluation on a dataloader and return metrics.

        Args:
            dataloader: DataLoader yielding ``(x, _, target, mask, _)`` tuples.
            phase: ``"val"`` or ``"test"`` — used for logging and WandB.
        """
        self.global_model.eval()
        self.global_model.is_training = False
        self.global_model.cuda()
        self.evaluator.to("cuda")

        with torch.no_grad():
            with tqdm(dataloader, unit="batch", desc=f"Eval ({phase})") as tepoch:
                for x, _, target, mask, _ in tepoch:
                    x = x.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                    mask = mask.cuda(non_blocking=True)

                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self.global_model(x, mask)

                    output = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
                    combined_pred, combined_target = self._to_brats_composite(output, target)
                    self.evaluator.update(combined_pred, combined_target)

        metrics = self.evaluator.compute()
        for name, value in metrics.items():
            logger.info("%s %s: %s", phase.capitalize(), name, value)

        if self.wandb:
            self.wandb.log({
                f"server/{phase}/{k}": v.item() if hasattr(v, "item") else v
                for k, v in metrics.items()
            })

        self.evaluator.reset()
        self.global_model.cpu()
        return metrics

    def validate_model(self) -> float:
        """Validate the global model and return ``dice_average``."""
        logger.info("Validating server model.")
        metrics = self._run_eval(self.val_dataloader, phase="val")
        self.val_track.append(metrics["dice_average"])
        return metrics["dice_average"]

    def test(self) -> None:
        """Test the global model."""
        logger.info("Testing server model.")
        self._run_eval(self.test_dataloader, phase="test")

    # ------------------------------------------------------------------
    # Client orchestration
    # ------------------------------------------------------------------

    def _request(self, eval: bool = False) -> Optional[Dict[int, int]]:
        """Update or evaluate all clients sequentially."""
        if eval:
            for client in self.clients:
                if client.model is None:
                    client.download(self.global_model)
                client.evaluate()
                client.model = None
            return None
        else:
            update_sizes: Dict[int, int] = {}
            for client in self.clients:
                if client.model is None:
                    client.download(self.global_model)
                client.update(self.round)
                update_sizes[client.id] = len(client.training_set)
            return update_sizes

    def _aggregate(self) -> Dict[str, torch.Tensor]:
        """Collect client models and compute a weighted average."""
        logger.info(
            "[%s] [Round: %s] Aggregate updated signals!",
            self.args.algorithm.upper(), str(self.round).zfill(4),
        )
        omega: List[int] = []
        w: List[Dict[str, torch.Tensor]] = []

        for client in self.clients:
            logger.info("[Round %d] Aggregating client %s", self.round, client.id)
            omega.append(len(client))
            w.append(client.upload())

        return self.server_optimizer.aggregate(
            client_models=w, omega=np.array(omega),
        )

    def dispatch(self) -> None:
        """Broadcast the current global model to all clients."""
        for client in self.clients:
            client.download(self.global_model)

    def update(self) -> None:
        """Execute one communication round of federated learning."""
        self.dispatch()
        self._request(eval=False)
        gc.collect()
        torch.cuda.empty_cache()

        self.global_model.load_state_dict(self._aggregate())
        self.round += 1

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_best(self, round_num: int) -> None:
        """Save the best global model checkpoint."""
        ckpt_path = os.path.join(self.save_dir, "model_best.pth")
        torch.save({"net": self.global_model.state_dict(), "comms": round_num}, ckpt_path)
        logger.info(
            "[%s] [%s] [Round: %s] Saved best checkpoint.",
            self.args.algorithm.upper(),
            self.config.dataset.dset_name.upper(),
            str(round_num).zfill(4),
        )

    def load_best(self) -> None:
        """Load the best model checkpoint."""
        ckpt_path = os.path.join(self.save_dir, "model_best.pth")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.global_model.load_state_dict(checkpoint["net"])
        logger.info("Best model at comms round: %s", checkpoint["comms"])

    def save_log(self) -> None:
        """Persist validation metrics to disk."""
        log_path = os.path.join(self.save_dir, "val_metrics.pkl")
        with open(log_path, "wb") as f:
            pickle.dump(self.val_track, f)

    def evaluate(self) -> None:
        """Evaluate the best global model on the server's holdout set."""
        cur_metric = self.validate_model()
        if cur_metric > self.val_metric:
            self.val_metric = cur_metric
            self.save_best(self.round)
        self.test()
        self.save_log()
        gc.collect()

    def finalize(self) -> None:
        """Load the best checkpoint, test, and save logs."""
        self.load_best()
        self.test()
        self.save_log()