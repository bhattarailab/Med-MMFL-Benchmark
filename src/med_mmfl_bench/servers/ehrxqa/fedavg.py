"""FedAvg server for EHRXQA generative visual question answering.

Orchestrates federated learning across clients using the FedAvg
aggregation strategy. Handles model initialization, client management,
validation/test evaluation with QA metrics, and checkpoint saving.
"""

import gc
import os
import pickle
import random
from collections import ChainMap
from importlib import import_module
from typing import Any, Dict, List, Optional

import concurrent.futures
import numpy as np
import torch
from torchmetrics import MetricCollection
from tqdm import tqdm

from med_mmfl_bench.metrics import get_metrics
from med_mmfl_bench.models import get_model
from med_mmfl_bench.servers.base import BaseServer
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["FedavgServer"]


class FedavgServer(BaseServer):
    """FedAvg server for EHRXQA federated VQA training.

    Manages the global BLIP-based VQA model, dispatches it to clients,
    collects and aggregates updated weights, and evaluates using
    QA metrics (EM, F1, BLEU).

    Args:
        args: Experiment arguments (algorithm name, directories, etc.).
        config: Full configuration object.
        val_dataset: Server-side validation dataset.
        test_dataset: Server-side test dataset.
        client_datasets: List of per-client dataset dictionaries.
        wandb: WandB run object or False to disable logging.
    """

    def __init__(
        self,
        args: Any,
        config: Any,
        val_dataset: Any,
        test_dataset: Any,
        client_datasets: List[Dict[str, Any]],
        wandb: Any = False,
    ) -> None:
        super().__init__()
        self.args = args
        self.config = config
        self.wandb = wandb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.round = 0
        self.opt_kwargs = dict(
            lr=self.config.optimizer.learning_rate, momentum=0.9
        )

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
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_model(self) -> None:
        """Build the global model from configuration."""
        self.global_model = get_model(self.config.model.name, self.config.model)

    def _get_dataloader(
        self,
    ) -> tuple:
        """Create validation and test DataLoaders.

        Returns:
            Tuple of (val_dataloader, test_dataloader).
        """
        loader_kwargs = dict(
            batch_size=self.config.dataloader.eval_batch_size,
            shuffle=False,
            num_workers=self.config.dataloader.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        val_loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset, **loader_kwargs
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset, **loader_kwargs
        )
        return val_loader, test_loader

    def _build_evaluator(self) -> MetricCollection:
        """Build a MetricCollection from the configured metric names."""
        if not isinstance(self.config.metrics, list):
            self.config.metrics = [self.config.metrics]
        logger.info("Metrics: %s", self.config.metrics)
        metric_map = get_metrics(self.config.metrics)
        return MetricCollection(metric_map).to(self.device)

    def _get_algorithm(self, **kwargs: Any) -> Any:
        """Dynamically load the FL aggregation optimizer from ``algorithms/``.

        Returns:
            Initialized server optimizer instance.
        """
        algorithm_module = import_module(
            f"med_mmfl_bench.algorithms.{self.args.algorithm}", package=__package__
        )
        algorithm_cls = algorithm_module.__dict__[
            f"{self.args.algorithm.title()}Optimizer"
        ]
        self.server_optimizer = algorithm_cls(model=self.global_model, **kwargs)
        return self.server_optimizer

    def _create_clients(
        self, client_datasets: List[Dict[str, Any]]
    ) -> tuple:
        """Instantiate client objects for each partition.

        Args:
            client_datasets: List of per-client dataset dictionaries with
                ``client_id``, ``train_set``, and ``val_set`` keys.

        Returns:
            Tuple of (clients_list, client_ids_list).
        """
        client_module = import_module(
            f"med_mmfl_bench.clients.ehrxqa.{self.args.algorithm}", package=__package__
        )
        client_cls = client_module.__dict__[
            f"{self.args.algorithm.title()}Client"
        ]

        clients: List[Any] = []
        client_ids: List[Any] = []
        for datasets in client_datasets:
            client = client_cls(
                args=self.args,
                config=self.config,
                client_id=datasets["client_id"],
                training_set=datasets["train_set"],
                test_set=datasets["val_set"],
                wandb=self.wandb,
            )
            client.id = datasets["client_id"]
            clients.append(client)
            client_ids.append(client.id)

        return clients, client_ids

    # ------------------------------------------------------------------
    # Federated learning loop
    # ------------------------------------------------------------------

    def _request(
        self,
        eval: bool = False,
        retain_model: bool = False,
        save_raw: bool = False,
    ) -> Optional[Dict[str, int]]:
        """Dispatch update or evaluation requests to clients.

        Args:
            eval: If True, evaluate clients; otherwise train them.
            retain_model: If True, keep client models in memory after eval.
            save_raw: Unused; kept for API compatibility.

        Returns:
            Dictionary of {client_id: dataset_size} for training requests,
            or None for evaluation requests.
        """
        def _update_client(client: Any) -> tuple:
            if client.model is None:
                client.download(self.global_model)
            update_result = client.update(self.round)
            return (
                {client.id: len(client.training_set)},
                {client.id: update_result},
            )

        def _evaluate_client(client: Any) -> tuple:
            if client.model is None:
                client.download(self.global_model)
            eval_result = client.evaluate()
            if not retain_model:
                client.model = None
            return (
                {client.id: len(client.test_set)},
                {client.id: eval_result},
            )

        max_workers = max(
            1,
            min(
                len(self.clients) // (1 if eval else 2),
                os.cpu_count() - 1,
            ),
        )
        fn = _evaluate_client if eval else _update_client

        jobs, results = [], []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for client in self.clients:
                jobs.append(pool.submit(fn, client))
            for job in concurrent.futures.as_completed(jobs):
                results.append(job.result())

        sizes, outcomes = list(map(list, zip(*results)))
        sizes = dict(ChainMap(*sizes))

        if eval:
            return None
        return sizes

    def _aggregate(self) -> Dict[str, torch.Tensor]:
        """Aggregate client model weights using FedAvg.

        Returns:
            Averaged state dictionary for the global model.
        """
        logger.info(
            "[%s] [Round: %04d] Aggregate updated signals!",
            self.args.algorithm.upper(),
            self.round,
        )

        omega = []
        w = []
        for client in self.clients:
            omega.append(len(client))
            w.append(client.upload())

        omega_arr = np.array(omega)
        return self.server_optimizer.aggregate(client_models=w, omega=omega_arr)

    def _sample_clients(self) -> List[Any]:
        """Sample a fraction of clients for the current round.

        Returns:
            Sorted list of sampled client IDs.
        """
        k = self.num_of_clients
        c = getattr(self.config.train, "client_fraction", None) or 1.0
        num_sampled = max(int(c * k), 1)
        all_ids = [client.id for client in self.clients]
        return sorted(random.sample(all_ids, num_sampled))

    def dispatch(self) -> None:
        """Broadcast the global model to all clients."""
        for client in self.clients:
            client.download(self.global_model)

    def update(self) -> None:
        """Execute one FedAvg communication round.

        Steps:
            1. Dispatch global model to clients.
            2. Clients perform local training.
            3. Aggregate updated weights into the global model.
        """
        self.dispatch()
        self._request(eval=False, retain_model=True)

        gc.collect()
        torch.cuda.empty_cache()

        self.global_model.load_state_dict(self._aggregate())
        self.evaluate()
        self.round += 1

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def validate_model(self) -> float:
        """Validate the global model on the server validation set.

        Returns:
            The primary validation metric (F1 score).
        """
        self.global_model.eval()
        self.global_model.to(self.device)
        self.evaluator.to(self.device)

        logger.info("Validating server model (round %d)...", self.round)

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, unit="batch", desc="Val"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    generated_ids = self.global_model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        max_length=int(self.config.model.max_length),
                    )

                preds = self.global_model.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                refs = self.global_model.tokenizer.batch_decode(
                    batch["labels"], skip_special_tokens=True
                )
                self.evaluator.update(preds, refs)

        metrics = self.evaluator.compute()
        for name, value in metrics.items():
            logger.info("  %s: %s", name, value)

        if self.wandb:
            self.wandb.log({
                f"server/val/{k}": v.item() if hasattr(v, "item") else v
                for k, v in metrics.items()
            })

        self.val_track.append(metrics["F1"])
        self.evaluator.reset()
        self.global_model.to("cpu")
        return metrics["F1"]

    def test(self) -> None:
        """Test the global model on the server test set."""
        self.global_model.eval()
        self.global_model.to(self.device)
        self.evaluator.to(self.device)

        logger.info("Testing server model...")

        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, unit="batch", desc="Test"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    generated_ids = self.global_model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        max_length=int(self.config.model.max_length),
                    )

                preds = self.global_model.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                refs = self.global_model.tokenizer.batch_decode(
                    batch["labels"], skip_special_tokens=True
                )
                self.evaluator.update(preds, refs)

        metrics = self.evaluator.compute()
        for name, value in metrics.items():
            logger.info("  %s: %s", name, value)

        if self.wandb:
            self.wandb.log({
                f"server/test/{k}": v.item() if hasattr(v, "item") else v
                for k, v in metrics.items()
            })

        self.evaluator.reset()
        self.global_model.to("cpu")

    def evaluate(self) -> None:
        """Run validation and test; save best model checkpoint."""
        cur_metric = self.validate_model()
        if cur_metric > self.val_metric:
            self.val_metric = cur_metric
            self.save_best(self.round)

        self.test()
        self.save_log()
        gc.collect()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_best(self, round: int) -> None:
        """Save the current global model as the best checkpoint.

        Args:
            round: Communication round number.
        """
        ckpt_path = os.path.join(self.save_dir, "model_best.pth")
        torch.save(
            {"net": self.global_model.state_dict(), "comms": round},
            ckpt_path,
        )
        logger.info(
            "[%s] [%s] [Round: %04d] Saved best global model checkpoint.",
            self.args.algorithm.upper(),
            self.config.dataset.dset_name.upper(),
            round,
        )

    def load_best(self) -> None:
        """Load the best model checkpoint."""
        ckpt_path = os.path.join(self.save_dir, "model_best.pth")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.global_model.load_state_dict(checkpoint["net"])
        logger.info("Best model is at comms round: %s", checkpoint["comms"])

    def save_log(self) -> None:
        """Save validation metric history to disk."""
        log_path = os.path.join(self.save_dir, "val_metrics.pkl")
        with open(log_path, "wb") as f:
            pickle.dump(self.val_track, f)

    def finalize(self) -> None:
        """Load the best model, run final test, and save logs."""
        self.load_best()
        self.test()
        self.save_log()