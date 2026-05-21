"""MIMIC-CXR classification trainer for federated and standalone training.

Provides the :class:`ClassificationTrainer` which encapsulates the full
training, validation, and evaluation pipeline for multi-label chest X-ray
pathology classification using image-text multimodal models on the
MIMIC-CXR dataset.

Supports three data-loading modes:
    - **Federated (server)**: Uses server-side partition indices.
    - **Federated (client)**: Uses per-client partition indices.
    - **Standalone**: Merges all client partitions for centralized training.
"""

import gc
import os
import pickle
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAUROC
from tqdm import tqdm

from med_mmfl_bench.datasets import MimicMultiModal
from med_mmfl_bench.losses import get_criterion
from med_mmfl_bench.models import get_model, get_tokenizer
from med_mmfl_bench.utils import get_optimizer
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)


class ClassificationTrainer:
    """Trainer for MIMIC-CXR multi-label pathology classification.

    Handles data loading, model initialization, training loops (both
    federated local rounds and standalone full training), validation,
    testing, and checkpoint management.

    Args:
        args: Parsed command-line arguments containing ``exp_dir`` and
            other runtime settings.
        config: Configuration object (``munch.Munch``) with sections for
            ``dataset``, ``dataloader``, ``model``, ``optimizer``,
            ``criterion``, and ``train``.
        wandb: Optional Weights & Biases run object for logging. Pass
            ``False`` to disable WandB logging.
        client_id: Client identifier. Use ``-1`` for the server role,
            or a non-negative integer for a specific client.

    Attributes:
        model: The multimodal classification model.
        tokenizer: BERT tokenizer for text encoding.
        criterion: Loss function (e.g., ``BCEWithLogitsLoss``).
        optimizer: PyTorch optimizer.
        evaluator: ``torchmetrics.MetricCollection`` for validation AUC.
        save_dir: Directory for checkpoints and logs.

    Example:
        >>> trainer = ClassificationTrainer(args, config, wandb=None)
        >>> trainer.run_standalone()  # Centralized training
        >>> trainer.run(comms=0)      # One federated round
    """

    def __init__(
        self,
        args: Any,
        config: Any,
        wandb: Any = False,
        client_id: int = -1,
    ) -> None:
        self.args = args
        self.wandb = wandb
        self.client_id = client_id
        self.config = config
        self.dset_name = self.config.dataset.dset_name

        self.load_data()
        self.load_model()

        self.evaluator = MetricCollection({
            "AUC": MultilabelAUROC(
                num_labels=14, average="macro", thresholds=None
            ),
        })
        self.cur_epoch = 0

        if self.client_id == -1:
            self.save_dir = os.path.join(self.args.exp_dir, "server")
        else:
            self.save_dir = os.path.join(
                os.path.dirname(self.args.exp_dir),
                f"client_{self.client_id}",
            )
        os.makedirs(self.save_dir, exist_ok=True)

        self.val_track: List[float] = []
        self.local_epoch = 0

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------

    def load_data(self) -> None:
        """Load and partition the MIMIC-CXR dataset for federated training.

        Creates train/val/test splits based on the client's role:
            - **Server** (``client_id == -1``): Uses the server partition for
              training, and loads full validation and test sets.
            - **Client**: Uses client-specific train/val partition indices.

        Sets the following instance attributes:
            ``train_set``, ``val_set``, ``test_set`` (if server),
            ``train_loader``, ``val_loader``, ``test_loader`` (if server).

        Raises:
            NotImplementedError: If ``dset_name`` is not ``"mimic-cxr"``.
        """
        if self.dset_name != "mimic-cxr":
            raise NotImplementedError(
                f"Dataset '{self.dset_name}' is not supported. "
                "Only 'mimic-cxr' is currently implemented."
            )

        train_set = MimicMultiModal(
            self.config.dataset.img_path,
            self.config.dataset.ann_path,
            self.config.dataset.view,
            "train",
        )

        partition_path = (
            f"partitions/"
            f"{self.dset_name}-{self.config.dataset.view}-"
            f"{self.config.dataset.partition}.pkl"
        )
        with open(partition_path, "rb") as f:
            data_partition = pickle.load(f)

        if self.client_id == -1:
            train_idx = data_partition["server"]
            self.train_set = Subset(train_set, train_idx)
            self.val_set = MimicMultiModal(
                self.config.dataset.img_path,
                self.config.dataset.ann_path,
                self.config.dataset.view,
                "val",
            )
            self.test_set = MimicMultiModal(
                self.config.dataset.img_path,
                self.config.dataset.ann_path,
                self.config.dataset.view,
                "test",
            )
        else:
            train_idx = data_partition["client"][self.client_id]["train"]
            val_idx = data_partition["client"][self.client_id]["val"]
            self.train_set = Subset(train_set, val_idx)
            self.val_set = Subset(train_set, val_idx)

        self._build_dataloaders()
        logger.info("Data loaded successfully.")

    def load_data_for_new_split(self) -> None:
        """Reload data with a different partition path variant.

        Uses the ``partitions/{dset_name}_{view}_{partition}.pkl`` path
        format (without the ``mimic-cxr-no-leak/`` subdirectory). Otherwise
        behaves identically to :meth:`load_data`.

        Raises:
            NotImplementedError: If ``dset_name`` is not ``"mimic-cxr"``.
        """
        if self.dset_name != "mimic-cxr":
            raise NotImplementedError(
                f"Dataset '{self.dset_name}' is not supported."
            )

        train_set = MimicMultiModal(
            self.config.dataset.img_path,
            self.config.dataset.ann_path,
            self.config.dataset.view,
            "train",
        )

        partition_path = (
            f"partitions/{self.dset_name}_{self.config.dataset.view}_"
            f"{self.config.dataset.partition}.pkl"
        )
        with open(partition_path, "rb") as f:
            data_partition = pickle.load(f)

        if self.client_id == -1:
            train_idx = data_partition["server"]
            self.train_set = Subset(train_set, train_idx)
            self.val_set = MimicMultiModal(
                self.config.dataset.img_path,
                self.config.dataset.ann_path,
                self.config.dataset.view,
                "val",
            )
            self.test_set = MimicMultiModal(
                self.config.dataset.img_path,
                self.config.dataset.ann_path,
                self.config.dataset.view,
                "test",
            )
        else:
            train_idx = data_partition["client"][self.client_id]["train"]
            val_idx = data_partition["client"][self.client_id]["val"]
            self.train_set = Subset(train_set, train_idx)
            self.val_set = Subset(train_set, val_idx)

        self._build_dataloaders()
        logger.info("Data reloaded for new split.")

    def load_data_standalone(self) -> None:
        """Load data for standalone (centralized) training.

        Merges all client training partitions into a single training set,
        then uses the standard validation and test sets.

        Raises:
            NotImplementedError: If ``dset_name`` is not ``"mimic-cxr"``.
        """
        if self.dset_name != "mimic-cxr":
            raise NotImplementedError(
                f"Dataset '{self.dset_name}' is not supported."
            )

        partition_path = (
            f"partitions/"
            f"{self.dset_name}-{self.config.dataset.view}-"
            f"{self.config.dataset.partition}.pkl"
        )
        with open(partition_path, "rb") as f:
            data_partition = pickle.load(f)

        train_set = MimicMultiModal(
            self.config.dataset.img_path,
            self.config.dataset.ann_path,
            self.config.dataset.view,
            "train",
        )

        indices = list(chain.from_iterable(
            data_partition["client"][cid]["train"]
            for cid in data_partition["client"]
        ))

        self.train_set = Subset(train_set, indices)
        self.val_set = MimicMultiModal(
            self.config.dataset.img_path,
            self.config.dataset.ann_path,
            self.config.dataset.view,
            "val",
        )
        self.test_set = MimicMultiModal(
            self.config.dataset.img_path,
            self.config.dataset.ann_path,
            self.config.dataset.view,
            "test",
        )

        self._build_dataloaders()
        logger.info("Standalone data loaded successfully.")

    def _build_dataloaders(self) -> None:
        """Create DataLoaders from the current train/val/test sets.

        Only creates a loader if the corresponding dataset attribute exists
        and is non-empty. This is an internal helper used by all
        ``load_data*`` methods.
        """
        batch_cfg = self.config.dataloader

        if hasattr(self, "train_set") and len(self.train_set) > 0:
            self.train_loader = DataLoader(
                self.train_set,
                batch_size=batch_cfg.batch_size,
                shuffle=True,
                num_workers=batch_cfg.num_workers,
                pin_memory=True,
                drop_last=False,
            )

        self.val_loader = DataLoader(
            self.val_set,
            batch_size=batch_cfg.eval_batch_size,
            shuffle=True,
            num_workers=batch_cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        if hasattr(self, "test_set") and self.test_set is not None:
            self.test_loader = DataLoader(
                self.test_set,
                batch_size=batch_cfg.eval_batch_size,
                shuffle=True,
                num_workers=batch_cfg.num_workers,
                pin_memory=True,
                drop_last=False,
            )

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Initialize the model, tokenizer, criterion, optimizer, and scaler.

        Creates and assigns the following instance attributes:
            ``model``, ``tokenizer``, ``criterion``, ``optimizer``,
            ``grad_scaler``.
        """
        self.model = get_model(
            model_name=self.config.model.name, config=self.config.model
        )
        self.tokenizer = get_tokenizer(config=self.config.model)
        self.criterion = get_criterion(self.config.criterion.name)
        self.optimizer = get_optimizer(
            self.config.optimizer.name,
            self.model.parameters(),
            self.config.optimizer,
        )
        self.grad_scaler = torch.cuda.amp.GradScaler()
        logger.info("Model loaded successfully.")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_best(self, comms: int) -> None:
        """Save the current model as the best checkpoint.

        Args:
            comms: Communication round number (or epoch number in standalone
                mode) at which this checkpoint was recorded.
        """
        ckpt_path = os.path.join(self.save_dir, "model_best.pth")
        torch.save({"net": self.model.state_dict(), "comms": comms}, ckpt_path)

    def load_best(self) -> None:
        """Load the best model checkpoint from disk.

        Restores model weights from ``model_best.pth`` in :attr:`save_dir`
        and logs the communication round at which it was saved.
        """
        ckpt_path = os.path.join(self.save_dir, "model_best.pth")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["net"])
        logger.info("Best model loaded from comms round: %s", checkpoint["comms"])

    def save_log(self) -> None:
        """Persist the validation AUC history to a pickle file."""
        log_path = os.path.join(self.save_dir, "val_aucs.pkl")
        with open(log_path, "wb") as f:
            pickle.dump(self.val_track, f)

    # ------------------------------------------------------------------
    # Training Loops
    # ------------------------------------------------------------------

    def run_standalone(self) -> None:
        """Execute the full standalone (centralized) training pipeline.

        Trains for ``config.train.total_epoch`` epochs, validating after
        each epoch. Saves the best model based on validation AUC, then
        runs final test evaluation.
        """
        logger.info("Starting standalone training.")
        self.val_auc = 0.0
        self.model.cuda()

        for epoch in range(self.config.train.total_epoch):
            logger.info("Epoch %d / %d", epoch + 1, self.config.train.total_epoch)
            self.train_epoch()
            cur_auc = self.val()
            self.val_track.append(cur_auc)
            if cur_auc > self.val_auc:
                self.val_auc = cur_auc
                self.save_best(epoch)
            self.cur_epoch += 1

        self.save_log()
        self.load_best()
        self.test()
        logger.info("Standalone training complete.")

    def run(self, comms: int) -> None:
        """Execute one federated communication round of local training.

        Trains for ``config.train.local_epoch`` epochs on the local
        dataset, then moves the model back to CPU.

        Args:
            comms: Current communication round number (for logging).
        """
        self.model.cuda()

        for i in range(self.config.train.local_epoch):
            if self.client_id == -1:
                logger.info(
                    "Server — round %d, local_epoch %d, round_epoch %d",
                    comms, self.local_epoch, i,
                )
            else:
                logger.info(
                    "Client %d — local_epoch %d, round %d, round_epoch %d",
                    self.client_id, self.local_epoch, comms, i,
                )
            self.train_epoch()
            self.local_epoch += 1

        self.model.cpu()
        gc.collect()

    def train_epoch(self) -> None:
        """Execute one training epoch over the training DataLoader.

        Uses mixed-precision (FP16) training with gradient scaling and
        optional gradient clipping (controlled by ``config.train.grad_clip``).
        """
        self.model.train()
        self.model.cuda()

        with tqdm(self.train_loader, unit="batch", desc="Training") as tepoch:
            for frames, label, text, _ in tepoch:
                self.optimizer.zero_grad()
                images = frames.cuda()
                label = label.cuda()

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = self.model(self.tokenizer, images, text)
                    loss = self.criterion(output["logits"], label)

                self.grad_scaler.scale(loss).backward()

                if self.config.train.grad_clip > 0:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.train.grad_clip,
                    )

                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                tepoch.set_postfix(Loss=loss.item())

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def val(self) -> float:
        """Run validation and return the macro AUC score.

        Evaluates the model on :attr:`val_loader`, computes the macro-averaged
        multilabel AUROC, optionally logs to WandB, and resets the evaluator.

        Returns:
            Macro-averaged validation AUC as a float.
        """
        self.model.cuda()
        self.model.eval()

        with tqdm(self.val_loader, unit="batch", desc="Validating") as tepoch:
            for frames, label, text, _ in tepoch:
                images = frames.cuda()
                label = label.cuda()
                with torch.no_grad():
                    output = self.model(self.tokenizer, images, text)
                self.evaluator.update(output["logits"], label.long())

        metrics = self.evaluator.compute()
        val_auc = metrics["AUC"].item()
        logger.info("Val AUC: %.4f", val_auc)

        if self.wandb:
            self.wandb.log(
                {"Val AUC(Server)": val_auc}, step=self.cur_epoch
            )

        self.evaluator.reset()
        self.model.cpu()
        return val_auc

    def test(self) -> None:
        """Run evaluation on the test set and log results.

        Computes both macro-averaged and per-label AUROC on the test
        DataLoader. Results are printed and logged to WandB.
        """
        self.model.cuda()
        self.model.eval()

        test_evaluator = MetricCollection({
            "AUC": MultilabelAUROC(
                num_labels=14, average="macro", thresholds=None
            ),
            "AUCperLabel": MultilabelAUROC(
                num_labels=14, average="none", thresholds=None
            ),
        })

        with tqdm(self.test_loader, unit="batch", desc="Testing") as tepoch:
            for frames, label, text, _ in tepoch:
                images = frames.cuda()
                label = label.cuda()
                with torch.no_grad():
                    output = self.model(self.tokenizer, images, text)
                test_evaluator.update(output["logits"], label.long())

        metrics = test_evaluator.compute()
        logger.info("Test AUC: %s", metrics["AUC"])
        logger.info("Test AUC per Label: %s", metrics["AUCperLabel"])

        if self.wandb:
            self.wandb.log({"Test AUC(Aggregated)": metrics["AUC"].item()})

        self.model.cpu()
        self.evaluator.reset()

    def eval(
        self, eval_loader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract image and text feature embeddings from a DataLoader.

        Used for knowledge distillation in CreamFL and similar algorithms
        that require feature representations from a public holdout set.

        Args:
            eval_loader: DataLoader yielding ``(frames, label, text, index)``
                tuples.

        Returns:
            Tuple of ``(image_features, text_features)`` where each tensor
            has shape ``(N, embed_dim)`` with ``N`` being the total number
            of samples in the loader.
        """
        self.model.cuda()
        self.model.eval()

        out_img_feat: List[torch.Tensor] = []
        out_txt_feat: List[torch.Tensor] = []

        with tqdm(eval_loader, unit="batch", desc="Extracting features") as tepoch:
            for frames, label, text, _ in tepoch:
                images = frames.cuda()
                label = label.cuda()
                with torch.no_grad():
                    output = self.model(self.tokenizer, images, text)
                out_img_feat.append(output["image_features"])
                out_txt_feat.append(output["caption_features"])

        image_features = torch.cat(out_img_feat, dim=0)
        text_features = torch.cat(out_txt_feat, dim=0)

        self.model.cpu()
        return image_features, text_features