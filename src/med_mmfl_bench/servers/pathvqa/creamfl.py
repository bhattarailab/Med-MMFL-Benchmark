"""CreamFL server for PathVQA federated VQA.

CreamFL uses contrastive-weighted feature aggregation and knowledge
distillation to improve the global model beyond simple weight averaging.
"""

import gc
import os
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Subset
from tqdm import tqdm

from med_mmfl_bench.losses import get_criterion
from med_mmfl_bench.servers.pathvqa.fedavg import FedavgServer
from med_mmfl_bench.utils.logging import get_logger
from med_mmfl_bench.utils.optimizers import get_optimizer

logger = get_logger(__name__)

__all__ = ["CreamflServer"]


class CreamflServer(FedavgServer):
    """CreamFL server for PathVQA.

    Extends FedAvg with server-side training, global feature extraction,
    contrastive-weighted feature aggregation, and knowledge distillation.

    Args:
        args: Experiment arguments.
        config: Full configuration object.
        val_dataset: Server-side validation dataset.
        test_dataset: Server-side test dataset.
        client_datasets: List of per-client dataset dictionaries.
        server_trainset: Optional public training dataset.
        wandb: WandB run object or False.
    """

    def __init__(
        self,
        args: Any,
        config: Any,
        val_dataset: Any,
        test_dataset: Any,
        client_datasets: List[Dict[str, Any]],
        server_trainset: Any = None,
        wandb: Any = False,
    ) -> None:
        super().__init__(args, config, val_dataset, test_dataset, client_datasets, wandb)

        if server_trainset is None:
            self.train_dataset = self.clients[-1].training_set
            removed_client = self.clients.pop(-1)
            del removed_client
            logger.info("Server training dataset size: %d", len(self.train_dataset))
        else:
            self.train_dataset = server_trainset

        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.dataloader.get("batch_size", 32),
            shuffle=True,
            num_workers=self.config.dataloader.num_workers,
            pin_memory=False,
            drop_last=False,
        )

        self.optimizer = get_optimizer(
            self.config.optimizer.name,
            self.global_model.parameters(),
            self.config.optimizer,
        )
        self.criterion = get_criterion(self.config.criterion.name)
        self.grad_scaler = torch.cuda.amp.GradScaler()

        # Create eval subset (10% of training data) for feature extraction
        num_eval = int(0.1 * len(self.train_dataset))
        eval_dataset = Subset(self.train_dataset, list(range(num_eval)))
        self.eval_loader = self._create_eval_dataloader(eval_dataset)

        for client in self.clients:
            client.eval_loader = self.eval_loader

        self.local_text_feat: List[torch.Tensor] = []
        self.local_vision_feat: List[torch.Tensor] = []
        self.text_vec: Optional[torch.Tensor] = None
        self.vision_vec: Optional[torch.Tensor] = None

    def _create_eval_dataloader(self, dataset: Any) -> torch.utils.data.DataLoader:
        """Create a DataLoader for the evaluation/public dataset."""
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.config.dataloader.get("eval_batch_size", 32),
            shuffle=False,
            num_workers=self.config.dataloader.num_workers,
            pin_memory=False,
            drop_last=False,
        )

    def extract_global_features(self) -> tuple:
        """Extract global text and vision features from the server model.

        Returns:
            Tuple of (text_features, vision_features) as CPU tensors.
        """
        self.global_model.eval()
        self.global_model.to(self.device)

        text_feats, vision_feats = [], []

        logger.info("Extracting global features from server model...")
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, unit="batch", desc="Global features"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                self.global_model.return_repr = True
                text_repr, vision_repr, _ = self.global_model(**batch)
                self.global_model.return_repr = False

                text_feats.append(text_repr.cpu())
                vision_feats.append(vision_repr.cpu())

        self.global_model.cpu()
        return torch.cat(text_feats, dim=0), torch.cat(vision_feats, dim=0)

    def aggregation(self) -> Dict[str, torch.Tensor]:
        """Aggregate local features using contrastive weighting."""
        device = self.local_text_feat[0].device
        num_clients = len(self.local_text_feat)

        text_feats = torch.stack(self.local_text_feat).to(device)
        vision_feats = torch.stack(self.local_vision_feat).to(device)

        def calc_weighted_features(
            main_feats: torch.Tensor, ref_feats: torch.Tensor
        ) -> torch.Tensor:
            """Compute contrastive-weighted feature aggregation."""
            logger.debug("Aggregating features with shape: %s", main_feats.shape)

            contrastive_w = torch.zeros(main_feats.shape[0], main_feats.shape[1])

            for i_idx, client_feat in enumerate(main_feats):
                logits = client_feat @ ref_feats.to(self.device).T
                logits = logits - torch.max(logits, dim=1, keepdim=True).values
                exp_logits = torch.exp(logits)
                log_prob = logits - torch.log(
                    torch.sum(exp_logits, dim=1, keepdim=True)
                )
                contrastive_w[i_idx] = torch.diagonal(log_prob)

                del log_prob, logits, exp_logits
                torch.cuda.empty_cache()
                gc.collect()

            contrastive_w = torch.softmax(contrastive_w, dim=0)

            if torch.isnan(contrastive_w).any():
                logger.warning(
                    "NaN in contrastive weights (count: %d). Using uniform.",
                    torch.isnan(contrastive_w).sum().item(),
                )
                contrastive_w = torch.ones_like(contrastive_w) / num_clients

            weighted_feats = []
            for i in range(num_clients):
                weighted = (
                    main_feats[i].to(device)
                    * contrastive_w[i].reshape(-1, 1).to(device)
                )
                weighted_feats.append(weighted.unsqueeze(0))

            return torch.sum(torch.cat(weighted_feats, dim=0), dim=0)

        weighted_text = calc_weighted_features(text_feats, self.global_text_feat)
        weighted_vision = calc_weighted_features(vision_feats, self.global_vision_feat)

        return {"text": weighted_text, "vision": weighted_vision}

    def distill(self) -> None:
        """Distill aggregated local features back into the global model."""
        mse_loss = torch.nn.MSELoss()

        agg = self.aggregation()
        self.text_vec = agg["text"]
        self.vision_vec = agg["vision"]

        logger.info("Distilling aggregated features into global model...")

        def feature_distill_loss(
            output: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
            output = output.sum(axis=1) if len(output.shape) == 3 else output
            target = target.type_as(output)
            return mse_loss(output, target)

        self.global_model.train()
        self.global_model.to(self.device)

        distill_optimizer = get_optimizer(
            self.config.optimizer.name,
            self.global_model.parameters(),
            self.config.optimizer,
        )

        for b_idx, inputs in enumerate(
            tqdm(self.eval_loader, unit="batch", desc="Distilling")
        ):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            self.global_model.return_repr = True
            text_repr, vision_repr, _ = self.global_model(**inputs)
            self.global_model.return_repr = False

            bs = inputs["labels"].size(0)
            start = b_idx * bs
            end = start + bs

            text_repr = torch.clamp(text_repr, min=-1e6, max=1e6)
            vision_repr = torch.clamp(vision_repr, min=-1e6, max=1e6)

            kd_weight = float(self.config.train.get("kd_weight", 1.0))

            text_target = torch.clamp(
                self.text_vec[start:end], min=-1e6, max=1e6
            ).type_as(text_repr)
            vision_target = torch.clamp(
                self.vision_vec[start:end], min=-1e6, max=1e6
            ).type_as(vision_repr)

            loss_text = feature_distill_loss(text_repr, text_target)
            loss_vision = feature_distill_loss(vision_repr, vision_target)
            loss = kd_weight * (loss_text + loss_vision)

            if not torch.isnan(loss):
                distill_optimizer.zero_grad()
                loss.backward()

                if self.config.train.get("grad_clip", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.global_model.parameters(),
                        self.config.train.grad_clip,
                    )

                distill_optimizer.step()
            else:
                logger.warning("NaN loss at batch %d, skipping", b_idx)

            del loss
            gc.collect()

        self.global_model.eval()
        self.global_model.cpu()

    def update(self) -> None:
        """Execute one CreamFL communication round."""
        logger.info("=" * 50)
        logger.info("CreamFL Round %d", self.round + 1)
        logger.info("=" * 50)

        # Step 1: Optional server-side training
        if self.config.train.get("server_train", False):
            logger.info("Training global model on server data...")
            self.global_model.train()
            self.global_model.to(self.device)

            for inputs in tqdm(self.train_loader, unit="batch"):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                self.optimizer.zero_grad()

                outputs = self.global_model(**inputs)
                loss = self.criterion(outputs["logits"], inputs["labels"])
                loss.backward()

                if self.config.train.get("grad_clip", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.global_model.parameters(),
                        self.config.train.grad_clip,
                    )

                self.optimizer.step()

            self.global_model.eval()
            self.global_model.cpu()

        # Step 2: Extract global features
        self.global_text_feat, self.global_vision_feat = self.extract_global_features()

        # Steps 3 & 4: Update clients and collect features
        self.local_text_feat = []
        self.local_vision_feat = []

        for client in self.clients:
            logger.info("Updating Client %s...", client.client_id)

            client.update(
                self.global_text_feat,
                self.global_vision_feat,
                self.eval_loader,
            )

            local_text, local_vision = client.extract_features(
                client.model, self.eval_loader
            )
            self.local_text_feat.append(local_text)
            self.local_vision_feat.append(local_vision)

        # Steps 5 & 6: Aggregate and distill
        self.distill()

        # Step 7: Evaluate
        logger.info("Evaluating Round %d...", self.round + 1)
        val_metric = self.validate_model()
        if val_metric > self.val_metric:
            self.val_metric = val_metric
            self.save_best(self.round)
            logger.info("New best model saved! Metric: %.4f", val_metric)

        self.test()
        self.save_log()

        self.round += 1
        gc.collect()
        torch.cuda.empty_cache()

    def finalize(self) -> None:
        """Load the best model, run final test, and save logs."""
        logger.info("Finalizing CreamFL Training")
        self.load_best()
        self.test()
        self.save_log()
        logger.info("Training completed successfully!")