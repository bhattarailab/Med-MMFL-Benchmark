"""CreamFL server for MIMIC-CXR multimodal classification.

Implements CreamFL (Contrastive Representation Ensembling And Mining for FL),
which uses public data to extract representations from both the global and
local models, then distils the aggregated representations back into the
global model using an MSE loss.
"""

import copy
import gc
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from med_mmfl_bench.servers.mimiccxrjpg.fedavg import FedavgServer
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)


class CreamflServer(FedavgServer):
    """CreamFL server with contrastive representation distillation.

    Uses a subset of the training data as a public evaluation set for
    computing global/local feature representations and performing
    knowledge distillation.
    """

    def __init__(self, args: Any, config: Any, wandb: Any = False) -> None:
        super().__init__(args, config, wandb)

        self.local_img_feat: List[torch.Tensor] = []
        self.local_txt_feat: List[torch.Tensor] = []
        self.img_vec: torch.Tensor = None
        self.txt_vec: torch.Tensor = None

        logger.info("Server train set length: %d", len(self.trainer.train_set))

        # Use the last client's data as the server's training data
        self.trainer.train_set = copy.deepcopy(self.clients[-1].trainer.train_set)
        self.trainer.train_loader = copy.deepcopy(self.clients[-1].trainer.train_loader)
        del_client = self.clients.pop(-1)
        del del_client

        logger.info("Server train set length (after swap): %d", len(self.trainer.train_set))

        # Build a small evaluation subset for feature extraction
        num_eval = int(0.1 * len(self.trainer.train_set))
        eval_indices = list(range(num_eval))
        eval_dataset = Subset(self.trainer.train_set, eval_indices)
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.dataloader.eval_batch_size,
            shuffle=True,
            num_workers=self.config.dataloader.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        for client in self.clients:
            client.trainer.eval_loader = copy.deepcopy(self.eval_loader)

    # ------------------------------------------------------------------
    # CreamFL aggregation
    # ------------------------------------------------------------------

    def aggregation(self) -> Dict[str, torch.Tensor]:
        """Compute contrastive-weighted feature aggregation.

        Weights each client's local features by their contrastive similarity
        to the global opposite-modality features.
        """
        device = self.local_img_feat[0].device
        num_img_vec = len(self.local_img_feat)

        img_feats = torch.stack(self.local_img_feat).to(device)
        txt_feats = torch.stack(self.local_txt_feat).to(device)

        def calc_weighted_features(
            main: torch.Tensor, sub: torch.Tensor,
        ) -> torch.Tensor:
            """Calculate contrastive-weighted average of feature vectors."""
            contrastive_w = torch.zeros(main.shape[0], main.shape[1])
            for i_idx, vec in enumerate(main):
                logits_sub = vec @ sub.T
                logits_sub = logits_sub - torch.max(logits_sub, dim=1, keepdim=True).values
                exp_logits_sub = torch.exp(logits_sub)

                # log(num / denom) = log(num) - log(denom)
                log_prob_sb = logits_sub - torch.log(
                    torch.sum(exp_logits_sub, dim=1, keepdim=True)
                )
                contrastive_w[i_idx] = torch.diagonal(log_prob_sb).reshape(-1)
                del log_prob_sb, logits_sub, exp_logits_sub
                torch.cuda.empty_cache()
                gc.collect()

            contrastive_w = torch.softmax(contrastive_w, dim=0)

            if torch.isnan(contrastive_w).any():
                logger.warning(
                    "NaN in contrastive weights! Count: %d",
                    torch.isnan(contrastive_w).sum().item(),
                )

            weighted_vecs = []
            for i in range(num_img_vec):
                weighted_vecs.append(
                    (main[i].to(device) * contrastive_w[i].reshape(-1, 1).to(device)).unsqueeze(0)
                )
            return torch.sum(torch.cat(weighted_vecs, dim=0), dim=0)

        weighted_img = calc_weighted_features(img_feats, self.global_txt_feat)
        weighted_txt = calc_weighted_features(txt_feats, self.global_img_feat)

        return {"img": weighted_img, "txt": weighted_txt}

    # ------------------------------------------------------------------
    # Distillation
    # ------------------------------------------------------------------

    def distill(self) -> None:
        """Distil aggregated client representations into the global model."""
        mse_loss = torch.nn.MSELoss()
        agg = self.aggregation()
        self.img_vec, self.txt_vec = agg["img"], agg["txt"]

        def code_sim(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """MSE similarity between output and target features."""
            output = output.sum(axis=1) if len(output.shape) == 3 else output
            return mse_loss(output, target.type_as(output))

        logger.info("Start distilling.")
        self.trainer.model.cuda()
        self.trainer.model.train()
        kd_weight = float(self.config.train.kd_weight)

        with tqdm(self.eval_loader, unit="batch", desc="Distilling") as tepoch:
            for b_idx, inputs in enumerate(tepoch):
                frames, label, text, _ = inputs
                images = frames.cuda()
                label = label.cuda()

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = self.trainer.model(self.trainer.tokenizer, images, text)

                bs = images.size(0)
                start = b_idx * bs
                end = start + bs

                sim_img = code_sim(
                    output["image_features"],
                    torch.clamp(self.img_vec[start:end], min=-1e6, max=1e6).type_as(
                        output["image_features"]
                    ),
                )
                sim_txt = code_sim(
                    output["caption_features"],
                    torch.clamp(self.txt_vec[start:end], min=-1e6, max=1e6).type_as(
                        output["caption_features"]
                    ),
                )
                loss = kd_weight * (sim_img + sim_txt)

                self.trainer.optimizer.zero_grad()
                loss.backward()
                self.trainer.optimizer.step()
                del loss
                gc.collect()

        self.trainer.model.cpu()
        self.trainer.model.eval()

    # ------------------------------------------------------------------
    # Federation protocol
    # ------------------------------------------------------------------

    def update(self) -> None:
        """Execute one CreamFL communication round.

        1. Train the global model on the server's data.
        2. Extract global representations.
        3. Each client trains locally and extracts local representations.
        4. Distil the aggregated local representations into the global model.
        5. Validate and test.
        """
        # Global train
        self.trainer.run(self.round)

        # Extract global representations
        self.global_img_feat, self.global_txt_feat = self.trainer.eval(self.eval_loader)

        # Client updates
        self.local_img_feat, self.local_txt_feat = [], []
        self.num_img: List[int] = []
        self.num_txt: List[int] = []

        for client in self.clients:
            client.update(
                global_img_feat=self.global_img_feat,
                global_txt_feat=self.global_txt_feat,
                comm_round=self.round,
            )
            local_img, local_txt = client.trainer.eval(self.eval_loader)
            self.local_img_feat.append(local_img)
            self.local_txt_feat.append(local_txt)
            self.num_img.append(len(client.trainer.train_loader))
            self.num_txt.append(len(client.trainer.train_loader))

        # Distillation
        self.distill()

        logger.info(":::: Validating Model :::: Round : %d", self.round)
        val_auc = self.trainer.val()
        if val_auc > self.best_val_auc:
            self.best_val_auc = val_auc
            self.best_epoch = self.round
            self.trainer.save_best(self.round)
            self.trainer.save_log()

        logger.info(":::: Testing Model :::: Round : %d", self.round)
        self.trainer.test()
        self.round += 1
        self.trainer.cur_epoch = self.round

    def evaluate(self) -> None:
        """Evaluate the best global model."""
        self.trainer.load_best()
        self.trainer.test()

    def finalize(self) -> None:
        """Finalize training — evaluate the best model."""
        self.evaluate()
