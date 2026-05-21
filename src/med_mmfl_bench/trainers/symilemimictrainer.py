import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from med_mmfl_bench.datasets.symile_mimic import SymileMIMICRetrievalDataset
from med_mmfl_bench.losses.symile_loss import zeroshot_retrieval_logits
from med_mmfl_bench.trainers import BaseTrainer
from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["ClientTrainer"]


class ClientTrainer(BaseTrainer):
    """Trainer for SymileMIMIC multimodal retrieval tasks."""

    def __init__(self, args: Any, config: Any, wandb: Any = False) -> None:
        """Initialize the SymileMIMIC trainer.

        Args:
            args: Command line arguments containing experiment paths.
            config: YAML configuration containing dataset/training options.
            wandb: Weights & Biases instance for logging, or False.
        """
        self.args = args
        self.config = config
        self.wandb = wandb
        self.setup()

    def setup(self):
        self.save_dir = self.args.exp_dir
        self.best_val_loss = 10000000000000

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.track_val_data = []

    def run(self) -> None:
        """Run the full local training loop for all epochs."""
        self.best_epoch = 0
        self.best_val_loss = 10000000000000
        for epoch in range(self.config.train.local_epoch):
            logger.info(":::: Training Model :::: Epoch : %d", epoch)
            self.train()
            logger.info(":::: Validating Model :::: Epoch : %d", epoch)
            val_loss = self.val()
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
            self.save(epoch)
        
        logger.info("Best Epoch at %d with Val Loss: %f", self.best_epoch, self.best_val_loss)
        logger.info(":::: Testing Best Model ::::")
        self.model.cpu()
        self.model.eval()
        self.test()

    def run_train(self) -> None:
        """Execute local training epochs without evaluation (used by clients)."""
        for _ in range(self.config.train.local_epoch):
            self.train()

    def loss_fn(
        self, r_c: torch.Tensor, r_e: torch.Tensor, r_l: torch.Tensor,
        logit_scale_exp: torch.Tensor, negative_sampling: int
    ) -> torch.Tensor:
        """Compute the Symile loss."""
        return self.criterion(r_c, r_e, r_l, logit_scale_exp, negative_sampling)
        
    def train(self) -> None:
        """Execute one training epoch."""
        self.model.cuda()
        self.model.train()
        with tqdm(self.train_loader, unit="batch") as tepoch:
            for inputs in tepoch:
                cxr, ecg, labs_percentiles, labs_missingness , hadm_id, _  = inputs
                
                cxr, ecg, labs_percentiles, labs_missingness , hadm_id = cxr.cuda(), ecg.cuda(), labs_percentiles.cuda(), labs_missingness.cuda(), hadm_id.cuda()
                batch = [cxr, ecg, labs_percentiles, labs_missingness , hadm_id]
                
                self.optimizer.zero_grad()
                
                r_c, r_e, r_l, logit_scale_exp  = self.model(batch)
                loss = self.loss_fn(r_c, r_e, r_l, logit_scale_exp, self.config.train.negative_sampling)
                log_n = np.log(len(batch[0]))
                               
                loss.backward()
                self.optimizer.step()
                if self.wandb:
                    self.wandb.log(
                        {f"[CLIENT {self.client_id}] train_loss": loss.item(), f"[CLIENT {self.client_id}] logit_scale_exp": logit_scale_exp, f"[CLIENT {self.client_id}] log_n": log_n}
                    )
                
                tepoch.set_postfix(Loss=loss.item())
        self.model.eval()
        self.model.cpu()

    def eval(self, eval_loader):
        self.model.cuda()
        self.model.eval()
        distill_index = []
        out_cxr_feat = []
        out_ecg_feat = []
        out_labs_feat = []
        with tqdm(eval_loader, unit="batch") as tepoch:
            for inputs in tepoch:
                cxr, ecg, labs_percentiles, labs_missingness , hadm_id, index = inputs
                
                cxr, ecg, labs_percentiles, labs_missingness , hadm_id = cxr.cuda(), ecg.cuda(), labs_percentiles.cuda(), labs_missingness.cuda(), hadm_id.cuda()
                batch = [cxr, ecg, labs_percentiles, labs_missingness , hadm_id]

                with torch.no_grad():
                    r_c, r_e, r_l, logit_scale_exp  = self.model(batch)
                out_cxr_feat.append(r_c)
                out_ecg_feat.append(r_e)
                out_labs_feat.append(r_l)
                distill_index.extend(index)
                tepoch.set_postfix(Loss=0)  # No loss in eval mode
        out_cxr_feat = torch.cat(out_cxr_feat, dim=0)
        out_ecg_feat = torch.cat(out_ecg_feat, dim=0)
        out_labs_feat = torch.cat(out_labs_feat, dim=0)
        self.model.cpu()
        return out_cxr_feat, out_ecg_feat, out_labs_feat, distill_index

    def val(self) -> float:
        """Run validation and compute zero-shot retrieval accuracy."""
        self.model.cuda()
        self.model.eval()
        total_val_loss = 0.0
        with tqdm(self.val_loader, unit="batch", desc="Validating") as tepoch:
            for inputs in tepoch:
                cxr, ecg, labs_percentiles, labs_missingness, hadm_id, _ = inputs
                cxr, ecg, labs_percentiles, labs_missingness, hadm_id = cxr.cuda(), ecg.cuda(), labs_percentiles.cuda(), labs_missingness.cuda(), hadm_id.cuda()
                batch = [cxr, ecg, labs_percentiles, labs_missingness, hadm_id]

                with torch.no_grad():
                    r_c, r_e, r_l, logit_scale_exp = self.model(batch)
                
                val_loss = self.criterion(r_c, r_e, r_l, logit_scale_exp, self.config.train.negative_sampling)
                total_val_loss += val_loss.item()
                tepoch.set_postfix(Loss=val_loss.item())
        
        self.model.cpu()
        mean_val_loss = total_val_loss / len(self.val_loader)

        acc = self.zeroshot_retrieval("val_retrieval")
        
        log_dict = {
            "val_loss": mean_val_loss,
            "val_acc": acc
        }

        if self.wandb:
            self.wandb.log(log_dict)
        
        logger.info("Validation Metrics: %s", log_dict)
        self.track_val_data.append(log_dict)
        return mean_val_loss
            

    def get_retrieval_dataset(self, split):
        if split == "val_retrieval":
            batch_sz = self.config.dataloader.batch_size_val
        elif split == "test":
            batch_sz = self.config.dataloader.batch_size_test

        retrieval_ds = SymileMIMICRetrievalDataset(self.config.dataset.img_path, split)

        r_c = []
        r_e = []
        r_l = []
        hadm_id = []
        label_hadm_id = []
        label = []
        self.model.eval()
        self.model.cuda()

        # setting generator manually so that PyTorch uses it for _base_seed creation
        # (avoids altering global seed; helps ensure reproducibility)
        for batch in tqdm(DataLoader(retrieval_ds, batch_size=batch_sz, shuffle=False,
                                drop_last=False, generator=torch.Generator()), desc="Retrieval"):
            with torch.no_grad():
                out_rc = self.model.cxr_encoder(batch["cxr"].cuda()).cpu()
                out_re = self.model.ecg_encoder(batch["ecg"].cuda()).cpu()

                labs = torch.cat([batch["labs_percentiles"], batch["labs_missingness"]], dim=1)
                out_lab = self.model.labs_encoder(labs.cuda()).cpu()
                
                r_c.append(out_rc)
                r_e.append(out_re)
                r_l.append(out_lab)


                hadm_id.append(batch["hadm_id"])
                label_hadm_id.append(batch["label_hadm_id"])
                label.append(batch["label"])

        r_c = torch.cat(r_c, dim=0)
        r_e = torch.cat(r_e, dim=0)
        r_l = torch.cat(r_l, dim=0)
        hadm_id = torch.cat(hadm_id, dim=0)
        label_hadm_id = torch.cat(label_hadm_id, dim=0)
        label = torch.cat(label, dim=0)
        
        assert len(r_c) == len(r_e) == len(r_l) == len(retrieval_ds), \
            "r_c, r_e, r_l, and retrieval_ds should have the same length"

        self.model.cpu()

        return {"r_c": r_c, "r_e": r_e, "r_l": r_l, "hadm_id": hadm_id,
                "label_hadm_id": label_hadm_id, "label": label}

    
    def zeroshot_retrieval(self, split, bootstrap=False):
        """
        Calculates zero-shot retrieval accuracy for a given dataset split ('val'
        or 'test'), where the task is to retrieve the true corresponding CXR
        image for each query ECG and labs pair.

        Args:
            split (str): The dataset split to evaluate ('val' or 'test').
            bootstrap (bool): Whether to bootstrap resample the test retrieval dataset.

        Returns:
            retrieval_acc (float): The retrieval accuracy for the specified split.
        """
        retrieval_ds = self.get_retrieval_dataset(split)

        if bootstrap:
            retrieval_ds = self.resample_retrieval_ds(retrieval_ds)

        # get query data (positive samples)
        mask = retrieval_ds["label"] == 1
        query_r_c = retrieval_ds["r_c"][mask]
        query_r_e = retrieval_ds["r_e"][mask]
        query_r_l = retrieval_ds["r_l"][mask]
        query_hadm_id = retrieval_ds["hadm_id"][mask]

        correct_pred = 0
        print_warning = False

        # loop through each query sample
        for ix, true_hadm_id in enumerate(query_hadm_id):
            r_c = query_r_c[ix] # (d,)
            r_e = query_r_e[ix] # (d,)
            r_l = query_r_l[ix] # (d,)

            # find negative candidates for this query, and add to positive candidate
            mask = (retrieval_ds["label_hadm_id"] == true_hadm_id) & (retrieval_ds["label"] == 0)
            neg_r_c = retrieval_ds["r_c"][mask] # (candidate_n - 1, d)
            r_c = torch.cat([r_c.unsqueeze(0), neg_r_c], dim=0) # (candidate_n, d)

            candidate_label = torch.zeros(len(r_c), dtype=torch.long)
            candidate_label[0] = 1

            assert torch.sum(candidate_label) == 1 and torch.count_nonzero(candidate_label) == 1, \
                "candidate_label must have exactly one 1 and all other elements as 0."

            logits = zeroshot_retrieval_logits(r_c, [r_e, r_l], self.model.logit_scale.exp(),
                                               ).cpu()

            # find all indices with the maximum value; if multiple indices have
            # the same max value, randomly select one of them (note: must use
            # np.random.choice instead of torch.randint to avoid altering the global random seed)
            max_value = torch.max(logits)
            max_indices = (logits == max_value).nonzero(as_tuple=True)[1]

            if len(max_indices) > 1:
                print_warning = True

            pred_ix = max_indices[np.random.choice(len(max_indices))].item()
            true_ix = torch.nonzero(candidate_label, as_tuple=True)[0].item()

            if pred_ix == true_ix:
                correct_pred += 1

        retrieval_acc = float(correct_pred / len(query_hadm_id))

        if print_warning:
            logger.warning("Multiple indices with max value. Random index selected.")

        return retrieval_acc
    
    
    def resample_retrieval_ds(self, ds):
        # get all query samples
        mask = ds["label"] == 1
        query_r_c = ds["r_c"][mask]
        query_r_e = ds["r_e"][mask]
        query_r_l = ds["r_l"][mask]
        query_hadm_id = ds["hadm_id"][mask]
        query_label_hadm_id = ds["label_hadm_id"][mask]
        query_label = ds["label"][mask]

        # randomly sample from the query subset with replacement
        n_samples = len(query_label)
        sample_indices = torch.randint(0, n_samples, (n_samples,), dtype=torch.long)

        # apply the sampled indices consistently across all keys
        sampled_r_c = query_r_c[sample_indices]
        sampled_r_e = query_r_e[sample_indices]
        sampled_r_l = query_r_l[sample_indices]
        sampled_hadm_id = query_hadm_id[sample_indices]
        sampled_label_hadm_id = query_label_hadm_id[sample_indices]
        sampled_label = query_label[sample_indices]

        # get the negative candidate samples
        negative_mask = ds["label"] == 0
        negative_r_c = ds["r_c"][negative_mask]
        negative_r_e = ds["r_e"][negative_mask]
        negative_r_l = ds["r_l"][negative_mask]
        negative_hadm_id = ds["hadm_id"][negative_mask]
        negative_label_hadm_id = ds["label_hadm_id"][negative_mask]
        negative_label = ds["label"][negative_mask]

        # combine positive and negative samples
        final_r_c = torch.cat([sampled_r_c, negative_r_c])
        final_r_e = torch.cat([sampled_r_e, negative_r_e])
        final_r_l = torch.cat([sampled_r_l, negative_r_l])
        final_hadm_id = torch.cat([sampled_hadm_id, negative_hadm_id])
        final_label_hadm_id = torch.cat([sampled_label_hadm_id, negative_label_hadm_id])
        final_label = torch.cat([sampled_label, negative_label])

        return {"r_c": final_r_c,
                "r_e": final_r_e,
                "r_l": final_r_l,
                "hadm_id": final_hadm_id,
                "label_hadm_id": final_label_hadm_id,
                "label": final_label}


    def save(self, epoch):
        ckpt_path = os.path.join(self.save_dir, f"model_ckpt_epoch_best.pt")
        state_dict = {"net":self.model.state_dict(),
                        # "optimizer": self.optimizer.state_dict(),
                        "val_logs":self.track_val_data[-1],
                        "epoch": epoch}
        torch.save(state_dict, ckpt_path)
        with open(os.path.join(self.save_dir, "val_logs.pkl"), "wb") as f:
            pickle.dump(self.track_val_data, f)

    def load_model(self, path, forTraining = False):
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt['net'])
        if forTraining:
            self.optimizer.load_state_dict(ckpt['optimizer'])

    def run_test(self) -> None:
        """Run testing loop by loading validation logs."""
        with open(os.path.join(self.save_dir, "val_logs.pkl"), "rb") as f:
            self.track_val_data = pickle.load(f)

        best_epoch = min(range(len(self.track_val_data)), key=lambda i: self.track_val_data[i]['val_loss'])
        log_dict = {
            "best_val": self.track_val_data[best_epoch]['val_loss']
        }
        logger.info("Best epoch at: %d", best_epoch)
        if self.wandb:
            self.wandb.log(log_dict)
        self.test(best_epoch, True)
      
    def test(self, best_epoch=False, boostrap=False):
        if best_epoch is not False:
            best_path = os.path.join(self.save_dir, f"model_ckpt_epoch_{best_epoch}.pt")
        else:
            best_path = os.path.join(self.save_dir, "model_ckpt_epoch_best.pt")
            if not os.path.exists(best_path):
                raise FileNotFoundError(f"{best_path} not found. Make sure you've renamed the best checkpoint.")
        ckpt = torch.load(best_path, map_location="cpu")
        self.model.load_state_dict(ckpt['net'])

        self.model.eval()

        acc = self.zeroshot_retrieval("test", boostrap)

                
        log_dict = {
            "test_acc": acc
        }

        if self.wandb:
            self.wandb.log(log_dict)
        
        logger.info("-------------------------------------------------")
        logger.info("Test Metrics: \n%s", log_dict)
        logger.info("-------------------------------------------------")
