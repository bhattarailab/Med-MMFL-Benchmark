# Reproducing Med-MMFL Benchmark Results

This guide explains how to reproduce the results from the Med-MMFL paper, including setup, configuration, and execution.

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Dataset Preparation](#dataset-preparation)
3. [Understanding Configuration Files](#understanding-configuration-files)
4. [Running Experiments](#running-experiments)
5. [Using Shell Scripts](#using-shell-scripts)
6. [Monitoring Results](#monitoring-results)

---

## Environment Setup

### Prerequisites

- Python 3.9 or higher
- CUDA 12.1 (recommended for GPU support)
- 36-48 GB RAM (per GPU)
- GPU with at least 24GB VRAM (higher is better)

### Installation

#### Option 1: From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/bhattarailab/Med-MMFL-Benchmark.git
cd Med-MMFL-Benchmark

# Create and activate conda environment
conda env create -f benchmark-env.yml
conda activate benchmark-env

# Install in development mode
pip install -e ".[dev]"

# Optional: Install WandB support for experiment tracking
pip install -e ".[wandb]"
```

#### Option 2: Automated Setup

```bash
# Use the provided setup script
bash ./scripts/create-env.sh
```

#### Verify Installation

```bash
# Test the installation
python -c "from med_mmfl_bench.models import get_model; print('Installation successful!')"

# Check available algorithms
python main.py --help
```

---
## Understanding Configuration Files

All experiments are controlled via YAML configuration files in the `configs/` directory. Here's the complete structure:

### Configuration Structure

```yaml
# Dataset Configuration
dataset:
  img_path: /path/to/images              # Path to image data
  ann_path: /path/to/annotations         # Path to split files
  dset_name: brats24 | mimic-cxr | ...   # Dataset name
  view: frontal | APPA | ...             # Specific view/modality
  partition: natural | iid-c3 | non-iid_02-c5  # Data distribution

# DataLoader Configuration
dataloader:
  batch_size: 2                          # Training batch size
  eval_batch_size: 2                     # Evaluation batch size
  num_workers: 2                         # DataLoader workers
  crop_size: 128                         # Image crop size

# Model Configuration
model:
  name: rfnet | mimic_mmclf | blip      # Model architecture
  # Model-specific parameters:
  # For RFNet:
  in_channels: 4                         # Number of input channels (T1, T1ce, T2, FLAIR)
  n_classes: 4                           # Number of output classes
  n_channels: 48                         # Base channel count
  
  # For MIMIC-MMCLF:
  embed_dim: 256                         # Embedding dimension
  cnn_type: resnet50                     # CNN backbone
  txt_type: bert-base-uncased            # Text encoder

# Optimizer Configuration
optimizer:
  name: SGD | adam | adamw                # Optimizer type
  learning_rate: 0.0001                  # Initial learning rate
  weight_decay: 0.00001                  # L2 regularization
  momentum: 0.9                          # SGD momentum

# Learning Rate Scheduler
lr_scheduler:
  name: cosine_annealing | step          # Scheduler type
  T_max: 30                              # Max iterations/epochs
  warmup: 0                              # Warmup rounds

# Loss Function
criterion:
  name: DiceBCELoss | BCEWithLogitsLoss | CrossEntropyLoss
                                         # Loss function type

# Training Configuration
train:
  local_epoch: 3                         # Local epochs per client per round
  total_epoch: 30                        # Total epochs (for centralized)
  model_save_path: model_last.pth        # Last checkpoint path
  best_model_save_path: model_best.pth   # Best checkpoint path
  pretrain_epochs: 0                     # Pretraining epochs
  finetune_epochs: 30                    # Fine-tuning epochs
  finetune_lr_decay: 0.1                 # Fine-tuning LR decay factor
  log_step: 1000                         # Logging frequency
  grad_clip: 2                           # Gradient clipping value
  val_epochs: 10                         # Validation frequency
  use_fp16: True                         # Mixed precision training
  output_file: model.log                 # Log file name
  partition: homo | hetero               # Partition type

# Metrics to Track
metrics:
  - dice                                 # Dice score
  - jaccard                              # Jaccard/IoU
  - accuracy                             # Classification accuracy
  - f1                                   # F1 score
```

### Partition Types

| Partition | Description | Use Case |
|-----------|-------------|----------|
| `natural` | Original non-IID distribution from dataset | Realistic scenarios |
| `iid-c3` | IID distribution, 3 clients | Controlled baseline |
| `iid-c5` | IID distribution, 5 clients | Controlled baseline |
| `non-iid_02-c3` | Non-IID (α=0.2), 3 clients | Heterogeneous data |
| `non-iid_08-c5` | Non-IID (α=0.8), 5 clients | Mild heterogeneity |

### Dataset-Specific Configs

#### BraTS24 Configuration (`configs/fedavg.yml`)

```yaml
dataset:
  img_path: /scratch/user123/bratsdata
  ann_path: /scratch/user123/mmfl_benchmark/splits
  dset_name: brats24
  view: federated
  partition: natural

model:
  name: rfnet
  in_channels: 4        # T1, T1ce, T2, FLAIR
  n_classes: 4         
  n_channels: 48

train:
  local_epoch: 5
  use_fp16: True
```

#### MIMIC-CXR Configuration (`configs/mimic-cxr_fedavg.yml`)

```yaml
dataset:
  img_path: /path/to/mimic-cxr-resized/files
  ann_path: /path/to/mimic_splits
  dset_name: mimic-cxr
  view: APPA            # Anterior-Posterior, Posterior-Anterior
  partition: iid-c3     # Or: non-iid_02-c3, non-iid_08-c5

model:
  name: mimic_mmclf     # Multimodal classifier
  embed_dim: 256
  cnn_type: resnet50
  txt_type: bert-base-uncased

train:
  local_epoch: 3
  total_epoch: 30
  finetune_epochs: 30
```

---

## Running Experiments

### Quick Start

The simplest way to run an experiment:

```bash
# Single experiment with FedAvg on BraTS
python main.py \
    --config configs/fedavg.yml \
    --algorithm fedavg \
    --comm-rounds 30 \
    --seed 42 \
    --name my_first_experiment \
    --exp-dir ./experiments/my_first_experiment
```

### Available Algorithms

| Algorithm | Class | Paper | Best For |
|-----------|-------|-------|----------|
| **FedAvg** | `FedavgServer` | [Paper](https://arxiv.org/abs/1602.05629) | Baseline, IID data |
| **FedProx** | `FedproxServer` | [Paper](https://arxiv.org/abs/1812.06127) | System heterogeneity |
| **SCAFFOLD** | `ScaffoldServer` | [Paper](https://arxiv.org/abs/1910.06378) | High variance reduction |
| **FedNova** | `FednovaServer` | [Paper](https://arxiv.org/abs/1907.01154) | Non-IID data |
| **MOON** | `MoonServer` | [Paper](https://arxiv.org/abs/2103.16257) | Contrastive learning |
| **CreamFL** | `CreamflServer` | [Paper](https://arxiv.org/abs/2210.15798) | Multimodal distillation |

### Detailed Command Reference

```bash
python main.py \
    --config <path/to/config.yml>           # [Required] Config file
    --algorithm <algorithm>                 # [Required] FL algorithm
    --comm-rounds <rounds>                  # Communication rounds (default: 30)
    --seed <seed>                           # Random seed (default: 42)
    --name <experiment_name>                # Experiment identifier
    --exp-dir <directory>                   # Output directory
    --wandb                                 # Enable W&B logging
    --wandb-project <project_name>          # W&B project name
    --wandb-entity <entity>                 # W&B entity name
```

### Example Experiments

#### Experiment 1: MIMIC-CXR-JPG with Different Algorithms

```bash
# FedAvg baseline
python main.py \
    --config configs/mimic-cxr.yml \
    --algorithm fedavg \
    --comm-rounds 30 \
    --seed 42 \
    --name brats_fedavg_baseline

# FedProx (system heterogeneity)
python main.py \
    --config configs/mimic-cxr.yml \
    --algorithm fedprox \
    --comm-rounds 30 \
    --seed 42 \
    --name mimic-cxr_fedprox

# SCAFFOLD (variance reduction)
python main.py \
    --config configs/mimic-cxr.yml \
    --algorithm scaffold \
    --comm-rounds 30 \
    --seed 42 \
    --name mimic-cxr_scaffold

# CreamFL (multimodal)
python main.py \
    --config configs/mimic-cxr.yml \
    --algorithm creamfl \
    --comm-rounds 30 \
    --seed 42 \
    --name mimic-cxr_creamfl
```


#### Experiment 2: Multiple Seeds for Robustness

```bash
# Run with multiple seeds for statistical significance
for SEED in 42 123 456; do
    python main.py \
        --config configs/mimic-cxr.yml \
        --algorithm fedavg \
        --comm-rounds 30 \
        --seed ${SEED} \
        --name fedavg_seed_${SEED} \
        --exp-dir ./experiments/fedavg_seed_${SEED}
done
```

---

## Using Shell Scripts

The repository includes SLURM job scripts for HPC clusters. These scripts automate multi-seed experiments.

### Available Scripts

| Script | Dataset | Purpose |
|--------|---------|---------|
| `scripts/run_fedavg.sh` | BraTS24 | FedAvg baseline |
| `scripts/run_fedprox.sh` | BraTS24 | FedProx algorithm |
| `scripts/run_scaffold.sh` | BraTS24 | SCAFFOLD algorithm |
| `scripts/run_creamfl.sh` | BraTS24 | CreamFL multimodal |
| `scripts/run_mimic.sh` | MIMIC-CXR | Multi-config MIMIC-CXR |

---

## Monitoring Results

### Output Structure

Each experiment creates the following structure:

```
experiments/
├── fedavg_run1/
│   ├── seed_42/
│   │   ├── model_last.pth              # Final model checkpoint
│   │   ├── model_best.pth              # Best performing model
│   │   ├── metrics.json                # Evaluation metrics
│   │   └── logs/
│   │       └── model.log               # Training log
│   ├── seed_123/
│   └── seed_456/
└── ...
```

### Log Files

Training logs are saved in: `experiments/<exp_name>/seed_<seed>/logs/model.log`

Example log format:

```
[Round 1/30] Dispatch to 5 clients
[Round 1/30] Client 1: loss=0.523, dice=0.654
[Round 1/30] Client 2: loss=0.456, dice=0.712
...
[Round 1/30] Global evaluation: loss=0.489, dice=0.683
[Round 2/30] Dispatch to 5 clients
...
```

### Weights & Biases Integration (Optional)

Track experiments in real-time:

```bash
# Set API key
export WANDB_API_KEY="your_wandb_api_key"

# Run with W&B logging
python main.py \
    --config configs/fedavg.yml \
    --algorithm fedavg \
    --comm-rounds 30 \
    --seed 42 \
    --wandb \
    --wandb-project fed-multimodal \
    --wandb-entity your-entity
```

Then view results at: https://wandb.ai/your-entity/fed-multimodal

### Python API for Analysis

```python
from med_mmfl_bench.utils.config import parse_config
from med_mmfl_bench.utils.seed import set_seed
import json

# Load configuration
config = parse_config("configs/fedavg.yml")
set_seed(42)

# Load results
with open("experiments/fedavg_run1/seed_42/metrics.json") as f:
    metrics = json.load(f)

print(f"Final Dice Score: {metrics['dice']}")
print(f"Final Jaccard: {metrics['jaccard']}")
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Solution:** Reduce batch size in config

```yaml
dataloader:
  batch_size: 2    # Decrease from current value
  eval_batch_size: 2
```

#### 2. CUDA Not Available

**Solution:** Check CUDA installation

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name())"
```

#### 3. Dataset Not Found

**Solution:** Verify paths in config file

```bash
# Check if paths exist
ls -la \<path-to-dataset\>
ls -la mmfl_benchmark/splits
```

#### 4. Import Errors

**Solution:** Reinstall package

```bash
pip install -e ".[dev]" --force-reinstall
```

---

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{med_mmfl_2025,
  title={Med-MMFL: Multimodal Federated Learning Benchmark in Healthcare},
  author={Chhetri, Aavash and Niroula, Bibek and Shrestha, Pratik},
  journal={arXiv preprint},
  year={2025}
}
```

---

## Additional Resources

- **Main Paper**: https://arxiv.org/abs/2602.04416
- **Repository**: https://github.com/bhattarailab/Med-MMFL-Benchmark
- **Issues**: https://github.com/bhattarailab/Med-MMFL-Benchmark/issues
- **Config Validation**: `configs/readme.md`

---

## Support

For questions or issues:

1. Check existing [GitHub Issues](https://github.com/bhattarailab/Med-MMFL-Benchmark/issues)
2. Consult the [main README](README.md)
3. Review [config examples](configs/)
4. Check logs in `experiments/<exp_name>/seed_<seed>/logs/`
