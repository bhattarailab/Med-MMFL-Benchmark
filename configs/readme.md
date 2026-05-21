# Configuration Guide for Med-MMFL

This document provides comprehensive guidance on creating and modifying experiment configurations for the Med-MMFL Benchmark.

---

## Configuration Overview

All experiments are controlled via YAML configuration files. Each config specifies:
- **Dataset** and data loading parameters
- **Model** architecture and hyperparameters
- **Optimizer** and learning rate schedule
- **Training** hyperparameters
- **Evaluation metrics**

---

## Configuration Structure

Here's the complete structure with all available options:

```yaml
# ============================================================================
# 1. DATASET CONFIGURATION
# ============================================================================
dataset:
  img_path: /path/to/images              # [Required] Path to image directory
  ann_path: /path/to/splits              # [Required] Path to annotation/split files
  dset_name: <dataset_name>              # [Required] Dataset identifier
  view: <view_type>                      # [Optional] Specific modality/view
  partition: <partition_type>            # [Required] Data distribution type
  clients: <int>                         # [Optional] Number of clients

# ============================================================================
# 2. DATALOADER CONFIGURATION
# ============================================================================
dataloader:
  # Standard keys (for BraTS24, MIMIC-CXR, EHRXQA, PathVQA):
  batch_size: <int>                      # Training batch size per client
  eval_batch_size: <int>                 # Evaluation batch size
  
  # Symile-specific keys (for SYMILE-MIMIC):
  batch_size_train: <int>                # Training batch size
  batch_size_val: <int>                  # Validation batch size
  batch_size_test: <int>                 # Test batch size
  
  num_workers: <int>                     # Number of dataloader workers
  crop_size: <int>                       # Image crop/resize size (e.g. 128)
  collate_fn: <str>                      # [Optional] Custom collate function (e.g. vqa_collate_fn)

# ============================================================================
# 3. MODEL CONFIGURATION
# ============================================================================
model:
  name: <model_name>                     # Model architecture to use
  # Model-specific hyperparameters below...
  # (e.g., in_channels, n_classes, embed_dim, pretrained_model_name, max_length)

# ============================================================================
# 4. OPTIMIZER CONFIGURATION
# ============================================================================
optimizer:
  name: <optimizer_name>                 # Type of optimizer
  learning_rate: <float>                 # Initial learning rate
  weight_decay: <float>                  # L2 regularization coefficient
  # Optimizer-specific parameters below...

# ============================================================================
# 5. LEARNING RATE SCHEDULER
# ============================================================================
lr_scheduler:
  name: <scheduler_name>                 # Type of scheduler
  T_max: <int>                           # Max iterations/epochs
  warmup: <int>                          # Warmup rounds/epochs

# ============================================================================
# 6. LOSS FUNCTION
# ============================================================================
criterion:
  name: <criterion_name>                 # Loss function type
  # Loss-specific parameters below...

# ============================================================================
# 7. TRAINING CONFIGURATION
# ============================================================================
train:
  local_epoch: <int>                     # Local epochs per client per round
  total_epoch: <int>                     # Total epochs (centralized training)
  model_save_path: <str>                 # Path for last checkpoint
  best_model_save_path: <str>            # Path for best checkpoint
  pretrain_epochs: <int>                 # Pretraining epochs
  finetune_epochs: <int>                 # Fine-tuning epochs
  finetune_lr_decay: <float>             # LR decay during fine-tuning
  log_step: <int>                        # Logging frequency
  grad_clip: <float>                     # Gradient clipping threshold
  val_epochs: <int>                      # Validation frequency
  use_fp16: <bool>                       # Mixed precision training
  output_file: <str>                     # Training log filename
  partition: homo | hetero               # Partition type label

# ============================================================================
# 8. METRICS
# ============================================================================
metrics:
  - <metric_name1>                       # List of metrics to track
  - <metric_name2>
  - ...
```

---


## Model Configuration

### 1. RFNet (BraTS Segmentation)

```yaml
model:
  name: rfnet                            # model
  in_channels: 4                         # T1, T1ce, T2, FLAIR
  n_classes: 4                           
  n_channels: 48                         # Base channel count
```

**Hyperparameter Tuning Guide:**
- `n_channels`: 32-64 (higher = more capacity but more memory)
- Typical values: 32, 48, 64

### 2. MIMIC Multimodal Classifier

```yaml
model:
  name: mimic_mmclf                      # or 'mimic_image_classifier', 'mimic_text_classifier'
  embed_dim: 256                         # Output embedding dimension
  cnn_type: resnet50                     # CNN backbone
  txt_type: bert-base-uncased            # Text encoder
```

**CNN Backbones:**
- `resnet18`, `resnet34`, `resnet50`, `resnet101`

**Text Encoders:**
- `bert-base-uncased`: Standard BERT
- `distilbert-base-uncased`: Faster, smaller variant

### 3. SYMILE-MIMIC Model

```yaml
model:
  name: symile_mimic
  d: 8192                                # Projection dimension
  freeze_logit_scale: False              # Whether to freeze scaling parameter
  logit_scale_init: -7                   # Logit scale initialization
  pretrained: False                      # Start from pretrained weights
```

### 4. EHRXQA VQA Model

```yaml
model:
  name: blip_ehrxqa
  pretrained_model_name: Salesforce/blip-vqa-base
  max_length: 64
```

### 5. PathVQA VQA Model

```yaml
model:
  name: blip_yesno_vqa
  pretrained_model_name: Salesforce/blip-vqa-base
```

---

## Training Configuration

### Local Training

```yaml
train:
  local_epoch: 5                         # Epochs per client per round
  model_save_path: model_last.pth
  best_model_save_path: model_best.pth
  grad_clip: 2                           # Gradient clipping
  use_fp16: True                         # Mixed precision
  log_step: 1000                         # Log every N batches
  val_epochs: 10                         # Validate every N rounds
  output_file: model.log
```


## Metrics Configuration

```yaml
metrics:
  - dice                                 # Dice coefficient (segmentation)
  - jaccard                              # Jaccard/IoU (segmentation)
  - accuracy                             # Accuracy (classification)
  - f1                                   # F1 score (classification)
  - auc                                  # AUC-ROC (binary classification)
```

### Metric Definitions

| Metric | Formula | Use Case |
|--------|---------|----------|
| **Dice** | $\frac{2 \|A \cap B\|}{\|A\| + \|B\|}$ | Segmentation overlap |
| **Jaccard** | $\frac{\|A \cap B\|}{\|A \cup B\|}$ | Segmentation IoU |
| **Accuracy** | $\frac{\text{correct}}{\text{total}}$ | Classification rate |
| **F1** | $2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$ | Balance precision/recall |
| **AUC** | Area under ROC curve | Binary classification ranking |

---

## Creating Custom Configurations

### Step 1: Choose Base Configuration

Start with an existing config matching your dataset:
```bash
cp configs/fedavg.yml configs/my_experiment.yml
```

### Step 2: Update Dataset Paths

```yaml
dataset:
  img_path: /path/to/your/images
  ann_path: /path/to/your/splits
  dset_name: <your_dataset>             # Must exist in _SERVER_REGISTRY
  partition: <partition_type>
```

### Step 3: Adjust Model for Your Task

```yaml
model:
  name: <model_name>                    # Choose from available models
  # Adjust architecture parameters:
  in_channels: <your_channels>
  n_classes: <your_classes>
```

### Step 4: Tune Hyperparameters

```yaml
optimizer:
  learning_rate: 0.0001                 # Start conservative
  
train:
  local_epoch: 5                        # Standard for FL
  
lr_scheduler:
  T_max: <your_comm_rounds>            # Match --comm-rounds
```

### Step 5: Validate Configuration

```bash
# Test-run configuration (1 round only)
python main.py \
    --config configs/my_experiment.yml \
    --algorithm fedavg \
    --comm-rounds 1 \
    --seed 42 \
    --name test_config
```

---

## Running with Your Configuration

```bash
python main.py \
    --config configs/my_experiment.yml \
    --algorithm fedavg \
    --comm-rounds 30 \
    --seed 42 \
    --name my_experiment
```

For more details, see [REPRODUCE.md](../REPRODUCE.md).
