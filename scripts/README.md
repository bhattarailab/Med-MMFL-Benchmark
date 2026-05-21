# Scripts and Job Submission Guide

This directory contains utility scripts and SLURM job templates for running Med-MMFL benchmark experiments on high-performance computing (HPC) clusters or local servers.

---

## Setup Instructions

### 1. Create the Environment
You can initialize the Conda environment using [create-env.sh](./create-env.sh). It will install dependencies from [benchmark-env.yml](../benchmark-env.yml) and install the codebase in editable mode:

```bash
# Run the setup helper
bash scripts/create-env.sh
```

### 2. Configure Dataset Partitions
Data splits and client assignments are pre-computed and loaded from `.pkl` files inside the `partitions/` directory.

---

## Running Experiments

Experiments can be executed via `python main.py` or using the CLI entrypoint `med-mmfl-bench`.

### On a SLURM Cluster

1. Open the desired runner script (e.g., `scripts/run_sample.sh`).
2. Edit the SLURM parameters at the top, specifically:
   - `#SBATCH --account <YOUR_ACCOUNT>` (set to your billing/allocation account)
   - Ensure the `source activate benchmark-env` statement matches your Conda installation.
3. Submit the job:

```bash
sbatch scripts/run_sample.sh
```

To run a multi-config sweep (e.g. MIMIC-CXR partitions):
```bash
sbatch scripts/run_mimic.sh
```

### Running Locally

To run without SLURM, simply run `main.py` directly using the CLI:

```bash
# Run BraTS24 FedAvg baseline
python main.py --config configs/fedavg.yml --algorithm fedavg --comm-rounds 30 --seed 42 --name brats_local

# Or using the installed CLI entrypoint
med-mmfl-bench --config configs/fedavg.yml --algorithm fedavg --comm-rounds 30 --seed 42 --name brats_local
```

---

## Monitoring and Logs

- **SLURM Log Files**: SLURM output files are written to log subfolders (e.g. `logs/brats/`, `logs/mimic_cxr/`). Make sure to create these directories before launching jobs:
  ```bash
  mkdir -p logs/brats logs/mimic_cxr logs/symile
  ```
- **Validation History**: Validation metric histories are persisted by the server in `.pkl` format under the experiment directory (e.g. `val_metrics.pkl` or `val_aucs.pkl`).
- **Weights & Biases (WandB)**: You can enable real-time dashboard plotting by adding the `--wandb` flag and configuring `--wandb-project` and `--wandb-entity`.
