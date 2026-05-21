#!/bin/bash --login
# SLURM job script for MIMIC-CXR experiments
#
# Runs FedAvg across multiple partitions (IID and non-IID)
# with 3 and 5 client configurations.
#
# Usage:
#   sbatch scripts/run_mimic.sh
#
# NOTE: Update paths, conda env, and SLURM account below
# for your HPC environment before running.

#SBATCH --account <YOUR_ACCOUNT>
#SBATCH --job-name mimic_cxr_fedavg
#SBATCH --output logs/mimic_cxr/fedavg_%A_%a.out

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 48G
#SBATCH --time 12:00:00

source activate benchmark-env

SEEDS=(1171 14059 4691)
CM=30
ALGORITHM=fedavg

for SEED in "${SEEDS[@]}"; do
    for NUM_CLIENTS in 3 5; do
        for PART in iid-c${NUM_CLIENTS} non-iid_02-c${NUM_CLIENTS} non-iid_08-c${NUM_CLIENTS}; do
            NAME="mimic_${ALGORITHM}_CM${CM}_c${NUM_CLIENTS}_${PART}"
            echo "Running: ${NAME} (seed=${SEED})"
            python main.py \
                --name ${NAME} \
                --exp-dir ./experiments/mimic_cxr/${NAME}/ckpt_${SEED} \
                --config configs/mimic-cxr.yml \
                --seed ${SEED} \
                --algorithm ${ALGORITHM} \
                --comm-rounds ${CM}
        done
    done
done
