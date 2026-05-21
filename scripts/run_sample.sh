#!/bin/bash --login
# SLURM job script for CreamFL on SYMILE-MIMIC dataset
#
# Usage:
#   sbatch scripts/run_creamfl.sh
#
# NOTE: Update paths, conda env, and SLURM account below
# for your HPC environment before running.

#SBATCH --account <YOUR_ACCOUNT>
#SBATCH --job-name symile_creamfl
#SBATCH --output logs/symile/train_symile_c3%A_%a.out

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 48G
#SBATCH --time 8:00:00

source activate benchmark-env

CM=120
SEEDS=(1191 11059 4694)
ALGORITHM=creamfl
SEED=${SEEDS[0]}
PARTITION=homoc3
kd_weight=0.3
interintra_weight=0.5
NAME="symile_${ALGORITHM}_le3_cm${CM}_${PARTITION}_kd${kd_weight}_interintra${interintra_weight}"

python main.py \
    --name ${NAME} \
    --exp-dir ./experiments/symile_mimic/${NAME}/ckpt_${SEED} \
    --config configs/symile.yml \
    --seed ${SEED} \
    --comm-rounds ${CM} \
    --algorithm ${ALGORITHM}
