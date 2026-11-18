#!/bin/bash -l
#SBATCH --job-name=smoke
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

# Synthetic-data test of every model: training, checkpointing, evaluation. No dataset needed.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source cluster/env.sh
set -u
nvidia-smi
"$BENCHMARK_PYTHON" -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
"$BENCHMARK_PYTHON" -u tests/smoke_test.py
