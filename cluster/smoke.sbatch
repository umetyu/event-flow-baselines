#!/bin/bash -l
#SBATCH --job-name=smoke
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out

# Synthetic-data test: training, checkpointing, evaluation. No dataset needed.
# By default one model of each kind used in the study; SMOKE_MODELS="" tests all of them (slower).
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source cluster/env.sh
set -u
nvidia-smi
"$BENCHMARK_PYTHON" -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
models="${SMOKE_MODELS-RecEVFlowNet:cnt:2 LeakyRecEVFlowNet:cnt:2 FireNet:cnt:2 EVFlowNet:voxel:9}"
# shellcheck disable=SC2086
"$BENCHMARK_PYTHON" -u tests/smoke_test.py ${models:+--models $models}
