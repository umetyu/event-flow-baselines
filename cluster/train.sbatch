#!/bin/bash -l
#SBATCH --job-name=evflow-train
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out

# Submit through cluster/submit_train.sh, which sets CONFIG and the job name.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source cluster/env.sh
set -u
: "${CONFIG:?Launch through cluster/submit_train.sh}"
nvidia-smi
echo "config: $CONFIG | commit: $(git rev-parse HEAD) | data: $EVFLOW_DATA"
"$BENCHMARK_PYTHON" -u train_flow.py --config "$CONFIG" --path_mlflow "$EVFLOW_MLRUNS" \
    --run_name "$(basename "$CONFIG" .yml)"
