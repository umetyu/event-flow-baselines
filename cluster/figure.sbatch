#!/bin/bash -l
#SBATCH --job-name=evflow-figure
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out

# Figure with the estimated flow of several runs next to the ground truth (experiments/make_figure.py).
# From the repository root, all arguments of the script are passed through:
#   sbatch cluster/figure.sbatch --sequence indoor_flying1 --index 600 1500 \
#       --runs EVFlowNet opt-cmax-timestamp_n30000 --labels "ConvGRU EV-FlowNet" "CMax optimization"
# The figure is written to results/flow_comparison.png (or --out).
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source cluster/env.sh
set -u
"$BENCHMARK_PYTHON" -u experiments/make_figure.py "$@"
