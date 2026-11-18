#!/bin/bash -l
#SBATCH --job-name=evflow-optim
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out

# Evaluates an optimization-based estimator (no training) with the same data and metrics as the
# networks. From the repository root:
#   sbatch --job-name=opt-cmax --export=ALL,PRESET=cmax-timestamp,SUITES="mvsec_dt1",STRIDE=50 cluster/optim.sbatch
# PRESET: cmax-timestamp | cmax-variance | shiba (shiba needs cluster/setup_shiba.sh first)
# STRIDE: score (and optimize) only every n-th ground-truth map; 1 or unset = every map
# NUM_EVENTS: events per optimization (default 30000)
# Results of a suite are written when the suite has finished: a job that hits its time limit loses
# the suite it was working on. Measure the time per window with a large STRIDE first.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source cluster/env.sh
set -u
: "${PRESET:?Set PRESET (cmax-timestamp, cmax-variance or shiba)}"
: "${SUITES:?Set SUITES, e.g. SUITES=\"mvsec_dt1 mvsec_dt4\"}"
num_events="${NUM_EVENTS:-30000}"
name="opt-${PRESET}_n${num_events}"
export EVFLOW_EVAL_STRIDE="${STRIDE:-1}"
nvidia-smi
echo "estimator: $name | suites: $SUITES | stride: $EVFLOW_EVAL_STRIDE | commit: $(git rev-parse HEAD)"
"$BENCHMARK_PYTHON" experiments/make_optimizer_run.py "$PRESET" --num_events "$num_events" \
    --path_mlflow "$EVFLOW_MLRUNS" --exist_ok
status=0
# shellcheck disable=SC2086
"$BENCHMARK_PYTHON" -u experiments/evaluate.py --suites $SUITES --runs "$name" \
    --path_mlflow "$EVFLOW_MLRUNS" || status=$?
"$BENCHMARK_PYTHON" experiments/collect.py --path_mlflow "$EVFLOW_MLRUNS"
exit $status
