#!/bin/bash -l
#SBATCH --job-name=evflow-eval
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

# Evaluate runs on evaluation suites and refresh the result tables. From the repository root:
#   sbatch --export=ALL,SUITES="mvsec_dt1 mvsec_dt4",RUNS="step2" cluster/eval.sbatch
# RUNS: run ids, run names or experiment names (wildcards allowed); empty = every run in mlruns/.
# MODELS: model classes to keep, or "core" for the six models of the study; empty = all.
# Add REFERENCE=1 to compare the MVSEC AEE with Table 1 of the paper.
# STRIDE=n scores only every n-th ground-truth map (to compare with strided optimizer runs).
# FLOW_DT_REF=s (seconds) rescales the output by dt_input / s (time-surface diagnostic, separate suite).
# CHECKPOINTS="all" (or names such as "epoch_0002 val_best") evaluates intermediate weights (step 5b).
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source cluster/env.sh
set -u
: "${SUITES:?Set SUITES, e.g. SUITES=\"mvsec_dt1 mvsec_dt4\"}"
export EVFLOW_EVAL_STRIDE="${STRIDE:-1}"
export EVFLOW_FLOW_DT_REF="${FLOW_DT_REF:-0}"
nvidia-smi
status=0
set -f  # checkpoint patterns such as epoch_000? must reach evaluate.py unexpanded
# shellcheck disable=SC2086
"$BENCHMARK_PYTHON" -u experiments/evaluate.py --suites $SUITES --runs ${RUNS:-} --models ${MODELS:-} \
    ${CHECKPOINTS:+--checkpoints $CHECKPOINTS} --path_mlflow "$EVFLOW_MLRUNS" || status=$?
# shellcheck disable=SC2086
# the tables always cover every run, so that jobs for different runs do not overwrite each other's rows
"$BENCHMARK_PYTHON" experiments/collect.py --path_mlflow "$EVFLOW_MLRUNS" ${REFERENCE:+--reference}
exit $status
