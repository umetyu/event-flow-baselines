#!/bin/bash
# One-time setup on an Izar login shell, from the repository root:  bash cluster/setup_izar.sh
# Creates (or reuses) the virtualenv, installs the dependencies and submits the smoke test.
# Only binary wheels are installed; nothing is compiled and no GPU work runs on the login node.
set -eo pipefail

if ! type module >/dev/null 2>&1; then
    echo 'Run this from an Izar login shell (bash -l) so that the module command is available.' >&2
    exit 2
fi
source cluster/env.sh
set -u

python -c 'import sys; assert (3, 9) <= sys.version_info[:2] <= (3, 12), sys.version; print(sys.version)'
if [[ ! -e "$EVENT_FLOW_VENV" ]]; then
    mkdir -p "$(dirname -- "$EVENT_FLOW_VENV")"
    python -m venv "$EVENT_FLOW_VENV"
elif [[ ! -x "$BENCHMARK_PYTHON" ]]; then
    echo "Existing path is not a valid virtualenv: $EVENT_FLOW_VENV" >&2
    exit 2
fi
echo "Environment: $EVENT_FLOW_VENV"

"$BENCHMARK_PYTHON" -m pip install --no-cache-dir --only-binary=:all: --upgrade 'pip==25.0.1'
if ! "$BENCHMARK_PYTHON" -c 'import torch' 2>/dev/null; then
    "$BENCHMARK_PYTHON" -m pip install --no-cache-dir --only-binary=:all: 'torch==2.5.1+cu121' \
        --index-url https://download.pytorch.org/whl/cu121
fi
"$BENCHMARK_PYTHON" -m pip install --no-cache-dir --only-binary=:all: -r requirements-izar.txt
"$BENCHMARK_PYTHON" -m pip check

mkdir -p logs results
"$BENCHMARK_PYTHON" -m pip freeze > results/izar-environment.txt
"$BENCHMARK_PYTHON" experiments/matrix.py --list > /dev/null

echo
echo "Datasets expected at:      $EVFLOW_DATA   (found: $(ls "$EVFLOW_DATA" 2>/dev/null | tr '\n' ' '))"
echo "Pretrained runs expected:  $EVFLOW_MLRUNS"
echo "Free space:"; df -h "$(dirname -- "$EVFLOW_DATA")" 2>/dev/null | tail -1 || true
echo
if [[ -z "${NO_SMOKE:-}" ]]; then
    sbatch --export=ALL cluster/smoke.sbatch
    echo 'Check the result with:  cat logs/smoke-<JOB_ID>.out   (must end with "All smoke tests passed.")'
fi
