#!/bin/bash
# Submit one training job per config, from the repository root:
#   bash cluster/submit_train.sh configs/exp/s2_*_seed0.yml
# Extra sbatch options can be passed through SBATCH_ARGS, e.g. SBATCH_ARGS="--time=12:00:00".
set -euo pipefail
if [[ $# -eq 0 ]]; then
    echo "usage: bash cluster/submit_train.sh <config.yml> [<config.yml> ...]" >&2
    exit 2
fi
mkdir -p logs
for config in "$@"; do
    [[ -f "$config" ]] || { echo "Missing config: $config" >&2; exit 2; }
    name="$(basename "$config" .yml)"
    # shellcheck disable=SC2086
    sbatch ${SBATCH_ARGS:-} --job-name="$name" --export=ALL,CONFIG="$config" cluster/train.sbatch
done
