#!/bin/bash
# Puts the datasets where the configs expect them, without modifying the source folder:
#   $EVFLOW_DATA/training -> symlink to the UZH-FPV training sequences
#   $EVFLOW_DATA/MVSEC/   -> MVSEC.zip extracted (skipped if .h5 files are already there)
# and remembers the location in cluster/local.env for all later jobs.
#
#   bash cluster/link_data.sh [SOURCE_DIR]
# SOURCE_DIR must contain author_training/ and MVSEC.zip (or an extracted MVSEC/).
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

SRC="${1:-${EVFLOW_SOURCE:-/scratch/izar/$USER/uzh_mvsec_partition_seed0/data}}"
DATA="${EVFLOW_DATA:-/scratch/izar/$USER/event-flow-baselines-data}"

[[ -d "$SRC/author_training" ]] || { echo "Missing $SRC/author_training" >&2; exit 2; }
mkdir -p "$DATA"

if [[ ! -e "$DATA/training" ]]; then
    ln -s "$SRC/author_training" "$DATA/training"
fi

if [[ -z "$(find -L "$DATA/MVSEC" -name '*.h5' -print -quit 2>/dev/null)" ]]; then
    if [[ -d "$SRC/MVSEC" && -n "$(find -L "$SRC/MVSEC" -name '*.h5' -print -quit)" ]]; then
        ln -sfn "$SRC/MVSEC" "$DATA/MVSEC"
    elif [[ -f "$SRC/MVSEC.zip" ]]; then
        mkdir -p "$DATA/MVSEC"
        echo "Extracting $SRC/MVSEC.zip ..."
        unzip -n -q "$SRC/MVSEC.zip" -d "$DATA/MVSEC"
    else
        echo "Missing $SRC/MVSEC.zip (or $SRC/MVSEC/)" >&2; exit 2
    fi
fi

printf 'export EVFLOW_DATA=%q\n' "$DATA" > cluster/local.env
echo "EVFLOW_DATA=$DATA"
echo "training: $(find -L "$DATA/training" -name '*.h5' | wc -l) .h5 files"
echo "MVSEC:    $(find -L "$DATA/MVSEC" -name '*.h5' | wc -l) .h5 files"
