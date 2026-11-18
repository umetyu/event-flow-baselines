#!/bin/bash
# Fetches the public code of Shiba et al., "Secrets of Event-Based Optical Flow" (ECCV 2022), at a
# pinned commit and installs what it needs. From the repository root on an Izar login shell:
#   bash cluster/setup_shiba.sh
# The code is GPL-3.0. It is kept outside of version control (third_party/) and only imported at
# run time by models/optimization.py (ExternalCMax).
set -eo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
source cluster/env.sh
set -u

URL=https://github.com/tub-rip/event_based_optical_flow.git
COMMIT=8c12d21a91c22922bb150acb9b1bada2f4b23def

if [[ ! -d "$EVFLOW_SHIBA_REPO/.git" ]]; then
    mkdir -p "$(dirname -- "$EVFLOW_SHIBA_REPO")"
    git clone "$URL" "$EVFLOW_SHIBA_REPO"
fi
git -C "$EVFLOW_SHIBA_REPO" fetch -q origin
git -C "$EVFLOW_SHIBA_REPO" checkout -q "$COMMIT"
[[ "$(git -C "$EVFLOW_SHIBA_REPO" rev-parse HEAD)" == "$COMMIT" ]] || { echo "Checkout failed" >&2; exit 2; }
[[ -z "$(git -C "$EVFLOW_SHIBA_REPO" status --porcelain)" ]] || { echo "External code was modified" >&2; exit 2; }

# torchvision has to match the installed torch (0.20.1 for torch 2.5.1)
torch_version=$("$BENCHMARK_PYTHON" -c 'import torch; print(torch.__version__)')
if ! "$BENCHMARK_PYTHON" -c 'import torchvision' 2>/dev/null; then
    case "$torch_version" in
        2.5.1+cu121) "$BENCHMARK_PYTHON" -m pip install --no-cache-dir --only-binary=:all: \
            'torchvision==0.20.1+cu121' --index-url https://download.pytorch.org/whl/cu121 ;;
        *) echo "Install the torchvision that matches torch $torch_version by hand" >&2; exit 2 ;;
    esac
fi
"$BENCHMARK_PYTHON" -m pip install --no-cache-dir --only-binary=:all: \
    'optuna>=3,<5' 'scipy>=1.10,<2' 'scikit-image>=0.21,<0.26' 'plotly>=5,<7' 'tqdm' 'Pillow'
"$BENCHMARK_PYTHON" -m pip check

# the external solver must be importable together with the installed torch
EVFLOW_SHIBA_REPO="$EVFLOW_SHIBA_REPO" "$BENCHMARK_PYTHON" - <<'PY'
import os, sys
sys.path.insert(0, os.environ["EVFLOW_SHIBA_REPO"])
from src import solver
assert "pyramidal_patch_contrast_maximization" in solver.collections
print("External solver ready:", os.environ["EVFLOW_SHIBA_REPO"])
PY
