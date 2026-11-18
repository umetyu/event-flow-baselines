"""
Registers an optimization-based estimator (models/optimization.py) as a run, so that it can be
evaluated and tabulated with the same tools as the networks.

    python experiments/make_optimizer_run.py cmax-timestamp
    python experiments/make_optimizer_run.py cmax-variance
    python experiments/make_optimizer_run.py shiba          # needs cluster/setup_shiba.sh
    python experiments/make_optimizer_run.py cmax-timestamp --num_events 15000 --name my-run

These estimators have no weights; the run only stores the settings. Evaluate with
    python experiments/evaluate.py --suites mvsec_dt1 mvsec_dt4 --runs 'opt-*'
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow  # noqa: E402

from experiments.common import ROOT, list_runs  # noqa: E402
from utils.utils import resolve_tracking_uri, save_model  # noqa: E402

PRESETS = {
    # own implementation, objective = training loss of the networks
    "cmax-timestamp": {"name": "CMaxOptimizer", "objective": "timestamp", "regul_weight": 0.001},
    # own implementation, objective = variance of the image of warped events (Gallego et al. 2018)
    "cmax-variance": {"name": "CMaxOptimizer", "objective": "variance", "regul_weight": 0.001},
    # public code of Shiba et al. (ECCV 2022), their MVSEC configuration without time-aware flow
    "shiba": {"name": "ExternalCMax", "config": "configs/mvsec_indoor_no_timeaware.yaml"},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("preset", choices=sorted(PRESETS))
    parser.add_argument("--num_events", type=int, default=30000, help="events used for every optimization")
    parser.add_argument("--iters", type=int, default=None, help="CMaxOptimizer: Adam steps per window")
    parser.add_argument("--regul_weight", type=float, default=None, help="CMaxOptimizer: smoothness weight")
    parser.add_argument("--name", default="")
    parser.add_argument("--exist_ok", action="store_true", help="do nothing if the run already exists")
    parser.add_argument("--path_mlflow", default=os.environ.get("EVFLOW_MLRUNS", os.path.join(ROOT, "mlruns")))
    args = parser.parse_args()

    model = dict(PRESETS[args.preset])
    model.update({"num_events": args.num_events, "encoding": "cnt", "round_encoding": False, "norm_input": False,
                  "num_bins": 2, "mask_output": True})
    if model["name"] == "CMaxOptimizer":
        model.update({"scales": [1, 2, 4, 8, 16], "iters": 40, "iters_first": 200, "lr": 0.2})
        if args.iters is not None:
            model["iters"] = args.iters
        if args.regul_weight is not None:
            model["regul_weight"] = args.regul_weight
    name = args.name if args.name else "opt-{}_n{}".format(args.preset, args.num_events)

    if any(run["name"] == name for run in list_runs(args.path_mlflow)):
        if args.exist_ok:
            print("Run {} already exists".format(name))
            return
        raise SystemExit("A run named {} already exists in {}".format(name, args.path_mlflow))

    config = {
        "experiment": "optimization",
        "data": {"path": "", "mode": "events", "window": 1000, "window_loss": 10000},
        "model": model,
        "spiking_neuron": None,
        "loss": {"flow_regul_weight": model.get("regul_weight", 0.0), "clip_grad": 100.0,
                 "overwrite_intermediate": False},
        "optimizer": {"name": "none"},
        "loader": {"n_epochs": 0, "batch_size": 1, "resolution": [256, 256], "augment": [], "augment_prob": [],
                   "gpu": 0, "seed": 0},
    }

    import models.optimization as zoo

    mlflow.set_tracking_uri(resolve_tracking_uri(args.path_mlflow))
    mlflow.set_experiment("optimization")
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(config)
        mlflow.log_param("n_params", 0)
        save_model(getattr(zoo, model["name"])(model))  # empty state dict: the estimators have no weights
        print("Created run {} ({})".format(name, run.info.run_id))


if __name__ == "__main__":
    main()
