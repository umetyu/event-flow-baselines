"""
Plot the training loss per epoch of MLflow runs and mark the epoch whose weights were kept.

train_flow.py (like the upstream code) saves the model only when the epoch-average training loss
reaches a new minimum, so the kept weights are those of the epoch with the lowest training loss.
There is no validation set (as in Hagenaars et al. 2021), so no inference-time loss is logged.

    python experiments/plot_loss.py --runs EVFlowNet s2_evflownet-voxel5_w1000_seed0 \
        --labels "ConvGRU EV-FlowNet" "EV-FlowNet + voxel (5 bins)" --out results/loss.png

A run is given by its run name (tag mlflow.runName) or its run ID.
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_run(mlruns, key):
    for run_dir in sorted(glob.glob(os.path.join(mlruns, "*", "*"))):
        if not os.path.isdir(os.path.join(run_dir, "metrics")):
            continue
        name_file = os.path.join(run_dir, "tags", "mlflow.runName")
        name = open(name_file).read().strip() if os.path.exists(name_file) else ""
        if key in (os.path.basename(run_dir), name):
            return run_dir
    raise FileNotFoundError("run {} not found in {}".format(key, mlruns))


def read_metric(run_dir, metric):
    path = os.path.join(run_dir, "metrics", metric)
    rows = [line.split() for line in open(path) if line.strip()]
    points = {}
    for _, value, step in rows:  # timestamp value step; keep the last value logged for a step
        points[int(step)] = float(value)
    return sorted(points.items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--mlruns", default=os.environ.get("EVFLOW_MLRUNS", "mlruns"))
    parser.add_argument("--metric", default="loss")
    parser.add_argument("--out", default="results/loss.png")
    args = parser.parse_args()
    labels = args.labels or args.runs

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for key, label in zip(args.runs, labels):
        points = read_metric(find_run(args.mlruns, key), args.metric)
        epochs = [e for e, _ in points]
        values = [v for _, v in points]
        best = min(range(len(values)), key=values.__getitem__)
        line, = ax.plot(epochs, values, lw=1.5, label=label)
        ax.plot(epochs[best], values[best], "o", color=line.get_color(), ms=6)
        print("{}: {} epochs | best epoch {} loss {:.4f} (weights kept) | last epoch {} loss {:.4f}".format(
            label, len(values), epochs[best], values[best], epochs[-1], values[-1]))
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (epoch average)")
    ax.set_title("Training loss; dot = epoch whose weights are used")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("Written:", args.out)


if __name__ == "__main__":
    main()
