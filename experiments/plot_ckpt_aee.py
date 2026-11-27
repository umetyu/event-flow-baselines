"""
MVSEC AEE of the intermediate checkpoints of a run against the epoch, with the training and validation
loss on a second axis (overfitting check, step 5b). Reads results/runs.csv (experiments/collect.py) and
the loss curves in mlruns/.

    python experiments/plot_ckpt_aee.py --runs s5b_evflownet-ts5pol_w1000_seed0 \
        s5b_evflownet-cnt_w1000_seed0 s5b_evflownet-voxel5_w1000_seed0 \
        --labels "time surface 5tau (pol)" count "voxel 5 bins" --out results/ckpt_aee.png
"""

import argparse
import csv
import os
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_loss import find_run, read_metric  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="run names")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--csv", default="results/runs.csv")
    parser.add_argument("--mlruns", default=os.environ.get("EVFLOW_MLRUNS", "mlruns"))
    parser.add_argument("--suite", default="MVSEC_dt1")
    parser.add_argument("--out", default="results/ckpt_aee.png")
    args = parser.parse_args()
    labels = args.labels or args.runs

    rows = list(csv.DictReader(open(args.csv)))
    prefix = args.suite + "_ckpt-"
    fig, (ax_aee, ax_loss) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for key, label in zip(args.runs, labels):
        per = {}
        for r in rows:
            if r["run"] == key and r["metric"] == "AEE" and r["suite"].startswith(prefix):
                per.setdefault(r["suite"][len(prefix):], []).append(float(r["value"]))
        epochs = sorted((int(c[len("epoch_"):]), statistics.mean(v)) for c, v in per.items() if c.startswith("epoch_"))
        if not epochs:
            print("{}: no checkpoint results for suite {} in {}".format(label, args.suite, args.csv))
            continue
        line, = ax_aee.plot([e for e, _ in epochs], [a for _, a in epochs], "o-", ms=3, lw=1.5, label=label)
        best = min(epochs, key=lambda p: p[1])
        print("{}: AEE {} per epoch".format(label, args.suite))
        print("  " + "  ".join("{}:{:.3f}".format(e, a) for e, a in epochs))
        print("  lowest AEE at epoch {} ({:.3f}); last checkpoint epoch {} ({:.3f})".format(
            best[0], best[1], epochs[-1][0], epochs[-1][1]))
        main_rows = [float(r["value"]) for r in rows if r["run"] == key and r["metric"] == "AEE" and r["suite"] == args.suite]
        if main_rows:
            print("  weights of the lowest training loss: {:.3f}".format(statistics.mean(main_rows)))
        for name in sorted(c for c in per if c.startswith("val_")):
            print("  weights of the lowest {} loss: {:.3f}".format(name[:-5], statistics.mean(per[name])))

        run_dir = find_run(args.mlruns, key)
        loss = read_metric(run_dir, "loss")
        ax_loss.plot([e for e, _ in loss], [v for _, v in loss], lw=1.5, color=line.get_color(), label=label + " (train)")
        for metric, style, what in [("val_loss", "--", "val, MVSEC test chunks"), ("val_loss_od2", ":", "val, outdoor_day2")]:
            if os.path.exists(os.path.join(run_dir, "metrics", metric)):
                val = read_metric(run_dir, metric)
                ax_loss.plot([e for e, _ in val], [v for _, v in val], style, lw=1.5, color=line.get_color(),
                             label="{} ({})".format(label, what))
    ax_aee.set_ylabel("AEE " + args.suite.replace("_", " "))
    ax_aee.set_title("MVSEC AEE of the weights after each epoch")
    ax_aee.grid(alpha=0.3)
    ax_aee.legend(frameon=False)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("CMax loss (epoch average)")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("Written: " + args.out)


if __name__ == "__main__":
    main()
