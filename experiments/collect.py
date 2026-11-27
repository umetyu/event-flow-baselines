"""
Collects training curves and evaluation metrics of all runs into tables.

    python experiments/collect.py                    # all runs
    python experiments/collect.py --runs step2       # one experiment
    python experiments/collect.py --reference        # also compare MVSEC AEE with Table 1 of the paper

Writes results/runs.csv (one row per run and metric) and results/summary.md (mean and standard
deviation over seeds for every configuration). Lower is better for AEE and RSAT, higher for FWL.
"""

import argparse
import csv
import glob
import os
import re
import statistics
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (  # noqa: E402
    ROOT,
    data_field,
    list_runs,
    model_field,
    read_metric,
    reference_key,
    select_runs,
)


def suite_name(config):
    data = config.get("data", {})
    folder = os.path.basename(os.path.normpath(str(data.get("path", "unknown"))))
    mode = data.get("mode", "")
    stride = int(config.get("metrics", {}).get("stride", 1))
    suffix = "" if stride == 1 else "_stride{}".format(stride)
    dt_ref = float(config.get("metrics", {}).get("flow_dt_ref", 0) or 0)
    if dt_ref > 0:
        suffix += "_dtref{:g}ms".format(1000 * dt_ref)
    checkpoint = config.get("metrics", {}).get("checkpoint", None)
    if checkpoint:
        suffix += "_ckpt-" + checkpoint
    if mode in ["gtflow_dt1", "gtflow_dt4"]:
        return "{}_{}{}".format(folder, mode[-3:], suffix)
    return folder + suffix


def run_rows(run, path_results):
    """
    :return: list of (suite, metric, sequence, value)
    """

    rows = []
    for path in sorted(glob.glob(os.path.join(path_results, run["run_id"], "metrics_*.yml"))):
        eval_id = os.path.basename(path)[8:-4]
        config_path = os.path.join(path_results, run["run_id"], "eval_{}.yml".format(eval_id))
        if not os.path.isfile(config_path):
            continue
        with open(config_path) as fid:
            suite = suite_name(yaml.safe_load(fid))
        with open(path) as fid:
            metrics = yaml.safe_load(fid) or {}
        for metric, per_file in metrics.items():
            for fname, value in per_file.items():
                rows.append((suite, metric, os.path.splitext(fname)[0], float(value)))
    # a suite evaluated more than once: keep the latest evaluation
    latest = {}
    for suite, metric, seq, value in rows:
        latest[(suite, metric, seq)] = value
    return [(s, m, q, v) for (s, m, q), v in latest.items()]


def fmt(values):
    if len(values) == 1:
        return "{:.3f}".format(values[0])
    return "{:.3f} ± {:.3f} (n={})".format(statistics.mean(values), statistics.stdev(values), len(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument(
        "--models", nargs="*", default=[], help='model classes to keep, "core" for the six core models (default: all)'
    )
    parser.add_argument("--path_mlflow", default=os.path.join(ROOT, "mlruns"))
    parser.add_argument("--path_results", default=os.path.join(ROOT, "results_inference"))
    parser.add_argument("--out", default=os.path.join(ROOT, "results"))
    parser.add_argument("--reference", action="store_true", help="compare MVSEC AEE with the paper")
    args = parser.parse_args()

    runs = select_runs(list_runs(args.path_mlflow), args.runs, args.models)
    if not runs:
        raise SystemExit("No runs match {} (models: {}) in {}".format(args.runs, args.models or "all", args.path_mlflow))
    os.makedirs(args.out, exist_ok=True)

    table, groups, info, encodings = [], {}, {}, {}
    for run in runs:
        label = run["name"] or run["run_id"]
        group = re.sub(r"_seed\d+$", "", label)
        loss = read_metric(run["dir"], "loss")
        times = read_metric(run["dir"], "epoch_time_s")
        info[group] = {
            "model": model_field(run, "name", ""),
            "input": "{}{}".format(model_field(run, "encoding", ""), model_field(run, "num_bins", "")),
            "window": data_field(run, "window", ""),
            "n_params": run["params"].get("n_params", ""),
        }
        base = [run["experiment"], label, run["run_id"], info[group]["model"], info[group]["input"], info[group]["window"]]
        encodings[label] = model_field(run, "encoding", "")
        if loss:
            table.append(base + ["train", "epochs", "-", len(loss)])
            table.append(base + ["train", "best_loss", "-", min(v for _, v in loss)])
            groups.setdefault((group, "train", "best_loss"), []).append(min(v for _, v in loss))
            groups.setdefault((group, "train", "epochs"), []).append(len(loss))
        if times:
            table.append(base + ["train", "epoch_time_s", "-", statistics.mean(v for _, v in times)])
            groups.setdefault((group, "train", "epoch_time_s"), []).append(statistics.mean(v for _, v in times))
        per_suite = {}
        for suite, metric, seq, value in run_rows(run, args.path_results):
            table.append(base + [suite, metric, seq, value])
            per_suite.setdefault((suite, metric), []).append(value)
        for (suite, metric), values in per_suite.items():
            # one number per run (mean over sequences), then statistics over seeds
            groups.setdefault((group, suite, metric), []).append(statistics.mean(values))

    with open(os.path.join(args.out, "runs.csv"), "w", newline="") as fid:
        writer = csv.writer(fid)
        writer.writerow(["experiment", "run", "run_id", "model", "input", "window", "suite", "metric", "sequence", "value"])
        writer.writerows(table)

    # per-epoch checkpoints are only in runs.csv (experiments/plot_ckpt_aee.py), not in the table
    columns = sorted({(suite, metric) for (_, suite, metric) in groups if "_ckpt-epoch_" not in suite})
    lines = ["# Results", "", "Mean over sequences per run; `±` is the standard deviation over seeds.", ""]
    lines.append("| config | model | input | window | params | " + " | ".join("{} {}".format(*c) for c in columns) + " |")
    lines.append("|" + "---|" * (5 + len(columns)))
    for group in sorted(info):
        cells = [fmt(groups[(group, s, m)]) if (group, s, m) in groups else "" for s, m in columns]
        i = info[group]
        lines.append("| {} | {} | {} | {} | {} | ".format(group, i["model"], i["input"], i["window"], i["n_params"]) + " | ".join(cells) + " |")

    if args.reference:
        with open(os.path.join(ROOT, "experiments", "reference", "hagenaars2021_table1.yml")) as fid:
            reference = yaml.safe_load(fid)
        lines += ["", "## MVSEC AEE vs. Table 1 of Hagenaars et al. 2021", "",
                  "| run | model (paper row) | dt | sequence | ours | paper | diff |", "|---|---|---|---|---|---|---|"]
        worst, literature = 0.0, False
        for row in table:
            _, label, _, model, _, _, suite, metric, seq, value = row
            key = reference_key(model, encodings.get(label, ""))
            if metric != "AEE" or not suite.startswith("MVSEC_dt"):
                continue
            if key in reference["models"]:
                entry, own = reference["models"][key], True
            elif key in reference["literature"]:
                entry, own = reference["literature"][key], False
            else:
                continue
            if "_stride" in suite or "_dtref" in suite or "_ckpt-" in suite:
                continue  # only full, unmodified evaluations are compared with the paper
            dt = suite[-3:]
            for i, name in enumerate(reference["sequences"]):
                if name in seq:
                    paper = entry[dt][i]
                    if own:
                        worst = max(worst, abs(value - paper))
                    literature = literature or not own
                    lines.append("| {} | {} ({}) | {} | {} | {:.3f} | {:.2f} | {:+.3f} |".format(
                        label, model, entry["paper"], dt, name, value, paper, value - paper))
        lines += ["", "Largest absolute difference to the paper's own models: {:.3f} px (the paper reports two decimals).".format(worst)]
        if literature:
            lines += ["Rows marked `*` are literature values of networks trained on MVSEC; they are shown for "
                      "orientation and are not expected to match a network trained here."]

    with open(os.path.join(args.out, "summary.md"), "w") as fid:
        fid.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nWritten: {} and {}".format(os.path.join(args.out, "runs.csv"), os.path.join(args.out, "summary.md")))


if __name__ == "__main__":
    main()
