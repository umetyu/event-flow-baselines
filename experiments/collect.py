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

from experiments.common import ROOT, data_field, list_runs, model_field, read_metric, select_runs  # noqa: E402


def suite_name(config):
    data = config.get("data", {})
    folder = os.path.basename(os.path.normpath(str(data.get("path", "unknown"))))
    mode = data.get("mode", "")
    if mode in ["gtflow_dt1", "gtflow_dt4"]:
        return "{}_{}".format(folder, mode[-3:])
    return folder


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
    parser.add_argument("--path_mlflow", default=os.path.join(ROOT, "mlruns"))
    parser.add_argument("--path_results", default=os.path.join(ROOT, "results_inference"))
    parser.add_argument("--out", default=os.path.join(ROOT, "results"))
    parser.add_argument("--reference", action="store_true", help="compare MVSEC AEE with the paper")
    args = parser.parse_args()

    runs = select_runs(list_runs(args.path_mlflow), args.runs)
    if not runs:
        raise SystemExit("No runs match {} in {}".format(args.runs, args.path_mlflow))
    os.makedirs(args.out, exist_ok=True)

    table, groups, info = [], {}, {}
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

    columns = sorted({(suite, metric) for (_, suite, metric) in groups})
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
        worst = 0.0
        for row in table:
            _, label, _, model, _, _, suite, metric, seq, value = row
            if metric != "AEE" or model not in reference["models"]:
                continue
            dt = suite[-3:]
            for i, name in enumerate(reference["sequences"]):
                if name in seq:
                    paper = reference["models"][model][dt][i]
                    worst = max(worst, abs(value - paper))
                    lines.append("| {} | {} ({}) | {} | {} | {:.3f} | {:.2f} | {:+.3f} |".format(
                        label, model, reference["models"][model]["paper"], dt, name, value, paper, value - paper))
        lines += ["", "Largest absolute difference: {:.3f} px (the paper reports two decimals).".format(worst)]

    with open(os.path.join(args.out, "summary.md"), "w") as fid:
        fid.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nWritten: {} and {}".format(os.path.join(args.out, "runs.csv"), os.path.join(args.out, "summary.md")))


if __name__ == "__main__":
    main()
