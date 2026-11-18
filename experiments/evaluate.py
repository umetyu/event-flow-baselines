"""
Evaluates trained (or pretrained) runs on one or more evaluation suites by calling eval_flow.py.

    python experiments/evaluate.py --suites mvsec_dt1 mvsec_dt4 --runs 's2_*'
    python experiments/evaluate.py --suites mvsec_val_dt1 --runs step2  # all runs of an experiment
    python experiments/evaluate.py --suites mvsec_dt1 --runs LIFFireNet  # run id or run name

Suites: mvsec_dt1, mvsec_dt4 (AEE on the four test sequences of the paper),
        mvsec_val_dt1, mvsec_val_dt4 (AEE on the other MVSEC sequences, for tuning),
        ecd, hqf (FWL/RSAT, as in the paper), affine (FWL/RSAT on AffineObjects).
Results land in results_inference/<run id>/ (metrics_<n>.yml + eval_<n>.yml) and are summarized by
experiments/collect.py. A suite that was already evaluated for a run is skipped unless --force.
"""

import argparse
import glob
import os
import subprocess
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import ROOT, data_field, list_runs, select_runs  # noqa: E402

FIXED = {
    "mvsec_dt1": "eval_mvsec_dt1",
    "mvsec_dt4": "eval_mvsec_dt4",
    "mvsec_val_dt1": "eval_mvsec_val_dt1",
    "mvsec_val_dt4": "eval_mvsec_val_dt4",
    "ecd": "eval_ecd",
    "hqf": "eval_hqf",
}


def suite_config(suite, run):
    """
    Config file of a suite. Suites without ground truth must use the forward-pass window the model
    was trained with, which the evaluation config has to state explicitly.
    """

    if suite in FIXED:
        return FIXED[suite]
    window = int(data_field(run, "window", 1000))
    if suite == "affine":
        # recurrent models: 10k events per pass (5 passes); single-pass models: the whole 50k window
        return "eval_affine_w{}".format(50000 if window >= 50000 else 10000)
    raise ValueError("Unknown suite: " + suite)


def already_done(path_results, run_id, config):
    with open(config) as fid:
        wanted = yaml.safe_load(fid)["data"]
    for path in glob.glob(os.path.join(path_results, run_id, "eval_*.yml")):
        with open(path) as fid:
            done = yaml.safe_load(fid).get("data", {})
        eval_id = os.path.basename(path)[5:-4]
        same = (
            os.path.basename(os.path.normpath(str(done.get("path", "")))) == os.path.basename(os.path.normpath(wanted["path"]))
            and done.get("mode") == wanted["mode"]
            and done.get("window") == wanted["window"]
            and done.get("window_eval") == wanted["window_eval"]
        )
        if same and os.path.isfile(os.path.join(path_results, run_id, "metrics_{}.yml".format(eval_id))):
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suites", nargs="+", required=True)
    parser.add_argument("--runs", nargs="*", default=[], help="run ids, run names or experiment names (wildcards ok)")
    parser.add_argument("--path_mlflow", default=os.path.join(ROOT, "mlruns"))
    parser.add_argument("--path_results", default=os.path.join(ROOT, "results_inference") + "/")
    parser.add_argument("--config_dir", default=os.path.join(ROOT, "configs", "exp"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    runs = select_runs(list_runs(args.path_mlflow), args.runs)
    if not runs:
        raise SystemExit("No runs match {} in {}".format(args.runs, args.path_mlflow))

    failed = []
    for run in runs:
        label = run["name"] or run["run_id"]
        for suite in args.suites:
            config_name = suite_config(suite, run)
            config = os.path.join(args.config_dir, config_name + ".yml")
            if not os.path.isfile(config):
                print("SKIP {} / {}: {} does not exist (add the window to experiments/matrix.py)".format(label, suite, config))
                failed.append((label, suite))
                continue
            if not args.force and already_done(args.path_results, run["run_id"], config):
                print("done {} / {}".format(label, suite), flush=True)
                continue
            print("eval {} / {} ({})".format(label, suite, config_name), flush=True)
            cmd = [
                sys.executable, "-u", "eval_flow.py", run["run_id"],
                "--config", config, "--path_mlflow", args.path_mlflow, "--path_results", args.path_results,
            ]
            if subprocess.run(cmd, cwd=ROOT).returncode != 0:
                failed.append((label, suite))

    if failed:
        print("\nFAILED: " + ", ".join("{}/{}".format(*f) for f in failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
