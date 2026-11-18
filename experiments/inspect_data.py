"""
Prints what is actually inside the datasets and the mlruns folder, so that assumptions of the
configs (file names, resolution, ground-truth layout) can be checked before any GPU time is spent.

    python experiments/inspect_data.py            # writes results/data_report.txt and prints it

Only metadata and a small sample of every file are read (safe on a login node).
"""

import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import ROOT, list_runs, model_field  # noqa: E402

TEST = ["outdoor_day1", "indoor_flying1", "indoor_flying2", "indoor_flying3"]


def h5_files(path):
    files = []
    for root, _, names in os.walk(path, followlinks=True):
        files += [os.path.join(root, name) for name in names if name.endswith(".h5")]
    return sorted(files)


def describe(path, sample=200000):
    out = {}
    with h5py.File(path, "r") as f:
        out["groups"] = sorted(f.keys())
        out["attrs"] = {k: (v.item() if hasattr(v, "item") and np.size(v) == 1 else str(v)) for k, v in f.attrs.items()}
        if "events" in f and all(k in f["events"] for k in ["xs", "ys", "ts", "ps"]):
            n = len(f["events/xs"])
            out["events"] = n
            m = min(n, sample)
            xs, ys, ps = f["events/xs"][:m], f["events/ys"][:m], f["events/ps"][:m]
            out["x_range"] = [int(xs.min()), int(xs.max())] if m else None
            out["y_range"] = [int(ys.min()), int(ys.max())] if m else None
            out["ps_values"] = sorted(set(np.unique(ps).tolist()))
            out["dtypes"] = {k: str(f["events/" + k].dtype) for k in ["xs", "ys", "ts", "ps"]}
            if n:
                t0, t1 = float(f["events/ts"][0]), float(f["events/ts"][-1])
                out["duration_s"] = round(t1 - t0, 3)
        else:
            out["events"] = "MISSING events/{xs,ys,ts,ps}"
        for name in ["flow_dt1", "flow_dt4"]:
            if name in f:
                keys = sorted(f[name].keys())
                first = f[name][keys[0]]
                out[name] = {
                    "maps": len(keys),
                    "shape": list(first.shape),
                    "dtype": str(first.dtype),
                    "has_timestamp": "timestamp" in first.attrs,
                }
        if "images" in f:
            out["images"] = len(f["images"])
    return out


def main():
    data = os.environ.get("EVFLOW_DATA", "datasets/data")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=data)
    parser.add_argument("--path_mlflow", default=os.environ.get("EVFLOW_MLRUNS", os.path.join(ROOT, "mlruns")))
    parser.add_argument("--out", default=os.path.join(ROOT, "results", "data_report.txt"))
    args = parser.parse_args()

    lines, problems = [], []

    # ---- training
    files = h5_files(os.path.join(args.data, "training"))
    lines.append("== training: {} files in {}".format(len(files), os.path.join(args.data, "training")))
    if files:
        lines.append("   names: " + ", ".join(os.path.basename(f) for f in files[:4]) + (" ..." if len(files) > 4 else ""))
        counts, xmax, ymax = [], 0, 0
        for path in files[:: max(1, len(files) // 12)]:
            d = describe(path)
            if isinstance(d["events"], str):
                problems.append("training file without events: " + path)
                continue
            counts.append(d["events"])
            xmax, ymax = max(xmax, d["x_range"][1]), max(ymax, d["y_range"][1])
        d = describe(files[0])
        lines.append("   first file: groups {} | attrs {}".format(d["groups"], sorted(d["attrs"])))
        lines.append("   first file: dtypes {} | polarity values {}".format(d.get("dtypes"), d.get("ps_values")))
        if counts:
            lines.append("   sampled {} files: events per file {}..{} | max x {} | max y {}".format(
                len(counts), min(counts), max(counts), xmax, ymax))
            if xmax >= 128 or ymax >= 128:
                problems.append("training events exceed 128x128 (max x {}, max y {}): the configs assume 128x128 crops".format(xmax, ymax))
        if len(files) != 140:
            lines.append("   note: the paper describes 140 training sequences, found {}".format(len(files)))
        if len(files) < 8:
            problems.append("fewer than 8 training files: batch size 8 cannot be used")
    else:
        problems.append("no training files found")

    # ---- MVSEC
    files = h5_files(os.path.join(args.data, "MVSEC"))
    lines.append("== MVSEC: {} files in {}".format(len(files), os.path.join(args.data, "MVSEC")))
    found = set()
    for path in files:
        d = describe(path)
        name = os.path.basename(path)
        found.update(seq for seq in TEST if seq in name)
        lines.append("   {}: events {} | duration {} s | x {} y {} | flow_dt1 {} | flow_dt4 {} | attrs {}".format(
            name, d["events"], d.get("duration_s"), d.get("x_range"), d.get("y_range"),
            d.get("flow_dt1", "MISSING"), d.get("flow_dt4", "MISSING"), sorted(d["attrs"])))
        if "flow_dt1" not in d or "flow_dt4" not in d:
            problems.append("MVSEC file without flow_dt1/flow_dt4: " + name)
        elif d["flow_dt1"]["shape"][-2:] != [256, 256]:
            problems.append("{}: ground-truth maps are {} but the evaluation configs assume 256x256".format(name, d["flow_dt1"]["shape"]))
        if d.get("x_range") and (d["x_range"][1] >= 256 or d["y_range"][1] >= 256):
            problems.append("{}: events exceed 256x256".format(name))
    missing = [seq for seq in TEST if seq not in found]
    if missing:
        problems.append("MVSEC sequences of Table 1 not found by name: " + ", ".join(missing))

    # ---- pretrained / trained runs
    runs = list_runs(args.path_mlflow) if os.path.isdir(args.path_mlflow) else []
    lines.append("== mlruns: {} runs in {}".format(len(runs), args.path_mlflow))
    for run in runs[:40]:
        weights = os.path.isfile(os.path.join(run["dir"], "artifacts", "model", "data", "model.pth"))
        lines.append("   {} | name {!r} | experiment {} | model {} {} | weights {}".format(
            run["run_id"], run["name"], run["experiment"], model_field(run, "name"),
            model_field(run, "encoding"), "yes" if weights else "NO"))
    if not runs:
        lines.append("   (none: the pretrained models of the paper are not here, so step 1 cannot run yet)")

    lines.append("== problems: " + ("none found" if not problems else ""))
    lines += ["   - " + p for p in problems]

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fid:
        fid.write(report + "\n")
    print(report)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
