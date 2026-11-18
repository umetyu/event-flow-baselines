"""
Splits the MVSEC sequences into test and validation (symlinks, nothing is copied).

    python experiments/split_mvsec.py             # ${EVFLOW_DATA}/MVSEC -> ${EVFLOW_DATA}/splits/mvsec_{test,val}
    python experiments/split_mvsec.py --dry-run

test: the four sequences of Table 1 in Hagenaars et al. 2021 (outdoor_day1, indoor_flying1-3).
      Only used for the final numbers.
val:  every other sequence that has ground-truth flow (e.g. outdoor_day2, indoor_flying4).
      Used for model selection and tuning, with the same AEE metric as the test.

Training uses UZH-FPV only, so no MVSEC sequence is seen during training. Validation with ground
truth is preferred over the ground-truth-free metrics (FWL/RSAT): those are contrast-based like the
training loss and can improve while the flow gets worse.

If the downloaded data only contains the four test sequences, the validation split is empty and the
script stops: the missing sequences then have to be converted from the raw MVSEC recordings with the
same preprocessing as the distributed files.
"""

import argparse
import json
import os

import h5py

TEST = ["outdoor_day1", "indoor_flying1", "indoor_flying2", "indoor_flying3"]


def main():
    data = os.environ.get("EVFLOW_DATA", "datasets/data")
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=os.path.join(data, "MVSEC"))
    parser.add_argument("--out", default=os.path.join(data, "splits"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = []
    for root, _, names in os.walk(args.source, followlinks=True):
        files += [os.path.join(root, name) for name in names if name.endswith(".h5")]
    files.sort()
    if not files:
        raise FileNotFoundError("No .h5 files in " + args.source)

    split, skipped = {"test": [], "val": []}, []
    for path in files:
        with h5py.File(path, "r") as fid:
            has_gt = "flow_dt1" in fid and "flow_dt4" in fid
            num_gt = len(fid["flow_dt1"]) if has_gt else 0
        name = os.path.splitext(os.path.basename(path))[0]
        matches = [seq for seq in TEST if seq in name]
        if not has_gt:
            skipped.append(path)
            key = "skip (no flow_dt1/flow_dt4)"
        else:
            key = "test" if matches else "val"
            split[key].append(path)
        print("  {:28s} {:40s} {:6d} ground-truth maps".format(key, name, num_gt))

    found = sorted({seq for path in split["test"] for seq in TEST if seq in os.path.basename(path)})
    print("test: {} files | val: {} files | skipped: {}".format(len(split["test"]), len(split["val"]), len(skipped)))
    if found != sorted(TEST) or len(split["test"]) != len(TEST):
        raise RuntimeError("Expected exactly one file for each of {} in the test split, found {}.".format(TEST, found))
    if not split["val"]:
        raise RuntimeError(
            "No MVSEC sequence with ground truth besides the four test sequences in {}. Validation needs "
            "at least one more (e.g. outdoor_day2, indoor_flying4) in the same HDF5 layout.".format(args.source)
        )
    if args.dry_run:
        return

    for key, paths in split.items():
        folder = os.path.join(args.out, "mvsec_" + key)
        os.makedirs(folder, exist_ok=True)
        for old in os.listdir(folder):
            if os.path.islink(os.path.join(folder, old)):
                os.remove(os.path.join(folder, old))
        for path in paths:
            os.symlink(os.path.abspath(path), os.path.join(folder, os.path.basename(path)))
    with open(os.path.join(args.out, "mvsec_split.json"), "w") as fid:
        json.dump({key: [os.path.basename(p) for p in paths] for key, paths in split.items()}, fid, indent=2)
    print("Split written to " + args.out)


if __name__ == "__main__":
    main()
