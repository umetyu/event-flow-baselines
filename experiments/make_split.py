"""
Deterministic train/validation split of the training sequences (symlinks, nothing is copied).

    python experiments/make_split.py                 # ${EVFLOW_DATA}/training -> ${EVFLOW_DATA}/splits/{train,val}
    python experiments/make_split.py --dry-run       # only show the groups and the split

Why: the protocol of the paper trains on UZH-FPV and tests on MVSEC, with no validation set. To tune
anything (later: the Hebbian mechanism) without looking at MVSEC, part of the training data is held
out and scored with ground-truth-free metrics (FWL/RSAT).

The training files are crops of longer recordings. Crops of one recording must not end up on both
sides, so files are grouped by recording before splitting. The recording is taken from the file name
with --group-regex; CHECK THE PRINTED GROUPS, the default is a guess about the naming scheme.
"""

import argparse
import json
import os
import random
import re


def main():
    data = os.environ.get("EVFLOW_DATA", "datasets/data")
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=os.path.join(data, "training"))
    parser.add_argument("--out", default=os.path.join(data, "splits"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--group-regex",
        default=r"^(.*?)(?:[_-]\d+)?\.h5$",
        help="first capture group = recording name (default: file name without a trailing _<number>)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = []
    for root, _, names in os.walk(args.source, followlinks=True):
        files += [os.path.join(root, name) for name in names if name.endswith(".h5")]
    files.sort()
    if not files:
        raise FileNotFoundError("No .h5 files in " + args.source)

    groups = {}
    for path in files:
        match = re.match(args.group_regex, os.path.basename(path))
        groups.setdefault(match.group(1) if match else os.path.basename(path), []).append(path)

    names = sorted(groups)
    random.Random(args.seed).shuffle(names)
    target = args.val_fraction * len(files)
    val_groups, count = [], 0
    for name in names:
        if count >= target:
            break
        val_groups.append(name)
        count += len(groups[name])
    split = {
        "val": sorted(p for name in val_groups for p in groups[name]),
        "train": sorted(p for name in names if name not in val_groups for p in groups[name]),
    }

    print("{} files in {} groups (regex {!r})".format(len(files), len(groups), args.group_regex))
    for name in sorted(groups):
        print("  {:4s} {:40s} {:4d} files".format("val" if name in val_groups else "", name, len(groups[name])))
    print("train: {} files | val: {} files".format(len(split["train"]), len(split["val"])))
    if len(groups) == len(files):
        print("WARNING: every file is its own group. If files are crops of shared recordings, set --group-regex.")
    if not split["train"] or not split["val"]:
        raise RuntimeError("Empty split: adjust --val-fraction or --group-regex.")
    if args.dry_run:
        return

    for key, paths in split.items():
        folder = os.path.join(args.out, key)
        os.makedirs(folder, exist_ok=True)
        for old in os.listdir(folder):
            if os.path.islink(os.path.join(folder, old)):
                os.remove(os.path.join(folder, old))
        for path in paths:
            # keep names unique even if the source has sub-folders
            link = os.path.join(folder, os.path.relpath(path, args.source).replace(os.sep, "__"))
            os.symlink(os.path.abspath(path), link)
    manifest = {
        "source": os.path.abspath(args.source),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "group_regex": args.group_regex,
        "val_groups": sorted(val_groups),
        "train": [os.path.basename(p) for p in split["train"]],
        "val": [os.path.basename(p) for p in split["val"]],
    }
    with open(os.path.join(args.out, "split.json"), "w") as fid:
        json.dump(manifest, fid, indent=2)
    print("Split written to " + args.out)


if __name__ == "__main__":
    main()
