"""
Cut MVSEC into fixed-size event chunks for the validation loss, in the layout of the training files.

    python experiments/make_val_chunks.py --src $EVFLOW_DATA/MVSEC --dst $EVFLOW_DATA/validation_mvsec

From each sequence, --chunks pieces of --events events are taken, evenly spaced in the middle of the
sequence (positions i / (chunks + 1), i = 1..chunks), so the start of the indoor sequences, where the
drone is still on the ground, is avoided. The training data (UZH-FPV) is made of 500k-event pieces,
hence the default size.

The chunks are only used to monitor a validation loss during training. They come from the MVSEC
evaluation sequences, so they must not be used to choose weights or hyper-parameters.

outdoor_day2 (not an evaluation sequence; converted with tools/convert_mvsec_raw.py) can be used to
choose weights:

    python experiments/make_val_chunks.py --src $EVFLOW_DATA/MVSEC_od2 --dst $EVFLOW_DATA/validation_od2 --chunks 8

(8 chunks, because the validation loader needs at least as many files as its batch size.)
"""

import argparse
import glob
import os

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="folder with the MVSEC .h5 files")
    parser.add_argument("--dst", required=True)
    parser.add_argument("--chunks", type=int, default=2, help="chunks per sequence")
    parser.add_argument("--events", type=int, default=500000, help="events per chunk")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "*.h5")))
    if not files:
        raise FileNotFoundError("no .h5 files in " + args.src)
    os.makedirs(args.dst, exist_ok=True)

    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        with h5py.File(path, "r") as src:
            num = len(src["events/ts"])
            if num < args.events:
                raise ValueError("{} has only {} events".format(name, num))
            for i in range(1, args.chunks + 1):
                start = int(round(i / (args.chunks + 1) * num - args.events / 2))
                start = min(max(start, 0), num - args.events)
                stop = start + args.events
                out = os.path.join(args.dst, "{}_val{}.h5".format(name, i - 1))
                with h5py.File(out, "w") as dst:
                    for key in ["xs", "ys", "ts", "ps"]:
                        dst.create_dataset("events/" + key, data=src["events/" + key][start:stop])
                    ts = dst["events/ts"][:]
                    ps = dst["events/ps"][:]
                    dst.attrs["t0"] = ts[0]
                    dst.attrs["tk"] = ts[-1]
                    dst.attrs["duration"] = ts[-1] - ts[0]
                    dst.attrs["num_events"] = args.events
                    dst.attrs["num_pos"] = int(np.sum(ps > 0))
                    dst.attrs["num_neg"] = int(np.sum(ps <= 0))
                    if "sensor_resolution" in src.attrs:
                        dst.attrs["sensor_resolution"] = src.attrs["sensor_resolution"]
                    dst.attrs["source"] = "{} events [{}, {})".format(os.path.basename(path), start, stop)
                print("{}: events [{}, {}) | {:.2f} s".format(os.path.basename(out), start, stop, ts[-1] - ts[0]))


if __name__ == "__main__":
    main()
