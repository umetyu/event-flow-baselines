"""
Convert an official MVSEC HDF5 file (e.g. outdoor_day2_data.hdf5 from the MVSEC download page) into the
event layout of this repository (events/{xs,ys,ts,ps}, attrs t0/tk/duration), left DAVIS camera only.

    python tools/convert_mvsec_raw.py --src /scratch/.../outdoor_day2_data.hdf5 \
        --dst $EVFLOW_DATA/MVSEC_od2/outdoor_day2_data.h5

Optionally (--flow outdoor_day2_gt_flow_dist.npz) the ground-truth flow of EV-FlowNet (distorted image
coordinates, one map per grayscale frame) is written as flow_dt1/<idx>, as in the evaluation files.

The official file stores the events as an N x 4 array (x, y, t [s], p in {-1, 1}); it is read in blocks,
so the conversion needs little memory. Coordinates are kept as they are (no undistortion), as in the
MVSEC files used for evaluation.
"""

import argparse
import os

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="official MVSEC *_data.hdf5")
    parser.add_argument("--dst", required=True, help="output .h5")
    parser.add_argument("--flow", default=None, help="optional *_gt_flow_dist.npz (EV-FlowNet ground truth)")
    parser.add_argument("--block", type=int, default=10_000_000, help="events read per block")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    with h5py.File(args.src, "r") as src, h5py.File(args.dst + ".tmp", "w") as dst:
        events = src["davis/left/events"]
        num = events.shape[0]
        print("{}: {} events".format(args.src, num), flush=True)
        xs = dst.create_dataset("events/xs", (num,), dtype=np.int16)
        ys = dst.create_dataset("events/ys", (num,), dtype=np.int16)
        ts = dst.create_dataset("events/ts", (num,), dtype=np.float64)
        ps = dst.create_dataset("events/ps", (num,), dtype=bool)
        num_pos, last_t = 0, -np.inf
        for start in range(0, num, args.block):
            block = events[start : start + args.block]
            if block[0, 2] < last_t or np.any(np.diff(block[:, 2]) < 0):
                raise ValueError("timestamps are not sorted around event {}".format(start))
            last_t = block[-1, 2]
            xs[start : start + len(block)] = block[:, 0].astype(np.int16)
            ys[start : start + len(block)] = block[:, 1].astype(np.int16)
            ts[start : start + len(block)] = block[:, 2]
            pol = block[:, 3] > 0
            ps[start : start + len(block)] = pol
            num_pos += int(pol.sum())
            print("  {:.0f}%".format(100 * min(start + args.block, num) / num), end="\r", flush=True)
        t0, tk = float(ts[0]), float(ts[num - 1])
        dst.attrs["t0"] = t0
        dst.attrs["tk"] = tk
        dst.attrs["duration"] = tk - t0
        dst.attrs["num_events"] = num
        dst.attrs["num_pos"] = num_pos
        dst.attrs["num_neg"] = num - num_pos
        dst.attrs["sensor_resolution"] = [260, 346]
        dst.attrs["source"] = os.path.basename(args.src) + " (davis/left/events)"

        if args.flow:
            gt = np.load(args.flow)
            stamps, fx, fy = gt["timestamps"], gt["x_flow_dist"], gt["y_flow_dist"]
            for i, t in enumerate(stamps):
                dset = dst.create_dataset(
                    "flow_dt1/{:09d}".format(i), data=np.stack([fx[i], fy[i]]).astype(np.float32), compression="gzip"
                )
                dset.attrs["timestamp"] = float(t)
            print("flow_dt1: {} maps".format(len(stamps)))
        print("\n{:.1f} s of events, {:.1f}% positive".format(tk - t0, 100 * num_pos / num))
    os.replace(args.dst + ".tmp", args.dst)
    print("Written: " + args.dst)


if __name__ == "__main__":
    main()
