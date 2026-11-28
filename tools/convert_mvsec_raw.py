"""
Convert an official MVSEC HDF5 file (e.g. outdoor_day2_data.hdf5 from the MVSEC download page) into the
event layout of this repository (events/{xs,ys,ts,ps}, attrs t0/tk/duration), left DAVIS camera only.

    python tools/convert_mvsec_raw.py --src /scratch/.../outdoor_day2_data.hdf5 \
        --dst $EVFLOW_DATA/MVSEC_od2/outdoor_day2_data.h5

Optionally (--flow outdoor_day2_gt_flow_dist.npz) the ground-truth flow of EV-FlowNet (distorted image
coordinates, one map per grayscale frame) is written as flow_dt1/<idx>, as in the evaluation files.

The official file stores the events as an N x 4 array (x, y, t [s], p in {-1, 1}) on the full 346 x 260
sensor; it is read in blocks, so the conversion needs little memory. As in the MVSEC files used for
evaluation (x, y in [0, 255]), only the central 256 x 256 crop is kept (EV-FlowNet convention: rows 2..257,
columns 45..300) and shifted to the origin; coordinates are not undistorted.
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
    parser.add_argument("--crop", type=int, nargs=2, default=[256, 256], metavar=("H", "W"),
                        help="central crop (0 0 = keep the full sensor)")
    parser.add_argument("--sensor", type=int, nargs=2, default=[260, 346], metavar=("H", "W"))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    with h5py.File(args.src, "r") as src, h5py.File(args.dst + ".tmp", "w") as dst:
        events = src["davis/left/events"]
        total = events.shape[0]
        crop_h, crop_w = args.crop if args.crop[0] > 0 else args.sensor
        oy, ox = (args.sensor[0] - crop_h) // 2, (args.sensor[1] - crop_w) // 2
        print("{}: {} events, keeping the {}x{} crop at rows {}..{}, columns {}..{}".format(
            args.src, total, crop_h, crop_w, oy, oy + crop_h - 1, ox, ox + crop_w - 1), flush=True)
        dsets = {}
        for key, dtype in [("xs", np.int16), ("ys", np.int16), ("ts", np.float64), ("ps", bool)]:
            dsets[key] = dst.create_dataset("events/" + key, (0,), maxshape=(None,), dtype=dtype, chunks=(1 << 20,))
        num, num_pos, last_t = 0, 0, -np.inf
        for start in range(0, total, args.block):
            block = events[start : start + args.block]
            if block[0, 2] < last_t or np.any(np.diff(block[:, 2]) < 0):
                raise ValueError("timestamps are not sorted around event {}".format(start))
            last_t = block[-1, 2]
            x = block[:, 0].astype(np.int64) - ox
            y = block[:, 1].astype(np.int64) - oy
            keep = (x >= 0) & (x < crop_w) & (y >= 0) & (y < crop_h)
            n = int(keep.sum())
            for key, values in [("xs", x[keep]), ("ys", y[keep]), ("ts", block[keep, 2]), ("ps", block[keep, 3] > 0)]:
                dsets[key].resize((num + n,))
                dsets[key][num:] = values
            num_pos += int((block[keep, 3] > 0).sum())
            num += n
            print("  {:.0f}%".format(100 * min(start + args.block, total) / total), end="\r", flush=True)
        ts = dsets["ts"]
        print("\nkept {} of {} events ({:.1f}%)".format(num, total, 100 * num / total))
        t0, tk = float(ts[0]), float(ts[num - 1])
        dst.attrs["t0"] = t0
        dst.attrs["tk"] = tk
        dst.attrs["duration"] = tk - t0
        dst.attrs["num_events"] = num
        dst.attrs["num_pos"] = num_pos
        dst.attrs["num_neg"] = num - num_pos
        dst.attrs["sensor_resolution"] = [crop_h, crop_w]
        dst.attrs["crop_offset_yx"] = [oy, ox]
        dst.attrs["source"] = os.path.basename(args.src) + " (davis/left/events)"

        if args.flow:
            gt = np.load(args.flow)
            stamps, fx, fy = gt["timestamps"], gt["x_flow_dist"], gt["y_flow_dist"]
            for i, t in enumerate(stamps):
                flow = np.stack([fx[i], fy[i]])[:, oy : oy + crop_h, ox : ox + crop_w]
                dset = dst.create_dataset("flow_dt1/{:09d}".format(i), data=flow.astype(np.float32), compression="gzip")
                dset.attrs["timestamp"] = float(t)
            print("flow_dt1: {} maps".format(len(stamps)))
        print("\n{:.1f} s of events, {:.1f}% positive".format(tk - t0, 100 * num_pos / num))
    os.replace(args.dst + ".tmp", args.dst)
    print("Written: " + args.dst)


if __name__ == "__main__":
    main()
