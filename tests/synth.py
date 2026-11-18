"""
Synthetic event sequences in the HDF5 layout expected by dataloader/h5.py.

Only meant for smoke tests of the pipeline (no realism): a cloud of points translates with constant
velocity and every point emits events at random times, so that the ground-truth optical flow is the
same constant vector at every pixel.

Layout written here (inferred from dataloader/h5.py):
    events/{xs,ys,ts,ps}        xs/ys: pixel coordinates, ts: seconds, ps: {0, 1}
    attrs: t0, tk, duration
    flow_dt1/<idx>, flow_dt4/<idx>   [2 x H x W] (x, y) displacement, attrs: timestamp
"""

import h5py
import numpy as np


def make_sequence(path, resolution=(64, 64), num_events=12000, duration=1.0, velocity=(20.0, -8.0), seed=0, rate=45.0):
    """
    :param resolution: (H, W)
    :param velocity: (vx, vy) in px/s
    :param rate: frequency [Hz] of the ground-truth flow maps (MVSEC-like)
    """

    rng = np.random.default_rng(seed)
    height, width = resolution

    num_points = 200
    x0 = rng.uniform(0, width, num_points)
    y0 = rng.uniform(0, height, num_points)

    point = rng.integers(0, num_points, num_events)
    ts = np.sort(rng.uniform(0.0, duration, num_events))
    xs = (x0[point] + velocity[0] * ts) % width
    ys = (y0[point] + velocity[1] * ts) % height
    ps = rng.integers(0, 2, num_events)

    xs = np.clip(np.floor(xs), 0, width - 1).astype(np.int16)
    ys = np.clip(np.floor(ys), 0, height - 1).astype(np.int16)

    t0 = 10.0  # sequences do not start at zero in the real datasets
    ts = ts + t0

    with h5py.File(path, "w") as f:
        f.create_dataset("events/xs", data=xs)
        f.create_dataset("events/ys", data=ys)
        f.create_dataset("events/ts", data=ts.astype(np.float64))
        f.create_dataset("events/ps", data=ps.astype(bool))
        f.attrs["t0"] = t0
        f.attrs["tk"] = t0 + duration
        f.attrs["duration"] = duration

        frame_ts = t0 + np.arange(0.0, duration, 1.0 / rate)
        for name, step in [("flow_dt1", 1), ("flow_dt4", 4)]:
            for i, t in enumerate(frame_ts):
                flow = np.zeros((2, height, width), dtype=np.float32)
                flow[0] = velocity[0] * step / rate
                flow[1] = velocity[1] * step / rate
                dset = f.create_dataset(name + "/{:09d}".format(i), data=flow)
                dset.attrs["timestamp"] = t

    return path
