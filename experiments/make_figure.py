"""
Qualitative comparison: the flow estimated by several runs next to the ground truth.

    python experiments/make_figure.py --sequence indoor_flying1 --index 600 1500 \\
        --runs EVFlowNet s2_evflownet-voxel9_w7500_seed0 s2_evflownet-cnt_w1000_seed0 opt-cmax-timestamp_n30000 \\
        --labels "ConvGRU EV-FlowNet" "EV-FlowNet + voxel" "EV-FlowNet + count" "CMax optimization" \\
        --out results/flow_comparison.png

One row per requested ground-truth index, columns: events | ground truth | one column per run.
Every run processes the sequence from its start up to the last requested index, exactly as in the
evaluation (recurrent networks and optimization-based estimators need the history). The flow is
shown as the displacement over the ground-truth interval, only at pixels that are scored by the AEE
(at least one event and valid ground truth). Colour = direction, brightness = magnitude, with one
scale per row shared by all panels. The AEE printed above each panel is computed by the metric of
the evaluation for that single window.

Runs are given by run id or run name; --mode selects gtflow_dt1 (default) or gtflow_dt4.
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.colors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402

import eval_flow as zoo  # noqa: E402  (namespace with all model classes)
from configs.parser import YAMLParser  # noqa: E402
from dataloader.h5 import H5Loader  # noqa: E402
from experiments.common import ROOT, list_runs, read_metric  # noqa: E402
from loss.flow import AEE  # noqa: E402
from utils.utils import get_run_params, load_model, resolve_tracking_uri  # noqa: E402


def flow_to_rgb(flow, mask, max_mag):
    """
    :param flow: [2 x H x W] (x, y) displacement
    :param mask: [H x W] pixels to show
    :param max_mag: magnitude mapped to full brightness
    :return [H x W x 3] colour = direction, brightness = magnitude, black where masked out
    """

    mag = np.sqrt(flow[0] ** 2 + flow[1] ** 2)
    hsv = np.zeros(flow.shape[1:] + (3,))
    hsv[..., 0] = (np.arctan2(flow[1], flow[0]) + np.pi) / (2 * np.pi)
    hsv[..., 1] = 1.0
    hsv[..., 2] = np.clip(mag / (max_mag + 1e-9), 0.0, 1.0)
    rgb = matplotlib.colors.hsv_to_rgb(hsv)
    rgb[~mask] = 0.0
    return rgb


def events_to_rgb(event_cnt):
    """
    :param event_cnt: [2 x H x W] per-polarity event counts
    :return [H x W x 3] white background, red/blue for the two polarities
    """

    rgb = np.ones(event_cnt.shape[1:] + (3,))
    pos, neg = event_cnt[0] > 0, event_cnt[1] > 0
    rgb[pos] = (0.85, 0.15, 0.15)
    rgb[neg] = (0.15, 0.25, 0.85)
    rgb[pos & neg] = (0.45, 0.1, 0.55)
    return rgb


def colour_wheel(size=101):
    y, x = np.mgrid[-1 : 1 : size * 1j, -1 : 1 : size * 1j]
    radius = np.sqrt(x ** 2 + y ** 2)
    rgb = flow_to_rgb(np.stack([x, y]), radius <= 1.0, 1.0)
    return np.concatenate([rgb, (radius <= 1.0)[..., None].astype(float)], axis=2)


def run_one(run_id, sequence_dir, indices, args):
    """
    :return dictionary index -> {flow, gt, mask, events, aee}
    """

    parser = YAMLParser(args.config)
    config = parser.merge_configs(get_run_params(run_id))
    config["data"]["path"] = sequence_dir
    config["data"]["mode"] = args.mode
    config["data"]["window"] = 1  # all events between two ground-truth maps in one pass
    config["loader"]["batch_size"] = 1
    config["vis"]["bars"] = False
    device = parser.device

    model = getattr(zoo, config["model"]["name"])(config["model"]).to(device)
    model = load_model(run_id, model, device, strict=True)
    model.eval()
    needs_events = getattr(model, "needs_events", False)
    if needs_events:
        model.configure(config)

    criterion = AEE(config, device, flow_scaling=config["metrics"]["flow_scaling"])
    data = H5Loader(config, config["model"]["num_bins"], config["model"]["round_encoding"])
    loader = torch.utils.data.DataLoader(
        data, drop_last=True, batch_size=1, collate_fn=data.custom_collate, worker_init_fn=parser.worker_init_fn
    )

    captured, k = {}, 0
    with torch.no_grad():
        for inputs in loader:
            if data.new_seq:
                data.new_seq = False
                model.reset_states()
            if data.seq_num >= len(data.files) or k > max(indices):
                break
            if needs_events:
                x = model(inputs["event_voxel"].to(device), inputs["event_cnt"].to(device), events=inputs,
                          optimize=k in indices)
            else:
                x = model(inputs["event_voxel"].to(device), inputs["event_cnt"].to(device))
            if k in indices:
                criterion.event_flow_association(x["flow"], inputs)
                aee = float(criterion()[0][0])
                criterion.reset()
                scale = config["metrics"]["flow_scaling"] * float(inputs["dt_gt"]) / float(inputs["dt_input"])
                gt = inputs["gtflow"][0].float().cpu().numpy()
                mask = (inputs["event_mask"][0, 0].cpu().numpy() > 0) & ~((gt[0] == 0.0) & (gt[1] == 0.0))
                captured[k] = {
                    "flow": x["flow"][-1][0].float().cpu().numpy() * scale,
                    "gt": gt,
                    "mask": mask,
                    "events": inputs["event_cnt"][0].cpu().numpy(),
                    "aee": aee,
                }
            k += 1
    missing = [i for i in indices if i not in captured]
    if missing:
        raise RuntimeError("Indices {} are beyond the end of the sequence ({} windows)".format(missing, k))
    return captured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="run ids or run names (one column each)")
    parser.add_argument("--labels", nargs="*", default=[], help="column titles (default: run names)")
    parser.add_argument("--sequence", required=True, help="part of the file name, e.g. indoor_flying1")
    parser.add_argument("--index", nargs="+", type=int, required=True, help="ground-truth indices (one row each)")
    parser.add_argument("--mode", default="gtflow_dt1", choices=["gtflow_dt1", "gtflow_dt4"])
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "exp", "eval_mvsec_dt1.yml"))
    parser.add_argument("--data", default="", help="folder with the sequences (default: data.path of --config)")
    parser.add_argument("--path_mlflow", default=os.environ.get("EVFLOW_MLRUNS", os.path.join(ROOT, "mlruns")))
    parser.add_argument("--out", default=os.path.join(ROOT, "results", "flow_comparison.png"))
    args = parser.parse_args()
    assert not args.labels or len(args.labels) == len(args.runs), "Give one label per run"

    mlflow.set_tracking_uri(resolve_tracking_uri(args.path_mlflow))
    known = list_runs(args.path_mlflow)
    run_ids, unfinished = [], []
    for name in args.runs:
        match = [r for r in known if name in (r["run_id"], r["name"])]
        if len(match) != 1:
            raise SystemExit("Run '{}' matches {} runs in {}".format(name, len(match), args.path_mlflow))
        run_ids.append(match[0]["run_id"])
        # a run that is still training has a checkpoint of an early epoch: say so in the figure
        epochs = len(read_metric(match[0]["dir"], "loss"))
        loader = match[0]["params"].get("loader", {})
        planned = loader.get("n_epochs", 0) if isinstance(loader, dict) else 0
        unfinished.append(" (epoch {}/{})".format(epochs, planned) if 0 < epochs < planned else "")
    labels = args.labels if args.labels else args.runs
    labels = [label + note for label, note in zip(labels, unfinished)]

    # folder with only the requested sequence
    folder = args.data if args.data else YAMLParser(args.config).config["data"]["path"]
    files = []
    for root, _, names in os.walk(folder, followlinks=True):
        files += [os.path.join(root, n) for n in names if n.endswith(".h5") and args.sequence in n]
    if len(files) != 1:
        raise SystemExit("'{}' matches {} files in {}".format(args.sequence, len(files), folder))
    sequence_dir = tempfile.mkdtemp(prefix="evflow_fig_")
    os.symlink(os.path.abspath(files[0]), os.path.join(sequence_dir, os.path.basename(files[0])))

    results = []
    for run_id, label in zip(run_ids, labels):
        print("Running {} ...".format(label), flush=True)
        results.append(run_one(run_id, sequence_dir + "/", args.index, args))

    # ---- figure
    rows, cols = len(args.index), 2 + len(run_ids)
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.75 * rows + 0.5), squeeze=False)
    for r, index in enumerate(args.index):
        first = results[0][index]
        gt_mag = np.sqrt(first["gt"][0] ** 2 + first["gt"][1] ** 2)[first["mask"]]
        max_mag = float(np.percentile(gt_mag, 95)) if gt_mag.size else 1.0
        panels = [(events_to_rgb(first["events"]), "Events (window {})".format(index))]
        panels.append((flow_to_rgb(first["gt"], first["mask"], max_mag), "Ground truth"))
        for result, label in zip(results, labels):
            entry = result[index]
            panels.append((flow_to_rgb(entry["flow"], entry["mask"], max_mag), "{}\nAEE {:.2f} px".format(label, entry["aee"])))
        for c, (image, title) in enumerate(panels):
            axes[r, c].imshow(image, interpolation="nearest")
            axes[r, c].set_title(title, fontsize=9)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        axes[r, 1].set_ylabel("full brightness = {:.1f} px".format(max_mag), fontsize=7)

    fig.suptitle(
        "{} ({}): displacement over the ground-truth interval, shown where the AEE is computed".format(
            os.path.basename(files[0]).replace(".h5", ""), args.mode[-3:]
        ),
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 0.94, 0.96))
    wheel = fig.add_axes([0.945, 0.42, 0.05, 0.05 * fig.get_figwidth() / fig.get_figheight()])
    wheel.imshow(colour_wheel(), interpolation="bilinear")
    wheel.set_title("direction", fontsize=7)
    wheel.axis("off")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print("Figure written to " + args.out)
    for r, index in enumerate(args.index):
        print("  window {}: ".format(index) + ", ".join(
            "{} AEE {:.3f}".format(label, result[index]["aee"]) for result, label in zip(results, labels)))


if __name__ == "__main__":
    main()
