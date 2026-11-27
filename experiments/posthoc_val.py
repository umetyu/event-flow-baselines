"""
Validation loss of the stored weights of a run after training (step 5b): for every
artifacts/checkpoints/epoch_XXXX.pth, the CMax loss on a validation set is computed exactly as during
training (train_flow.validate), logged to the run as metric val_loss_<name> (step = epoch), and the weights
of the lowest loss are copied to artifacts/checkpoints/val_<name>_best.pth.

    python experiments/posthoc_val.py --run s5b_evflownet-ts5pol_w1000_seed0 \
        --name od2 --path $EVFLOW_DATA/validation_od2/

Epochs already logged are skipped, so the script can be run again after an interruption.
"""

import argparse
import glob
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch  # noqa: E402

import train_flow  # noqa: E402
from configs.parser import YAMLParser  # noqa: E402
from plot_loss import find_run, read_metric  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="run name or id")
    parser.add_argument("--config", default=None, help="training config (default: configs/exp/<run name>.yml)")
    parser.add_argument("--name", default="od2", help="suffix of the metric and checkpoint names")
    parser.add_argument("--path", required=True, help="folder with the validation .h5 chunks")
    parser.add_argument("--resolution", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--mlruns", default=os.environ.get("EVFLOW_MLRUNS", "mlruns"))
    args = parser.parse_args()

    run_dir = find_run(args.mlruns, args.run)
    name_file = os.path.join(run_dir, "tags", "mlflow.runName")
    run_name = open(name_file).read().strip() if os.path.exists(name_file) else args.run
    config_parser = YAMLParser(args.config or os.path.join(ROOT, "configs", "exp", run_name + ".yml"))
    config, device = config_parser.config, config_parser.device
    val = {"enabled": True, "path": args.path, "resolution": args.resolution, "batch_size": args.batch_size}
    config_val = train_flow.build_validation_config(config, val)

    metric = "val_loss_" + args.name
    metric_file = os.path.join(run_dir, "metrics", metric)
    done = dict(read_metric(run_dir, metric)) if os.path.exists(metric_file) else {}
    ckpt_dir = os.path.join(run_dir, "artifacts", "checkpoints")
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pth")))
    if not ckpts:
        raise SystemExit("no epoch checkpoints in " + ckpt_dir)
    print("{}: {} checkpoints, validation data {}".format(run_name, len(ckpts), config_val["data"]["path"]), flush=True)

    model = eval(config["model"]["name"], vars(train_flow))(config["model"].copy()).to(device)
    for path in ckpts:
        epoch = int(os.path.basename(path)[len("epoch_"):-4])
        if epoch in done:
            continue
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.train()  # as during training: validate() switches to eval mode and back
        t = time.time()
        loss = train_flow.validate(model, config_val, device)
        with open(metric_file, "a") as fid:
            fid.write("{} {} {}\n".format(int(time.time() * 1000), loss, epoch))
        done[epoch] = loss
        print("epoch {:04d} | {} {:.6f} ({:.0f} s)".format(epoch, metric, loss, time.time() - t), flush=True)

    valid = {e: v for e, v in done.items() if v == v}  # drop nan
    best = min(valid, key=valid.get)
    target = os.path.join(ckpt_dir, "val_{}_best.pth".format(args.name))
    shutil.copyfile(os.path.join(ckpt_dir, "epoch_{:04d}.pth".format(best)), target + ".tmp")
    os.replace(target + ".tmp", target)
    print("lowest {} at epoch {} ({:.6f}) -> {}".format(metric, best, valid[best], target))


if __name__ == "__main__":
    main()
