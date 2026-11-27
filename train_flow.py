import argparse
import copy
import os
import random
import time

import numpy as np

import mlflow
import torch
from torch.optim import *

from configs.parser import YAMLParser
from dataloader.h5 import H5Loader
from loss.flow import EventWarping
from models.model import (
    FireNet,
    RNNFireNet,
    LeakyFireNet,
    FireFlowNet,
    LeakyFireFlowNet,
    E2VID,
    EVFlowNet,
    RecEVFlowNet,
    LeakyRecEVFlowNet,
    RNNRecEVFlowNet,
)
from models.model import (
    LIFFireNet,
    PLIFFireNet,
    ALIFFireNet,
    XLIFFireNet,
    LIFFireFlowNet,
    SpikingRecEVFlowNet,
    PLIFRecEVFlowNet,
    ALIFRecEVFlowNet,
    XLIFRecEVFlowNet,
)
from utils.gradients import get_grads
from utils.utils import load_model, resolve_tracking_uri, save_csv, save_diff, save_model
from utils.visualization import Visualization


def build_validation_config(config):
    """
    Config of the validation pass, or None. The validation data is read with the training pipeline
    (same windows, same loss) but without augmentation; it is only monitored and never used to pick
    the saved weights, which stay those of the lowest training loss (as in the paper's code).
    """

    val = config.get("validation", None)
    if not val or not val.get("enabled", False):
        return None
    config_val = copy.deepcopy(config)
    path = os.path.expandvars(val["path"])
    if "$" in path:
        raise ValueError("Environment variable not set in validation path: " + path)
    config_val["data"]["path"] = path
    config_val["loader"]["resolution"] = list(val.get("resolution", config["loader"]["resolution"]))
    config_val["loader"]["batch_size"] = int(val.get("batch_size", config["loader"]["batch_size"]))
    config_val["loader"]["augment"] = []
    config_val["loader"]["augment_prob"] = []
    config_val["vis"]["enabled"] = False
    config_val["vis"]["verbose"] = False
    return config_val


def validate(model, config_val, device):
    """
    Average CMax loss per sample over one pass on the validation sequences, computed exactly as the
    training loss (window, loss window, regularization) but without gradients. The random number
    generators and the recurrent states of the model are restored afterwards, so the validation
    pass does not change the training run.
    """

    rng_state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        states = model.states
    except AttributeError:
        states = None
    model.eval()

    data = H5Loader(config_val, config_val["model"]["num_bins"], config_val["model"]["round_encoding"])
    dataloader = torch.utils.data.DataLoader(
        data, drop_last=True, batch_size=config_val["loader"]["batch_size"], collate_fn=data.custom_collate
    )
    loss_function = EventWarping(config_val, device)
    model.reset_states()

    total, samples, done = 0.0, 0, False
    with torch.no_grad():
        while not done:
            for inputs in dataloader:
                if data.new_seq:
                    data.new_seq = False
                    loss_function.reset()
                    model.reset_states()
                if data.seq_num >= len(data.files):
                    done = True
                    break
                x = model(inputs["event_voxel"].to(device), inputs["event_cnt"].to(device))
                loss_function.event_flow_association(
                    x["flow"],
                    inputs["event_list"].to(device),
                    inputs["event_list_pol_mask"].to(device),
                    inputs["event_mask"].to(device),
                )
                if loss_function.num_events >= config_val["data"]["window_loss"]:
                    if config_val["loss"]["overwrite_intermediate"]:
                        loss_function.overwrite_intermediate_flow(x["flow"])
                    total += loss_function().item()
                    samples += config_val["loader"]["batch_size"]
                    loss_function.reset()
    for f in data.open_files:
        f.close()

    model.train()
    model.reset_states()
    if states is not None:
        model.states = states
    random.setstate(rng_state[0])
    np.random.set_state(rng_state[1])
    torch.set_rng_state(rng_state[2])
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    return total / max(samples, 1)


def checkpoint_epoch(epoch, config):
    """
    Whether to keep the weights after this epoch (config "checkpoints": every epoch below `dense`, then
    every `every` epochs, and the last epoch). Off without the config block.
    """

    ckpt = config.get("checkpoints", None)
    if not ckpt:
        return False
    dense, every = int(ckpt.get("dense", 0)), int(ckpt.get("every", 0))
    last = epoch == config["loader"]["n_epochs"] - 1
    return epoch < dense or (every > 0 and (epoch + 1) % every == 0) or last


def update_ts_stats(stats, inputs, config, num_passes=500):
    """
    Accumulates, per time constant, the mean |value| and the share of pixels above 0.05 and 0.95 over
    the pixels that have had at least one event, plus the input window duration; prints them once.
    """

    surface = inputs["event_voxel"].abs()
    seen = (surface.sum(dim=1, keepdim=True) > 0).float()
    n = seen.sum().clamp(min=1)
    for key, value in [
        ("mean", (surface * seen).sum(dim=(0, 2, 3)) / n),
        ("above_0.05", ((surface > 0.05).float() * seen).sum(dim=(0, 2, 3)) / n),
        ("above_0.95", ((surface > 0.95).float() * seen).sum(dim=(0, 2, 3)) / n),
    ]:
        stats[key] = stats.get(key, 0) + value
    stats["dt_input"] = stats.get("dt_input", 0) + inputs["dt_input"].float().mean().item()
    stats["passes"] += 1
    if stats["passes"] == num_passes:
        print("time surface statistics over {} passes (pixels with events so far):".format(num_passes))
        print("  input window: {:.2f} ms on average".format(1000 * stats["dt_input"] / num_passes))
        taus = config["model"]["ts_taus"]
        split = config["model"].get("ts_split_polarity", False)
        labels = (["pos"] * len(taus) + ["neg"] * len(taus)) if split else [""] * len(taus)
        for i, (tau, label) in enumerate(zip(list(taus) * (2 if split else 1), labels)):
            print(
                "  tau {:7.1f} ms {:3s} | mean |value| {:.3f} | >0.05 {:.3f} | >0.95 {:.3f}".format(
                    1000 * tau,
                    label,
                    stats["mean"][i].item() / num_passes,
                    stats["above_0.05"][i].item() / num_passes,
                    stats["above_0.95"][i].item() / num_passes,
                )
            )
        return None
    return stats


def train(args, config_parser):
    mlflow.set_tracking_uri(resolve_tracking_uri(args.path_mlflow))

    # configs
    config = config_parser.config
    if config["data"]["mode"] == "frames":
        print("Config error: Training pipeline not compatible with frames mode.")
        raise AttributeError

    # log config
    mlflow.set_experiment(config["experiment"])
    mlflow.start_run(run_name=args.run_name if args.run_name else None)
    mlflow.log_params(config)
    mlflow.log_param("prev_runid", args.prev_runid)
    config = config_parser.combine_entries(config)
    print("MLflow dir:", mlflow.active_run().info.artifact_uri[:-9])
    print("MLflow run ID:", mlflow.active_run().info.run_id, flush=True)

    # log git diff
    save_diff("train_diff.txt")

    # MLflow draws from Python's global random generator (run names, experiment ids) a number of
    # times that depends on the state of the tracking folder. Seed again here so that the sequence
    # order, the augmentation and the weight initialization depend on loader.seed only.
    config_parser.init_seeds()

    # initialize settings
    device = config_parser.device
    kwargs = config_parser.loader_kwargs

    # visualization tool
    if config["vis"]["enabled"]:
        vis = Visualization(config)

    # data loader
    data = H5Loader(config, config["model"]["num_bins"], config["model"]["round_encoding"])
    dataloader = torch.utils.data.DataLoader(
        data,
        drop_last=True,
        batch_size=config["loader"]["batch_size"],
        collate_fn=data.custom_collate,
        worker_init_fn=config_parser.worker_init_fn,
        **kwargs,
    )

    # loss function
    loss_function = EventWarping(config, device)

    # optional validation pass after every epoch (monitoring only)
    config_val = build_validation_config(config)
    if config_val is not None:
        print("validation data:", config_val["data"]["path"], flush=True)

    # model initialization and settings
    model = eval(config["model"]["name"])(config["model"].copy()).to(device)
    model = load_model(args.prev_runid, model, device, strict=args.prev_runid != "")
    mlflow.log_param("n_params", sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.train()

    # optimizers
    optimizer = eval(config["optimizer"]["name"])(model.parameters(), lr=config["optimizer"]["lr"])
    optimizer.zero_grad()

    # simulation variables
    train_loss = 0
    best_loss = 1.0e6
    best_val_loss = 1.0e6
    end_train = False
    grads_w = []
    t_epoch = time.time()

    ts_stats = {"passes": 0} if config["model"].get("encoding", None) == "ts" else None

    # training loop
    data.shuffle()
    while True:
        for inputs in dataloader:

            if data.new_seq:
                data.new_seq = False

                loss_function.reset()
                model.reset_states()
                optimizer.zero_grad()

            if data.seq_num >= len(data.files):
                epoch_time = time.time() - t_epoch
                mlflow.log_metric("loss", train_loss / (data.samples + 1), step=data.epoch)
                mlflow.log_metric("epoch_time_s", epoch_time, step=data.epoch)
                val_text = ""
                if config_val is not None:
                    t_val = time.time()
                    val_loss = validate(model, config_val, device)
                    mlflow.log_metric("val_loss", val_loss, step=data.epoch)
                    val_text = " | val_loss {:.6f} ({:.0f} s)".format(val_loss, time.time() - t_val)
                    if config.get("checkpoints", None) and val_loss < best_val_loss:
                        # weights of the lowest validation loss (diagnostic: the validation data are
                        # MVSEC chunks of the test sequences, so not for reporting)
                        with torch.no_grad():
                            save_model(model, "val_best")
                        best_val_loss = val_loss
                        val_text += " (val best)"
                print(
                    "Epoch {:04d} done | loss {:.6f} | {:.0f} s{}".format(
                        data.epoch, train_loss / (data.samples + 1), epoch_time, val_text
                    ),
                    flush=True,
                )
                t_epoch = time.time()

                with torch.no_grad():
                    if train_loss / (data.samples + 1) < best_loss:
                        save_model(model)
                        best_loss = train_loss / (data.samples + 1)
                    if checkpoint_epoch(data.epoch, config):
                        save_model(model, "epoch_{:04d}".format(data.epoch))

                data.epoch += 1
                data.samples = 0
                train_loss = 0
                data.seq_num = data.seq_num % len(data.files)

                # save grads to file
                if config["vis"]["store_grads"]:
                    save_csv(grads_w, "grads_w.csv")
                    grads_w = []

                # finish training loop
                if data.epoch == config["loader"]["n_epochs"]:
                    end_train = True

            # input statistics of the time surface channels over the first passes (sanity check of the time
            # constants: a channel that is ~0 or ~1 everywhere carries no information)
            if ts_stats is not None:
                ts_stats = update_ts_stats(ts_stats, inputs, config)

            # forward pass
            x = model(inputs["event_voxel"].to(device), inputs["event_cnt"].to(device))

            # event flow association
            loss_function.event_flow_association(
                x["flow"],
                inputs["event_list"].to(device),
                inputs["event_list_pol_mask"].to(device),
                inputs["event_mask"].to(device),
            )

            # backward pass
            if loss_function.num_events >= config["data"]["window_loss"]:

                # overwrite intermediate flow estimates with the final ones
                if config["loss"]["overwrite_intermediate"]:
                    loss_function.overwrite_intermediate_flow(x["flow"])

                # loss
                loss = loss_function()
                train_loss += loss.item()

                # update number of loss samples seen by the network
                data.samples += config["loader"]["batch_size"]

                loss.backward()

                # clip and save grads
                if config["loss"]["clip_grad"] is not None:
                    torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), config["loss"]["clip_grad"])
                if config["vis"]["store_grads"]:
                    grads_w.append(get_grads(model.named_parameters()))

                optimizer.step()
                optimizer.zero_grad()

                # mask flow for visualization
                flow_vis = x["flow"][-1].clone()
                if model.mask and config["vis"]["enabled"] and config["loader"]["batch_size"] == 1:
                    flow_vis *= loss_function.event_mask

                model.detach_states()
                loss_function.reset()

                # visualize
                with torch.no_grad():
                    if config["vis"]["enabled"] and config["loader"]["batch_size"] == 1:
                        vis.update(inputs, flow_vis, None)

            # print training info
            if config["vis"]["verbose"]:
                print(
                    "Train Epoch: {:04d} [{:03d}/{:03d} ({:03d}%)] Loss: {:.6f}".format(
                        data.epoch,
                        data.seq_num,
                        len(data.files),
                        int(100 * data.seq_num / len(data.files)),
                        train_loss / (data.samples + 1),
                    ),
                    end="\r",
                )

        if end_train:
            break

    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/train_flow.yml",
        help="training configuration",
    )
    parser.add_argument(
        "--path_mlflow",
        default="",
        help="location of the mlflow ui",
    )
    parser.add_argument(
        "--prev_runid",
        default="",
        help="pre-trained model to use as starting point",
    )
    parser.add_argument(
        "--run_name",
        default="",
        help="human-readable name of the mlflow run",
    )
    args = parser.parse_args()

    # launch training
    train(args, YAMLParser(args.config))
