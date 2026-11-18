"""
Generates the training/evaluation configs of the baseline study into configs/exp/.

    python experiments/matrix.py            # (re)generate all configs
    python experiments/matrix.py --list     # only print the experiment table

The generated files are committed so that every run can be traced back to a file in the repository.
Edit this script (not the generated files) to change the study.

Naming: <step>_<model>-<input>_w<events per forward pass>_seed<seed>.yml

Step 2 (input representation x memory), training setup identical to Hagenaars et al. 2021:
    leaky / RNN / GRU, count input  -> past kept in the network state. These six are models of the
                                       paper (Table 1); its spiking variants are not retrained here.
    no memory, count input          -> NOT in the paper: lower bound, no motion information in the input
    no memory, voxel input          -> past encoded in the input by hand. evflownet-voxel9 is the
                                       counterpart of the literature row "EV-FlowNet [56]" of Table 1;
                                       evflownet-voxel5 / fireflownet-voxel5 are NOT in the paper
The paper reports one result per model and no seeds; the three seeds are an addition.
Step 3 (how much past in the input): the same networks with longer input windows. NOT in the paper.
Step 4 (AffineObjects): initial configs on the simulated dataset of EV-LayerSegNet. NOT yet run.
"""

import argparse
import copy
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "configs", "exp")

# Training setup of the paper (configs/train_ANN.yml): 1k events per forward pass, loss every 10k
# events (10 forward passes per backward pass), 128x128 crops, batch 8, 100 epochs, Adam 2e-4.
BASE = {
    "experiment": "Default",
    "data": {"path": "${EVFLOW_DATA}/training/", "mode": "events", "window": 1000, "window_loss": 10000},
    "model": {
        "name": "FireNet",
        "encoding": "cnt",
        "round_encoding": False,
        "norm_input": False,
        "num_bins": 2,
        "base_num_channels": 32,
        "kernel_size": 3,
        "activations": ["relu", None],
        "mask_output": True,
    },
    "spiking_neuron": None,
    "loss": {"flow_regul_weight": 0.001, "clip_grad": 100.0, "overwrite_intermediate": False},
    "optimizer": {"name": "Adam", "lr": 0.0002},
    "loader": {
        "n_epochs": 100,
        "batch_size": 8,
        "resolution": [128, 128],
        "augment": ["Horizontal", "Vertical", "Polarity"],
        "augment_prob": [0.5, 0.5, 0.5],
        "gpu": 0,
        "seed": 0,
    },
    "vis": {"verbose": False, "enabled": False, "px": 400, "store_grads": False},
    "hot_filter": {"enabled": False, "max_px": 100, "min_obvs": 5, "max_rate": 0.8},
}

# tag: (class in models/model.py, input encoding, input channels, learning rate, memory)
MODELS = {
    # EV-FlowNet family (encoder-decoder)
    "evflownet-cnt": ("EVFlowNet", "cnt", 2, 2e-4, "none"),
    "evflownet-voxel5": ("EVFlowNet", "voxel", 5, 2e-4, "none"),
    "evflownet-voxel9": ("EVFlowNet", "voxel", 9, 2e-4, "none"),  # input of Zhu et al. 2019, see ZHU2019
    "leaky-evflownet-cnt": ("LeakyRecEVFlowNet", "cnt", 2, 2e-4, "leaky"),
    "rnn-evflownet-cnt": ("RNNRecEVFlowNet", "cnt", 2, 5e-5, "rnn"),  # lr from the upstream config comment
    "gru-evflownet-cnt": ("RecEVFlowNet", "cnt", 2, 2e-4, "gru"),
    # FireNet family (lightweight, full resolution)
    "fireflownet-cnt": ("FireFlowNet", "cnt", 2, 2e-4, "none"),
    "fireflownet-voxel5": ("FireFlowNet", "voxel", 5, 2e-4, "none"),
    "leaky-firenet-cnt": ("LeakyFireNet", "cnt", 2, 2e-4, "leaky"),
    "rnn-firenet-cnt": ("RNNFireNet", "cnt", 2, 2e-4, "rnn"),
    "gru-firenet-cnt": ("FireNet", "cnt", 2, 2e-4, "gru"),
}

STEP2_SEEDS = [0, 1, 2]

# Counterpart of the literature row "EV-FlowNet [56]" (Zhu et al., CVPR 2019): non-recurrent network,
# voxel grid with B=9 bins, one forward pass per loss. Zhu et al. use N=30000 events on 256x256; the
# same number of events per pixel on the 128x128 training crops is 7500. Everything else (training
# data, loss, optimizer) is the pipeline of this repository, NOT the one of Zhu et al.
ZHU2019 = ("evflownet-voxel9", 7500)
STEP3_MODELS = ["evflownet-cnt", "evflownet-voxel5", "gru-evflownet-cnt"]
STEP3_WINDOWS = [2500, 5000, 10000]  # divisors of the 10k-event loss window; 1000 is covered by step 2

# AffineObjects is 640x480, i.e. ~19x more pixels than the 128x128 crops above. The windows are
# scaled accordingly, and the loss window equals the 200k events used to train EV-LayerSegNet.
STEP4_MODELS = {
    "evflownet-voxel5": (200000, 200000, 8),  # events per pass, events per loss, batch size
    "gru-evflownet-cnt": (20000, 200000, 2),
    "gru-firenet-cnt": (20000, 200000, 2),
}


def train_config(tag, window, seed, experiment):
    name, encoding, num_bins, lr, _ = MODELS[tag]
    config = copy.deepcopy(BASE)
    config["experiment"] = experiment
    config["data"]["window"] = window
    config["model"]["name"] = name
    config["model"]["encoding"] = encoding
    config["model"]["num_bins"] = num_bins
    config["optimizer"]["lr"] = lr
    config["loader"]["seed"] = seed
    return config


def eval_config(path, mode, window, window_eval, metrics, resolution, flow_scaling):
    return {
        "data": {"path": path, "mode": mode, "window": window, "window_eval": window_eval},
        "model": {"mask_output": True},
        "metrics": {"name": metrics, "flow_scaling": flow_scaling},
        "loader": {"batch_size": 1, "resolution": resolution, "augment": [], "gpu": 0},
        "vis": {"enabled": False, "px": 400, "bars": False, "activity": False, "store": False},
        "hot_filter": {"enabled": True, "max_px": 100, "min_obvs": 5, "max_rate": 0.8},
    }


def build():
    train, evals = {}, {}

    for tag in MODELS:
        if tag == ZHU2019[0]:
            continue
        for seed in STEP2_SEEDS:
            train["s2_{}_w1000_seed{}".format(tag, seed)] = train_config(tag, 1000, seed, "step2")

    tag, window = ZHU2019
    config = train_config(tag, window, 0, "step2")
    config["data"]["window_loss"] = window
    train["s2_{}_w{}_seed0".format(tag, window)] = config

    for tag in STEP3_MODELS:
        for window in STEP3_WINDOWS:
            train["s3_{}_w{}_seed0".format(tag, window)] = train_config(tag, window, 0, "step3")

    for tag, (window, window_loss, batch_size) in STEP4_MODELS.items():
        config = train_config(tag, window, 0, "step4")
        config["data"]["path"] = "${AFFINE_DATA}/train/h5/"
        config["data"]["window_loss"] = window_loss
        config["loader"]["resolution"] = [480, 640]
        config["loader"]["batch_size"] = batch_size
        train["s4_affine_{}_w{}_seed0".format(tag, window)] = config

    # MVSEC with ground truth, as in the paper (configs/eval_MVSEC.yml without the display).
    # The paper has no validation set: training uses UZH-FPV only and MVSEC is used for evaluation only.
    evals["eval_mvsec_dt1"] = eval_config("${EVFLOW_DATA}/MVSEC/", "gtflow_dt1", 1, 15000, ["AEE"], [256, 256], 128)
    evals["eval_mvsec_dt4"] = eval_config("${EVFLOW_DATA}/MVSEC/", "gtflow_dt4", 0.25, 15000, ["AEE"], [256, 256], 128)

    # datasets without ground truth, as in the paper (configs/eval_ECD.yml, configs/eval_HQF.yml)
    for name in ["ECD", "HQF"]:
        evals["eval_" + name.lower()] = eval_config(
            "${EVFLOW_DATA}/" + name + "/", "events", 15000, 15000, ["FWL", "RSAT"], [180, 240], 128
        )

    # AffineObjects test set: metrics over the 50k events used to test EV-LayerSegNet
    for window in [10000, 50000]:
        evals["eval_affine_w{}".format(window)] = eval_config(
            "${AFFINE_DATA}/test/recordings_h5/", "events", window, 50000, ["FWL", "RSAT"], [480, 640], 640
        )

    return train, evals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    train, evals = build()
    if not args.list:
        os.makedirs(OUT, exist_ok=True)
        for fname in os.listdir(OUT):
            if fname.endswith(".yml"):
                os.remove(os.path.join(OUT, fname))
        for name, config in {**train, **evals}.items():
            with open(os.path.join(OUT, name + ".yml"), "w") as fid:
                fid.write("# Generated by experiments/matrix.py - do not edit by hand.\n")
                yaml.safe_dump(config, fid, sort_keys=False)

    print("{:44s} {:20s} {:6s} {:>4s} {:>7s} {:>7s} {:>8s} {:>5s}".format(
        "config", "model", "input", "bins", "window", "loss", "lr", "batch"))
    for name, config in train.items():
        print("{:44s} {:20s} {:6s} {:>4d} {:>7d} {:>7d} {:>8.0e} {:>5d}".format(
            name,
            config["model"]["name"],
            config["model"]["encoding"],
            config["model"]["num_bins"],
            config["data"]["window"],
            config["data"]["window_loss"],
            config["optimizer"]["lr"],
            config["loader"]["batch_size"],
        ))
    print("\n{} training configs, {} evaluation configs{}".format(
        len(train), len(evals), "" if args.list else " written to configs/exp/"))


if __name__ == "__main__":
    main()
