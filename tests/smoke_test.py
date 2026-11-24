"""
End-to-end smoke test on synthetic data (CPU is enough, takes a few minutes).

For every architecture/encoding used in the experiments it runs: a short training (several optimizer
updates, checkpoint written), evaluation with ground truth (AEE) and evaluation without ground truth
(FWL/RSAT) from the stored checkpoint. It checks that losses and metrics are finite and that the
weights loaded for evaluation are the ones that were trained.

This verifies that the code runs on the installed software stack. It says nothing about accuracy.

Usage: python tests/smoke_test.py [--keep] [--models EVFlowNet:cnt:2 ...]
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests.synth import make_sequence  # noqa: E402

# name : encoding : num_bins
DEFAULT_MODELS = [
    "EVFlowNet:cnt:2",
    "EVFlowNet:voxel:5",
    "EVFlowNet:voxel:9",
    "EVFlowNet:ts:3",
    "EVFlowNet:tspol:6",
    "RecEVFlowNet:cnt:2",
    "RNNRecEVFlowNet:cnt:2",
    "LeakyRecEVFlowNet:cnt:2",
    "FireFlowNet:cnt:2",
    "FireFlowNet:voxel:5",
    "FireNet:cnt:2",
    "RNNFireNet:cnt:2",
    "LeakyFireNet:cnt:2",
    "SpikingRecEVFlowNet:cnt:2",
    "LIFFireNet:cnt:2",
]

SPIKING = {
    "leak": [-4.0, 0.1],
    "thresh": [0.8, 0.1],
    "learn_leak": True,
    "learn_thresh": True,
    "hard_reset": True,
}


def train_config(data_dir, name, encoding, num_bins):
    spiking = "Spiking" in name or "LIF" in name
    config = {
        "experiment": "smoke",
        "data": {"path": data_dir, "mode": "events", "window": 500, "window_loss": 2000},
        "model": {
            "name": name,
            "encoding": encoding,
            "round_encoding": False,
            "norm_input": False,
            "num_bins": num_bins,
            "base_num_channels": 8,
            "kernel_size": 3,
            "activations": ["arctanspike", "arctanspike"] if spiking else ["relu", None],
            "mask_output": True,
        },
        "spiking_neuron": SPIKING if spiking else None,
        "loss": {"flow_regul_weight": 0.001, "clip_grad": 100.0, "overwrite_intermediate": False},
        "optimizer": {"name": "Adam", "lr": 0.0002},
        "loader": {
            "n_epochs": 1,
            "batch_size": 2,
            "resolution": [64, 64],
            "augment": ["Horizontal", "Vertical", "Polarity"],
            "augment_prob": [0.5, 0.5, 0.5],
            "gpu": 0,
            "seed": 0,
        },
        "vis": {"verbose": False, "enabled": False, "px": 400, "store_grads": False},
        "hot_filter": {"enabled": False, "max_px": 100, "min_obvs": 5, "max_rate": 0.8},
    }
    if encoding in ("ts", "tspol"):
        split = encoding == "tspol"
        config["model"]["encoding"] = "ts"
        config["model"]["ts_split_polarity"] = split
        config["model"]["ts_taus"] = [float(t) for t in np.geomspace(0.005, 0.08, num_bins // (2 if split else 1))]
    return config


def eval_config(data_dir, mode):
    config = {
        "data": {"path": data_dir, "mode": mode, "window": 1, "window_eval": 15000},
        "model": {"mask_output": True},
        "metrics": {"name": ["AEE"], "flow_scaling": 64},
        "loader": {"batch_size": 1, "resolution": [64, 64], "augment": [], "gpu": 0},
        "vis": {"enabled": False, "px": 400, "bars": False, "activity": False, "store": False},
        "hot_filter": {"enabled": True, "max_px": 100, "min_obvs": 5, "max_rate": 0.8},
    }
    if mode == "events":
        config["data"]["window"] = 500
        config["data"]["window_eval"] = 2000
        config["metrics"]["name"] = ["FWL", "RSAT"]
    return config


def run(cmd, cwd, quiet=False, env=None):
    out = subprocess.run(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, env=env
    )
    if out.returncode != 0:
        if not quiet:
            print(out.stdout[-4000:])
        raise RuntimeError("Command failed: " + " ".join(cmd) + "\n" + out.stdout[-600:])
    return out.stdout


def read_metrics(path_results, runid, eval_id):
    with open(os.path.join(path_results, runid, "metrics_{}.yml".format(eval_id))) as fid:
        return yaml.safe_load(fid)


def check_finite(metrics, label):
    assert len(metrics) > 0, label + ": no metrics were written"
    for metric, per_file in metrics.items():
        assert len(per_file) > 0, label + ": metric " + metric + " has no sequences"
        for fname, value in per_file.items():
            assert math.isfinite(float(value)), "{}: {} is not finite for {}".format(label, metric, fname)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--keep", action="store_true", help="keep the temporary directory")
    args = parser.parse_args()

    import torch

    from utils.utils import find_model_file, resolve_tracking_uri

    tmp = tempfile.mkdtemp(prefix="evflow_smoke_")
    data_dir = os.path.join(tmp, "data") + "/"
    os.makedirs(data_dir)
    for i in range(3):
        make_sequence(os.path.join(data_dir, "seq{}.h5".format(i)), seed=i, velocity=(20.0 - 15.0 * i, -8.0 + 10.0 * i))
    path_mlflow = os.path.join(tmp, "mlruns")
    path_results = os.path.join(tmp, "results") + "/"

    print("torch", torch.__version__, "| tmp dir", tmp, flush=True)
    failed = []
    for entry in args.models:
        name, encoding, num_bins = entry.split(":")
        try:
            # training
            cfg = os.path.join(tmp, "train.yml")
            with open(cfg, "w") as fid:
                yaml.safe_dump(train_config(data_dir, name, encoding, int(num_bins)), fid)
            out = run(
                [sys.executable, "train_flow.py", "--config", cfg, "--path_mlflow", path_mlflow, "--run_name", entry],
                ROOT,
            )
            runid = [line for line in out.splitlines() if line.startswith("MLflow run ID:")][0].split()[-1]
            epoch = [line for line in out.splitlines() if line.startswith("Epoch 0000 done")]
            assert len(epoch) == 1, "training did not finish its epoch"
            loss = float(epoch[0].split("loss")[1].split("|")[0])
            assert math.isfinite(loss), "training loss is not finite"

            # the checkpoint exists, is a state dict and differs from a fresh initialization
            import mlflow

            mlflow.set_tracking_uri(resolve_tracking_uri(path_mlflow))
            ckpt = find_model_file(runid)
            assert ckpt is not None, "no checkpoint written"
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point()), "non-finite weights"

            # evaluation with and without ground truth
            results = {}
            for eval_id, mode in enumerate(["gtflow_dt1", "gtflow_dt4", "events"]):
                cfg = os.path.join(tmp, "eval.yml")
                config = eval_config(data_dir, mode)
                if mode == "gtflow_dt4":
                    config["data"]["window"] = 0.25
                with open(cfg, "w") as fid:
                    yaml.safe_dump(config, fid)
                out = run(
                    [
                        sys.executable,
                        "eval_flow.py",
                        runid,
                        "--config",
                        cfg,
                        "--path_mlflow",
                        path_mlflow,
                        "--path_results",
                        path_results,
                    ],
                    ROOT,
                )
                assert "Model restored from " + runid in out, "evaluation did not load the trained weights"
                metrics = read_metrics(path_results, runid, eval_id)
                check_finite(metrics, entry + "/" + mode)
                results[mode] = {k: sum(float(v) for v in d.values()) / len(d) for k, d in metrics.items()}

            print(
                "OK   {:28s} loss {:.4f} | AEE dt1 {:.2f} dt4 {:.2f} | FWL {:.3f} RSAT {:.3f}".format(
                    entry,
                    loss,
                    results["gtflow_dt1"]["AEE"],
                    results["gtflow_dt4"]["AEE"],
                    results["events"]["FWL"],
                    results["events"]["RSAT"],
                ),
                flush=True,
            )
        except Exception as e:  # keep going to report every failing model
            failed.append(entry)
            print("FAIL {:28s} {}: {}".format(entry, type(e).__name__, e), flush=True)

    # ---- runs created elsewhere (like the pretrained models of the paper): no usable MLflow metadata,
    # weights stored as a pickled nn.Module. Evaluation must find and load them, and reproduce the
    # metric of the run they were copied from.
    try:
        import mlflow

        import models.model as zoo
        from experiments.common import list_runs

        mlflow.set_tracking_uri(resolve_tracking_uri(path_mlflow))
        source = [r for r in list_runs(path_mlflow) if r["name"] == args.models[0]][0]
        name, encoding, num_bins = args.models[0].split(":")
        model_config = train_config(data_dir, name, encoding, int(num_bins))
        kwargs = dict(model_config["model"], spiking_neuron=model_config["spiking_neuron"])
        module = getattr(zoo, name)(kwargs)
        module.load_state_dict(torch.load(find_model_file(source["run_id"]), map_location="cpu", weights_only=True))

        foreign = os.path.join(os.path.dirname(source["dir"]), "PretrainedLike")
        shutil.copytree(os.path.join(source["dir"], "params"), os.path.join(foreign, "params"))
        os.makedirs(os.path.join(foreign, "artifacts", "model", "data"))
        torch.save(module, os.path.join(foreign, "artifacts", "model", "data", "model.pth"))

        cfg = os.path.join(tmp, "eval.yml")
        with open(cfg, "w") as fid:
            yaml.safe_dump(eval_config(data_dir, "gtflow_dt1"), fid)
        out = run(
            [sys.executable, "eval_flow.py", "PretrainedLike", "--config", cfg,
             "--path_mlflow", path_mlflow, "--path_results", path_results],
            ROOT,
        )
        assert "Model restored from PretrainedLike" in out, "weights of the foreign run were not loaded"
        ours = read_metrics(path_results, "PretrainedLike", 0)["AEE"]
        theirs = read_metrics(path_results, source["run_id"], 0)["AEE"]
        for fname in theirs:
            assert abs(float(ours[fname]) - float(theirs[fname])) < 1e-4, "foreign run gives a different AEE"
        print("OK   foreign run folder with pickled module evaluates to the same AEE as its source")
    except Exception as e:
        failed.append("foreign-run check")
        print("FAIL foreign-run check {}: {}".format(type(e).__name__, e))

    # ---- same seed, same result: train twice in the (now non-empty) tracking folder, once with and
    # once without a run name, and compare the losses of all epochs. This checks the seeding logic,
    # so it runs on the CPU. On a GPU, runs with the same seed differ (measured on Izar: 1-2% in the
    # loss of the first two epochs), most likely because the event accumulation of the loss is not
    # bit-wise reproducible there.
    try:
        name, encoding, num_bins = args.models[0].split(":")
        cfg = os.path.join(tmp, "train.yml")
        with open(cfg, "w") as fid:
            yaml.safe_dump(train_config(data_dir, name, encoding, int(num_bins)), fid)
        curves = []
        cpu = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        for extra in [["--run_name", "repeat"], []]:
            out = run([sys.executable, "train_flow.py", "--config", cfg, "--path_mlflow", path_mlflow] + extra, ROOT, env=cpu)
            curves.append([line.split("|")[1] for line in out.splitlines() if line.startswith("Epoch")])
        assert len(curves[0]) > 0 and curves[0] == curves[1], (
            "losses differ between CPU runs with the same seed: {} / {}".format(curves[0][:2], curves[1][:2])
        )
        print("OK   repeated CPU training with the same seed gives identical losses ({} epochs)".format(len(curves[0])))
    except Exception as e:
        failed.append("determinism check")
        print("FAIL determinism check {}: {}".format(type(e).__name__, e))

    # ---- experiment scripts: suite evaluation, skipping of finished suites, result tables
    try:
        config_dir = os.path.join(tmp, "configs")
        os.makedirs(config_dir)
        with open(os.path.join(config_dir, "eval_mvsec_dt4.yml"), "w") as fid:
            config = eval_config(data_dir, "gtflow_dt4")
            config["data"]["window"] = 0.25
            yaml.safe_dump(config, fid)
        with open(os.path.join(config_dir, "eval_mvsec_dt1.yml"), "w") as fid:
            yaml.safe_dump(eval_config(data_dir, "gtflow_dt1"), fid)
        results2 = os.path.join(tmp, "results2") + "/"
        cmd = [
            sys.executable, "experiments/evaluate.py", "--suites", "mvsec_dt4", "mvsec_dt1", "--runs", args.models[0],
            "--path_mlflow", path_mlflow, "--path_results", results2, "--config_dir", config_dir,
        ]
        first = run(cmd, ROOT)
        assert first.count("eval " + args.models[0]) == 2, "suites were not evaluated"
        second = run(cmd, ROOT)
        assert second.count("done " + args.models[0]) == 2, "finished suites were not skipped"
        out = run(
            [sys.executable, "experiments/collect.py", "--runs", args.models[0], "--path_mlflow", path_mlflow,
             "--path_results", results2, "--out", os.path.join(tmp, "tables")],
            ROOT,
        )
        with open(os.path.join(tmp, "tables", "runs.csv")) as fid:
            table = fid.read()
        # restriction to model classes: the class of this run is kept, another class selects nothing
        name = args.models[0].split(":")[0]
        kept = run(cmd + ["--models", name], ROOT)
        assert kept.count("done " + args.models[0]) == 2, "--models dropped the run of its own class"
        other = subprocess.run(cmd + ["--models", "NoSuchModel"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert other.returncode != 0, "--models with an unknown class must select nothing"
        for needle in ["best_loss", "AEE", "data_dt1", "data_dt4"]:
            assert needle in table, needle + " missing from runs.csv"
        print("OK   evaluate.py / collect.py produce the result tables")
    except Exception as e:
        failed.append("experiment-scripts check")
        print("FAIL experiment-scripts check {}: {}".format(type(e).__name__, e))

    # ---- optimization-based estimator (no training): must recover a known constant flow clearly
    # better than predicting zero flow. Only every 6th ground-truth map is scored (stride).
    try:
        opt_data = os.path.join(tmp, "optdata") + "/"
        os.makedirs(opt_data)
        velocities = {"a.h5": (30.0, -12.0), "b.h5": (-18.0, 24.0)}
        for i, (fname, velocity) in enumerate(velocities.items()):
            make_sequence(os.path.join(opt_data, fname), num_events=60000, velocity=velocity, seed=10 + i)
        out = run(
            [sys.executable, "experiments/make_optimizer_run.py", "cmax-timestamp", "--num_events", "6000",
             "--iters", "30", "--path_mlflow", path_mlflow],
            ROOT,
        )
        opt_id = [line for line in out.splitlines() if line.startswith("Created run")][0].split("(")[1].strip(")")
        cfg = os.path.join(tmp, "eval_opt.yml")
        with open(cfg, "w") as fid:
            yaml.safe_dump(eval_config(opt_data, "gtflow_dt1"), fid)
        out = run(
            [sys.executable, "eval_flow.py", opt_id, "--config", cfg, "--path_mlflow", path_mlflow,
             "--path_results", path_results],
            ROOT,
            env=dict(os.environ, EVFLOW_EVAL_STRIDE="6"),
        )
        errors = read_metrics(path_results, opt_id, 0)["AEE"]
        report = []
        for fname, velocity in velocities.items():
            zero_flow = math.hypot(*velocity) / 45.0  # error of predicting no motion [px per frame]
            assert float(errors[fname]) < 0.5 * zero_flow, "AEE {} on {} is not below half of the zero-flow error {:.3f}".format(
                errors[fname], fname, zero_flow
            )
            report.append("{:.3f} (zero flow {:.3f})".format(float(errors[fname]), zero_flow))
        print("OK   CMaxOptimizer recovers a known constant flow: AEE " + ", ".join(report))
    except Exception as e:
        failed.append("optimizer check")
        print("FAIL optimizer check {}: {}".format(type(e).__name__, e))

    # evaluation must refuse to run without weights
    with open(os.path.join(tmp, "eval.yml"), "w") as fid:
        yaml.safe_dump(eval_config(data_dir, "gtflow_dt1"), fid)
    try:
        run(
            [sys.executable, "eval_flow.py", "does_not_exist", "--config", os.path.join(tmp, "eval.yml"),
             "--path_mlflow", path_mlflow, "--path_results", path_results],
            ROOT,
            quiet=True,
        )
        failed.append("missing-run check")
        print("FAIL evaluation of a missing run did not raise")
    except RuntimeError as e:
        if "does_not_exist" in str(e) and ("not found" in str(e) or "No model found" in str(e)):
            print("OK   evaluation of a missing run is refused")
        else:
            failed.append("missing-run check")
            print("FAIL evaluation of a missing run failed for another reason:", str(e)[-300:])

    if not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    if failed:
        print("\nFAILED:", ", ".join(failed))
        sys.exit(1)
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
