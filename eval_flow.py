import argparse
import os
import time

import mlflow
import numpy as np
import torch
from torch.optim import *

from configs.parser import YAMLParser
from dataloader.h5 import H5Loader
from loss.flow import FWL, RSAT, AEE
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
from models.optimization import CMaxOptimizer, ExternalCMax
from utils.iwe import compute_pol_iwe
from utils.utils import load_model, create_model_dir, get_run_params, resolve_tracking_uri
from utils.mlflow import log_config, log_results
from utils.visualization import Visualization, vis_activity


def test(args, config_parser):
    mlflow.set_tracking_uri(resolve_tracking_uri(args.path_mlflow))

    config = config_parser.merge_configs(get_run_params(args.runid))

    # configs
    if config["loader"]["batch_size"] > 1:
        config["vis"]["enabled"] = False
        config["vis"]["store"] = False
        config["vis"]["bars"] = False  # progress bars not yet compatible batch_size > 1

    # asserts
    if "AEE" in config["metrics"]["name"]:
        assert (
            config["data"]["mode"] == "gtflow_dt1" or config["data"]["mode"] == "gtflow_dt4"
        ), "AEE computation not possible without ground truth mode"

    if "AEE" in config["metrics"]["name"]:
        assert config["data"]["window"] <= 1, "AEE computation not compatible with window > 1"
        assert np.isclose(
            (1.0 / config["data"]["window"]) % 1.0, 0.0
        ), "AEE computation not compatible with windows whose inverse is not a round number"

    if config["data"]["mode"] == "frames":
        if config["data"]["window"] <= 1.0:
            assert np.isclose(
                (1.0 / config["data"]["window"]) % 1.0, 0.0
            ), "Frames mode not compatible with < 1 windows whose inverse is not a round number"
        else:
            assert np.isclose(
                config["data"]["window"] % 1.0, 0.0
            ), "Frames mode not compatible with > 1 fractional windows"

    # optionally score only every n-th ground-truth map (EVFLOW_EVAL_STRIDE). Networks still process
    # every window; optimization-based estimators only optimize the windows that are scored. The
    # stride is stored with the evaluation settings, so strided results are never mixed up with
    # full evaluations.
    stride = int(os.environ.get("EVFLOW_EVAL_STRIDE", "1"))
    assert stride >= 1
    if stride > 1:
        assert "metrics" in config.keys() and config["metrics"]["name"] == ["AEE"], "A stride needs AEE only"
        config["metrics"]["stride"] = stride

    # Diagnostic for inputs that do not show the window length (time surfaces): the network predicts the
    # displacement over its input window, but a time surface only shows how long ago each pixel fired, so
    # the network can only learn the displacement over the average training window. With
    # EVFLOW_FLOW_DT_REF = that average (seconds), the output is rescaled by dt_input / dt_ref, i.e.
    # read as a velocity times the actual window length. Stored with the evaluation settings and
    # reported as a separate suite.
    dt_ref = float(os.environ.get("EVFLOW_FLOW_DT_REF", "0") or 0)
    if dt_ref > 0:
        assert "metrics" in config.keys(), "EVFLOW_FLOW_DT_REF needs metrics"
        config["metrics"]["flow_dt_ref"] = dt_ref

    if not args.debug:
        # create directory for inference results
        path_results = create_model_dir(args.path_results, args.runid)

        # store validation settings
        eval_id = log_config(path_results, args.runid, config)
    else:
        path_results = None
        eval_id = -1

    # initialize settings
    device = config_parser.device
    kwargs = config_parser.loader_kwargs

    # visualization tool
    if config["vis"]["enabled"] or config["vis"]["store"]:
        vis = Visualization(config, eval_id=eval_id, path_results=path_results)

    # model initialization and settings
    model = eval(config["model"]["name"])(config["model"]).to(device)
    model = load_model(args.runid, model, device, strict=True)
    model.eval()

    # estimators without weights that optimize the flow of every window (models/optimization.py)
    needs_events = getattr(model, "needs_events", False)
    if needs_events:
        model.configure(config)
        t_start = time.time()

    # validation metric
    criteria = []
    if "metrics" in config.keys():
        for metric in config["metrics"]["name"]:
            criteria.append(eval(metric)(config, device, flow_scaling=config["metrics"]["flow_scaling"]))

    # data loader
    data = H5Loader(config, config["model"]["num_bins"])
    dataloader = torch.utils.data.DataLoader(
        data,
        drop_last=True,
        batch_size=config["loader"]["batch_size"],
        collate_fn=data.custom_collate,
        worker_init_fn=config_parser.worker_init_fn,
        **kwargs,
    )

    # inference loop
    idx_AEE = 0
    idx_gt = 0  # index of the ground-truth map of the current sequence (for the stride)
    val_results = {}
    end_test = False
    activity_log = None
    with torch.no_grad():
        while True:
            for inputs in dataloader:

                if data.new_seq:
                    data.new_seq = False
                    activity_log = None
                    model.reset_states()
                    idx_gt = 0

                # finish inference loop
                if data.seq_num >= len(data.files):
                    end_test = True
                    break

                # forward pass
                if needs_events:
                    # the optimization is only run for windows that are scored; the events of the other
                    # windows are still added to the buffer of the estimator. With AEE, a window is
                    # scored when enough events were collected since the last score (window_eval),
                    # the required number of (fractional) windows has passed, and the stride allows it:
                    # the same conditions as in the validation block below.
                    will_score = True
                    if "AEE" in config["metrics"]["name"]:
                        collected = criteria[config["metrics"]["name"].index("AEE")].num_events
                        will_score = (
                            collected + inputs["event_list"].shape[1] >= config["data"]["window_eval"]
                            and bool(inputs["dt_gt"] > 0.0)
                            and (idx_AEE + 1 == np.round(1.0 / config["data"]["window"]))
                            and idx_gt % stride == 0
                        )
                    done = model.windows
                    x = model(
                        inputs["event_voxel"].to(device),
                        inputs["event_cnt"].to(device),
                        events=inputs,
                        optimize=will_score,
                    )
                    if model.windows != done and (model.windows % 20 == 0 or model.windows == 1):
                        elapsed = time.time() - t_start
                        print(
                            "Optimized {} windows | {:.0f} s | {:.2f} s per window".format(
                                model.windows, elapsed, elapsed / model.windows
                            ),
                            flush=True,
                        )
                else:
                    x = model(
                        inputs["event_voxel"].to(device),
                        inputs["event_cnt"].to(device),
                        log=config["vis"]["activity"],
                    )
                    if dt_ref > 0:
                        scale = (inputs["dt_input"].float() / dt_ref).to(device).view(-1, 1, 1, 1)
                        x["flow"] = [flow * scale for flow in x["flow"]]

                # mask flow for visualization
                flow_vis = x["flow"][-1].clone()
                if model.mask:
                    flow_vis *= inputs["event_mask"].to(device)

                # image of warped events
                iwe = compute_pol_iwe(
                    x["flow"][-1],
                    inputs["event_list"].to(device),
                    config["loader"]["resolution"],
                    inputs["event_list_pol_mask"][:, :, 0:1].to(device),
                    inputs["event_list_pol_mask"][:, :, 1:2].to(device),
                    flow_scaling=config["metrics"]["flow_scaling"],
                    round_idx=True,
                )

                iwe_window_vis = None
                events_window_vis = None
                masked_window_flow_vis = None
                if "metrics" in config.keys():

                    # event flow association
                    for metric in criteria:
                        metric.event_flow_association(x["flow"], inputs)

                    # validation
                    for i, metric in enumerate(config["metrics"]["name"]):
                        if criteria[i].num_events >= config["data"]["window_eval"]:

                            # overwrite intermedia flow estimates with the final ones
                            if config["loss"]["overwrite_intermediate"]:
                                criteria[i].overwrite_intermediate_flow(x["flow"])
                            if metric == "AEE" and inputs["dt_gt"] <= 0.0:
                                continue
                            if metric == "AEE":
                                idx_AEE += 1
                                if idx_AEE != np.round(1.0 / config["data"]["window"]):
                                    continue
                                idx_gt += 1
                                if (idx_gt - 1) % stride != 0:
                                    # ground-truth map skipped by the stride
                                    idx_AEE = 0
                                    criteria[i].reset()
                                    continue

                            # compute metric
                            val_metric = criteria[i]()
                            if metric == "AEE":
                                idx_AEE = 0

                            # accumulate results
                            for batch in range(config["loader"]["batch_size"]):
                                filename = data.files[data.batch_idx[batch] % len(data.files)].split("/")[-1]
                                if filename not in val_results.keys():
                                    val_results[filename] = {}
                                    for metric in config["metrics"]["name"]:
                                        val_results[filename][metric] = {}
                                        val_results[filename][metric]["metric"] = 0
                                        val_results[filename][metric]["it"] = 0
                                        if metric == "AEE":
                                            val_results[filename][metric]["percent"] = 0

                                val_results[filename][metric]["it"] += 1
                                if metric == "AEE":
                                    val_results[filename][metric]["metric"] += val_metric[0][batch].cpu().numpy()
                                    val_results[filename][metric]["percent"] += val_metric[1][batch].cpu().numpy()
                                else:
                                    val_results[filename][metric]["metric"] += val_metric[batch].cpu().numpy()

                            # visualize
                            if (
                                i == 0
                                and config["data"]["mode"] == "events"
                                and (config["vis"]["enabled"] or config["vis"]["store"])
                                and config["data"]["window"] < config["data"]["window_eval"]
                            ):
                                events_window_vis = criteria[i].compute_window_events()
                                iwe_window_vis = criteria[i].compute_window_iwe()
                                masked_window_flow_vis = criteria[i].compute_masked_window_flow()

                            # reset criteria
                            criteria[i].reset()

                # visualize
                if config["vis"]["bars"]:
                    for bar in data.open_files_bar:
                        bar.next()
                if config["vis"]["enabled"]:
                    vis.update(inputs, flow_vis, iwe, events_window_vis, masked_window_flow_vis, iwe_window_vis)
                if config["vis"]["store"]:
                    sequence = data.files[data.batch_idx[0] % len(data.files)].split("/")[-1].split(".")[0]
                    vis.store(
                        inputs,
                        flow_vis,
                        iwe,
                        sequence,
                        events_window_vis,
                        masked_window_flow_vis,
                        iwe_window_vis,
                        ts=data.last_proc_timestamp,
                    )

                # visualize activity
                if config["vis"]["activity"]:
                    activity_log = vis_activity(x["activity"], activity_log)

            if end_test:
                break

    if config["vis"]["bars"]:
        for bar in data.open_files_bar:
            bar.finish()

    # store validation config and results
    results = {}
    if not args.debug and "metrics" in config.keys():
        for metric in config["metrics"]["name"]:
            results[metric] = {}
            if metric == "AEE":
                results[metric + "_percent"] = {}
            for key in val_results.keys():
                results[metric][key] = str(val_results[key][metric]["metric"] / val_results[key][metric]["it"])
                if metric == "AEE":
                    results[metric + "_percent"][key] = str(
                        val_results[key][metric]["percent"] / val_results[key][metric]["it"]
                    )
            log_results(args.runid, results, path_results, eval_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runid", help="mlflow run")
    parser.add_argument(
        "--config",
        default="configs/eval_flow.yml",
        help="config file, overwrites mlflow settings",
    )
    parser.add_argument(
        "--path_mlflow",
        default="",
        help="location of the mlflow ui",
    )
    parser.add_argument("--path_results", default="results_inference/")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="don't save stuff",
    )
    args = parser.parse_args()

    # launch testing
    test(args, YAMLParser(args.config))
