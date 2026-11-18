"""
Helpers shared by the experiment scripts: reading runs from the local MLflow file store without
depending on the MLflow version that created them.
"""

import ast
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Models of Hagenaars et al. 2021 that differ only in how the network keeps the past (class names in
# models/model.py): gated recurrence (the paper's "EV-FlowNet"/"FireNet"), vanilla recurrence and
# leaky neurons, for both architectures. The spiking models of the paper are left out.
# In addition, the non-recurrent EV-FlowNet with a voxel-grid input: the kind of network behind the
# literature row "EV-FlowNet [56]" of Table 1 (Zhu et al., CVPR 2019), where the past is encoded in
# the input instead of the network state. Value = required input encoding (None = any).
CORE_MODELS = {
    "RecEVFlowNet": None,
    "RNNRecEVFlowNet": None,
    "LeakyRecEVFlowNet": None,
    "FireNet": None,
    "RNNFireNet": None,
    "LeakyFireNet": None,
    "EVFlowNet": "voxel",
}


def reference_key(model, encoding):
    """
    Key of a run in experiments/reference/hagenaars2021_table1.yml.
    """

    if model == "EVFlowNet":
        return "EVFlowNet_voxel" if str(encoding).startswith("voxel") else None
    return model


def _read(path):
    with open(path) as fid:
        return fid.read().strip()


def parse_value(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def list_runs(path_mlflow):
    """
    :return: list of dicts {run_id, name, experiment, dir, params} for every run folder
    """

    runs = []
    for params_dir in sorted(glob.glob(os.path.join(path_mlflow, "*", "*", "params"))):
        run_dir = os.path.dirname(params_dir)
        exp_dir = os.path.dirname(run_dir)
        if os.path.basename(exp_dir) in [".trash", "models"]:
            continue
        run_id = os.path.basename(run_dir)
        params = {name: parse_value(_read(os.path.join(params_dir, name))) for name in sorted(os.listdir(params_dir))}

        name = ""
        tag = os.path.join(run_dir, "tags", "mlflow.runName")
        if os.path.isfile(tag):
            name = _read(tag)

        experiment = os.path.basename(exp_dir)
        meta = os.path.join(exp_dir, "meta.yaml")
        if os.path.isfile(meta):
            for line in _read(meta).splitlines():
                if line.startswith("name:"):
                    experiment = line.split(":", 1)[1].strip().strip("'\"")

        runs.append({"run_id": run_id, "name": name, "experiment": experiment, "dir": run_dir, "params": params})
    return runs


def read_metric(run_dir, key):
    """
    :return: list of (step, value) from the MLflow file store ("timestamp value step" per line)
    """

    path = os.path.join(run_dir, "metrics", key)
    if not os.path.isfile(path):
        return []
    values = []
    for line in _read(path).splitlines():
        parts = line.split()
        if len(parts) >= 3:
            values.append((int(parts[2]), float(parts[1])))
    return sorted(values)


def select_runs(runs, patterns, models=None):
    """
    Runs whose id, name or experiment matches one of the shell-style patterns (all runs if empty),
    optionally restricted to model classes ("core" = CORE_MODELS, "all"/empty = no restriction).
    """

    import fnmatch

    if models and "all" not in models:
        wanted = {}
        for model in models:
            wanted.update(CORE_MODELS if model == "core" else {model: None})
        runs = [
            run
            for run in runs
            if model_field(run, "name") in wanted
            and wanted[model_field(run, "name")] in [None, model_field(run, "encoding")]
        ]
    if not patterns:
        return runs
    return [
        run
        for run in runs
        if any(
            fnmatch.fnmatch(run["run_id"], p) or fnmatch.fnmatch(run["name"], p) or fnmatch.fnmatch(run["experiment"], p)
            for p in patterns
        )
    ]


def model_field(run, key, default=None):
    model = run["params"].get("model", {})
    return model.get(key, default) if isinstance(model, dict) else default


def data_field(run, key, default=None):
    data = run["params"].get("data", {})
    return data.get(key, default) if isinstance(data, dict) else default
