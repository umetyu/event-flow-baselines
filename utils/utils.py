import glob
import os
from pathlib import Path

# the experiments rely on the local file store (run folders are read directly); newer MLflow
# versions only allow it when this variable is set
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402


def resolve_tracking_uri(path_mlflow):
    """
    Always use a local file store. An empty argument means <cwd>/mlruns (newer MLflow versions no
    longer default to it), and plain paths are converted to file:// URIs.
    """

    if path_mlflow is None or path_mlflow == "":
        return Path("mlruns").resolve().as_uri()
    if "://" in path_mlflow:
        return path_mlflow
    return Path(path_mlflow).resolve().as_uri()


def _uri_to_path(uri):
    return uri[7:] if uri[:7] == "file://" else uri


def find_run_dir(runid):
    """
    Folder of a run inside the local file store (<tracking dir>/<experiment>/<runid>), or None.
    """

    root = _uri_to_path(mlflow.get_tracking_uri())
    for candidate in sorted(glob.glob(os.path.join(root, "*", runid))):
        if os.path.isdir(os.path.join(candidate, "params")):
            return candidate
    return None


def get_run_params(runid):
    """
    Parameters of a run as a {key: string} dict. Falls back to reading the file store directly when
    MLflow cannot parse the run metadata (e.g. mlruns folders created with a much older version).
    """

    try:
        return dict(mlflow.get_run(runid).data.params)
    except Exception as error:
        run_dir = find_run_dir(runid)
        if run_dir is None:
            raise FileNotFoundError(
                "Run " + runid + " not found in " + mlflow.get_tracking_uri() + " (" + str(error) + ")"
            )
        params = {}
        for fname in sorted(os.listdir(os.path.join(run_dir, "params"))):
            with open(os.path.join(run_dir, "params", fname)) as fid:
                params[fname] = fid.read()
        print("MLflow could not open run " + runid + "; parameters read from " + run_dir + "\n")
        return params


def find_model_file(runid):
    """
    Locate the weights of an MLflow run. The artifact location stored in the run metadata is tried
    first; if the mlruns folder was moved/downloaded from another machine (e.g. the pretrained
    models), we fall back to searching the current tracking directory.
    """

    candidates = []
    try:
        run = mlflow.get_run(runid)
        candidates.append(os.path.join(_uri_to_path(run.info.artifact_uri), "model", "data", "model.pth"))
    except Exception:
        pass

    root = _uri_to_path(mlflow.get_tracking_uri())
    candidates += sorted(glob.glob(os.path.join(root, "*", runid, "artifacts", "model", "data", "model.pth")))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def load_model(prev_runid, model, device, strict=False):
    """
    :param strict: raise if the weights cannot be found (always used for evaluation, so that we never
    report metrics of a randomly initialized network by accident)
    """

    if prev_runid is None or prev_runid == "":
        if strict:
            raise FileNotFoundError("No MLflow run specified.")
        return model

    model_dir = find_model_file(prev_runid)
    if model_dir is None:
        msg = "No model found for run " + prev_runid + " in " + mlflow.get_tracking_uri()
        if strict:
            raise FileNotFoundError(msg)
        print(msg + "\n")
        return model

    try:
        model_loaded = torch.load(model_dir, map_location=device, weights_only=True)
    except Exception:
        # checkpoints of the original authors are pickled nn.Modules (trusted source)
        model_loaded = torch.load(model_dir, map_location=device, weights_only=False)
    if isinstance(model_loaded, torch.nn.Module):
        model_loaded = model_loaded.state_dict()
    model.load_state_dict(model_loaded)
    print("Model restored from " + prev_runid + " (" + model_dir + ")\n")

    return model


def create_model_dir(path_results, runid):
    path_results += runid + "/"
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    print("Results stored at " + path_results + "\n")
    return path_results


def save_model(model):
    """
    Stores the state dict at <run>/artifacts/model/data/model.pth (same location as the original
    mlflow.pytorch.log_model call, but independent of the MLflow/PyTorch version). The file is
    written atomically so that a job killed mid-write cannot corrupt the best checkpoint.
    """

    path = os.path.join(_uri_to_path(mlflow.get_artifact_uri()), "model", "data")
    os.makedirs(path, exist_ok=True)
    tmp = os.path.join(path, "model.pth.tmp")
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, os.path.join(path, "model.pth"))


def save_csv(data, fname):
    # create file if not there
    path = mlflow.get_artifact_uri(artifact_path=fname)
    if path[:7] == "file://":  # to_csv() doesn't work with 'file://'
        path = path[7:]
    if not os.path.isfile(path):
        mlflow.log_text("", fname)
        pd.DataFrame(data).to_csv(path)
    # else append
    else:
        pd.DataFrame(data).to_csv(path, mode="a", header=False)


def save_diff(fname="git_diff.txt"):
    # .txt to allow showing in mlflow
    path = mlflow.get_artifact_uri(artifact_path=fname)
    if path[:7] == "file://":
        path = path[7:]
    mlflow.log_text("", fname)
    os.system(f"git diff > {path}")
