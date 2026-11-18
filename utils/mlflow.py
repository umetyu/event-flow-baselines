import os
import yaml

import torch
import mlflow


def log_config(path_results, runid, config):
    """
    Log configuration file to MlFlow run.
    """

    eval_id = 0
    for file in os.listdir(path_results):
        if file.endswith(".yml"):
            tmp = int(file.split(".")[0].split("_")[-1])
            eval_id = tmp + 1 if tmp + 1 > eval_id else eval_id
    yaml_filename = path_results + "eval_" + str(eval_id) + ".yml"
    with open(yaml_filename, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    _log_artifact(runid, yaml_filename)

    return eval_id


def log_results(runid, results, path, eval_id):
    """
    Log validation results as artifacts to MlFlow run.
    """

    yaml_filename = path + "metrics_" + str(eval_id) + ".yml"
    with open(yaml_filename, "w") as outfile:
        yaml.dump(results, outfile, default_flow_style=False)

    _log_artifact(runid, yaml_filename)


def _log_artifact(runid, filename):
    """
    The file in the results folder is the reference copy. Attaching it to the MLflow run can fail for
    runs created elsewhere (e.g. pretrained models whose metadata points to another machine).
    """

    try:
        mlflow.start_run(runid)
        mlflow.log_artifact(filename)
    except Exception as error:
        print("Could not attach " + filename + " to run " + runid + ": " + str(error))
    finally:
        mlflow.end_run()
