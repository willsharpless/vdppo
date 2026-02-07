import datetime
import json
import pathlib
import tempfile
from typing import TypedDict

import ipdb
from loguru import logger


class EvalResult(TypedDict):
    num_valid: int
    total: int
    data: list[int | float]


class EvalResultEntry(TypedDict):
    timestamp: str
    run_path: str
    eval_results: dict[str, EvalResult]


AllEvalResults = dict[str, list[EvalResultEntry]]


def get_eval_results_path(env_name: str) -> pathlib.Path:
    # return pathlib.Path("/datadrive/vd") / env_name / f"eval_results.json"
    return pathlib.Path("/datadrive/vd") / env_name / f"eval_results2.json"


def get_eval_label(alg: str, env_name: str, ablation_type: str) -> str:
    if "ablation" in env_name:
        return f"{alg}_{ablation_type}_{env_name}"
    else:
        return f"{alg}_{env_name}"


def save_eval_results(
    run_path: pathlib.Path, eval_results_path: pathlib.Path, eval_results: dict[str, dict[str, float]], label: str
):
    if eval_results_path.exists():
        with open(eval_results_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    # Add the results for the current run.
    now = datetime.datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")

    entry = {
        "timestamp": stamp,
        "run_path": str(run_path),
        "eval_results": eval_results,
    }

    # label = f"{alg}_{env_name}" if not "ablation" in env_name else f"{alg}_{ablation_type}_{env_name}"
    if label not in all_results:
        all_results[label] = []

    assert isinstance(all_results[label], list)
    all_results[label].append(entry)

    with tempfile.NamedTemporaryFile("w", dir=eval_results_path.parent, delete=False) as tmp_file:
        json.dump(all_results, tmp_file, indent=4)
        temp_file_path = pathlib.Path(tmp_file.name)

    # Atomically replace the original file with the temporary file
    temp_file_path.replace(eval_results_path)
    logger.info(f"Wrote results to {eval_results_path}")


def has_eval_results(label: str, run_path: pathlib.Path, env_name: str):
    eval_results_path = get_eval_results_path(env_name)
    if not eval_results_path.exists():
        return False

    with open(eval_results_path, "r") as f:
        all_eval_results: AllEvalResults = json.load(f)

    if label not in all_eval_results:
        logger.debug(f"No eval results for label {label} in {eval_results_path}")
        return False

    eval_result_entries = all_eval_results[label]
    for entry in eval_result_entries:
        if entry["run_path"] == run_path:
            return True

        if pathlib.Path(entry["run_path"]) == run_path:
            return True

    logger.debug(f"No eval results for run path {run_path} under label {label} in {eval_results_path}")
    return False


def load_eval_results(env_name: str, latest_only: bool = True) -> AllEvalResults:
    score_file = get_eval_results_path(env_name)
    assert score_file.exists(), f"Score file {score_file} does not exist."

    with open(score_file, "r") as f:
        all_results_env: AllEvalResults = json.load(f)

    if not latest_only:
        return all_results_env

    # For each label, there may be multiple entries from a single run_path. Keep the latest only.
    filtered_results_env: AllEvalResults = {}
    for label, entries in all_results_env.items():
        latest_entries: dict[str, EvalResultEntry] = {}
        for entry in entries:
            run_path = entry["run_path"]
            timestamp = entry["timestamp"]
            if run_path not in latest_entries:
                latest_entries[run_path] = entry
            else:
                if timestamp > latest_entries[run_path]["timestamp"]:
                    latest_entries[run_path] = entry

        filtered_results_env[label] = list(latest_entries.values())
    return filtered_results_env
