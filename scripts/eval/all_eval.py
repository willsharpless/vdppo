import json
import pathlib
import re
import tempfile
from doctest import debug

import cyclopts
import ipdb
import jax
import jax.random as jr
import numpy as np
from loguru import logger
from tqdm.auto import tqdm

from vdppo.training.eval import eval
from vdppo.training.eval_results import get_eval_results_path

app = cyclopts.App()


@app.default()
def main(
    algs: list[str] = ["vdppo", "lcrl", "mppi"],
    envs: list[str] = [
        "gridworld_map1",
        "gridworld_map5",
        "gridworld_map6",
        "gridworld_map7",
        "herding",
        "delivery",
        # "manip_scene",
        "ablation",
        "ablation_depth",
    ],
    ablation_types: list[str] = ["spec", "ag"],
    seed: int = 123,
    n_envs_test: int = 128,
    debug: bool = False,
    plot_failures: bool = False,
    check_missing_runs: bool = False,
    check_missing_scores: bool = False,
):

    algs = [algs] if not isinstance(algs, list) else algs
    envs = [envs] if not isinstance(envs, list) else envs
    ablation_types = [ablation_types] if not isinstance(ablation_types, list) else ablation_types

    missing_scores = {}
    for env_name in envs:

        if check_missing_scores:
            logger.info(f"Checking missing scores for {env_name}...")

            score_file = get_eval_results_path(env_name)

            if score_file.exists():
                with open(score_file, "r") as f:
                    all_results_env = json.load(f)
                env_missing_none = True
            else:
                logger.warning(f"    no score file found for {env_name}! Skipping...")
                continue

        for alg in tqdm(algs, desc=f"Evaluating algs for {env_name}", unit="alg"):

            # ablation
            if env_name == "ablation":
                ablation_its = ablation_types
            else:
                ablation_its = ["spec"]

            for ablation_type in ablation_its:

                if check_missing_scores:
                    label = f"{alg}_{env_name}" if not env_name == "ablation" else f"{alg}_{ablation_type}_{env_name}"
                    if label not in all_results_env:
                        missing_scores[label] = (alg, env_name, ablation_type)
                        logger.warning(f"    missing {label}...")
                        env_missing_none = False
                    continue

                means, stds, missing_runs = eval(
                    alg=alg,
                    env_name=env_name,
                    ablation_type=ablation_type,
                    seed=seed,
                    n_envs_test=n_envs_test,
                    eval_T=None,  # defaults to env specific
                    debug=plot_failures,
                    missing=check_missing_runs,
                    min_ckpt=50000 if not debug else 0,
                )

        if check_missing_scores and env_missing_none:
            logger.info(f"    no missing scores!")

    if check_missing_scores and missing_scores:
        print(f"\nMISSING SCORES for:")
        print(missing_scores)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
