from collections import defaultdict

import cyclopts
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from loguru import logger

from rraa_rl.eval_results import load_eval_results
from rraa_rl.paper_plot_utils import get_ci, set_ax_style
from rraa_rl.path_utils import get_paper_plot_dir

app = cyclopts.App()


@app.default()
def main():
    envs = [
        "gridworld_map1",
        "gridworld_map5",
        "gridworld_map6",
        "gridworld_map7",
        "herdos",
        "delivery",
    ]
    algs = ["vd", "lcrl", "mppi"]
    plot_dir = get_paper_plot_dir()

    for env in envs:
        all_results = load_eval_results(env, latest_only=True)

        figsize = 0.8 * np.array([4.0, 3.0])
        fig, ax = plt.subplots(figsize=figsize)
        set_ax_style(ax)

        for alg in tqdm.tqdm(algs):
            label = f"{alg}_{env}"
            assert label in all_results
            entries = all_results[label]

            step_dict: dict[int, list[float]] = defaultdict(list)
            for entry in entries:
                run_path = entry["run_path"]
                eval_results = entry["eval_results"]



if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
