from collections import defaultdict

import cyclopts
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from loguru import logger

from rraa_rl.training.eval_results import load_eval_results
from rraa_rl.training.paper_plot_utils import get_ci, set_ax_style
from rraa_rl.common.path_utils import get_paper_plot_dir

app = cyclopts.App()


@app.default()
def main():
    # The tuples are (spec_name, display_name)
    envs = {
        # "gridworld_map1":,
        # "gridworld_map5":,
        # "gridworld_map6":,
        # "gridworld_map7":,
        "herding": [("root", "Overall"), ("safety", "Safety"), ("herded", "Herded")],
        "delivery": [
            ("root", "Overall"),
            ("safety", "Safety"),
            ("cycled", "Agent\nCycled"),
        ],
        "manip_scene": [("root", "Overall"), ("drawer_open", "Open\nDrawer"), ("cube_in_drawer", "Cube in\nDrawer")],
    }
    algs = ["vdppo", "lcrl", "mppi"]
    plot_dir = get_paper_plot_dir()

    method_colors = {
        "vdppo": "C0",
        "lcrl": "C1",
        "mppi": "C2",
    }

    alg_display_name = {
        "vdppo": "VDPPO",
        "lcrl": "LCRL",
        "mppi": "MPPI",
    }

    for env, specs_to_plot in tqdm.tqdm(envs.items(), unit="Env"):
        specs_to_plot = specs_to_plot[::-1]
        all_results = load_eval_results(env, latest_only=True)

        figsize = 0.8 * np.array([4.0, 3.0])
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        set_ax_style(ax)

        # ci_los[metric][method] = value
        ci_los: dict[str, dict[str, float]] = defaultdict(dict)
        ci_his: dict[str, dict[str, float]] = defaultdict(dict)
        means: dict[str, dict[str, float]] = defaultdict(dict)

        for alg in tqdm.tqdm(algs):
            label = f"{alg}_{env}"

            if label == "mppi_manip_scene":
                b_num_dict = {"root": np.zeros(3), "drawer_open": np.zeros(3), "cube_in_drawer": np.zeros(3)}
                b_tot_dict = {
                    "root": np.full(3, 128),
                    "drawer_open": np.full(3, 128),
                    "cube_in_drawer": np.full(3, 128),
                }
            else:
                assert label in all_results
                entries = all_results[label]

                b_num_dict: dict[str, list[int]] = defaultdict(list)
                b_tot_dict: dict[str, list[int]] = defaultdict(list)

                step_dict: dict[int, list[float]] = defaultdict(list)
                for entry in entries:
                    run_path = entry["run_path"]
                    eval_results = entry["eval_results"]

                    for spec_name, results in eval_results.items():
                        num_valid = results["num_valid"]
                        total = results["total"]

                        b_num_dict[spec_name].append(num_valid)
                        b_tot_dict[spec_name].append(total)

                assert "root" in b_num_dict
                n_seeds = len(b_num_dict["root"])
                if n_seeds < 2:
                    logger.warning(f"{label} only has {n_seeds} seeds, skipping...")
                    continue
                if n_seeds < 3:
                    logger.warning(f"{label} only has {n_seeds} seeds, CI may be unreliable...")

            for spec_name in b_num_dict:
                b_num = np.array(b_num_dict[spec_name])
                b_tot = np.array(b_tot_dict[spec_name])
                n_seeds = len(b_num)

                means[spec_name][alg], ci_los[spec_name][alg], ci_his[spec_name][alg] = get_ci(
                    b_num, b_tot, B=1_000, K=1_000
                )

        # Plot.
        n_methods = len(algs)

        spec_names = [spec_name for spec_name, _ in specs_to_plot]
        n_specs = len(spec_names)

        group_gap = 1.0
        within_gap = 0.23
        box_height = 0.18
        median_frac = 0.8

        group_centers = np.arange(n_specs) * group_gap
        offsets = (np.arange(n_methods) - (n_methods - 1) / 2) * within_gap

        for ii, alg in enumerate(algs[::-1]):
            xs = []
            y_positions = []
            xerr_left = []
            xerr_right = []

            for jj, spec_name in enumerate(spec_names):
                if alg not in means[spec_name]:
                    continue

                x = means[spec_name][alg]
                ci_lo = ci_los[spec_name][alg]
                ci_hi = ci_his[spec_name][alg]

                xs.append(x)
                y_positions.append(group_centers[jj] + offsets[ii])
                xerr_left.append(x - ci_lo)
                xerr_right.append(ci_hi - x)

            if not xs:
                continue

            alg_color = method_colors[alg]
            marker = "o"
            markersize = 4
            linewidth = 1.5
            capsize = 3
            ax.errorbar(
                xs,
                y_positions,
                xerr=[xerr_left, xerr_right],
                fmt=marker,
                markersize=markersize,
                linewidth=linewidth,
                capsize=capsize,
                color=alg_color,
                ecolor=alg_color,
                markerfacecolor=alg_color,
                markeredgecolor=alg_color,
                linestyle="none",
                label=alg_display_name[alg],
                zorder=3,
            )

        # Y ticks at criterion centers
        ax.set_xlim(-0.05, 1.05)
        ax.set_yticks(group_centers)
        ax.set_yticklabels([display_name for _, display_name in specs_to_plot])

        ax.set_xlabel("Success Rate")
        # ax.set_ylabel("Criterion")
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        # fig_path = plot_dir / f"{env}_plot.pdf"
        fig_path = plot_dir / f"{env}_plot.png"
        fig.savefig(fig_path, bbox_inches="tight", pad_inches=1e-2, dpi=400)
        plt.close(fig)
        logger.success("Saved plot to {}", fig_path)

    # End of for loop over env.


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
