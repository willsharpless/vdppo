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
    # The tuples are (spec_name, display_name)
    envs = {
        # "gridworld_map1":,
        # "gridworld_map5":,
        # "gridworld_map6":,
        # "gridworld_map7":,
        "herdos": [("root", "Task"), ("safety", "Safety"), ("gate0->1", "Gate 0->1"), ("herded", "Herded")],
        # "delivery":,
    }
    algs = ["vd", "lcrl", "mppi"]
    plot_dir = get_paper_plot_dir()

    method_colors = {
        "vd": "C0",
        "lcrl": "C1",
        "mppi": "C2",
    }

    alg_display_name = {
        "vd": "VD",
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
            markersize = 5
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
        ax.set_ylabel("Criterion")
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        fig_path = plot_dir / f"{env}_plot.pdf"
        fig.savefig(fig_path, bbox_inches="tight", pad_inches=1e-2)
        plt.close(fig)
        logger.success("Saved plot to {}", fig_path)

    # End of for loop over env.


def plot_grouped_boxplots_horizontal(
    data,
    criteria_names=None,
    method_names=None,
    colors=None,
    figsize=(4, 3),
    group_gap=1.0,
    within_gap=0.23,
    box_height=0.18,
    median_frac=0.8,  # fraction of box width used by the median line
    showfliers=False,
    xlim=(0, 1),
    title="Method Comparison Across Criteria",
    xlabel="Score",
    ylabel="Criterion",
):
    """
    Horizontal grouped boxplots with:
      - criteria as groups on y-axis
      - methods as colored boxes within each group
      - black, shortened median lines
    """

    if criteria_names is None:
        criteria_names = list(data.keys())

    if method_names is None:
        seen = []
        for c in criteria_names:
            for m in data[c]:
                if m not in seen:
                    seen.append(m)
        method_names = seen

    n_criteria = len(criteria_names)
    n_methods = len(method_names)

    # Colors
    if colors is None:
        palette = plt.cm.tab10.colors
        method_colors = [palette[i % len(palette)] for i in range(n_methods)]
    elif isinstance(colors, dict):
        method_colors = [colors[m] for m in method_names]
    else:
        method_colors = colors

    fig, ax = plt.subplots(figsize=figsize)

    group_centers = np.arange(n_criteria) * group_gap

    if box_height >= within_gap:
        box_height = 0.9 * within_gap

    offsets = (np.arange(n_methods) - (n_methods - 1) / 2) * within_gap

    for i, (method, color) in enumerate(zip(method_names, method_colors)):
        method_data = []
        positions = []

        for j, crit in enumerate(criteria_names):
            method_data.append(data[crit].get(method, [np.nan]))
            positions.append(group_centers[j] + offsets[i])

        bp = ax.boxplot(
            method_data,
            vert=False,
            positions=positions,
            widths=box_height,
            patch_artist=True,
            showfliers=showfliers,
            manage_ticks=False,
            medianprops=dict(color="black", linewidth=1.5),
        )

        # Style boxes
        for box in bp["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.85)

        # --- SHORTEN MEDIAN LINES ---
        for median in bp["medians"]:
            y = median.get_ydata()
            center = np.mean(y)
            new_len = (y[1] - y[0]) * median_frac
            median.set_ydata([y[0], y[0] + new_len])

    ax.set_yticks(group_centers)
    ax.set_yticklabels(criteria_names)

    ax.set_xlim(-0.1, 1.1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.grid(axis="x", linestyle="--", alpha=0.5)

    handles = [plt.Line2D([0], [0], color=c, lw=10) for c in method_colors]
    ax.legend(handles, method_names, title="Method")

    plt.tight_layout()

    fig.savefig("test.pdf")


def main2():
    data = {
        "Safety": {
            "Method A": np.random.rand(20),
            "Method B": np.random.rand(20),
            "Method C": np.random.rand(20),
        },
        "ReachA": {
            "Method A": np.random.rand(20),
            "Method B": np.random.rand(20),
            "Method C": np.random.rand(20),
        },
        "ReachB": {
            "Method A": np.random.rand(20),
            "Method B": np.random.rand(20),
            "Method C": np.random.rand(20),
        },
        "ReachC": {
            "Method A": np.random.rand(20),
            "Method B": np.random.rand(20),
            "Method C": np.random.rand(20),
        },
    }

    plot_grouped_boxplots_horizontal(data)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
