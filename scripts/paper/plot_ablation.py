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


def extract_abl_params_from_run_path(run_path: str) -> tuple[int, int, int]:
    # e.g., chest_ablation_vd_spc3_ag1_seed1
    parts = run_path.split("_")
    seed = int(parts[-1].replace("seed", ""))
    ag = int(parts[-2].replace("ag", ""))
    spc = int(parts[-3].replace("spc", ""))

    return spc, ag, seed


@app.default()
def main():
    plot_dir = get_paper_plot_dir()

    algs = ["vd", "lcrl", "mppi"]

    # envs = ["ablation", "ablation_depth"]
    envs = ["ablation_depth"]
    for env_name in envs:
        all_results = load_eval_results(env_name, latest_only=True)

        if env_name == "ablation":
            ablation_types: list[str] = ["spec", "ag"]
        else:
            ablation_types = ["spec"]

            for ablation_type in ablation_types:
                figsize = 0.8 * np.array([4.0, 3.0])
                fig, ax = plt.subplots(figsize=figsize)
                set_ax_style(ax)

                for alg in tqdm.tqdm(algs):
                    label = f"{alg}_{ablation_type}_{env_name}"
                    assert label in all_results
                    entries = all_results[label]

                    a_means = {}
                    a_ci_hi = {}
                    a_ci_lo = {}

                    b_num_dict: dict[int, list[int]] = defaultdict(list)
                    b_tot_dict: dict[int, list[int]] = defaultdict(list)
                    for entry in entries:
                        # e.g., chest_ablation_vd_spc3_ag1_seed1
                        run_path = entry["run_path"]
                        try:
                            spc, ag, seed = extract_abl_params_from_run_path(run_path)
                        except Exception as e:
                            logger.warning(f"Failed to extract ablation params from {label}:{run_path}, skipping...")
                            continue

                        eval_results = entry["eval_results"]
                        root_results = eval_results["root"]
                        num_valid = root_results["num_valid"]
                        total = root_results["total"]

                        if ablation_type == "spec":
                            plot_key = spc
                        else:
                            plot_key = ag

                        b_num_dict[plot_key].append(num_valid)
                        b_tot_dict[plot_key].append(total)

                    for plot_key, b_num in b_num_dict.items():
                        b_num = np.array(b_num)
                        b_tot = np.array(b_tot_dict[plot_key])
                        n_seeds = len(b_num)
                        if n_seeds != 3:
                            logger.warning(f"Expected 3 seeds for {label} plot_key={plot_key}, got {n_seeds}...")

                        # model, idata = fit_beta_binomial(b_num, b_tot)
                        # cis = credible_intervals_for_plotting(idata, prob=0.95)
                        #
                        # a_means[spc] = cis["mu"]["mid"]
                        # a_ci_lo[spc] = cis["mu"]["low"]
                        # a_ci_hi[spc] = cis["mu"]["high"]

                        # out = logit_t_with_logistic_normal_mu_ci(
                        #     successes=b_num,
                        #     trials=b_tot,
                        #     alpha=0.05,
                        #     smooth="jeffreys",
                        #     B=10_000,
                        #     K=10_000,
                        #     random_seed=42,
                        # )
                        # a_means[plot_key] = out["mu_hat"]
                        # a_ci_lo[plot_key] = out["mu_ci"][0]
                        # a_ci_hi[plot_key] = out["mu_ci"][1]

                        a_means[plot_key], a_ci_lo[plot_key], a_ci_hi[plot_key] = get_ci(b_num, b_tot)

                        # b_means = b_num / b_tot
                        #
                        # p_mean = b_means.mean()
                        # s = b_means.std(ddof=1)
                        # se = s / np.sqrt(n_seeds)  # standard error of mean across seeds
                        # alpha = 0.05
                        # tcrit = stats.t.ppf(1 - alpha / 2, df=n_seeds - 1)
                        # ci_low, ci_high = p_mean - tcrit * se, p_mean + tcrit * se
                        #
                        # # Clip ci_low and ci_high to [0, 1].
                        # ci_low = max(0.0, ci_low)
                        # ci_high = min(1.0, ci_high)
                        #
                        # a_means[spc] = p_mean
                        # a_ci_lo[spc] = ci_low
                        # a_ci_hi[spc] = ci_high

                    # Plot
                    plot_key_values = sorted(a_means.keys())
                    means = [a_means[plot_key] for plot_key in plot_key_values]
                    ci_his = [a_ci_hi[plot_key] - a_means[plot_key] for plot_key in plot_key_values]
                    ci_los = [a_means[plot_key] - a_ci_lo[plot_key] for plot_key in plot_key_values]

                    # Plot a line plot with error bars.
                    ax.errorbar(
                        plot_key_values,
                        means,
                        yerr=[ci_los, ci_his],
                        label=alg.upper(),
                        capsize=5,
                        marker="o",
                        linestyle="-",
                    )
                    if ablation_type == "spec":
                        x_label = "Number of Specifications"
                    else:
                        x_label = "Number of Agents"

                    ax.set_xlabel(x_label)
                    ax.set_ylabel("Success Rate")

                # At top center.
                legend = ax.legend(
                    ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), borderaxespad=0, frameon=False
                )
                # fig_path = plot_dir / f"ablation_{ablation_type}.pdf"
                fig_path = plot_dir / f"{env_name}_{ablation_type}.png"
                fig.savefig(fig_path, bbox_inches="tight", pad_inches=1e-3, dpi=400)
                plt.close(fig)
                logger.success(f"Saved ablation plot to {fig_path}")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
