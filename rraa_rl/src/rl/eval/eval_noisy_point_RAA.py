import os
import jax
import sys
import copy
import numpy as np

from rraa_rl.src.rl.utils.arguments import get_args

from matplotlib import pyplot as plt
from matplotlib.colors import CenteredNorm, ListedColormap
import seaborn as sns

plt.style.use("seaborn-v0_8-darkgrid")

if __name__ == "__main__":
    config = vars(get_args(sys.argv[1:]))

    # Define a list of experiment directories to compare
    experiment_dirs = [
        "point_raa_noisy",
        "point_raa_noisy_MORL",
        "point_raa_noisy_Sparse",
    ]
    log_file_dir = "model"
    log_name = "training_scores.txt"

    # Define noise tags corresponding to each experiment
    noise_tags = ["_nz0", "_nz100", "_nz200", "_nz500", "_nz1000"]

    # seaborn deep color dict corresponding to original baseline indices
    alg_labels = ["DOHJPPO", "RA", "CPPO", "PPO-LAG", "PPO", "RCPPO", "RESPO", "MORL", "Sparse", "P2BPO", "LOGBAR"]
    palette = sns.color_palette("deep", n_colors=len(alg_labels))
    palette[0] = (0, 0, 0)
    colors = {label: color for label, color in zip(alg_labels, palette)}
    color_dict = {
        "point_raa_noisy": colors["DOHJPPO"], 
        "point_raa_noisy_MORL": colors["MORL"],
        "point_raa_noisy_Sparse": colors["Sparse"],
    }

    data_to_plot_ix = 1  # Index of the data column to plot
    with open(os.path.join(log_file_dir, experiment_dirs[0] + noise_tags[0], log_name), 'r') as f:
        f.readline()
        header = f.readline().strip()
        data_labels = header.split(',')
    print("Plotting data for:", data_labels[data_to_plot_ix])

    fig, axes = plt.subplots(1, len(noise_tags), figsize=(len(noise_tags)*5, 5))
    for ax, noise_tag in zip(axes, noise_tags):
        for exp_dir in experiment_dirs:
            
            path = os.path.join(log_file_dir, exp_dir + noise_tag, log_name)
            if not os.path.exists(path):
                print(f"Log file not found: {path}")
                continue

            # Read scores from the log file
            scores = np.loadtxt(path, delimiter=',', skiprows=2)
            if not scores.size:
                print(f"No valid scores found in: {path}")
                continue
            
            # Flip value for paper convention
            scores[:, 1] = -scores[:, 1]

            # Score Plot over epochs
            ax.plot(np.array(scores[:, 0], dtype=int), scores[:, data_to_plot_ix], 
                    label=exp_dir, color=color_dict[exp_dir], linewidth=2)

        ax.set_title(r"{}".format(noise_tag))
        ax.set_xlabel(r"Epoch")
        if noise_tag == noise_tags[0]:
            ax.set_ylabel(r"{}".format(data_labels[data_to_plot_ix]))
        if noise_tag == noise_tags[-1]:
            ax.legend(["DOHJ", "MORL", "SPARSE"], loc='upper left')
        # ax.grid(True, color='white', alpha=0.5)
        # ax.spines[['top', 'right', 'left']].set_visible(False)
        # ax.set_facecolor("#e6ecf2")

    plt.tight_layout()
    plt.savefig("./eval/noisy_point_raa_comparison.png", dpi=300)
