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

    data_to_plot_ix = -1  # [epoch, value_mean, value_std, crash_percent, reach_percent, rora_percent, raa_percent]
    both_result_types = False

    if both_result_types:
        fig, axes = plt.subplots(2, 5, figsize=(5*4, 5))
        axes = axes.flatten()
        axes_raa = plt.axes[:4]
        axes_rr = axes[5:-1]
    else:
        fig, axes = plt.subplots(2, 4, figsize=(4*4, 5))
        axes = axes.flatten()
        axes_raa = axes[:4]
        axes_rr = axes[4:]

    # Define a list of experiment directories to compare
    experiment_dirs_raa = {
        "DOHJ":"NOISY_halfcheetah_raa_rg_noisy",
        "SPARSE":"NOISY_halfcheetah_raa_sparse_noisy",
        "MORL":"NOISY_halfcheetah_raa_morl_noisy",
        "CPPO":"NOISY_halfcheetah_raa_cppo_kp100_noisy",
    }
    experiment_dirs_rr = {
        "DOHJ":"NOISY_halfcheetah_rr_rg_noisy",
        "DSTL":"NOISY_halfcheetah_rr_dstl_noisy",
        "SPARSE":"NOISY_halfcheetah_rr_sparse_noisy",
        "LOGBAR":"NOISY_halfcheetah_rr_logbar_noisy",
    }
    log_file_dir = "model"
    log_name = "training_scores.txt"

    # Define noise tags corresponding to each experiment
    noise_tags = ["_nz0", "_nz5", "_nz10", "_nz20"]

    # seaborn deep color dict corresponding to original baseline indices
    alg_labels = ["DOHJ", "DSTL", "CPPO", "PPO-LAG", "PPO", "RCPPO", "RESPO", "MORL", "Sparse", "P2BPO", "LOGBAR"]
    palette = sns.color_palette("deep", n_colors=len(alg_labels))
    palette[0] = (0, 0, 0)
    colors = {label: color for label, color in zip(alg_labels, palette)}
    color_dict = {
        "NOISY_halfcheetah_raa_rg_noisy": colors["DOHJ"], 
        "NOISY_halfcheetah_raa_sparse_noisy": colors["Sparse"],
        "NOISY_halfcheetah_raa_morl_noisy": colors["MORL"],
        "NOISY_halfcheetah_raa_cppo_kp100_noisy": colors["CPPO"],
        "NOISY_halfcheetah_rr_rg_noisy": colors["DOHJ"], 
        "NOISY_halfcheetah_rr_dstl_noisy": colors["DSTL"],
        "NOISY_halfcheetah_rr_sparse_noisy": colors["Sparse"],
        "NOISY_halfcheetah_rr_logbar_noisy": colors["LOGBAR"],
    }

    ## RAA RESULTS SUBPLOTS

    with open(os.path.join(log_file_dir, experiment_dirs_raa["DOHJ"] + noise_tags[0], log_name), 'r') as f:
        f.readline()
        header = f.readline().strip()
        data_labels = header.split(',')
    
    # insert lost value_std label after value mean
    data_labels.insert(2, "value_std")
    print("Plotting data for:", data_labels[data_to_plot_ix])

    for ax, noise_tag in zip(axes_raa, noise_tags):
        for alg_name, exp_dir in experiment_dirs_raa.items():

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
            scores[:, -1] = 100 * scores[:, -1]

            data = scores[:, data_to_plot_ix]
            if data_to_plot_ix==1 or data_to_plot_ix==-1:
                ax.plot(np.array(scores[:, 0], dtype=int), data, alpha=0.3,
                    label="", color=color_dict[exp_dir], linewidth=1) # plot raw data
                # running mean
                window_size = 10
                data = np.concatenate((np.min(data[:window_size-1]) + 0*data[:window_size-1], data)) # pad
                data = np.convolve(data, np.ones(window_size)/window_size, mode='valid')

            # Score Plot over epochs
            ax.plot(np.array(scores[:, 0], dtype=int), data,
                    label=alg_name, color=color_dict[exp_dir], linewidth=2)

        # Make noise title with Normal symbol \mathcal{N}(0, σ^2)
        if noise_tag == "_nz0":
            title_str = r"Deterministic Dynamics"
            fontsize = 12
        else:
            noise_level = noise_tag.replace("_nz", "")
            title_str = r"$\mathcal{N}(0, " + r"{}".format(float(noise_level)/10) + r"^2)$"
            fontsize = 15
        ax.set_title(title_str, fontsize=fontsize)
        
        # ax.set_xlabel(r"Epoch")
        if data_to_plot_ix == 1:
            ax.set_ylim([-0.3, 0.])
        if data_to_plot_ix == -1:
            ax.set_ylim([0, 100])
        if noise_tag == noise_tags[0]:
            nice_labels = ["Epoch", r"$\widetilde{V}_{RAA}$", "value_std", "Crash (%)", "reach_percent", "rora_percent", "RAA SUCCESS (%)"]
            ax.set_ylabel(r"{}".format(nice_labels[data_to_plot_ix]))
        if (noise_tag == noise_tags[-1] and data_to_plot_ix == -1) or \
            (noise_tag == noise_tags[0] and data_to_plot_ix == 1):
            loc_ = 'lower right' if data_to_plot_ix != -1 else 'upper right'
            # legend = ax.legend(["DOHJ", "SPARSE", "MORL", "CPPO"], loc='upper right')
            legend = ax.legend(loc=loc_, ncol=2,
                      handlelength=2, handletextpad=0.5, 
                      prop={'size': 10})
            # Make legend lines thicker
            for line in legend.get_lines():
                line.set_linewidth(2)
            legend.get_texts()[0].set_weight('bold')
        if noise_tag != noise_tags[0]:
            ax.set_yticklabels([])
    
    ## RR RESULTS SUBPLOTS

    with open(os.path.join(log_file_dir, experiment_dirs_rr["DOHJ"] + noise_tags[0], log_name), 'r') as f:
        f.readline()
        header = f.readline().strip()
        data_labels = header.split(',')
    
    # insert lost value_std label after value mean
    data_labels.insert(2, "value_std")
    print("Plotting data for:", data_labels[data_to_plot_ix])

    for ax, noise_tag in zip(axes_rr, noise_tags):
        for alg_name, exp_dir in experiment_dirs_rr.items():

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
            scores[:, -1] = 100 * scores[:, -1]

            data = scores[:, data_to_plot_ix]
            if data_to_plot_ix==1 or data_to_plot_ix==-1:
                ax.plot(np.array(scores[:, 0], dtype=int), data, alpha=0.3,
                    label="", color=color_dict[exp_dir], linewidth=1) # plot raw data
                # running mean
                window_size = 10
                data = np.concatenate((np.min(data[:window_size-1]) + 0*data[:window_size-1], data)) # pad
                data = np.convolve(data, np.ones(window_size)/window_size, mode='valid')

            # Score Plot over epochs
            ax.plot(np.array(scores[:, 0], dtype=int), data,
                    label=alg_name, color=color_dict[exp_dir], linewidth=2)

        # Make noise title with Normal symbol \mathcal{N}(0, σ^2)
        # if noise_tag == "_nz0":
        #     title_str = r"Deterministic Dynamics"
        #     fontsize = 12
        # else:
        #     noise_level = noise_tag.replace("_nz", "")
        #     title_str = r"$\mathcal{N}(0, " + r"{}".format(float(noise_level)/10) + r"^2)$"
        #     fontsize = 15
        # ax.set_title(title_str, fontsize=fontsize)

        # Set epoch ticks every 150 steps
        ax.set_xticks(np.arange(0, 800, 150))
        
        ax.set_xlabel(r"Epoch")
        if data_to_plot_ix == 1:
            ax.set_ylim(top=210)
        if data_to_plot_ix == -1:
            ax.set_ylim([0, 100])
        if noise_tag == noise_tags[0]:
            nice_labels = ["Epoch", r"$\widetilde{V}_{RR}$", "value_std", "Reach 2 (%)", "Reach 1 (%)", "Reached a Target (%)", "RR SUCCESS (%)"]
            ax.set_ylabel(r"{}".format(nice_labels[data_to_plot_ix]))
        if (noise_tag == noise_tags[-1] and data_to_plot_ix == -1) or \
            (noise_tag == noise_tags[0] and data_to_plot_ix == 1):
            loc_ = 'upper left' if data_to_plot_ix != -1 else 'upper right'
            # legend = ax.legend(["DOHJ", "SPARSE", "MORL", "CPPO"], loc='upper right')
            legend = ax.legend(loc=loc_, ncol=2,
                      handlelength=2, handletextpad=0.5, 
                      prop={'size': 10})
            # Make legend lines thicker
            for line in legend.get_lines():
                line.set_linewidth(2)
            legend.get_texts()[0].set_weight('bold')
        if noise_tag != noise_tags[0]:
            ax.set_yticklabels([])

    plt.tight_layout()
    plt.savefig("./eval/noisy_halfcheetah_comparison_{}.png".format(data_labels[data_to_plot_ix]), dpi=300)
