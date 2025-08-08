import os
import optax
import jax
from jax import lax
import sys
import numpy as np
import copy

from functools import partial
from flax.training.train_state import TrainState
from flax.training import checkpoints
import jax.numpy as jnp
import seaborn as sns

import matplotlib.pyplot as plt
from PIL import Image
import imageio

from rraa_rl.EFPPO.src.rl.arguments import get_args
from rraa_rl.EFPPO.src.env.env_list import get_env
from rraa_rl.EFPPO.src.model.actorcritic import Policy_Network, Value_Network, ActorCritic_Continuous, Policy_Network_Discrete, MoGPolicy_Network
from rraa_rl.EFPPO.src.rl.EFPPO_utils import _env_step_raa_vanilla, _env_step_raa_vanilla_deterministic, _env_step_cppo_RAA, _env_step_ra_vanilla, _env_step_adapted_raa, _env_step_raa_respo
# from rraa_rl.EFPPO.src.rl.plot_utils import calculate_reachreach
from rraa_rl.EFPPO.src.rl.root_finding import Bisection
from rraa_rl.EFPPO.src.rl.utils import tree_index1, tree_index2, optimizer

from rraa_rl.EFPPO.src.rl.MORL_PPO_RAA import sparse_replace_raa, morl_replace_raa

def calculate_reachavoid(traj_batch):
    reach_idx = (traj_batch.reach < 0).argmax(axis=0)
    crash_idx = (traj_batch.avoid > 0).argmax(axis=0)
    reach_idx = np.where(np.any((traj_batch.reach < 0) == 1, axis=0), reach_idx, np.inf)
    crash_idx = np.where(np.any((traj_batch.avoid > 0) == 1, axis=0), crash_idx, np.inf)
    # Find indices where reach < inf and avoid = inf
    reach_and_avoid_idx = np.where(crash_idx == np.inf, reach_idx, np.inf)

    reach_perc = ((reach_idx < np.inf).sum() / reach_idx.__len__()).item()
    crash_perc = ((crash_idx < np.inf).sum() / crash_idx.__len__()).item()
    reach_avoid_perc = ((reach_and_avoid_idx < np.inf).sum() / reach_and_avoid_idx.__len__()).item()

    reach_or_avoid_one = np.logical_or(reach_idx < np.inf, crash_idx == np.inf)
    reach_or_avoid_one_perc = reach_or_avoid_one.sum() / reach_or_avoid_one.__len__()

    min_values = np.maximum(np.min(traj_batch.reach, axis=0), np.max(traj_batch.avoid, axis=0))

    return (reach_perc, crash_perc, reach_avoid_perc), (reach_idx, crash_idx, reach_and_avoid_idx), reach_or_avoid_one_perc, min_values.mean().item(), min_values.std().item()

def plot_scores_RAA(traj_batches, config, title="HOPPER-RAA", sub_plots_adjust=True, wspace=0.3, hspace=0.6, tight_layout=True, pad=0.1, w_pad=0.1, h_pad=0.1, use_stochastic=False):

    (traj_batch_HJPPO, 
        traj_batch_HJPPO_d, 
        traj_batch_CPPO, 
        traj_batch_RA,
        traj_batch_PPOLAG,
        traj_batch_PPO,
        traj_batch_RCPPO,
        traj_batch_RESPO,
        traj_batch_MORL,
        traj_batch_SPARSE,
        traj_batch_P2BPO,
        traj_batch_LOGBAR,
    ) = traj_batches
    
    raa_scores_HJPPO = calculate_reachavoid(traj_batch_HJPPO)
    raa_scores_HJPPO_d = calculate_reachavoid(traj_batch_HJPPO_d)
    raa_scores_CPPO = calculate_reachavoid(traj_batch_CPPO)
    raa_scores_RA = calculate_reachavoid(traj_batch_RA)
    raa_scores_PPOLAG = calculate_reachavoid(traj_batch_PPOLAG)
    raa_scores_PPO = calculate_reachavoid(traj_batch_PPO)
    raa_scores_RCPPO = calculate_reachavoid(traj_batch_RCPPO)
    raa_scores_RESPO = calculate_reachavoid(traj_batch_RESPO)
    raa_scores_MORL = calculate_reachavoid(traj_batch_MORL)
    raa_scores_SPARSE = calculate_reachavoid(traj_batch_SPARSE)
    raa_scores_P2BPO = calculate_reachavoid(traj_batch_P2BPO)
    raa_scores_LOGBAR = calculate_reachavoid(traj_batch_LOGBAR)

    # (avoid perc, reach avoid idx)
    # raa_scores_HJPPO = (raa_scores_HJPPO[0][-1], raa_scores_HJPPO[1][-1])
    # raa_scores_HJPPO_d = (raa_scores_HJPPO_d[0][-1], raa_scores_HJPPO_d[1][-1])
    # raa_scores_CPPO = (raa_scores_CPPO[0][-1], raa_scores_CPPO[1][-1])
    # raa_scores_RA = (raa_scores_RA[0][-1], raa_scores_RA[1][-1])

    raa_scores_all = [
        ("LOGBAR", raa_scores_LOGBAR),
        ("P2BPO", raa_scores_P2BPO),
        ("SPARSE", raa_scores_SPARSE),
        ("MORL", raa_scores_MORL),
        ("RESPPO", raa_scores_RESPO),
        ("RCPPO", raa_scores_RCPPO),
        ("PPO", raa_scores_PPO),
        ("PPO-LAG", raa_scores_PPOLAG),
        ("C-PPO", raa_scores_CPPO),
        ("RA", raa_scores_RA),
        ("DOHJPPO", raa_scores_HJPPO_d),
        # ("DOHJPPO", raa_scores_HJPPO),
    ]

    # Extract data
    labels = []
    reach_avoid_percs = []
    reach_or_avoid_percs = []
    mean_idxs = []
    std_idxs = []
    min_val_means = []
    min_val_stds = []

    for tag, scores in raa_scores_all:
        
        reach_avoid_perc = scores[0][2]
        idxs = scores[1][2]

        reach_or_avoid_perc = scores[2]
        min_val_mean = scores[3]
        min_val_std = scores[4]

        # finite_mask = jnp.isfinite(idxs)
        # finite_idxs = idxs[finite_mask]
        # mean_idx = jnp.mean(finite_idxs) if finite_idxs.size > 0 else jnp.nan
        # std_idx = jnp.std(finite_idxs) if finite_idxs.size > 0 else jnp.nan

        replace_val = config["NUM_STEPS"]
        cleaned_idxs = jnp.where(jnp.isfinite(idxs), idxs, replace_val)

        mean_idx = jnp.mean(cleaned_idxs)
        std_idx = jnp.std(cleaned_idxs)

        labels.append(tag)
        reach_avoid_percs.append(reach_avoid_perc)
        reach_or_avoid_percs.append(reach_or_avoid_perc)
        min_val_means.append(min_val_mean)
        min_val_stds.append(min_val_std)

        mean_idxs.append(mean_idx)
        std_idxs.append(std_idx)

        # print(f"{tag:<10} - Success: {reach_avoid_perc:.2f}%, Mean Index: {mean_idx:.1f}, RorA: {reach_or_avoid_perc:.2f}%, Max Value Mean: {-min_val_mean:.3f}")
        print(f"{tag:<10} - R & A: {100*reach_avoid_perc:.0f}%, R: {100*scores[0][0]:.0f}%, A: {100*(1. - scores[0][1]):.0f}%")

    ## Individual Plots

    data_names = ["only_success", "only_speed", "only_success_partial", "only_value"]
    for data_name in data_names:
        fig, axes = plt.subplots(1, 1, figsize=(5, 3), sharex=False)
        palette = sns.color_palette("deep", n_colors=len(raa_scores_all))[::-1]
        palette[-1] = (0, 0, 0)
        colors = {label: color for label, color in zip(labels, palette)}
            
        # Reach percentage bar plot
        order = 0
        for i, label in enumerate(labels):
            if data_name == "only_success":
                axes.barh(label, reach_avoid_percs[i], color=colors[label])
            elif data_name == "only_speed":
                axes.barh(label, mean_idxs[i], xerr=std_idxs[i]/2, color=colors[label], capsize=4, zorder=1)
                axes.barh(label, mean_idxs[i], color=colors[label], capsize=4, zorder=2)
            elif data_name == "only_success_partial":
                axes.barh(label, reach_or_avoid_percs[i], color=colors[label])
            elif data_name == "only_value":
                axes.barh(label, -min_val_means[i], xerr=min_val_stds[i]/2, color=colors[label], capsize=4, zorder=1)
                axes.barh(label, -min_val_means[i], color=colors[label], capsize=4, zorder=2)
        
        if data_name == "only_success":
            axes.set_title(r"{}: Success, $R$ and $AA$ ($\rightarrow$)".format(title), fontsize=12)
            axes.set_xlabel(r"Percentage")
        elif data_name == "only_speed":
            axes.set_title(r"{}: Mean Steps to Success ($\leftarrow$)".format(title), fontsize=12)
            axes.set_xlabel(r"Steps")
        elif data_name == "only_success_partial":
            axes.set_title(r"{}: Partial Success, $R$ or $AA$ ($\rightarrow$)".format(title), fontsize=12)
            axes.set_xlabel(r"Percentage")
        elif data_name == "only_value":
            axes.set_title(r"{}: Maximum Value ($\leftarrow$)".format(title), fontsize=12)
            axes.set_xlabel(r"Value")

        axes.set_yticks(np.arange(len(labels)))
        axes.set_yticklabels(labels, ha='right', fontsize=10)
        if data_name == "success" or data_name == "success_one":
            axes.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[r'$0\%$', r'$20\%$', r'$40\%$', r'$60\%$', r'$80\%$', r'$100\%$'])
        
        axes.tick_params(axis='y', pad=10)  # move labels away from bars
        axes.grid(True, color='white', axis="x", linestyle="--", alpha=0.5)
        axes.spines[['top', 'right', 'left']].set_visible(False)
        axes.set_facecolor("#e6ecf2")

        for tick_label in axes.get_yticklabels():
            if tick_label.get_text() == "DOHJPPO":
                tick_label.set_fontweight("bold")
                tick_label.set_fontsize(12)

        axes.spines[['top', 'right']].set_visible(False)
        axes.tick_params(axis='both', which='both', labelsize=10)
    
        # plt.subplots_adjust(hspace=0.8)
        if tight_layout:
            plt.tight_layout(pad=pad, w_pad=w_pad, h_pad=h_pad)
        if sub_plots_adjust:
            plt.subplots_adjust(wspace=wspace, hspace=hspace)

        plt.savefig(f"eval/{config['EVAL_DIR']}/{config['NAME_TAG']}/{title}_{data_name}", dpi=300, bbox_inches="tight", pad_inches=0.1)

    ## Combined Plots

    fig, axes = plt.subplots(1, 3, figsize=(15, 3), sharex=False)
    palette = sns.color_palette("deep", n_colors=len(raa_scores_all))[::-1]
    palette[-1] = (0, 0, 0)
    colors = {label: color for label, color in zip(labels, palette)}
        
    # Reach percentage bar plot
    order = 0
    for i, label in enumerate(labels):
        axes[order].barh(label, reach_avoid_percs[i], color=colors[label])
    axes[order].set_title(r"{}: Success, $R$ and $AA$ ($\rightarrow$)".format(title), fontsize=12)
    axes[order].set_xlabel(r"Percentage")
    axes[order].set_yticks(np.arange(len(labels)))
    axes[order].set_yticklabels(labels, ha='right', fontsize=10)
    axes[order].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[r'$0\%$', r'$20\%$', r'$40\%$', r'$60\%$', r'$80\%$', r'$100\%$'])
    axes[order].tick_params(axis='y', pad=10)  # move labels away from bars
    axes[order].grid(True, color='white', axis="x", linestyle="--", alpha=0.5)
    axes[order].spines[['top', 'right', 'left']].set_visible(False)
    axes[order].set_facecolor("#e6ecf2")

    # Reach At least onepercentage bar plot
    order = 1
    for i, label in enumerate(labels):
        axes[order].barh(label, reach_or_avoid_percs[i], color=colors[label])
    # axes[0].set_xlim(0.7, 1.0)
    axes[order].set_title(r"{}: Partial Success, $R$ or $AA$ ($\rightarrow$)".format(title), fontsize=12)
    axes[order].set_xlabel(r"Percentage")
    axes[order].set_yticks(np.arange(len(labels)))
    axes[order].set_yticklabels(labels, ha='right', fontsize=10)
    axes[order].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[r'$0\%$', r'$20\%$', r'$40\%$', r'$60\%$', r'$80\%$', r'$100\%$'])
    axes[order].tick_params(axis='y', pad=10)  # move labels away from bars
    axes[order].grid(True, color='white', axis="x", linestyle="--", alpha=0.5)
    axes[order].spines[['top', 'right', 'left']].set_visible(False)
    axes[order].set_facecolor("#e6ecf2")

    # Speed Plot
    order = 2
    for i, label in enumerate(labels):
        axes[order].barh(label, mean_idxs[i], xerr=std_idxs[i]/2, color=colors[label], capsize=4, zorder=1)
        axes[order].barh(label, mean_idxs[i], color=colors[label], capsize=4, zorder=2)
    axes[order].set_title(r"{}: Mean Steps to Success ($\leftarrow$)".format(title), fontsize=12)
    axes[order].set_xlabel(r"Index")
    axes[order].set_yticks(np.arange(len(labels)))
    axes[order].set_yticklabels(labels, ha='right', fontsize=10)
    axes[order].tick_params(axis='y', pad=10)
    axes[order].grid(True, color='white', axis="x", linestyle="--", alpha=0.5)
    axes[order].spines[['top', 'right', 'left']].set_visible(False)
    axes[order].set_facecolor("#e6ecf2")

    for ax in axes:
        for tick_label in ax.get_yticklabels():
            if tick_label.get_text() == "DOHJPPO":
                tick_label.set_fontweight("bold")
                tick_label.set_fontsize(12)

    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='both', which='both', labelsize=10)
   
    # plt.subplots_adjust(hspace=0.8)
    if tight_layout:
        plt.tight_layout(pad=pad, w_pad=w_pad, h_pad=h_pad)
    if sub_plots_adjust:
        plt.subplots_adjust(wspace=wspace, hspace=hspace)

    plt.savefig(f"eval/{config['EVAL_DIR']}/{config['NAME_TAG']}/{title}_combined_plot", dpi=300, bbox_inches="tight", pad_inches=0.1)

    ## Score Plot

    fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharex=False)
    palette = sns.color_palette("deep", n_colors=len(raa_scores_all))[::-1]
    palette[-1] = (0, 0, 0)
    colors = {label: color for label, color in zip(labels, palette)}
        
    # Reach percentage bar plot
    order = 0
    for i, label in enumerate(labels):
        axes[order].barh(label, reach_avoid_percs[i], color=colors[label])
    axes[order].set_title(r"{}: Success, $R$ and $AA$ ($\rightarrow$)".format(title), fontsize=12)
    axes[order].set_xlabel(r"Percentage")
    axes[order].set_yticks(np.arange(len(labels)))
    axes[order].set_yticklabels(labels, ha='right', fontsize=10)
    axes[order].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[r'$0\%$', r'$20\%$', r'$40\%$', r'$60\%$', r'$80\%$', r'$100\%$'])
    axes[order].tick_params(axis='y', pad=10)  # move labels away from bars
    axes[order].grid(True, color='white', axis="x", linestyle="--", alpha=0.5)
    axes[order].spines[['top', 'right', 'left']].set_visible(False)
    axes[order].set_facecolor("#e6ecf2")

    # Reach At least onepercentage bar plot
    order = 1
    for i, label in enumerate(labels):
        axes[order].barh(label, reach_or_avoid_percs[i], color=colors[label])
    # axes[0].set_xlim(0.7, 1.0)
    axes[order].set_title(r"{}: Partial Success, $R$ or $AA$ ($\rightarrow$)".format(title), fontsize=12)
    axes[order].set_xlabel(r"Percentage")
    axes[order].set_yticks(np.arange(len(labels)))
    axes[order].set_yticklabels(labels, ha='right', fontsize=10)
    axes[order].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[r'$0\%$', r'$20\%$', r'$40\%$', r'$60\%$', r'$80\%$', r'$100\%$'])
    axes[order].tick_params(axis='y', pad=10)  # move labels away from bars
    axes[order].grid(True, color='white', axis="x", linestyle="--", alpha=0.5)
    axes[order].spines[['top', 'right', 'left']].set_visible(False)
    axes[order].set_facecolor("#e6ecf2")

    for ax in axes:
        for tick_label in ax.get_yticklabels():
            if tick_label.get_text() == "DOHJPPO":
                tick_label.set_fontweight("bold")
                tick_label.set_fontsize(12)

    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='both', which='both', labelsize=10)
   
    # plt.subplots_adjust(hspace=0.8)
    if tight_layout:
        plt.tight_layout(pad=pad, w_pad=w_pad, h_pad=h_pad)
    if sub_plots_adjust:
        plt.subplots_adjust(wspace=wspace, hspace=hspace)

    plt.savefig(f"eval/{config['EVAL_DIR']}/{config['NAME_TAG']}/{title}_successes_plot", dpi=300, bbox_inches="tight", pad_inches=0.1)

    ## Success & Speed Plot

    fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharex=False)
    palette = sns.color_palette("deep", n_colors=len(raa_scores_all))[::-1]
    palette[-1] = (0, 0, 0)
    colors = {label: color for label, color in zip(labels, palette)}
        
    # Reach percentage bar plot
    order = 0
    for i, label in enumerate(labels):
        axes[order].barh(label, reach_avoid_percs[i], color=colors[label])
    axes[order].set_title(r"{}: Success, $R$ and $AA$ ($\rightarrow$)".format(title), fontsize=12)
    axes[order].set_xlabel(r"Percentage")
    axes[order].set_yticks(np.arange(len(labels)))
    axes[order].set_yticklabels(labels, ha='right', fontsize=10)
    axes[order].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[r'$0\%$', r'$20\%$', r'$40\%$', r'$60\%$', r'$80\%$', r'$100\%$'])
    axes[order].tick_params(axis='y', pad=10)  # move labels away from bars
    axes[order].grid(True, color='white', axis="x", linestyle="--", alpha=0.5)
    axes[order].spines[['top', 'right', 'left']].set_visible(False)
    axes[order].set_facecolor("#e6ecf2")

    # Speed Plot
    order = 1
    for i, label in enumerate(labels):
        axes[order].barh(label, mean_idxs[i], xerr=std_idxs[i]/2, color=colors[label], capsize=4, zorder=1)
        axes[order].barh(label, mean_idxs[i], color=colors[label], capsize=4, zorder=2)
    axes[order].set_title(r"{}: Mean Steps to Success ($\leftarrow$)".format(title), fontsize=12)
    axes[order].set_xlabel(r"Index")
    axes[order].set_yticks(np.arange(len(labels)))
    axes[order].set_yticklabels(labels, ha='right', fontsize=10)
    axes[order].tick_params(axis='y', pad=10)
    axes[order].grid(True, color='white', axis="x", linestyle="--", alpha=0.5)
    axes[order].spines[['top', 'right', 'left']].set_visible(False)
    axes[order].set_facecolor("#e6ecf2")

    for ax in axes:
        for tick_label in ax.get_yticklabels():
            if tick_label.get_text() == "DOHJPPO":
                tick_label.set_fontweight("bold")
                tick_label.set_fontsize(12)

    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='both', which='both', labelsize=10)
   
    # plt.subplots_adjust(hspace=0.8)
    if tight_layout:
        plt.tight_layout(pad=pad, w_pad=w_pad, h_pad=h_pad)
    if sub_plots_adjust:
        plt.subplots_adjust(wspace=wspace, hspace=hspace)

    plt.savefig(f"eval/{config['EVAL_DIR']}/{config['NAME_TAG']}/{title}_success_speed_plot", dpi=300, bbox_inches="tight", pad_inches=0.1)

    return raa_scores_all

def save_traj(traj_batch, config, tag, sample_size=10):
    traj_data = {
        attr: getattr(traj_batch, attr)[:, :sample_size]
        for attr in dir(traj_batch)
        if not callable(getattr(traj_batch, attr)) and not attr.startswith("_") and not attr == 'info'
    }

    save_path = f"eval/{config['EVAL_DIR']}/{config['NAME_TAG']}/traj_sample/traj_{tag}" + ".npz"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        jnp.savez(f, **traj_data)
    print(f"Trajectory saved to {save_path}")

def load_traj(file_path):
    with jnp.load(file_path, allow_pickle=False) as traj_data:
        traj_batch = {key: traj_data[key] for key in traj_data.files}
    return traj_batch

def test_RAA(envs, env_paramss, config, rngs, saving_traj=False):
    rng_1, rng_2, rng_3, rng_4, rng_5, rng_6, rng_7, rng_8, rng_9, rng_10, rng_11, rng_12, rng_13, rng_14 = rngs

    (env_HJPPO, env_HJPPO_avoid,
        env_CPPO, env_RA,
        env_PPOLAG, env_PPO, env_RCPPO, env_RESPO,
        env_MORL, env_SPARSE,
        env_P2BPO, env_LOGBAR,
    ) = envs # COMPOSED (RAA) + 2 DECOMPOSED (R1 + R2)

    (env_params_HJPPO, env_params_HJPPO_avoid, env_params_CPPO,
        env_params_RA,
        env_params_PPOLAG, env_params_PPO, env_params_RCPPO, env_params_RESPO,
        env_params_MORL, env_params_SPARSE,
        env_params_P2BPO, env_params_LOGBAR
    ) = env_paramss

    # DEFINE ENV STEP WRAPPERS
    env_step_HJPPO = partial(_env_step_raa_vanilla, env_HJPPO, env_params_HJPPO)
    env_step_HJPPO_d = partial(_env_step_raa_vanilla_deterministic, env_HJPPO, env_params_HJPPO)
    env_step_CPPO= partial(_env_step_cppo_RAA, env_CPPO, env_params_CPPO)
    env_step_RA = partial(_env_step_ra_vanilla, env_RA, env_params_RA)
    env_step_PPOLAG = partial(_env_step_cppo_RAA, env_PPOLAG, env_params_PPOLAG)
    env_step_PPO = partial(_env_step_cppo_RAA, env_PPO, env_params_PPO)
    env_step_RCPPO = partial(_env_step_adapted_raa, env_RCPPO, env_params_RCPPO)
    env_step_RESPO = partial(_env_step_raa_respo, env_RESPO, env_params_RESPO)
    env_step_MORL = partial(_env_step_cppo_RAA, env_MORL, env_params_MORL)
    env_step_SPARSE = partial(_env_step_cppo_RAA, env_SPARSE, env_params_SPARSE)
    env_step_P2BPO = partial(_env_step_cppo_RAA, env_P2BPO, env_params_P2BPO)
    env_step_LOGBAR = partial(_env_step_cppo_RAA, env_LOGBAR, env_params_LOGBAR)
    tx = optimizer(config)

    ########################################## LOAD HJ-PPO #################################################

    raw_restored = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath('{}/{}/{}'.format(
        config["BASE_MODEL_DIR"], config["DIR_HJPPO"], config["DIR_MODEL_HJPPO"])), target=None)

    ## LOAD POLICY NETWORKS
    policy_network_HJPPO = Policy_Network(
        env_HJPPO.action_space(env_params_HJPPO).shape[0], activation=config["ACTIVATION"]
    )
    policy_network_HJPPO_avoid = Policy_Network(
        env_HJPPO_avoid.action_space(env_params_HJPPO_avoid).shape[0], activation=config["ACTIVATION"]
    )

    train_state_policy_HJPPO = TrainState.create(
        apply_fn=policy_network_HJPPO.apply,
        params=raw_restored['policy_network']['params'],
        tx=tx,
    )    
    train_state_policy_HJPPO_avoid = TrainState.create(
        apply_fn=policy_network_HJPPO_avoid.apply,
        params=raw_restored['policy_avoid_network']['params'],
        tx=tx,
    )

    ## LOAD VALUE NETWORKS
    value_network_HJPPO = Value_Network(activation=config["ACTIVATION"])
    train_state_value_HJPPO = TrainState.create(
        apply_fn=value_network_HJPPO.apply,
        params=raw_restored['value_network']['params'],
        tx=tx,
    )
    value_network_HJPPO_avoid = Value_Network(activation=config["ACTIVATION"])
    train_state_value_HJPPO_avoid = TrainState.create(
        apply_fn=value_network_HJPPO_avoid.apply,
        params=raw_restored['value_avoid_network']['params'],
        tx=tx,
    )
    ########################################## LOAD CPPO #################################################

    ## CPPO
    raw_restored_CPPO = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath('{}/{}/{}'.format(
        config["BASE_MODEL_DIR"], config["DIR_CPPO"], config["DIR_MODEL_CPPO"])), target=None)

    policy_network_CPPO = Policy_Network(
        env_CPPO.action_space(env_params_CPPO).shape[0], activation=config["ACTIVATION"]
    )

    train_state_policy_CPPO = TrainState.create(
        apply_fn=policy_network_CPPO.apply,
        params=raw_restored_CPPO['policy_network']['params'],
        tx=tx,
        # lambda_coef=0.,
    )

    value_network_CPPO = Value_Network(activation=config["ACTIVATION"])
    train_state_value_CPPO = TrainState.create(
        apply_fn=value_network_CPPO.apply,
        params=raw_restored_CPPO['value_network']['params'],
        tx=tx,
        # lambda_coef=0.,
    )

    value_network_cost_CPPO = Value_Network(activation=config["ACTIVATION"])
    train_state_cost_CPPO = TrainState.create(
        apply_fn=value_network_cost_CPPO.apply,
        params=raw_restored_CPPO['cost_network']['params'],
        tx=tx,
        # lambda_coef=0.,
    )

    ########################################## LOAD DECOMPOSED RA #################################################

    raw_restored_RA = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath('{}/{}/{}'.format(
        config["BASE_MODEL_DIR"], config["DIR_RA"], config["DIR_MODEL_RA"])), target=None)
    
    ## LOAD POLICY NETWORKS
    policy_network_RA = MoGPolicy_Network(
        env_RA.action_space(env_params_RA).shape[0], activation=config["ACTIVATION"]
    )

    train_state_policy_RA = TrainState.create(
        apply_fn=policy_network_RA.apply,
        params=raw_restored_RA['policy_network']['params'],
        tx=tx,
    )    

    ## LOAD VALUE NETWORKS
    value_network_RA = Value_Network(activation=config["ACTIVATION"])
    train_state_value_RA = TrainState.create(
        apply_fn=value_network_RA.apply,
        params=raw_restored_RA['value_network']['params'],
        tx=tx,
    )

    ########################################## LOAD PPOLAG & PPO #################################################

    ## PPOLAG
    raw_restored_PPOLAG = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath('{}/{}/{}'.format(
        config["BASE_MODEL_DIR"], config["DIR_PPOLAG"], config["DIR_MODEL_PPOLAG"])), target=None)
    
    policy_network_PPOLAG = Policy_Network(
        env_PPOLAG.action_space(env_params_PPOLAG).shape[0], activation=config["ACTIVATION"]
    )

    train_state_policy_PPOLAG = TrainState.create(
        apply_fn=policy_network_PPOLAG.apply,
        params=raw_restored_PPOLAG['policy_network']['params'],
        tx=tx,
        # lambda_coef=0.,
    )

    value_network_PPOLAG = Value_Network(activation=config["ACTIVATION"])
    train_state_value_PPOLAG = TrainState.create(
        apply_fn=value_network_PPOLAG.apply,
        params=raw_restored_PPOLAG['value_network']['params'],
        tx=tx,
        # lambda_coef=0.,
    )

    value_network_cost_PPOLAG = Value_Network(activation=config["ACTIVATION"])
    train_state_cost_PPOLAG = TrainState.create(
        apply_fn=value_network_cost_PPOLAG.apply,
        params=raw_restored_PPOLAG['cost_network']['params'],
        tx=tx,
        # lambda_coef=0.,
    )

    ## PPO
    raw_restored_PPO = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath('{}/{}/{}'.format(
        config["BASE_MODEL_DIR"], config["DIR_PPO"], config["DIR_MODEL_PPO"])), target=None)

    policy_network_PPO = Policy_Network(
        env_PPO.action_space(env_params_PPO).shape[0], activation=config["ACTIVATION"]
    )

    train_state_policy_PPO = TrainState.create(
        apply_fn=policy_network_PPO.apply,
        params=raw_restored_PPO['policy_network']['params'],
        tx=tx,
        # lambda_coef=0.,
    )

    value_network_PPO = Value_Network(activation=config["ACTIVATION"])
    train_state_value_PPO = TrainState.create(
        apply_fn=value_network_PPO.apply,
        params=raw_restored_PPO['value_network']['params'],
        tx=tx,
        # lambda_coef=0.,
    )

    value_network_cost_PPO = Value_Network(activation=config["ACTIVATION"])
    train_state_cost_PPO = TrainState.create(
        apply_fn=value_network_cost_PPO.apply,
        params=raw_restored_PPO['cost_network']['params'],
        tx=tx,
        # lambda_coef=0.,
    )

    ########################################## LOAD RCPPO & RESPO #################################################

    ## RCPPO
    raw_restored_RCPPO = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath('{}/{}/{}'.format(
        config["BASE_MODEL_DIR"], config["DIR_RCPPO"], config["DIR_MODEL_RCPPO"])), target=None)

    policy_network_RCPPO = Policy_Network(
        env_RCPPO.action_space(env_params_RCPPO).shape[0], activation=config["ACTIVATION"]
    )

    train_state_policy_RCPPO = TrainState.create(
        apply_fn=policy_network_RCPPO.apply,
        params=raw_restored_RCPPO['policy_network']['params'],
        tx=tx,
        # lambda_coef=0.,
    )

    value_network_energy_RCPPO = Value_Network(activation=config["ACTIVATION"])
    train_state_energy_RCPPO = TrainState.create(
        apply_fn=value_network_energy_RCPPO.apply,
        params=raw_restored_RCPPO['energy_network']['params'],
        tx=tx,
        # lambda_coef=0.,
    )

    value_network_h_RCPPO = Value_Network(activation=config["ACTIVATION"])
    train_state_h_RCPPO = TrainState.create(
        apply_fn=value_network_h_RCPPO.apply,
        params=raw_restored_RCPPO['reach_network']['params'],
        tx=tx,
        # lambda_coef=0.,
    )

    ## RESPO
    raw_restored_RESPO = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath('{}/{}/{}'.format(
        config["BASE_MODEL_DIR"], config["DIR_RESPO"], config["DIR_MODEL_RESPO"])), target=None)

    policy_network_RESPO = Policy_Network(
        env_RESPO.action_space(env_params_RESPO).shape[0], activation=config["ACTIVATION"]
    )

    train_state_policy_RESPO = TrainState.create(
        apply_fn=policy_network_RESPO.apply,
        params=raw_restored_RESPO['policy_network']['params'],
        tx=tx,
        # lambda_coef=0.,
    )

    value_network_RESPO = Value_Network(activation=config["ACTIVATION"])
    train_state_value_RESPO = TrainState.create(
        apply_fn=value_network_RESPO.apply,
        params=raw_restored_RESPO['value_network']['params'],
        tx=tx,
        # lambda_coef=0.,
    )

    value_network_prob_RESPO = Value_Network(activation=config["ACTIVATION"])
    train_state_prob_RESPO = TrainState.create(
        apply_fn=value_network_prob_RESPO.apply,
        params=raw_restored_RESPO['prob_network']['params'],
        tx=tx,
        # lambda_coef=0.,
    )

    value_network_cost_RESPO = Value_Network(activation=config["ACTIVATION"])
    train_state_cost_RESPO = TrainState.create(
        apply_fn=value_network_cost_RESPO.apply,
        params=raw_restored_RESPO['cost_network']['params'],
        tx=tx,
        # lambda_coef=0.,
    )

    ########################################## LOAD MORL & SPARSE #################################################

    ## MORL
    raw_restored_MORL = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath('{}/{}/{}'.format(
        config["BASE_MODEL_DIR"], config["DIR_MORL"], config["DIR_MODEL_MORL"])), target=None)

    policy_network_MORL = Policy_Network(
        env_MORL.action_space(env_params_MORL).shape[0], activation=config["ACTIVATION"]
    )

    train_state_policy_MORL = TrainState.create(
        apply_fn=policy_network_MORL.apply,
        params=raw_restored_MORL['policy_network']['params'],
        tx=tx,
        # lambda_coef=0.,
    )

    value_network_MORL = Value_Network(activation=config["ACTIVATION"])
    train_state_value_MORL = TrainState.create(
        apply_fn=value_network_MORL.apply,
        params=raw_restored_MORL['value_network']['params'],
        tx=tx,
        # lambda_coef=0.,
    )

    value_network_cost_MORL = Value_Network(activation=config["ACTIVATION"])
    train_state_cost_MORL = TrainState.create(
        apply_fn=value_network_cost_MORL.apply,
        params=raw_restored_MORL['cost_network']['params'],
        tx=tx,
        # lambda_coef=0.,
    )

    ## SPARSE
    raw_restored_SPARSE = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath('{}/{}/{}'.format(
        config["BASE_MODEL_DIR"], config["DIR_SPARSE"], config["DIR_MODEL_SPARSE"])), target=None)

    policy_network_SPARSE = Policy_Network(
        env_SPARSE.action_space(env_params_SPARSE).shape[0], activation=config["ACTIVATION"]
    )

    train_state_policy_SPARSE = TrainState.create(
        apply_fn=policy_network_SPARSE.apply,
        params=raw_restored_SPARSE['policy_network']['params'],
        tx=tx,
        # lambda_coef=0.,
    )

    value_network_SPARSE = Value_Network(activation=config["ACTIVATION"])
    train_state_value_SPARSE = TrainState.create(
        apply_fn=value_network_SPARSE.apply,
        params=raw_restored_SPARSE['value_network']['params'],
        tx=tx,
        # lambda_coef=0.,
    )

    value_network_cost_SPARSE = Value_Network(activation=config["ACTIVATION"])
    train_state_cost_SPARSE = TrainState.create(
        apply_fn=value_network_cost_SPARSE.apply,
        params=raw_restored_SPARSE['cost_network']['params'],
        tx=tx,
        # lambda_coef=0.,
    )

    ########################################## LOAD P2BPO & LOGBAR #################################################

    ## P2BPO
    raw_restored_P2BPO = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath('{}/{}/{}'.format(
        config["BASE_MODEL_DIR"], config["DIR_P2BPO"], config["DIR_MODEL_P2BPO"])), target=None)

    policy_network_P2BPO = Policy_Network(
        env_P2BPO.action_space(env_params_P2BPO).shape[0], activation=config["ACTIVATION"]
    )

    train_state_policy_P2BPO = TrainState.create(
        apply_fn=policy_network_P2BPO.apply,
        params=raw_restored_P2BPO['policy_network']['params'],
        tx=tx,
        # lambda_coef=0.,
    )

    value_network_P2BPO = Value_Network(activation=config["ACTIVATION"])
    train_state_value_P2BPO = TrainState.create(
        apply_fn=value_network_P2BPO.apply,
        params=raw_restored_P2BPO['value_network']['params'],
        tx=tx,
        # lambda_coef=0.,
    )

    value_network_cost_P2BPO = Value_Network(activation=config["ACTIVATION"])
    train_state_cost_P2BPO = TrainState.create(
        apply_fn=value_network_cost_P2BPO.apply,
        params=raw_restored_P2BPO['cost_network']['params'],
        tx=tx,
        # lambda_coef=0.,
    )

    ## LOGBAR
    raw_restored_LOGBAR = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath('{}/{}/{}'.format(
        config["BASE_MODEL_DIR"], config["DIR_LOGBAR"], config["DIR_MODEL_LOGBAR"])), target=None)

    policy_network_LOGBAR = Policy_Network(
        env_LOGBAR.action_space(env_params_LOGBAR).shape[0], activation=config["ACTIVATION"]
    )

    train_state_policy_LOGBAR = TrainState.create(
        apply_fn=policy_network_LOGBAR.apply,
        params=raw_restored_LOGBAR['policy_network']['params'],
        tx=tx,
        # lambda_coef=0.,
    )

    value_network_LOGBAR = Value_Network(activation=config["ACTIVATION"])
    train_state_value_LOGBAR = TrainState.create(
        apply_fn=value_network_LOGBAR.apply,
        params=raw_restored_LOGBAR['value_network']['params'],
        tx=tx,
        # lambda_coef=0.,
    )

    value_network_cost_LOGBAR = Value_Network(activation=config["ACTIVATION"])
    train_state_cost_LOGBAR = TrainState.create(
        apply_fn=value_network_cost_LOGBAR.apply,
        params=raw_restored_LOGBAR['cost_network']['params'],
        tx=tx,
        # lambda_coef=0.,
    )


    ########################################## ROLL OUT MODELS #################################################

    ## MODEL 1 : HJ-PPO : STOCHASTIC
    
    print("Rolling Out HJ-PPO (Stochastic)")
    rng_1, _rng_1 = jax.random.split(rng_1)
    reset_rng_1 = jax.random.split(_rng_1, config["NUM_ENVS"])
    obsv_1, env_state_1 = jax.vmap(env_HJPPO.reset, in_axes=(0, None))(reset_rng_1, env_params_HJPPO)
    
    rng_1, _rng_1 = jax.random.split(rng_1)
    runner_state_standard = (train_state_policy_HJPPO, train_state_value_HJPPO, env_state_1, obsv_1, _rng_1)
    
    decomposed_state = (train_state_policy_HJPPO_avoid, train_state_value_HJPPO_avoid)
    policy_controls = (False, False)
    runner_state = (*runner_state_standard, decomposed_state, policy_controls)

    runner_state, traj_batch_HJPPO = jax.lax.scan(
        env_step_HJPPO, runner_state, None, config["NUM_STEPS"]
    )

    if saving_traj: save_traj(traj_batch_HJPPO, config, 'DOHJPPO_s', sample_size=10)

    ## MODEL 2 : HJ-PPO : DETERMINISTIC

    print("Rolling Out HJ-PPO (Deterministic)")
    rng_2, _rng_2 = jax.random.split(rng_2)
    reset_rng_2 = jax.random.split(_rng_2, config["NUM_ENVS"])
    obsv_2, env_state_2 = jax.vmap(env_HJPPO.reset, in_axes=(0, None))(reset_rng_2, env_params_HJPPO)

    rng_2, _rng_2 = jax.random.split(rng_2)
    runner_state_standard = (train_state_policy_HJPPO, train_state_value_HJPPO, env_state_2, obsv_2, _rng_2)
    
    decomposed_state = (train_state_policy_HJPPO_avoid, train_state_value_HJPPO_avoid)
    policy_controls = (False, False,)
    runner_state = (*runner_state_standard, decomposed_state, policy_controls)

    runner_state, traj_batch_HJPPO_d = jax.lax.scan(
        env_step_HJPPO_d, runner_state, None, config["NUM_STEPS"]
    )

    if saving_traj: save_traj(traj_batch_HJPPO_d, config, 'DOHJPPO', sample_size=10)

    ## MODEL 3 : CPPO : Variant 1

    print("Rolling Out C-PPO (Variant 1)")
    rng_3, _rng_3 = jax.random.split(rng_3)
    reset_rng_3 = jax.random.split(_rng_3, config["NUM_ENVS"])
    obsv_3, env_state_3 = jax.vmap(env_CPPO.reset, in_axes=(0, None))(reset_rng_3, env_params_CPPO)

    rng_3, _rng_3 = jax.random.split(rng_3)
    runner_state = (train_state_policy_CPPO, train_state_value_CPPO, train_state_cost_CPPO, env_state_3, obsv_3, _rng_3)
    
    runner_state, traj_batch_CPPO = jax.lax.scan(
        env_step_CPPO, runner_state, None, config["NUM_STEPS"]
    )

    if saving_traj: save_traj(traj_batch_CPPO, config, 'CPPO', sample_size=10)

    ## MODEL 6 : RA
    print("Rolling Out RA (Variant 1)")
    rng_4, _rng_4 = jax.random.split(rng_4)
    reset_rng_4 = jax.random.split(_rng_4, config["NUM_ENVS"])
    obsv_4, env_state_4 = jax.vmap(env_RA.reset, in_axes=(0, None))(reset_rng_4, env_params_RA)
    rng_4, _rng_4 = jax.random.split(rng_4)
    runner_state = (train_state_policy_RA, train_state_value_RA, 
                    env_state_4, obsv_4, _rng_4)

    runner_state, traj_batch_RA = jax.lax.scan(
        env_step_RA, runner_state, None, config["NUM_STEPS"]
    )

    if saving_traj: save_traj(traj_batch_RA, config, 'RA', sample_size=10)

        ## MODEL 7 : PPOLAG

    print("Rolling Out PPO-LAG")
    rng_7, _rng_7 = jax.random.split(rng_7)
    reset_rng_7 = jax.random.split(_rng_7, config["NUM_ENVS"])
    obsv_7, env_state_7 = jax.vmap(env_PPOLAG.reset, in_axes=(0, None))(reset_rng_7, env_params_PPOLAG)

    rng_7, _rng_7 = jax.random.split(rng_7)
    runner_state = (train_state_policy_PPOLAG, train_state_value_PPOLAG, train_state_cost_PPOLAG, env_state_7, obsv_7, _rng_7)

    runner_state, traj_batch_PPOLAG = jax.lax.scan(
        env_step_PPOLAG, runner_state, None, config["NUM_STEPS"]
    )

    if saving_traj: save_traj(traj_batch_PPOLAG, config, 'PPOLAG', sample_size=10)

    ## MODEL 8 : PPO

    print("Rolling Out PPO")
    rng_8, _rng_8 = jax.random.split(rng_8)
    reset_rng_8 = jax.random.split(_rng_8, config["NUM_ENVS"])
    obsv_8, env_state_8 = jax.vmap(env_PPO.reset, in_axes=(0, None))(reset_rng_8, env_params_PPO)

    rng_8, _rng_8 = jax.random.split(rng_8)
    runner_state = (train_state_policy_PPO, train_state_value_PPO, train_state_cost_PPO, env_state_8, obsv_8, _rng_8)

    runner_state, traj_batch_PPO = jax.lax.scan(
        env_step_PPO, runner_state, None, config["NUM_STEPS"]
    )

    if saving_traj: save_traj(traj_batch_PPO, config, 'PPO', sample_size=10)

    ## MODEL 8 : RCPPO

    print("Rolling Out RC-PPO")
    rng_9, _rng_9 = jax.random.split(rng_9)
    reset_rng_9 = jax.random.split(_rng_9, config["NUM_ENVS"])
    obsv_9, env_state_9 = jax.vmap(env_RCPPO.reset, in_axes=(0, None))(reset_rng_9, env_params_RCPPO)

    rng_9, _rng_9 = jax.random.split(rng_9)
    runner_state = (train_state_policy_RCPPO, train_state_energy_RCPPO, train_state_h_RCPPO, env_state_9, obsv_9, _rng_9)

    runner_state, traj_batch_RCPPO = jax.lax.scan(
        env_step_RCPPO, runner_state, None, config["NUM_STEPS"]
    )

    if saving_traj: save_traj(traj_batch_RCPPO, config, 'RCPPO', sample_size=10)

    ## MODEL 8 : RESPO

    print("Rolling Out RESPO")
    rng_10, _rng_10 = jax.random.split(rng_10)
    reset_rng_10 = jax.random.split(_rng_10, config["NUM_ENVS"])
    obsv_10, env_state_10 = jax.vmap(env_RESPO.reset, in_axes=(0, None))(reset_rng_10, env_params_RESPO)

    rng_10, _rng_10 = jax.random.split(rng_10)
    runner_state = (train_state_policy_RESPO, train_state_value_RESPO, train_state_prob_RESPO, train_state_cost_RESPO, env_state_10, obsv_10, _rng_10)

    runner_state, traj_batch_RESPO = jax.lax.scan(
        env_step_RESPO, runner_state, None, config["NUM_STEPS"]
    )

    if saving_traj: save_traj(traj_batch_RESPO, config, 'RESPO', sample_size=10)

    ## MODEL 9 : MORL

    print("Rolling Out MORL")
    rng_11, _rng_11 = jax.random.split(rng_11)
    reset_rng_11 = jax.random.split(_rng_11, config["NUM_ENVS"])
    obsv_11, env_state_11 = jax.vmap(env_MORL.reset, in_axes=(0, None))(reset_rng_11, env_params_MORL)

    rng_11, _rng_11 = jax.random.split(rng_11)
    runner_state = (train_state_policy_MORL, train_state_value_MORL, train_state_cost_MORL, env_state_11, obsv_11, _rng_11)

    runner_state, traj_batch_MORL = jax.lax.scan(
        env_step_MORL, runner_state, None, config["NUM_STEPS"]
    )

    if saving_traj: save_traj(traj_batch_MORL, config, 'MORL', sample_size=10)

    ## MODEL 10 : SPARSE

    print("Rolling Out SPARSE")
    rng_12, _rng_12 = jax.random.split(rng_12)
    reset_rng_12 = jax.random.split(_rng_12, config["NUM_ENVS"])
    obsv_12, env_state_12 = jax.vmap(env_SPARSE.reset, in_axes=(0, None))(reset_rng_12, env_params_SPARSE)

    rng_12, _rng_12 = jax.random.split(rng_12)
    runner_state = (train_state_policy_SPARSE, train_state_value_SPARSE, train_state_cost_SPARSE, env_state_12, obsv_12, _rng_12)

    runner_state, traj_batch_SPARSE = jax.lax.scan(
        env_step_SPARSE, runner_state, None, config["NUM_STEPS"]
    )

    if saving_traj: save_traj(traj_batch_SPARSE, config, 'SPARSE', sample_size=10)

    ## MODEL 11 : P2BPO

    print("Rolling Out P2BPO")
    rng_13, _rng_13 = jax.random.split(rng_13)
    reset_rng_13 = jax.random.split(_rng_13, config["NUM_ENVS"])
    obsv_13, env_state_13 = jax.vmap(env_P2BPO.reset, in_axes=(0, None))(reset_rng_13, env_params_P2BPO)

    rng_13, _rng_13 = jax.random.split(rng_13)
    runner_state = (train_state_policy_P2BPO, train_state_value_P2BPO, train_state_cost_P2BPO, env_state_13, obsv_13, _rng_13)

    runner_state, traj_batch_P2BPO = jax.lax.scan(
        env_step_P2BPO, runner_state, None, config["NUM_STEPS"]
    )

    if saving_traj: save_traj(traj_batch_P2BPO, config, 'P2BPO', sample_size=10)

    ## MODEL 12 : LOGBAR

    print("Rolling Out LOGBAR")
    rng_14, _rng_14 = jax.random.split(rng_14)
    reset_rng_14 = jax.random.split(_rng_14, config["NUM_ENVS"])
    obsv_14, env_state_14 = jax.vmap(env_LOGBAR.reset, in_axes=(0, None))(reset_rng_14, env_params_LOGBAR)

    rng_14, _rng_14 = jax.random.split(rng_14)
    runner_state = (train_state_policy_LOGBAR, train_state_value_LOGBAR, train_state_cost_LOGBAR, env_state_14, obsv_14, _rng_14)

    runner_state, traj_batch_LOGBAR = jax.lax.scan(
        env_step_LOGBAR, runner_state, None, config["NUM_STEPS"]
    )

    if saving_traj: save_traj(traj_batch_LOGBAR, config, 'LOGBAR', sample_size=10)

    return (traj_batch_HJPPO, traj_batch_HJPPO_d, 
            traj_batch_CPPO, traj_batch_RA,
            traj_batch_PPOLAG, traj_batch_PPO, 
            traj_batch_RCPPO, traj_batch_RESPO, 
            traj_batch_MORL, traj_batch_SPARSE,
            traj_batch_P2BPO, traj_batch_LOGBAR)
