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

def plot_scores(traj_batches, config):

    (traj_batch_HJPPO, 
        traj_batch_HJPPO_d, 
        traj_batch_CPPO, 
        traj_batch_RA,
        traj_batch_PPOLAG,
        traj_batch_PPO,
        traj_batch_RCPPO,
        traj_batch_RESPO,
        traj_batch_MORL,
        traj_batch_SPARSE
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

    # (avoid perc, reach avoid idx)
    # raa_scores_HJPPO = (raa_scores_HJPPO[0][-1], raa_scores_HJPPO[1][-1])
    # raa_scores_HJPPO_d = (raa_scores_HJPPO_d[0][-1], raa_scores_HJPPO_d[1][-1])
    # raa_scores_CPPO = (raa_scores_CPPO[0][-1], raa_scores_CPPO[1][-1])
    # raa_scores_RA = (raa_scores_RA[0][-1], raa_scores_RA[1][-1])

    raa_scores_all = [
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

    # Plotting
    fig, axes = plt.subplots(4, 1, figsize=(12, 4.5), sharex=False)
    palette = sns.color_palette("deep", n_colors=len(raa_scores_all))[::-1]
    colors = {label: color for label, color in zip(labels, palette)}
        
    # Reach percentage bar plot
    for i, label in enumerate(labels):
        axes[0].barh(label, reach_avoid_percs[i], color=colors[label])
    axes[0].set_xlim(0, 1.)
    axes[0].set_title(r"HOPPER-RAA: Success Percentage ($\uparrow$)", fontsize=12)
    axes[0].set_xlabel(r"Percentage")
    axes[0].set_yticks(np.arange(len(labels)))
    axes[0].set_yticklabels(labels, ha='right', fontsize=10)
    axes[0].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[r'$0\%$', r'$20\%$', r'$40\%$', r'$60\%$', r'$80\%$', r'$100\%$'])
    axes[0].tick_params(axis='y', pad=10)  # move labels away from bars
    axes[0].grid(True, color='white', axis="x", linestyle="--", alpha=0.5)
    axes[0].spines[['top', 'right', 'left']].set_visible(False)
    axes[0].set_facecolor("#e6ecf2")
    for tick_label in axes[0].get_yticklabels():
        if tick_label.get_text() == "DOHJPPO":
            tick_label.set_fontweight("bold")
            tick_label.set_fontsize(12)

    # Mean reach index bar plot with error bars
    for i, label in enumerate(labels):
        axes[1].barh(label, mean_idxs[i], xerr=std_idxs[i]/2, color=colors[label], capsize=4)
    axes[1].set_title(r"HOPPER-RAA: Mean Steps to Success ($\downarrow$)", fontsize=12)
    axes[1].set_xlabel(r"Index")
    axes[1].set_yticks(np.arange(len(labels)))
    axes[1].set_yticklabels(labels, ha='right', fontsize=10)
    axes[1].tick_params(axis='y', pad=10)
    axes[1].grid(True, color='white', axis="x", linestyle="--", alpha=0.5)
    axes[1].spines[['top', 'right', 'left']].set_visible(False)
    axes[1].set_facecolor("#e6ecf2")
    for tick_label in axes[1].get_yticklabels():
        if tick_label.get_text() == "DOHJPPO":
            tick_label.set_fontweight("bold")
            tick_label.set_fontsize(12)

    # Mean optimal value bar plot with error bars
    for i, label in enumerate(labels):
        axes[2].barh(label, -min_val_means[i], xerr=min_val_stds[i]/2, color=colors[label], capsize=4)
    axes[2].set_title(r"HOPPER-RAA: Maximum Value ($\uparrow$)", fontsize=12)
    axes[2].set_xlabel(r"Value")
    axes[2].set_yticks(np.arange(len(labels)))
    axes[2].set_yticklabels(labels, ha='right', fontsize=10)
    axes[2].tick_params(axis='y', pad=10)
    axes[2].grid(True, color='white', axis="x", linestyle="--", alpha=0.5)
    axes[2].spines[['top', 'right', 'left']].set_visible(False)
    axes[2].set_facecolor("#e6ecf2")
    for tick_label in axes[2].get_yticklabels():
        if tick_label.get_text() == "DOHJPPO":
            tick_label.set_fontweight("bold")
            tick_label.set_fontsize(12)

    # Reach At least onepercentage bar plot
    for i, label in enumerate(labels):
        axes[3].barh(label, reach_or_avoid_percs[i], color=colors[label])
    # axes[0].set_xlim(0.7, 1.0)
    axes[3].set_title(r"HOPPER-RAA: Reached or Avoided Percentage ($\uparrow$)", fontsize=12)
    axes[3].set_xlabel(r"Percentage")
    axes[3].set_yticks(np.arange(len(labels)))
    axes[3].set_yticklabels(labels, ha='right', fontsize=10)
    # axes[0].set_xticklabels([r'0\%', r'20\%', r'40\%', r'60\%', r'80\%', r'100\%'])
    # axes[0].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[r'$0\%$', r'$20\%$', r'$40\%$', r'$60\%$', r'$80\%$', r'$100\%$'])
    # axes[0].set_xticks([0.75, 0.8, 0.85, 0.9, 0.95, 1.0], labels=[r'$75\%$', r'$80\%$', r'$85\%$', r'$90\%$', r'$95\%$', r'$100\%$'])
    axes[3].tick_params(axis='y', pad=10)  # move labels away from bars
    axes[3].grid(True, color='white', axis="x", linestyle="--", alpha=0.5)
    axes[3].spines[['top', 'right', 'left']].set_visible(False)
    axes[3].set_facecolor("#e6ecf2")
    for tick_label in axes[3].get_yticklabels():
        if tick_label.get_text() == "DOHJPPO":
            tick_label.set_fontweight("bold")
            tick_label.set_fontsize(12)

    # Style tweaks to match NeurIPS-style
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='both', which='both', labelsize=10)
   
    plt.subplots_adjust(hspace=0.8)

    plt.savefig(f"model/{config['TEST_DIR']}/{config['NAME_TAG']}/score_plot", dpi=300, bbox_inches="tight", pad_inches=0.1)
    return fig

def save_traj(traj_batch, config, tag, sample_size=5):
    traj_data = {
        attr: getattr(traj_batch, attr)[:, :sample_size]
        for attr in dir(traj_batch)
        if not callable(getattr(traj_batch, attr)) and not attr.startswith("_") and not attr == 'info'
    }

    save_path = f"model/{config['TEST_DIR']}/{config['NAME_TAG']}/traj_sample/traj_{tag}" + ".npz"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        jnp.savez(f, **traj_data)
    print(f"Trajectory saved to {save_path}")

def load_traj(file_path):
    with jnp.load(file_path, allow_pickle=False) as traj_data:
        traj_batch = {key: traj_data[key] for key in traj_data.files}
    return traj_batch

def test(envs, env_paramss, config, rngs, saving_traj=False):
    rng_1, rng_2, rng_3, rng_4, rng_5, rng_6, rng_7, rng_8, rng_9, rng_10, rng_11, rng_12 = rngs

    (env_HJPPO, env_HJPPO_avoid,
        env_CPPO, env_RA,
        env_PPOLAG, env_PPO, env_RCPPO, env_RESPO,
        env_MORL, env_SPARSE
    ) = envs # COMPOSED (RAA) + 2 DECOMPOSED (R1 + R2)

    env_params_HJPPO, env_params_HJPPO_avoid, env_params_CPPO, env_params_RA = env_paramss
    (env_params_HJPPO, env_params_HJPPO_avoid, env_params_CPPO,
        env_params_RA,
        env_params_PPOLAG, env_params_PPO, env_params_RCPPO, env_params_RESPO,
        env_params_MORL, env_params_SPARSE
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
    tx = optimizer(config)

    ########################################## LOAD HJ-PPO #################################################

    raw_restored = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath('model/{}/{}'.format(
        config["DIR_HJPPO"], config["DIR_MODEL_HJPPO"])), target=None)

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
    raw_restored_CPPO = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath('model/{}/{}'.format(
        config["DIR_CPPO"], config["DIR_MODEL_CPPO"])), target=None)

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

    ########################################## LOAD DECOMPOSED STL #################################################

    raw_restored_RA = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath('model/{}/{}'.format(
        config["DIR_RA"], config["DIR_MODEL_RA"])), target=None)
    
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

    if saving_traj: save_traj(traj_batch_HJPPO_d, config, 'DOHJPPO', sample_size=5)

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

    if saving_traj: save_traj(traj_batch_CPPO, config, 'CPPO', sample_size=5)

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

    if saving_traj: save_traj(traj_batch_RA, config, 'RA', sample_size=5)

    return traj_batch_HJPPO, traj_batch_HJPPO_d, traj_batch_CPPO, traj_batch_RA

if __name__ == "__main__":
    config = vars(get_args(sys.argv[1:]))

    debug = True
    if debug:
        config["EXP_NAME"]="HopperReachAlwaysAvoid"

        # config["DIR_HJPPO"]="hopper_raa_final_nikhil"#"stonkens_models/hopper_reachalwaysavoid"
        # config["DIR_MODEL_HJPPO"]="checkpoint_160"#"best_126" # FIXME NIKHIL'S CODE, but where is checkpoint_160?
        config["DIR_HJPPO"]="BASELINE_hopper_raa_final_nikhil"#"stonkens_models/hopper_reachalwaysavoid"
        config["DIR_MODEL_HJPPO"]="checkpoint_242"#"best_126"

        config["DIR_CPPO"]="BASELINE_hopper_raa_cppo_0_val100_dones_augmented"
        config["DIR_MODEL_CPPO"]="checkpoint_486"

        config["DIR_RA"]="BASELINE_hopper_reachavoid_final_nikhil" #"stonkens_models/hopper_reachavoid_final"
        config["DIR_MODEL_RA"]="checkpoint_240" #"best_244"

        print(os.path.abspath('model/{}/{}'.format(config["DIR_RA"], config["DIR_MODEL_RA"])))
        print(os.path.exists(os.path.abspath('model/{}/{}'.format(config["DIR_RA"], config["DIR_MODEL_RA"]))))

        config['TEST_DIR'] = "eval_all_figs"
        config['NAME_TAG'] = "Hopper_RAA_061925"

    config["NUM_ENVS"]=1000
    config["NUM_STEPS"]=500
    config["ACTIVATION"]="tanh"

    envs_HJPPO = get_env(config)
    env_HJPPO, env_HJPPO_avoid = envs_HJPPO

    # "BASELINE_final_hopper_rr_cppomax_raccum_cfnmax_caccum_umin_V1--LR=3e-4"
    config_CPPO = copy.deepcopy(config)
    config_CPPO["EXP_NAME"] = "HopperReachAlwaysAvoid_CPPO"
    # config_CPPO["ENV_REWARD_TYPE"] = "accumulated" # reward
    # config_CPPO["ENV_COST_FN"] = "max" # cost_fn
    # config_CPPO["ENV_COST_TYPE"] = "accumulated" # cost
    # config_CPPO["CPPO_UPDATE_TYPE"] = "min" # update
    # config_CPPO["USE_STL"] = False # stl 
    env_CPPO = get_env(config_CPPO)

    config_RA = copy.deepcopy(config)
    config_RA["EXP_NAME"] = "HopperReachAvoid"
    env_RA = get_env(config_RA)

    envs = (
        env_HJPPO, env_HJPPO_avoid, 
        env_CPPO, env_RA
    )
    env_paramss = (
        env_HJPPO.default_params, env_HJPPO_avoid.default_params, env_CPPO.default_params, env_RA.default_params 
    )

    rng_1 = jax.random.PRNGKey(20)
    rng_2 = jax.random.PRNGKey(20)
    rng_3 = jax.random.PRNGKey(20)
    rng_4 = jax.random.PRNGKey(20)
    rngs = (rng_1, rng_2, rng_3, rng_4)
    # folder = os.path.exists("model/{}/traj".format(config['DIR']))

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    print("\n\nCollecting Trajectories")
    traj_batches = test(envs, env_paramss, config, rngs, saving_traj=True)

    os.makedirs(f"model/{config['TEST_DIR']}/{config['NAME_TAG']}", exist_ok=True)
    # val_fig = plot_RR_value(result_traj_batch, result_traj_batch_deterministic, config)
    # traj_fig = plot_traj_sample(result_traj_batch, result_traj_batch_deterministic, config, sample_size=5, make_video=False)

    score_plot = plot_scores(traj_batches, config)
