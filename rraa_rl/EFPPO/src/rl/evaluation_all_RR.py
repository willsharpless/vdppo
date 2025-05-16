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
from rraa_rl.EFPPO.src.rl.EFPPO_utils import _env_step_rr_vanilla, _env_step_rr_deterministic, _env_step_cppo_RR, _env_step_rr_decomposed
# from rraa_rl.EFPPO.src.rl.plot_utils import calculate_reachreach
from rraa_rl.EFPPO.src.rl.root_finding import Bisection
from rraa_rl.EFPPO.src.rl.utils import tree_index1, tree_index2, optimizer

def calculate_reachreach(traj_batch, reach_type="both"):
    
    # Compute first reaching idx
    reach_idx_1 = (traj_batch.reach1 < 0).argmax(axis=0) if reach_type in ["both", "1"] else None
    reach_idx_2 = (traj_batch.reach2 < 0).argmax(axis=0) if reach_type in ["both", "2"] else None
    reach_idx_1 = jnp.where(jnp.any((traj_batch.reach1 < 0) == 1, axis=0), reach_idx_1, jnp.inf) if reach_type in ["both", "1"] else None
    reach_idx_2 = jnp.where(jnp.any((traj_batch.reach2 < 0) == 1, axis=0), reach_idx_2, jnp.inf) if reach_type in ["both", "2"] else None
    reach_idx = jnp.maximum(reach_idx_1, reach_idx_2) if reach_type in ["both"] else None

    # Compute
    reach_1_perc = (reach_idx_1 < jnp.inf).sum() / reach_idx_1.__len__() if reach_type in ["both", "1"] else None
    reach_2_perc = (reach_idx_2 < jnp.inf).sum() / reach_idx_2.__len__() if reach_type in ["both", "2"] else None
    reach_perc = (reach_idx < jnp.inf).sum() / reach_idx.__len__() if reach_type in ["both"] else None

    reach_percs = (reach_1_perc.item(), reach_2_perc.item(), reach_perc.item())
    reach_idxs = (reach_idx_1, reach_idx_2, reach_idx)
    return reach_percs, reach_idxs

def plot_scores(traj_batches, config):

    (traj_batch_HJPPO, 
        traj_batch_HJPPO_d, 
        traj_batch_CPPOv1, 
        traj_batch_CPPOv2, 
        traj_batch_CPPOv3, 
        traj_batch_dSTL) = traj_batches
    
    rr_scores_HJPPO = calculate_reachreach(traj_batch_HJPPO)
    rr_scores_HJPPO_d = calculate_reachreach(traj_batch_HJPPO_d)
    rr_scores_CPPOv1 = calculate_reachreach(traj_batch_CPPOv1)
    rr_scores_CPPOv2 = calculate_reachreach(traj_batch_CPPOv2)
    rr_scores_CPPOv3 = calculate_reachreach(traj_batch_CPPOv3)
    rr_scores_dSTL = calculate_reachreach(traj_batch_dSTL)

    rr_scores_all = [
        ("HJPPO", rr_scores_HJPPO),
        ("HJPPO_d", rr_scores_HJPPO_d),
        ("CPPOv1", rr_scores_CPPOv1),
        ("CPPOv2", rr_scores_CPPOv2),
        ("CPPOv3", rr_scores_CPPOv3),
        ("dSTL", rr_scores_dSTL),
    ]

    # Extract data
    labels = []
    reach_percs = []
    mean_idxs = []
    std_idxs = []

    for tag, scores in rr_scores_all:
        
        reach_perc = scores[0][2]
        idxs = scores[1][2]
        finite_mask = jnp.isfinite(idxs)
        finite_idxs = idxs[finite_mask]
        mean_idx = jnp.mean(finite_idxs) if finite_idxs.size > 0 else jnp.nan
        std_idx = jnp.std(finite_idxs) if finite_idxs.size > 0 else jnp.nan

        labels.append(tag)
        reach_percs.append(reach_perc)
        mean_idxs.append(mean_idx)
        std_idxs.append(std_idx)

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(7, 4.5), sharex=False)
    palette = sns.color_palette("deep", n_colors=6)
    colors = {label: color for label, color in zip(labels, palette)}

    # Reach percentage bar plot
    for i, label in enumerate(labels):
        axes[0].barh(label, reach_percs[i], color=colors[label])
    axes[0].set_xlim(0, 1.1)
    axes[0].set_title(r"Success Percentage", fontsize=12)
    axes[0].set_xlabel(r"Percentage")
    axes[0].set_yticks(np.arange(len(labels)))
    axes[0].set_yticklabels(labels, ha='right', fontsize=10)
    axes[0].tick_params(axis='y', pad=10)  # move labels away from bars
    axes[0].grid(True, axis="x", linestyle="--", alpha=0.5)
    axes[0].spines[['top', 'right', 'left']].set_visible(False)

    # Mean reach index bar plot with error bars
    for i, label in enumerate(labels):
        axes[1].barh(label, mean_idxs[i], xerr=std_idxs[i], color=colors[label], capsize=4)
    axes[1].set_title(r"Mean Steps to Success", fontsize=12)
    axes[1].set_xlabel(r"Index")
    axes[1].set_yticks(np.arange(len(labels)))
    axes[1].set_yticklabels(labels, ha='right', fontsize=10)
    axes[1].tick_params(axis='y', pad=10)
    axes[1].grid(True, axis="x", linestyle="--", alpha=0.5)
    axes[1].spines[['top', 'right', 'left']].set_visible(False)

    # Style tweaks to match NeurIPS-style
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='both', which='both', labelsize=10)

    plt.savefig(f"model/{config['TEST_DIR']}/score_plot", dpi=300)
    return fig

def test(envs, env_paramss, config, rngs):
    rng_1, rng_2, rng_3, rng_4, rng_5, rng_6 = rngs

    env_HJPPO, env_HJPPO_reach_1, env_HJPPO_reach_2, env_CPPO_max, env_CPPO_sum, env_dSTL, env_dSTL_1, env_dSTL_2 = envs # COMPOSED (RR) + 2 DECOMPOSED (R1 + R2)
    env_params_HJPPO, env_params_HJPPO_reach_1, env_params_HJPPO_reach_2, env_params_CPPO, env_params_dSTL, env_params_dSTL_1, env_params_dSTL_2 = env_paramss

    # DEFINE ENV STEP WRAPPERS
    env_step_HJPPO = partial(_env_step_rr_vanilla, env_HJPPO, env_params_HJPPO)
    env_step_HJPPO_d = partial(_env_step_rr_deterministic, env_HJPPO, env_params_HJPPO)
    env_step_CPPO_max = partial(_env_step_cppo_RR, env_CPPO_max, env_params_CPPO)
    env_step_CPPO_sum = partial(_env_step_cppo_RR, env_CPPO_sum, env_params_CPPO)
    env_step_dSTL = partial(_env_step_rr_decomposed, env_dSTL, env_params_dSTL)
    tx = optimizer(config)

    ########################################## LOAD HJ-PPO #################################################

    raw_restored = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath('model/{}/{}'.format(
        config["DIR_HJPPO"], config["DIR_MODEL_HJPPO"])), target=None)
    
    ## LOAD POLICY NETWORKS
    policy_network_HJPPO = MoGPolicy_Network(
        env_HJPPO.action_space(env_params_HJPPO).shape[0], activation=config["ACTIVATION"]
    )
    policy_network_HJPPO_reach1 = Policy_Network(
        env_HJPPO_reach_1.action_space(env_params_HJPPO_reach_1).shape[0], activation=config["ACTIVATION"]
    )
    policy_network_HJPPO_reach2 = Policy_Network(
        env_HJPPO_reach_2.action_space(env_params_HJPPO_reach_2).shape[0], activation=config["ACTIVATION"]
    )

    train_state_policy_HJPPO = TrainState.create(
        apply_fn=policy_network_HJPPO.apply,
        params=raw_restored['policy_network']['params'],
        tx=tx,
    )    
    train_state_policy_HJPPO_reach1 = TrainState.create(
        apply_fn=policy_network_HJPPO_reach1.apply,
        params=raw_restored['policy_reach1_network']['params'],
        tx=tx,
    )
    train_state_policy_HJPPO_reach2 = TrainState.create(
        apply_fn=policy_network_HJPPO_reach2.apply,
        params=raw_restored['policy_reach2_network']['params'],
        tx=tx,
    )

    ## LOAD VALUE NETWORKS
    value_network_HJPPO = Value_Network(activation=config["ACTIVATION"])
    train_state_value_HJPPO = TrainState.create(
        apply_fn=value_network_HJPPO.apply,
        params=raw_restored['value_network']['params'],
        tx=tx,
    )
    value_network_HJPPO_reach1 = Value_Network(activation=config["ACTIVATION"])
    train_state_value_HJPPO_reach1 = TrainState.create(
        apply_fn=value_network_HJPPO_reach1.apply,
        params=raw_restored['value_reach1_network']['params'],
        tx=tx,
    )
    value_network_HJPPO_reach2 = Value_Network(activation=config["ACTIVATION"])
    train_state_value_HJPPO_reach2 = TrainState.create(
        apply_fn=value_network_HJPPO_reach2.apply,
        params=raw_restored['value_reach2_network']['params'],
        tx=tx,
    )

    ########################################## LOAD CPPO #################################################

    ## CPO v1
    raw_restored_CPPOv1 = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath('model/{}/{}'.format(
        config["DIR_CPPOv1"], config["DIR_MODEL_CPPOv1"])), target=None)

    policy_network_CPPOv1 = Policy_Network(
        env_CPPO_max.action_space(env_params_CPPO).shape[0], activation=config["ACTIVATION"]
    )

    train_state_policy_CPPOv1 = TrainState.create(
        apply_fn=policy_network_CPPOv1.apply,
        params=raw_restored_CPPOv1['policy_network']['params'],
        tx=tx,
        lambda_coef=0.,
    )

    value_network_CPPOv1 = Value_Network(activation=config["ACTIVATION"])
    train_state_value_CPPOv1 = TrainState.create(
        apply_fn=value_network_CPPOv1.apply,
        params=raw_restored_CPPOv1['value_network']['params'],
        tx=tx,
        lambda_coef=0.,
    )

    value_network_cost_CPPOv1 = Value_Network(activation=config["ACTIVATION"])
    train_state_cost_CPPOv1 = TrainState.create(
        apply_fn=value_network_cost_CPPOv1.apply,
        params=raw_restored_CPPOv1['cost_network']['params'],
        tx=tx,
        lambda_coef=0.,
    )

    ## CPO v2
    raw_restored_CPPOv2 = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath('model/{}/{}'.format(
        config["DIR_CPPOv2"], config["DIR_MODEL_CPPOv2"])), target=None)

    policy_network_CPPOv2 = Policy_Network(
        env_CPPO_sum.action_space(env_params_CPPO).shape[0], activation=config["ACTIVATION"]
    )

    train_state_policy_CPPOv2 = TrainState.create(
        apply_fn=policy_network_CPPOv2.apply,
        params=raw_restored_CPPOv2['policy_network']['params'],
        tx=tx,
        lambda_coef=0.,
    )

    value_network_CPPOv2 = Value_Network(activation=config["ACTIVATION"])
    train_state_value_CPPOv2 = TrainState.create(
        apply_fn=value_network_CPPOv2.apply,
        params=raw_restored_CPPOv2['value_network']['params'],
        tx=tx,
        lambda_coef=0.,
    )

    value_network_cost_CPPOv2 = Value_Network(activation=config["ACTIVATION"])
    train_state_cost_CPPOv2 = TrainState.create(
        apply_fn=value_network_cost_CPPOv2.apply,
        params=raw_restored_CPPOv2['cost_network']['params'],
        tx=tx,
        lambda_coef=0.,
    )

    ## CPO v3
    raw_restored_CPPOv3 = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath('model/{}/{}'.format(
        config["DIR_CPPOv3"], config["DIR_MODEL_CPPOv3"])), target=None)

    policy_network_CPPOv3 = Policy_Network(
        env_CPPO_sum.action_space(env_params_CPPO).shape[0], activation=config["ACTIVATION"]
    )

    train_state_policy_CPPOv3 = TrainState.create(
        apply_fn=policy_network_CPPOv3.apply,
        params=raw_restored_CPPOv3['policy_network']['params'],
        tx=tx,
        lambda_coef=0.,
    )

    value_network_CPPOv3 = Value_Network(activation=config["ACTIVATION"])
    train_state_value_CPPOv3 = TrainState.create(
        apply_fn=value_network_CPPOv3.apply,
        params=raw_restored_CPPOv3['value_network']['params'],
        tx=tx,
        lambda_coef=0.,
    )

    value_network_cost_CPPOv3 = Value_Network(activation=config["ACTIVATION"])
    train_state_cost_CPPOv3 = TrainState.create(
        apply_fn=value_network_cost_CPPOv3.apply,
        params=raw_restored_CPPOv3['cost_network']['params'],
        tx=tx,
        lambda_coef=0.,
    )

    ########################################## LOAD DECOMPOSED STL #################################################

    raw_restored_dSTL = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath('model/{}/{}'.format(
        config["DIR_dSTL"], config["DIR_MODEL_dSTL"])), target=None)
    
    policy_network_dSTL_1 = Policy_Network(
        env_dSTL_1.action_space(env_params_dSTL_1).shape[0], activation=config["ACTIVATION"]
    )
    policy_network_dSTL_2 = Policy_Network(
        env_dSTL_2.action_space(env_params_dSTL_2).shape[0], activation=config["ACTIVATION"]
    )

    train_state_policy_dSTL_1 = TrainState.create(
        apply_fn=policy_network_dSTL_1.apply,
        params=raw_restored_dSTL['policy_reach1_network']['params'],
        tx=tx,
        count=1e-4,
    )

    train_state_policy_dSTL_2 = TrainState.create(
        apply_fn=policy_network_dSTL_2.apply,
        params=raw_restored_dSTL['policy_reach2_network']['params'],
        tx=tx,
        count=1e-4,
    )

    value_network_reach1 = Value_Network(activation=config["ACTIVATION"])
    train_state_value_dSTL_1 = TrainState.create(
        apply_fn=value_network_reach1.apply,
        params=raw_restored_dSTL['value_reach1_network']['params'],
        tx=tx,
        count=1e-4,
    )

    value_network_reach2 = Value_Network(activation=config["ACTIVATION"])
    train_state_value_dSTL_2 = TrainState.create(
        apply_fn=value_network_reach2.apply,
        params=raw_restored_dSTL['value_reach2_network']['params'],
        tx=tx,
        count=1e-4,
    )

    ########################################## ROLL OUT MODELS #################################################

    ## MODEL 1 : HJ-PPO : STOCHASTIC
    
    print("Rolling Out HJ-PPO (Stochastic)")
    rng_1, _rng_1 = jax.random.split(rng_1)
    reset_rng_1 = jax.random.split(_rng_1, config["NUM_ENVS"])
    obsv_1, env_state_1 = jax.vmap(env_HJPPO.reset, in_axes=(0, None))(reset_rng_1, env_params_HJPPO)
    
    rng_1, _rng_1 = jax.random.split(rng_1)
    runner_state_standard = (train_state_policy_HJPPO, train_state_value_HJPPO, env_state_1, obsv_1, _rng_1)
    
    decomposed_state = (train_state_policy_HJPPO_reach1, train_state_value_HJPPO_reach1, train_state_policy_HJPPO_reach2, train_state_value_HJPPO_reach2)
    policy_controls = (False, False, False)
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
    
    decomposed_state = (train_state_policy_HJPPO_reach1, train_state_value_HJPPO_reach1, train_state_policy_HJPPO_reach2, train_state_value_HJPPO_reach2)
    policy_controls = (False, False, False)
    runner_state = (*runner_state_standard, decomposed_state, policy_controls)

    runner_state, traj_batch_HJPPO_d = jax.lax.scan(
        env_step_HJPPO_d, runner_state, None, config["NUM_STEPS"]
    )

    ## MODEL 3 : CPPO : Variant 1

    print("Rolling Out C-PPO (Variant 1)")
    rng_3, _rng_3 = jax.random.split(rng_3)
    reset_rng_3 = jax.random.split(_rng_3, config["NUM_ENVS"])
    obsv_3, env_state_3 = jax.vmap(env_CPPO_max.reset, in_axes=(0, None))(reset_rng_3, env_params_CPPO)

    rng_3, _rng_3 = jax.random.split(rng_3)
    runner_state = (train_state_policy_CPPOv1, train_state_value_CPPOv1, train_state_cost_CPPOv1, env_state_3, obsv_3, _rng_3)
    
    runner_state, traj_batch_CPPOv1 = jax.lax.scan(
        env_step_CPPO_max, runner_state, None, config["NUM_STEPS"]
    )

    ## MODEL 4 : CPPO : Variant 2

    print("Rolling Out C-PPO (Variant 1)")
    rng_4, _rng_4 = jax.random.split(rng_4)
    reset_rng_4 = jax.random.split(_rng_4, config["NUM_ENVS"])
    obsv_4, env_state_4 = jax.vmap(env_CPPO_sum.reset, in_axes=(0, None))(reset_rng_3, env_params_CPPO)

    rng_4, _rng_4 = jax.random.split(rng_4)
    runner_state = (train_state_policy_CPPOv2, train_state_value_CPPOv2, train_state_cost_CPPOv2, env_state_4, obsv_4, _rng_4)
    
    runner_state, traj_batch_CPPOv2 = jax.lax.scan(
        env_step_CPPO_sum, runner_state, None, config["NUM_STEPS"]
    )

    ## MODEL 5 : CPPO : Variant 3

    print("Rolling Out C-PPO (Variant 3)")
    rng_5, _rng_5 = jax.random.split(rng_5)
    reset_rng_5 = jax.random.split(_rng_5, config["NUM_ENVS"])
    obsv_5, env_state_5 = jax.vmap(env_CPPO_sum.reset, in_axes=(0, None))(reset_rng_5, env_params_CPPO)

    rng_5, _rng_5 = jax.random.split(rng_5)
    runner_state = (train_state_policy_CPPOv3, train_state_value_CPPOv3, train_state_cost_CPPOv3, env_state_5, obsv_5, _rng_5)
    
    runner_state, traj_batch_CPPOv3 = jax.lax.scan(
        env_step_CPPO_sum, runner_state, None, config["NUM_STEPS"]
    )

    ## MODEL 6 : DSTL

    rng_6, _rng_6 = jax.random.split(rng_6)
    reset_rng_6 = jax.random.split(_rng_6, config["NUM_ENVS"])
    obsv_6, env_state_6 = jax.vmap(env_dSTL.reset, in_axes=(0, None))(reset_rng_6, env_params_dSTL)
    rng_6, _rng_6 = jax.random.split(rng_6)
    runner_state = (train_state_policy_dSTL_1, train_state_value_dSTL_1,
                    train_state_policy_dSTL_2, train_state_value_dSTL_2,
                    env_state_6, obsv_6, _rng_6)

    runner_state, traj_batch_dSTL = jax.lax.scan(
        env_step_dSTL, runner_state, None, config["NUM_STEPS"]
    )

    return traj_batch_HJPPO, traj_batch_HJPPO_d, traj_batch_CPPOv1, traj_batch_CPPOv2, traj_batch_CPPOv3, traj_batch_dSTL

if __name__ == "__main__":
    config = vars(get_args(sys.argv[1:]))

    debug = True
    if debug:
        config["EXP_NAME"]="HopperReachReach"

        config["DIR_HJPPO"]="BASELINE_hopper_reachavoid_final"
        config["DIR_MODEL"]="best_244"

        config["DIR_CPPOv1"]="BASELINE_final_hopper_rr_cppomax_raccum_cfnmax_caccum_umin_V1--LR=3e-4"
        config["DIR_MODEL_CPPOv1"]="checkpoint_243"

        config["DIR_CPPOv2"]="BASELINE_final_hopper_rr_cpposum_raccum_cfnmax_caccum_umin_V1"
        config["DIR_MODEL_CPPOv2"]="checkpoint_243"

        config["DIR_CPPOv3"]="BASELINE_final_hopper_rr_cpposum_raccum_cfnsum_caccum_umean_V2"
        config["DIR_MODEL_CPPOv3"]="checkpoint_214"

        config["DIR_DSTL"]="BASELINE_hopper_reachreach_decomposed"
        config["DIR_MODEL_DSTL"]="checkpoint_240"

    config["NUM_ENVS"]=1000
    config["NUM_STEPS"]=500
    config["ACTIVATION"]="tanh"

    envs_HJPPO = get_env(config)
    env_HJPPO, env_HJPPO_1, env_HJPPO_2 = envs_HJPPO

    config_dummy = copy.deepcopy(config)
    config_dummy["EXP_NAME"] = "HopperReachReach_max_CPPO"
    env_CPPO_max = get_env(config_dummy)
    config_dummy["EXP_NAME"] = "HopperReachReach_sum_CPPO"
    env_CPPO_sum = get_env(config_dummy)
    config_dummy["EXP_NAME"] = "HopperReachReachDecomposed"
    env_dSTL, env_dSTL_1, env_dSTL_2 = get_env(config_dummy)

    envs = (
        env_HJPPO, env_HJPPO_1, env_HJPPO_2, 
        env_CPPO_max, env_CPPO_sum, env_dSTL, env_dSTL_1, env_dSTL_2
    )
    env_paramss = (
        env_HJPPO.default_params, env_HJPPO_1.default_params, env_HJPPO_2.default_params,
        env_CPPO_max.default_params, env_CPPO_sum.default_params, env_dSTL.default_params, env_dSTL_1.default_params, env_dSTL_2.default_params, 
    )

    rng_1 = jax.random.PRNGKey(20)
    rng_2 = jax.random.PRNGKey(20)
    rng_3 = jax.random.PRNGKey(20)
    rng_4 = jax.random.PRNGKey(20)
    rng_5 = jax.random.PRNGKey(20)
    rng_6 = jax.random.PRNGKey(20)
    rngs = (rng_1, rng_2, rng_3, rng_4, rng_5, rng_6)
    # folder = os.path.exists("model/{}/traj".format(config['DIR']))

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    print("\n\nCollecting Trajectories")
    traj_batches = test(envs, env_paramss, config, rngs)

    os.makedirs(f"model/{config['TEST_DIR']}", exist_ok=True)
    # val_fig = plot_RR_value(result_traj_batch, result_traj_batch_deterministic, config)
    # traj_fig = plot_traj_sample(result_traj_batch, result_traj_batch_deterministic, config, sample_size=5, make_video=False)

    score_plot = plot_scores(traj_batches, config)
