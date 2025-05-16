import os
import optax
import jax
from jax import lax
import sys
import numpy as np

from functools import partial
from flax.training.train_state import TrainState
from flax.training import checkpoints
import jax.numpy as jnp

import matplotlib.pyplot as plt
from PIL import Image
import imageio

from rraa_rl.EFPPO.src.rl.arguments import get_args
from rraa_rl.EFPPO.src.env.env_list import get_env
from rraa_rl.EFPPO.src.model.actorcritic import Policy_Network, Value_Network, ActorCritic_Continuous, Policy_Network_Discrete, MoGPolicy_Network
from rraa_rl.EFPPO.src.rl.EFPPO_utils import _env_step_rr_vanilla, _env_step_rr_deterministic, _env_step_CPPO_rr
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
    
def plot_RR_value(traj_batch, traj_batch_d, config):

    plt.figure(figsize=(5, 10), constrained_layout=True)
    fig, axes = plt.subplots(1, 2)

    def draw_vals(batch, title, ax):

        # Cumulative min over time for each batch column
        cummin1 = lax.associative_scan(jnp.minimum, batch.reach1, axis=0)  # [T, B]
        cummin2 = lax.associative_scan(jnp.minimum, batch.reach2, axis=0)  # [T, B]
        # score = jnp.maximum(cummin1, cummin2)  # [T, B]

        # Compute summary statistics over batch for ribbon plot
        median_1 = jnp.median(cummin1, axis=1)
        q25_1 = jnp.percentile(cummin1, 25, axis=1)
        q75_1 = jnp.percentile(cummin1, 75, axis=1)

        median_2 = jnp.median(cummin2, axis=1)
        q25_2 = jnp.percentile(cummin2, 25, axis=1)
        q75_2 = jnp.percentile(cummin2, 75, axis=1)

        # Plot
        timesteps = jnp.arange(cummin1.shape[0])
        ax.fill_between(timesteps, q25_1, q75_1, alpha=0.3, label="min<t l1", color="green")
        ax.fill_between(timesteps, q25_2, q75_2, alpha=0.3, label="min<t l2", color="blue")
        ax.plot(timesteps, median_1, color="green")
        ax.plot(timesteps, median_2, color="blue")
        ax.set_xlabel("Trajectory Step")
        ax.set_title(title)
        ax.legend()

    draw_vals(traj_batch, "Stochastic Policy", axes[0])
    draw_vals(traj_batch_d, "Deterministic Policy", axes[1])

    plt.savefig(f"model/{config['DIR']}/test/{config['DIR_MODEL']}_value_plot", dpi=300)
    return fig

def test(envs, env_paramss, config, rngs):
    rng_1, rng_2, rng_3, rng_4, rng_5, rng_6, rng_7, rng_8 = rngs

    env_HJPPO, env_HJPPO_reach_1, env_HJPPO_reach_2, env_CPPO, env_dSTL = envs # COMPOSED (RR) + 2 DECOMPOSED (R1 + R2)
    env_params_HJPPO, env_params_HJPPO_reach_1, env_params_HJPPO_reach_2, env_params_CPPO, env_params_dSTL = env_paramss

    # DEFINE ENV STEP WRAPPERS
    env_step_HJPPO = partial(_env_step_rr_vanilla, env_HJPPO, env_params_HJPPO)
    env_step_HJPPO_d = partial(_env_step_rr_deterministic, env_HJPPO, env_params_HJPPO)
    env_step_CPPO = partial(_env_step_CPPO_rr, env_CPPO, env_params_CPPO)
    # env_step_dSTL = partial(FIXME, env_dSTL, env_params_dSTL)
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
        env.action_space(env_params_CPPO).shape[0], activation=config["ACTIVATION"]
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
        env.action_space(env_params_CPPO).shape[0], activation=config["ACTIVATION"]
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
        env.action_space(env_params_CPPO).shape[0], activation=config["ACTIVATION"]
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

    # raw_restored_dSTL = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath('model/{}/{}'.format(
    #     config["DIR_dSTL"], config["DIR_MODEL_dSTL"])), target=None)
    
    ## FIXME

    ########################################## ROLL OUT MODELS #################################################

    ## MODEL 1 : HJ-PPO : STOCHASTIC
    
    print("Rolling Out HJ-PPO (Stochastic)")
    rng_1, _rng_1 = jax.random.split(rng_1)
    reset_rng_1 = jax.random.split(_rng_1, config["NUM_ENVS"])
    obsv_1, env_state_1 = jax.vmap(env.reset, in_axes=(0, None))(reset_rng_1, env_params_HJPPO)
    
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
    obsv_2, env_state_2 = jax.vmap(env.reset, in_axes=(0, None))(reset_rng_2, env_params_HJPPO)

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
    obsv_3, env_state_3 = jax.vmap(env_CPPO.reset, in_axes=(0, None))(reset_rng_3, env_params_CPPO)

    rng_3, _rng_3 = jax.random.split(rng_3)
    runner_state = (train_state_policy_CPPOv1, train_state_value_CPPOv1, train_state_cost_CPPOv1, env_state_3, obsv_3, _rng_3)
    
    runner_state, traj_batch_CPPOv1 = jax.lax.scan(
        env_step_CPPO, runner_state, None, config["NUM_STEPS"]
    )

    ## MODEL 4 : CPPO : Variant 2

    print("Rolling Out C-PPO (Variant 1)")
    rng_4, _rng_4 = jax.random.split(rng_4)
    reset_rng_4 = jax.random.split(_rng_4, config["NUM_ENVS"])
    obsv_4, env_state_4 = jax.vmap(env_CPPO.reset, in_axes=(0, None))(reset_rng_3, env_params_CPPO)

    rng_4, _rng_4 = jax.random.split(rng_4)
    runner_state = (train_state_policy_CPPOv2, train_state_value_CPPOv2, train_state_cost_CPPOv2, env_state_4, obsv_4, _rng_4)
    
    runner_state, traj_batch_CPPOv2 = jax.lax.scan(
        env_step_CPPO, runner_state, None, config["NUM_STEPS"]
    )

    ## MODEL 5 : CPPO : Variant 3

    print("Rolling Out C-PPO (Variant 3)")
    rng_5, _rng_5 = jax.random.split(rng_5)
    reset_rng_5 = jax.random.split(_rng_5, config["NUM_ENVS"])
    obsv_5, env_state_5 = jax.vmap(env_CPPO.reset, in_axes=(0, None))(reset_rng_5, env_params_CPPO)

    rng_5, _rng_5 = jax.random.split(rng_5)
    runner_state = (train_state_policy_CPPOv3, train_state_value_CPPOv3, train_state_cost_CPPOv3, env_state_5, obsv_5, _rng_5)
    
    runner_state, traj_batch_CPPOv3 = jax.lax.scan(
        env_step_CPPO, runner_state, None, config["NUM_STEPS"]
    )

    ## FIXME ADD DECOMPOSED STL
    traj_batch_dSTL = traj_batch_HJPPO

    return traj_batch_HJPPO, traj_batch_HJPPO_d, traj_batch_CPPOv1, traj_batch_CPPOv2, traj_batch_CPPOv3, traj_batch_dSTL

if __name__ == "__main__":
    config = vars(get_args(sys.argv[1:]))

    debug = False
    if debug:
        config["EXP_NAME"]="HopperReachReach"
        config["DIR"]="hopper_reachreach_halfwidth_R10"
        config["DIR_MODEL"]="checkpoint_975"
        config["NUM_ENVS"]=128
        config["NUM_STEPS"]=500
        config["ACTIVATION"]="tanh"

    envs = get_env(config)
    env, env_1, env_2 = envs
    env_paramss = (env.default_params, env_1.default_params, env_2.default_params)
    rng = jax.random.PRNGKey(20)
    folder = os.path.exists("model/{}/traj".format(config['DIR']))

    # if config['EXP_NAME'] == 'WindField': 
    #     env_params = env_params.replace(index=config['SECTION'])
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    (result_traj_batch, result_traj_batch_deterministic) = test(envs, env_paramss, config, rng)

    ((reach_1_perc, reach_2_perc, reach_perc), 
     (reach_idx_1, reach_idx_2, reach_idx)) = calculate_reachreach(result_traj_batch)
    
    ((reach_1_perc_d, reach_2_perc_d, reach_perc_d), 
     (reach_idx_1_d, reach_idx_2_d, reach_idx_d)) = calculate_reachreach(result_traj_batch_deterministic)
    
    print("\nSCORES")
    print(f" STOCH - REACH-REACH : {100*reach_perc:0.1f}%")
    print(f" STOCH - REACH-1     : {100*reach_1_perc:0.1f}%")
    print(f" STOCH - REACH-2     : {100*reach_2_perc:0.1f}%")
    print(f" DETER - REACH-REACH : {100*reach_perc_d:0.1f}%")
    print(f" DETER - REACH-1     : {100*reach_1_perc_d:0.1f}%")
    print(f" DETER - REACH-2     : {100*reach_2_perc_d:0.1f}%")
    print("")

    os.makedirs(f"model/{config['DIR']}/test", exist_ok=True)
    val_fig = plot_RR_value(result_traj_batch, result_traj_batch_deterministic, config)
    traj_fig = plot_traj_sample(result_traj_batch, result_traj_batch_deterministic, config, sample_size=5, make_video=False)
