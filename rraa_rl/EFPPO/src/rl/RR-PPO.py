import sys
# sys.path.append("/home/mepear_gc")

import os
import time
import wandb
import jax
import jax.numpy as jnp
import numpy as np
import pdb
import matplotlib.pyplot as plt

from flax.training import train_state
from flax.training import checkpoints

from arguments import get_args
from functools import partial
from typing import Any

from rraa_rl.EFPPO.src.rl.EFPPO_utils import _ppo_vanilla_update, _env_step_rr_vanilla, _env_step_r1_vanilla, _env_step_r2_vanilla
from rraa_rl.EFPPO.src.env.env_list import get_env
from rraa_rl.EFPPO.src.model.actorcritic import Policy_Network, Value_Network, Policy_Network_Discrete, MoGPolicy_Network
from rraa_rl.EFPPO.src.rl.plot_utils import calculate_minimal_reach, calculate_consumption, calculate_reachreach, plot_target, plot_value_target, plot_contour, plot_contour_RRAA, plot_policy_decision, plot_video_contour_RRAA
from rraa_rl.EFPPO.src.rl.utils import optimizer, get_BuRd, tree_index1, tree_index2
from rraa_rl.EFPPO.src.rl.gae import (Transition_reach,
                              calculate_gae, calculate_gae2, calculate_gae3,
                              calculate_gae_reach, calculate_gae_reach2, calculate_gae_reach3, calculate_gae_reach4,
                              calculate_indexs, calculate_indexs2, calculate_indexs3, calculate_indexs3_rr, calculate_indexs_rr)

class TrainState(train_state.TrainState):
    mean: Any
    variance: Any
    count: Any

def train(envs, env_paramss, config, rng):
    env, env_reach_1, env_reach_2 = envs # COMPOSED (RR) + 2 DECOMPOSED (R1 + R2)
    env_params, env_params_reach_1, env_params_reach_2 = env_paramss

    def _train(train_state_total, ent_gamma):

        train_state_policy, train_state_value, \
            train_state_policy_reach1, train_state_value_reach1, \
            train_state_policy_reach2, train_state_value_reach2, \
            rng_og, timestep = train_state_total

        # RESET ENV
        rng, _rng = jax.random.split(rng_og)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        rng, _rng = jax.random.split(rng)
        runner_state_standard = (train_state_policy, train_state_value, env_state, obsv, _rng)
        
        # SPECIAL DECOMPOSED STATES
        decomposed_state = (train_state_policy_reach1, train_state_value_reach1, train_state_policy_reach2, train_state_value_reach2)
        force_combined = False #if timestep < 20 else False # ihibits switching until > 20 epochs
        force_reach1, force_reach2 = False, False
        policy_controls = (force_combined, force_reach1, force_reach2)
        runner_state = (*runner_state_standard, decomposed_state, policy_controls)

        # COLLECT TRAJECTORY COMPOSED
        runner_state, traj_batch = jax.lax.scan(
            env_step, runner_state, None, config["NUM_STEPS"]
        )

        init_type = "toinput_goal" # "standard", "toinput", "toinput_goal"
        # RESET ENV - 1
        rng, _rng = jax.random.split(rng_og)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])

        if init_type == "standard": 
            obsv_reach_1, env_state_reach_1 = jax.vmap(env_reach_1.reset, in_axes=(0, None))(reset_rng, env_params_reach_1)
        
        elif "toinput" in init_type: 
            rng_reach1, _rng_reach1 = jax.random.split(rng)
            
            ## Select first reach2 step in composed rollout for initial decomposed reach1 state 
            if init_type == "toinput_goal":
                random_index_pre = jax.random.randint(_rng_reach1, shape=(config["NUM_ENVS"],), minval=0, maxval=config["NUM_STEPS"])
                reach2_idx_pre = (traj_batch.reach2 < 0).argmax(axis=0)
                reach2_idx = jnp.where(jnp.any((traj_batch.reach2 < 0) == 1, axis=0), reach2_idx_pre, config["NUM_STEPS"])
                random_index = jnp.where(jnp.any(traj_batch.reach2 < 0, axis=0), reach2_idx, random_index_pre)
            
            ## Select random step in composed rollout for initial decomposed reach1 state
            else:
                random_index = jax.random.randint(_rng_reach1, shape=(config["NUM_ENVS"],), minval=0, maxval=config["NUM_STEPS"])
            # random_index = jax.random.randint(_rng_reach1, shape=(untrans_traj_batch_observations_full.shape[0],), minval=0, maxval=untrans_traj_batch_observations_full.shape[1])

            # Multiple random indices
            if "Hopper" in config["EXP_NAME"] or "HalfCheetah" in config["EXP_NAME"]:
                traj_batch_observations_full = traj_batch.obs 
                untrans_traj_batch_observations_full = env.untransform_obs(traj_batch_observations_full)
                untrans_traj_batch_observations_full = jnp.transpose(untrans_traj_batch_observations_full, axes=(1, 0, 2))
                untrans_traj_batch_observations = jax.vmap(lambda obs, idx: obs[idx])(untrans_traj_batch_observations_full, random_index)
                obsv_reach_1, env_state_reach_1 = jax.vmap(env_reach_1.reset_toinput, in_axes=(0, 0, None))(reset_rng, untrans_traj_batch_observations, env_params_reach_1) 

            elif "Humanoid" in config["EXP_NAME"]:
                # FIXME: humanoid obs need an action, meaning we would need to pass reset action too, for now just zeros
                traj_batch_observations_full = traj_batch.obs 
                untrans_traj_batch_observations_full = env.untransform_obs(traj_batch_observations_full)
                untrans_traj_batch_observations_full = jnp.transpose(untrans_traj_batch_observations_full, axes=(1, 0, 2))
                untrans_traj_batch_observations = jax.vmap(lambda obs, idx: obs[idx])(untrans_traj_batch_observations_full, random_index)
                obsv_reach_1, env_state_reach_1 = jax.vmap(env_reach_1.reset_toinput, in_axes=(0, 0, None))(reset_rng, untrans_traj_batch_observations, env_params_reach_1) 

            elif "F16" in config["EXP_NAME"]:
                traj_batch_states = traj_batch.info['state']
                traj_batch_states = jnp.transpose(traj_batch_states, axes=(1, 0, 2))
                reset_states = jax.vmap(lambda obs, idx: obs[idx])(traj_batch_states, random_index)
                obsv_reach_1, env_state_reach_1 = jax.vmap(env_reach_1.reset_env_toinput, in_axes=(0, None))(reset_states, env_params_reach_1) 
            
            else:
                raise NotImplementedError("Unknown environment type for toinput reset")
        
        rng, _rng = jax.random.split(rng)
        runner_state_standard_reach_1 = (train_state_policy, train_state_value, env_state_reach_1, obsv_reach_1, _rng)
        
        # SPECIAL DECOMPOSED STATES - 1
        decomposed_state = (train_state_policy_reach1, train_state_value_reach1, train_state_policy_reach2, train_state_value_reach2)
        force_reach1, force_reach2 = True, False
        policy_controls_reach1 = (force_combined, force_reach1, force_reach2)
        runner_state_reach1 = (*runner_state_standard_reach_1, decomposed_state, policy_controls_reach1)

        # COLLECT TRAJECTORY DECOMPOSED - 1
        runner_state_reach1, traj_batch_reach1 = jax.lax.scan(
            env_step_reach_1, runner_state_reach1, None, config["NUM_STEPS"]
        )

        # RESET ENV - 2
        rng, _rng = jax.random.split(rng_og)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])

        if init_type == "standard":
            obsv_reach_2, env_state_reach_2 = jax.vmap(env_reach_2.reset, in_axes=(0, None))(reset_rng, env_params_reach_2)
        # elif init_type == "toinput": 
        #     # Select random observations from standard rollout to use for initial avoid state 
        #     rng_reach2, _rng_reach2 = jax.random.split(rng)

        #     # Multiple random indices
        #     random_index = jax.random.randint(_rng_reach2, shape=(untrans_traj_batch_observations_full.shape[0],), minval=0, maxval=untrans_traj_batch_observations_full.shape[1])
        #     untrans_traj_batch_observations = jax.vmap(lambda obs, idx: obs[idx])(untrans_traj_batch_observations_full, random_index)

        #     obsv_reach_2, env_state_reach_2 = jax.vmap(env_reach_2.reset_toinput, in_axes=(0, 0, None))(reset_rng, untrans_traj_batch_observations, env_params_reach_2) 

        elif "toinput" in init_type: 
            rng_reach2, _rng_reach2 = jax.random.split(rng)
            
            ## Select first reach1 step in composed rollout for initial decomposed reach2 state 
            if init_type == "toinput_goal":
                random_index_pre = jax.random.randint(rng_reach2, shape=(config["NUM_ENVS"],), minval=0, maxval=config["NUM_STEPS"])
                reach1_idx_pre = (traj_batch.reach1 < 0).argmax(axis=0)
                reach1_idx = jnp.where(jnp.any((traj_batch.reach1 < 0) == 1, axis=0), reach1_idx_pre, config["NUM_STEPS"])
                random_index = jnp.where(jnp.any(traj_batch.reach1 < 0, axis=0), reach1_idx, random_index_pre)

            ## Select random step in composed rollout for initial decomposed reach2 state 
            else:
                random_index = jax.random.randint(rng_reach2, shape=(config["NUM_ENVS"],), minval=0, maxval=config["NUM_STEPS"])
            # random_index = jax.random.randint(_rng_reach1, shape=(untrans_traj_batch_observations_full.shape[0],), minval=0, maxval=untrans_traj_batch_observations_full.shape[1])

            # Multiple random indices
            if "Hopper" in config["EXP_NAME"] or "HalfCheetah" in config["EXP_NAME"]:
                traj_batch_observations_full = traj_batch.obs 
                untrans_traj_batch_observations_full = env.untransform_obs(traj_batch_observations_full)
                untrans_traj_batch_observations_full = jnp.transpose(untrans_traj_batch_observations_full, axes=(1, 0, 2))
                untrans_traj_batch_observations = jax.vmap(lambda obs, idx: obs[idx])(untrans_traj_batch_observations_full, random_index)
                # obsv_reach_1, env_state_reach_1 = jax.vmap(env_reach_1.reset_toinput, in_axes=(0, 0, None))(reset_rng, untrans_traj_batch_observations, env_params_reach_1) 
                obsv_reach_2, env_state_reach_2 = jax.vmap(env_reach_2.reset_toinput, in_axes=(0, 0, None))(reset_rng, untrans_traj_batch_observations, env_params_reach_2) 

            elif "Humanoid" in config["EXP_NAME"]:
                # FIXME: humanoid obs need an action, meaning we would need to pass reset action too, for now just zeros
                traj_batch_observations_full = traj_batch.obs 
                untrans_traj_batch_observations_full = env.untransform_obs(traj_batch_observations_full)
                untrans_traj_batch_observations_full = jnp.transpose(untrans_traj_batch_observations_full, axes=(1, 0, 2))
                untrans_traj_batch_observations = jax.vmap(lambda obs, idx: obs[idx])(untrans_traj_batch_observations_full, random_index)
                # obsv_reach_1, env_state_reach_1 = jax.vmap(env_reach_1.reset_toinput, in_axes=(0, 0, None))(reset_rng, untrans_traj_batch_observations, env_params_reach_1) 
                obsv_reach_2, env_state_reach_2 = jax.vmap(env_reach_2.reset_toinput, in_axes=(0, 0, None))(reset_rng, untrans_traj_batch_observations, env_params_reach_2) 

            elif "F16" in config["EXP_NAME"]:
                traj_batch_states = traj_batch.info['state']
                traj_batch_states = jnp.transpose(traj_batch_states, axes=(1, 0, 2))
                reset_states = jax.vmap(lambda obs, idx: obs[idx])(traj_batch_states, random_index)
                # obsv_reach_1, env_state_reach_1 = jax.vmap(env_reach_1.reset_env_toinput, in_axes=(0, None))(reset_states, env_params_reach_1) 
                obsv_reach_2, env_state_reach_2 = jax.vmap(env_reach_2.reset_env_toinput, in_axes=(0, None))(reset_states, env_params_reach_2) 
        
            else:
                raise NotImplementedError("Unknown environment type for toinput reset")

        rng, _rng = jax.random.split(rng)
        runner_state_standard_reach_2 = (train_state_policy, train_state_value, env_state_reach_2, obsv_reach_2, _rng)
        
        # SPECIAL DECOMPOSED STATES - 2
        decomposed_state = (train_state_policy_reach1, train_state_value_reach1, train_state_policy_reach2, train_state_value_reach2)
        force_reach1, force_reach2 = False, True
        policy_controls_reach2 = (force_combined, force_reach1, force_reach2)
        runner_state_reach2 = (*runner_state_standard_reach_2, decomposed_state, policy_controls_reach2)

        # COLLECT TRAJECTORY DECOMPOSED - 2
        runner_state_reach2, traj_batch_reach2 = jax.lax.scan(
            env_step_reach_2, runner_state_reach2, None, config["NUM_STEPS"]
        )

        # CALCULATE COMPOSED ADVANTAGE
        (train_state_policy, train_state_value, env_state, last_obs, rng,
          decomposed_state, policy_controls) = runner_state

        last_val = train_state_value.apply_fn(train_state_value.params, last_obs)
        last_val1 = train_state_value_reach1.apply_fn(train_state_value_reach1.params, last_obs)
        last_val2 = train_state_value_reach2.apply_fn(train_state_value_reach2.params, last_obs)

        # DECOMPOSED REACH VALUES ON COMPOSED PPO ACTOR ROLL OUT
        reach1_append = jnp.concatenate((traj_batch.reach1, jnp.expand_dims(env_state.reach1, axis=1).T))
        V_reach1_append = jnp.concatenate((traj_batch.value_reach1, jnp.expand_dims(last_val1, axis=1).T))

        reach2_append = jnp.concatenate((traj_batch.reach2, jnp.expand_dims(env_state.reach2, axis=1).T))
        V_reach2_append = jnp.concatenate((traj_batch.value_reach2, jnp.expand_dims(last_val2, axis=1).T))

        V_append = jnp.concatenate((traj_batch.value, jnp.expand_dims(last_val, axis=1).T))

        # SPECIAL BRT TARGET FOR BRRT PROBLEM
        l_tile_append = jnp.minimum(jnp.maximum(reach1_append, V_reach2_append), jnp.maximum(reach2_append, V_reach1_append))

        indexs, done = calculate_indexs3_rr(ent_gamma[1], traj_batch.reward, l_tile_append,
                                               jnp.expand_dims(last_val, axis=1).T) 
        # indexs, done = calculate_indexs_rr(ent_gamma[1], traj_batch.reward, l_tile_append,
        #                                        V_append)
        done = done[:-1, :]

        # FILTER UNHEALTHY
        # head_height = traj_batch.obs[:,:,1] + 0.2 * jnp.cos(traj_batch.obs[:,:,2]) # from calculate_position
        # head_threshold = 1. # aggresive to start
        # done = jnp.where(head_height < head_threshold, jnp.ones_like(done), done)

        # head_height = traj_batch.obs[:,:,1] + 0.2 * jnp.cos(traj_batch.obs[:,:,2]) # from calculate_position
        # jaw_height = traj_batch.obs[:,:,1] - 0.2 * jnp.cos(traj_batch.obs[:,:,2])
        # thg_height = jaw_height - 0.45 * jnp.cos(traj_batch.obs[:,:,2] - traj_batch.obs[:,:,3])
        # done = jnp.where(head_height < 0., jnp.ones_like(done), done)
        # done = jnp.where(jaw_height < 0., jnp.ones_like(done), done)
        # done = jnp.where(thg_height < 0., jnp.ones_like(done), done)

        advantages_V, targets_V = calculate_gae_reach4(ent_gamma[1], config["GAE_LAMBDA"], l_tile_append, V_append, done)

        # UPDATE COMPOSED NETWORK
        composed_policy_mask = jnp.where(traj_batch.policy_taken == 0, 1., 0.)
        update_state = (train_state_policy, train_state_value,
                        traj_batch, advantages_V, targets_V, advantages_V, composed_policy_mask, rng)

        xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent_gamma[0]
        update_state, loss_info = jax.lax.scan(
            update_epoch, update_state, xs, config["UPDATE_EPOCHS"]
        )
        train_state_policy = update_state[0]
        train_state_value = update_state[1]
        rng = update_state[-1]

        # CALCULATE DECOMPOSED ADVANTAGES - 1
        (_, _, env_state_1, last_obs_1, rng_1, _, _) = runner_state_reach1

        last_val1 = train_state_value_reach1.apply_fn(train_state_value_reach1.params, last_obs_1)
        reach1_append = jnp.concatenate((traj_batch_reach1.reach1, jnp.expand_dims(env_state_1.reach1, axis=1).T))
        V_reach1_append = jnp.concatenate((traj_batch_reach1.value, jnp.expand_dims(last_val1, axis=1).T))

        indexs, done_1 = calculate_indexs3_rr(ent_gamma[1], traj_batch_reach1.reward, reach1_append,
                                               jnp.expand_dims(last_val1, axis=1).T)
        # indexs, done_1 = calculate_indexs_rr(ent_gamma[1], traj_batch_reach1.reward, reach1_append,
        #                                        V_reach1_append)

        done_1 = done_1[:-1, :]

        advantages_V_reach1, targets_V_reach1 = calculate_gae_reach4(ent_gamma[1], config["GAE_LAMBDA"], reach1_append, V_reach1_append, done_1)

        # UPDATE DECOMPOSED NETWORK - 1
        dummy_mask = jnp.ones(traj_batch_reach1.reach1.shape)
        update_state_reach1 = (train_state_policy_reach1, train_state_value_reach1,
                        traj_batch_reach1, advantages_V_reach1, targets_V_reach1, advantages_V_reach1, dummy_mask, rng_1)

        xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent_gamma[0]
        update_state_reach1, loss_info_1 = jax.lax.scan(
            update_epoch_reach1, update_state_reach1, xs, config["UPDATE_EPOCHS"]
        )
        train_state_policy_reach1 = update_state_reach1[0]
        train_state_value_reach1 = update_state_reach1[1]
        rng_1 = update_state_reach1[-1]

        # CALCULATE DECOMPOSED ADVANTAGES - 2
        (_, _, env_state_2, last_obs_2, rng_2, _, _) = runner_state_reach2

        last_val2 = train_state_value_reach2.apply_fn(train_state_value_reach2.params, last_obs_2)
        reach2_append = jnp.concatenate((traj_batch_reach2.reach2, jnp.expand_dims(env_state_2.reach2, axis=1).T))
        V_reach2_append = jnp.concatenate((traj_batch_reach2.value, jnp.expand_dims(last_val2, axis=1).T))

        indexs, done_2 = calculate_indexs3_rr(ent_gamma[1], traj_batch_reach2.reward, reach2_append,
                                               jnp.expand_dims(last_val2, axis=1).T)
        # indexs, done_2 = calculate_indexs_rr(ent_gamma[1], traj_batch_reach2.reward, reach2_append,
        #                                        V_reach2_append)
        done_2 = done_2[:-1, :]

        advantages_V_reach2, targets_V_reach2 = calculate_gae_reach4(ent_gamma[1], config["GAE_LAMBDA"], reach2_append, V_reach2_append, done_2)

        # UPDATE DECOMPOSED NETWORK - 2
        dummy_mask = jnp.ones(traj_batch_reach2.reach2.shape)
        update_state_reach2 = (train_state_policy_reach2, train_state_value_reach2,
                        traj_batch_reach2, advantages_V_reach2, targets_V_reach2, advantages_V_reach2, dummy_mask, rng_2)

        xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent_gamma[0]
        update_state_reach2, loss_info_2 = jax.lax.scan(
            update_epoch_reach2, update_state_reach2, xs, config["UPDATE_EPOCHS"]
        )
        train_state_policy_reach2 = update_state_reach2[0]
        train_state_value_reach2 = update_state_reach2[1]
        rng_2 = update_state_reach2[-1]

        return ((train_state_policy, train_state_value, train_state_policy_reach1, train_state_value_reach1, train_state_policy_reach2, train_state_value_reach2, rng, timestep),
                {"batch_info": (traj_batch, targets_V, done), "loss_info": loss_info,
                 "batch_1_info": (traj_batch_reach1, targets_V_reach1, done_1), "loss_info_1": loss_info_1,
                 "batch_2_info": (traj_batch_reach2, targets_V_reach2, done_2), "loss_info_2": loss_info_2,
                 "reach_gamma": ent_gamma[1], "entropy_weight": ent_gamma[0]})
    
    # INIT JAX WRAPPERS
    update_epoch = partial(_ppo_vanilla_update, config)
    env_step = partial(_env_step_rr_vanilla, env, env_params)
    env_step_reach_1 = partial(_env_step_r1_vanilla, env_reach_1, env_params_reach_1)
    env_step_reach_2 = partial(_env_step_r2_vanilla, env_reach_2, env_params_reach_2)
    training = jax.jit(_train)

    tx = optimizer(config)

    # INIT POLICY NETWORK
    if config["DISCRETE"] == False:
        policy_network = MoGPolicy_Network( # MoG
        # policy_network = Policy_Network(
            env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
        )
        policy_network_reach1 = Policy_Network(
            env_reach_1.action_space(env_params_reach_1).shape[0], activation=config["ACTIVATION"]
        )
        policy_network_reach2 = Policy_Network(
            env_reach_2.action_space(env_params_reach_2).shape[0], activation=config["ACTIVATION"]
        )
    else:
        policy_network = Policy_Network_Discrete(
            env.action_space(env_params).n, activation=config["ACTIVATION"]
        )
        policy_network_reach1 = Policy_Network(
            env_reach_1.action_space(env_params_reach_1).n, activation=config["ACTIVATION"]
        )
        policy_network_reach2 = Policy_Network(
            env_reach_2.action_space(env_params_reach_2).n, activation=config["ACTIVATION"]
        )

    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(env.observation_space(env_params).shape)
    network_params_policy = policy_network.init(_rng, init_x)
    train_state_policy = TrainState.create(
        apply_fn=policy_network.apply,
        params=network_params_policy,
        tx=tx,
        mean=jnp.zeros(env.observation_space(env_params).shape),
        variance=jnp.zeros(env.observation_space(env_params).shape),
        count=1e-4,
    )

    # INIT VALUE CRITIC NETWORK
    value_network = Value_Network(activation=config["ACTIVATION"])
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(env.observation_space(env_params).shape)
    network_params = value_network.init(_rng, init_x)
    train_state_value = TrainState.create(
        apply_fn=value_network.apply,
        params=network_params,
        tx=tx,
        mean=jnp.zeros(env.observation_space(env_params).shape),
        variance=jnp.zeros(env.observation_space(env_params).shape),
        count=1e-4,
    )

    # INIT DECOMPOSED ACTOR AND CRITICS
    if not config["LOAD_DECOMPOSED"]:
    
        # DECOMPOSED POLICIES
        init_x_reach_1 = jnp.zeros(env_reach_1.observation_space(env_params_reach_1).shape)
        network_params_policy_reach1 = policy_network_reach1.init(_rng, init_x_reach_1)
        train_state_policy_reach1 = TrainState.create(
            apply_fn=policy_network_reach1.apply,
            params=network_params_policy_reach1,
            tx=tx,
            mean=jnp.zeros(env_reach_1.observation_space(env_params_reach_1).shape),
            variance=jnp.zeros(env_reach_1.observation_space(env_params_reach_1).shape),
            count=1e-4,
        )

        init_x_reach_2 = jnp.zeros(env_reach_2.observation_space(env_params_reach_2).shape)
        network_params_policy_reach2 = policy_network_reach2.init(_rng, init_x_reach_2)
        train_state_policy_reach2 = TrainState.create(
            apply_fn=policy_network_reach2.apply,
            params=network_params_policy_reach2,
            tx=tx,
            mean=jnp.zeros(env_reach_2.observation_space(env_params_reach_2).shape),
            variance=jnp.zeros(env_reach_2.observation_space(env_params_reach_2).shape),
            count=1e-4,
        )

        # DECOMPOSED VALUE CRITICS
        value_network_reach1 = Value_Network(activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env_reach_1.observation_space(env_params_reach_1).shape)
        network_params_reach1 = value_network_reach1.init(_rng, init_x)
        train_state_value_reach1 = TrainState.create(
            apply_fn=value_network_reach1.apply,
            params=network_params_reach1,
            tx=tx,
            mean=jnp.zeros(env_reach_1.observation_space(env_params_reach_1).shape),
            variance=jnp.zeros(env_reach_1.observation_space(env_params_reach_1).shape),
            count=1e-4,
        )

        value_network_reach2 = Value_Network(activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env_reach_2.observation_space(env_params_reach_2).shape)
        network_params_reach2 = value_network_reach2.init(_rng, init_x)
        train_state_value_reach2 = TrainState.create(
            apply_fn=value_network_reach2.apply,
            params=network_params_reach2,
            tx=tx,
            mean=jnp.zeros(env_reach_2.observation_space(env_params_reach_2).shape),
            variance=jnp.zeros(env_reach_2.observation_space(env_params_reach_2).shape),
            count=1e-4,
        )

    # LOAD DECOMPOSED ACTOR AND CRITICS
    else:
        raw_restored = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath('model/{}/{}'.format(
            config["LOAD_DEC_DIR"], config["LOAD_DEC_DIR_MODEL"])), target=None)
        
        train_state_policy_reach1 = TrainState.create(
            apply_fn=policy_network_reach1.apply,
            params=raw_restored['policy_reach1_network']['params'],
            mean=raw_restored['policy_reach1_network']["mean"],
            variance=raw_restored['policy_reach1_network']["variance"],
            count=raw_restored['policy_reach1_network']["count"],
            tx=tx,
        )
        train_state_policy_reach2 = TrainState.create(
            apply_fn=policy_network_reach2.apply,
            params=raw_restored['policy_reach2_network']['params'],
            mean=raw_restored['policy_reach2_network']["mean"],
            variance=raw_restored['policy_reach2_network']["variance"],
            count=raw_restored['policy_reach2_network']["count"],
            tx=tx,
        )

        value_network_reach1 = Value_Network(activation=config["ACTIVATION"])
        train_state_value_reach1 = TrainState.create(
            apply_fn=value_network_reach1.apply,
            params=raw_restored['value_reach1_network']['params'],
            mean=raw_restored['value_reach1_network']["mean"],
            variance=raw_restored['value_reach1_network']["variance"],
            count=raw_restored['value_reach1_network']["count"],
            tx=tx,
        )
        value_network_reach2 = Value_Network(activation=config["ACTIVATION"])
        train_state_value_reach2 = TrainState.create(
            apply_fn=value_network_reach2.apply,
            params=raw_restored['value_reach2_network']['params'],
            mean=raw_restored['value_reach2_network']["mean"],
            variance=raw_restored['value_reach2_network']["variance"],
            count=raw_restored['value_reach2_network']["count"],
            tx=tx,
        )

    # IF TRAINING DECOMPOSED, USE PPO
    if not config["LOAD_DECOMPOSED"]:
        update_epoch_reach1 = partial(_ppo_vanilla_update, config)
        update_epoch_reach2 = partial(_ppo_vanilla_update, config)

    # IF LOADING PRESOLVED DECOMPOSED, NO TRAINING
    else:
        def _no_update(config, update_state, ent):
            dummy_loss = {
                "actor_loss": 0.0,
                "value_loss": 0.0,
                "entropy_loss": 0.0,
            }
            return update_state, dummy_loss

        update_epoch_reach1 = partial(_no_update, config)
        update_epoch_reach2 = partial(_no_update, config)

    total_timesteps = config["NUM_UPDATES"] // config["STEP_SCAN"]

    best_score = -jnp.inf
    for timestep in range(config["NUM_UPDATES"] // config["STEP_SCAN"]):

        t0 = time.time()

        xs = jnp.zeros((config["STEP_SCAN"], 2))

        if config['ANNEAL_ENT'] == True:
            ent = jnp.ones(config["STEP_SCAN"]) * config["ENT_COEF"] * (total_timesteps - timestep) / total_timesteps
        else:
            ent = jnp.ones(config["STEP_SCAN"]) * config["ENT_COEF"]

        gamma_1 = jnp.ones(config["STEP_SCAN"]) * config["GAMMA_REACH_INIT"] + (config['GAMMA_REACH_FINAL'] - config["GAMMA_REACH_INIT"]) * timestep / total_timesteps
        gamma_2 = jnp.ones(config["STEP_SCAN"]) * jnp.minimum(config['GAMMA_REACH_FINAL'], config["GAMMA_REACH_INIT"] +
                              (config['GAMMA_REACH_FINAL'] - config["GAMMA_REACH_INIT"]) * timestep * 2 / total_timesteps)

        xs = xs.at[:, 0].set(ent)
        xs = xs.at[:, 1].set(gamma_2)

        update_state, result = jax.lax.scan(
            training, (train_state_policy, train_state_value,
                       train_state_policy_reach1, train_state_value_reach1,
                       train_state_policy_reach2, train_state_value_reach2, 
                       rng, timestep),
            xs, config["STEP_SCAN"]
        )

        (train_state_policy, train_state_value, 
            train_state_policy_reach1, train_state_value_reach1, 
            train_state_policy_reach2, train_state_value_reach2, 
            rng, timestep) = update_state

        loss_info = result['loss_info']
        loss_info_1 = result['loss_info_1']
        loss_info_2 = result['loss_info_2']

        result_traj = tree_index1(result['batch_info'], 0)
        result_traj_1 = tree_index1(result['batch_1_info'], 0)
        result_traj_2 = tree_index1(result['batch_2_info'], 0)
        
        traj_batch, targets_V, done = result_traj
        traj_batch_1, targets_V_1, done_1 = result_traj_1
        traj_batch_2, targets_V_2, done_2 = result_traj_2

        ((reach_1_perc, reach_2_perc, reach_perc),
            (reach_idx_1, reach_idx_2, reach_idx)) = calculate_reachreach(traj_batch)
        ((reach_1_perc_1, _, _),
            (reach_idx_1_1, _, _)) = calculate_reachreach(traj_batch_1, reach_type="1")
        ((_, reach_2_perc_2, _),
            (_, reach_idx_2_2, _)) = calculate_reachreach(traj_batch_2, reach_type="2")

        idx = 0

        # reach_idx = calculate_minimal_reach(traj_batch.reach[:, idx])

        info = tree_index2(traj_batch.info, idx)
        info_1 = tree_index2(traj_batch_1.info, idx)
        info_2 = tree_index2(traj_batch_2.info, idx)
        info['reach_index_1'], info['reach_index_2'] = reach_idx_1[idx], reach_idx_2[idx]
        info_1['reach_index_1'], info_1['reach_index_2'] = reach_idx_1_1[idx], np.array(-1)
        info_2['reach_index_1'], info_2['reach_index_2'] = np.array(-1), reach_idx_2_2[idx]

        if config['EXP_NAME'] == 'WindField' or config['EXP_NAME'] == 'WindFieldFull':
            info['u_air'] = env_params.u_air
            info['v_air'] = env_params.v_air
            info['obs'] = env_params.obstacle

        checkpoints.save_checkpoint(ckpt_dir=os.path.abspath(os.path.join("model", config["DIR"])),
                                    target={"policy_network":train_state_policy, 
                                            "value_network":train_state_value,
                                            "policy_reach1_network":train_state_policy_reach1, 
                                            "value_reach1_network":train_state_value_reach1,
                                            "policy_reach2_network":train_state_policy_reach2, 
                                            "value_reach2_network":train_state_value_reach2,
                                            },
                                    step=timestep,
                                    overwrite=True)
        
        if reach_perc > best_score:
            best_score = reach_perc
            checkpoints.save_checkpoint(ckpt_dir=os.path.abspath(os.path.join("model", config["DIR"])),
                                        target={"policy_network":train_state_policy,
                                                "value_network":train_state_value,
                                                "policy_reach1_network":train_state_policy_reach1,
                                                "value_reach1_network":train_state_value_reach1,
                                                "policy_reach2_network":train_state_policy_reach2,
                                                "value_reach2_network":train_state_value_reach2,
                                                },
                                        step=timestep,
                                        prefix="best_",
                                        overwrite=True,)

        policy_decision_sample = traj_batch.policy_taken[:,idx]
        # fig = plot_contour_RRAA((info, info_1, info_2), timestep, config)
        fig = plot_contour_RRAA((info, info_1, info_2), timestep, config, policy_decision_sample=policy_decision_sample)

        fig2 = plot_policy_decision(policy_decision_sample, timestep, config)

        # plot_target(targets_h[:, idx], traj_batch.value_reach[:, idx], traj_batch.reach1[:, idx], traj_batch.reach2[:, idx],
        #             timestep, traj_batch.energy[0, idx], done[:, idx], config)
        # plot_value_target(targets_V[:, idx], traj_batch.value[:, idx], timestep,
        #                   traj_batch.energy[0, idx], done[:, idx], config)
        t1 = time.time()

        if config["USE_WANDB"]:
            wandb.log({
                    #    "not reaching goal": cnt,
                    "actor_loss": jnp.mean(loss_info["actor_loss"]), "value_loss": jnp.mean(loss_info["value_loss"]),
                    #    "entropy_loss": jnp.mean(loss_info["entropy_loss"]),
                    "actor_1_loss": jnp.mean(loss_info_1["actor_loss"]), "value_1_loss": jnp.mean(loss_info_1["value_loss"]),
                    "actor_2_loss": jnp.mean(loss_info_2["actor_loss"]), "value_2_loss": jnp.mean(loss_info_2["value_loss"]),
                    "reach_gamma": result['reach_gamma'][0], "entropy_weight": result['entropy_weight'][0],
                    "Dec. Reach 1 Success %": reach_1_perc_1,
                    "Dec. Reach 2 Success %": reach_2_perc_2,
                    "Reach-Reach Success %": reach_perc,
                    }, step=timestep)
            
            if "F16" not in config["EXP_NAME"]: # FIXME make f16 methods uniform
                wandb.log({
                    'trajectory_sample':wandb.Image(fig),
                    'policy_decision_sample':wandb.Image(fig2),
                }, step=timestep)
            
        # Save video of trajectory 
        if "F16" not in config["EXP_NAME"]:
            if timestep % config['VIDEO_FREQ'] == 0 or timestep == total_timesteps - 1: 
                video_frames = plot_video_contour_RRAA((info, info_1, info_2), timestep, config, save_video=True, log_wandb=config["USE_WANDB"])

        plt.close("all")
        print(f"ITER TIME : {t1-t0:2.1f}s    SUCCESS : (DEC. R1)  {100*reach_1_perc_1:2.1f}%  (DEC. R2)  {100*reach_2_perc_2:2.1f}%  (COM. RR)  {100*reach_perc:2.1f}%")
        # print("Time {}".format(t1-t0))

    return

if __name__ == "__main__":
    config = vars(get_args(sys.argv[1:]))

    debug = True
    if debug:
        # config["EXP_NAME"]="HopperReachReach"
        # config["DIR"]="hopper_reachreach_debug"
        # config["LR"]=3e-4
        # # config["NUM_ENVS"]=128
        # config["NUM_ENVS"]=1
        # config["NUM_STEPS"]=400
        # config["TOTAL_TIMESTEPS"]=50_000_000
        # config["STEP_SCAN"]=4
        # config["UPDATE_EPOCHS"]=10
        # config["NUM_MINIBATCHES"]=32
        # config["GAMMA_ENERGY"]=1.0
        # config["GAMMA_REACH_INIT"]=0.995
        # config["GAMMA_REACH_FINAL"]=0.9995
        # config["GAE_LAMBDA"]=0.95
        # config["CLIP_EPS"]=0.2
        # config["ENT_COEF"]=0.0001
        # config["VF_COEF"]=2.0
        # config["MAX_GRAD_NORM"]=0.5
        # config["ACTIVATION"]="tanh"
        # config["CUDA_USE"]="0,1,2,3"
        # config["ANNEAL_LR"]=True,
        # config["ANNEAL_ENT"]=True
        # config["NAME"]="hopper_debug"
        # config["TEST_MODE"]=True # USES DETERMINISTIC MODELS

        # config["EXP_NAME"]="F16ReachReach"
        # config["DIR"]="F16_rr_verttargs_cutsamp_To80m80s_tjreset_LR2e-3"
        # config["LR"]=2e-3
        # config["NUM_ENVS"]=256
        # config["NUM_STEPS"]=200
        # config["TOTAL_TIMESTEPS"]=100_000_000
        # config["STEP_SCAN"]=10
        # config["UPDATE_EPOCHS"]=10
        # config["NUM_MINIBATCHES"]=64
        # config["GAMMA_ENERGY"]=1.0
        # config["GAMMA_REACH_INIT"]=0.995
        # config["GAMMA_REACH_FINAL"]=0.9995
        # config["GAE_LAMBDA"]=0.95
        # config["CLIP_EPS"]=0.2
        # config["ENT_COEF"]=0.001
        # config["VF_COEF"]=2.0
        # config["MAX_GRAD_NORM"]=0.5
        # config["ACTIVATION"]="tanh"
        # config["CUDA_USE"]="0"
        # config["ANNEAL_LR"]=True
        # config["ANNEAL_ENT"]=True
        # config["NAME"]="F16_rr_verttargs_cutsamp_To80m80s_tjreset_LR2e-3"

        # config["EXP_NAME"]="HalfCheetahReachReach"
        # config["DIR"]="halfcheetah_rr_resetgoal_reachv0.1"
        # config["LR"]=3e-4
        # config["NUM_ENVS"]=128
        # config["NUM_STEPS"]=400
        # config["TOTAL_TIMESTEPS"]=150_000_000
        # config["STEP_SCAN"]=4
        # config["UPDATE_EPOCHS"]=10
        # config["NUM_MINIBATCHES"]=32
        # config["GAMMA_ENERGY"]=1.0
        # config["GAMMA_REACH_INIT"]=0.995
        # config["GAMMA_REACH_FINAL"]=0.9995
        # config["GAE_LAMBDA"]=0.95
        # config["CLIP_EPS"]=0.2
        # config["ENT_COEF"]=0.005
        # config["VF_COEF"]=2.0
        # config["MAX_GRAD_NORM"]=0.5
        # config["ACTIVATION"]="tanh"
        # config["CUDA_USE"]="0"
        # config["ANNEAL_LR"]=True,
        # config["ANNEAL_ENT"]=True
        # config["NAME"]="halfcheetah_rr_resetgoal_reachv0.1"

        config["EXP_NAME"]="HumanoidReachReach"
        config["DIR"]="humanoid_rr_debug_gif_faster_fixed"
        config["LR"]=3e-4
        config["NUM_ENVS"]=128
        config["NUM_STEPS"]=400
        config["TOTAL_TIMESTEPS"]=150_000_000
        config["STEP_SCAN"]=4
        config["UPDATE_EPOCHS"]=10
        config["NUM_MINIBATCHES"]=32
        config["GAMMA_ENERGY"]=1.0
        config["GAMMA_REACH_INIT"]=0.995
        config["GAMMA_REACH_FINAL"]=0.9995
        config["GAE_LAMBDA"]=0.95
        config["CLIP_EPS"]=0.2
        config["ENT_COEF"]=0.005
        config["VF_COEF"]=2.0
        config["MAX_GRAD_NORM"]=0.5
        config["ACTIVATION"]="tanh"
        config["CUDA_USE"]="0"
        config["ANNEAL_LR"]=True,
        config["ANNEAL_ENT"]=True
        config["NAME"]="humanoid_rr_debug_gif_faster_fixed"
    #     # config["TEST_MODE"]=True # USES DETERMINISTIC MODELS

    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = int(
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA_USE']
    folder = os.path.exists("model/{}".format(config['DIR']))
    if not folder:
        os.makedirs("model/{}".format(config['DIR']))
        os.makedirs("model/{}/reach".format(config['DIR']))
        os.makedirs("model/{}/policy".format(config['DIR']))
        os.makedirs("model/{}/value".format(config['DIR']))
        os.makedirs("model/{}/total".format(config['DIR']))
        os.makedirs("model/{}/target".format(config['DIR']))
        os.makedirs("model/{}/value_target".format(config['DIR']))
        os.makedirs("model/{}/state_traj".format(config['DIR']))
    
    envs = get_env(config)
    env, env_reach_1, env_reach_2 = envs
    env_params = env.default_params
    env_params_reach_1 = env_reach_1.default_params
    env_params_reach_2 = env_reach_2.default_params

    if config['EXP_NAME'] == 'WindField':
        env_params = env_params.replace(index=config['SECTION'])
        env_params_reach_1 = env_params_reach_1.replace(index=config['SECTION'])
        env_params_reach_2 = env_params_reach_2.replace(index=config['SECTION'])
    env_paramss = (env_params, env_params_reach_1, env_params_reach_2)

    config["USE_WANDB"] = False #not debug # False for debugging
    if config["USE_WANDB"]:
        wandb.init(project='EC-EFPPO-{}'.format(config["EXP_NAME"]), name=config["NAME"], config=config,
                   entity='braat_brrt')

    config["LOAD_DECOMPOSED"] = False # TODO make arg
    if config["LOAD_DECOMPOSED"]:
        config["LOAD_DEC_DIR"] ="hopper_reachreach_idxsMAX_switchfix_augstate_obsfix_long"
        config["LOAD_DEC_DIR_MODEL"] ="checkpoint_859"

    if 'VIDEO_FREQ' not in config.keys():
        if 'Humanoid' in config['EXP_NAME']:
            config['VIDEO_FREQ'] = 200
        else:
            config['VIDEO_FREQ'] = 25

    rng = jax.random.PRNGKey(20)
    out = train(envs, env_paramss, config, rng) # TODO assumes same env params (should be tuple if diff)
    # NOTE passing multiple envs (composed + decomposed)
    # TODO more elegant use one env w/ diff env_params, but this is safe for now