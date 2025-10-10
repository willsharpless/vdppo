"""
File for Reach-Reach Always-Avoid (RRAA) PPO training.
"""
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

from rraa_rl.src.rl.utils.arguments import get_args
from functools import partial
from typing import Any

from rraa_rl.src.rl.utils.alg_utils import _ppo_vanilla_update, _env_step_rr_vanilla, _env_step_r1_vanilla, _env_step_r2_vanilla
from rraa_rl.src.rl.utils.alg_utils import _env_step_rraa, _env_step_raa, _env_step_a
from rraa_rl.src.env.env_list import get_env
from rraa_rl.src.model.actorcritic import Policy_Network, Value_Network, Policy_Network_Discrete, MoGPolicy_Network
from rraa_rl.src.rl.utils.plot_utils import calculate_minimal_reach, calculate_consumption, calculate_reachreach, plot_target, plot_value_target, plot_contour, plot_contour_RRAA, plot_policy_decision, plot_video_contour_RRAA
from rraa_rl.src.rl.utils.utils import optimizer, get_BuRd, tree_index1, tree_index2
from rraa_rl.src.rl.utils.gae import (Transition_reach,
                              calculate_gae, calculate_gae2, calculate_gae3,
                              calculate_gae_reach, calculate_gae_reach2, calculate_gae_reach3, calculate_gae_reach4, calculate_gae_reachavoid4, calculate_gae_avoid4,
                              calculate_indexs, calculate_indexs2, calculate_indexs3, calculate_indexs3_rr, calculate_indexs_rr)

from rraa_rl.src.env.reach_avoid.humanoid_RR import HUMANOID_TORSO_MIN_Z, HUMANOID_TORSO_MAX_Z

class TrainState(train_state.TrainState):
    mean: Any
    variance: Any
    count: Any

def train(envs, env_paramss, config, rng):
    (env_rraa, 
        env_raa1, env_raa2, 
        env_a, # SINGLE AVOID POLICY
        # env_a1, env_a2 # TWO AVOID POLICIES
    ) = envs
    (env_params_rraa, 
        env_params_raa1, env_params_raa2, 
        env_params_a,
        # env_params_a1, env_params_a2
    ) = env_paramss

    def _train(train_state_total, ent_gamma):
        
        (train_state_policy_rraa, train_state_value_rraa,
            train_state_policy_raa1, train_state_value_raa1,
            train_state_policy_raa2, train_state_value_raa2, 
            train_state_policy_a, train_state_value_a,
            # train_state_policy_a1, train_state_value_a1,
            # train_state_policy_a2, train_state_value_a2, 
            rng_og, timestep) = train_state_total
        
        ####################################################################################################################
        # ROLLOUT RRAA

        # RESET ENV -- RRAA
        rng, _rng = jax.random.split(rng_og)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state_rraa = jax.vmap(env_rraa.reset, in_axes=(0, None))(reset_rng, env_params_rraa)
        rng, _rng = jax.random.split(rng)
        runner_state_standard_rraa = (train_state_policy_rraa, train_state_value_rraa, env_state_rraa, obsv, _rng)
        
        # SPECIAL DECOMPOSED STATES
        decomposed_state_rraa = (train_state_policy_raa1, train_state_value_raa1, 
                            train_state_policy_raa2, train_state_value_raa2, 
                            train_state_policy_a, train_state_value_a,
                            # train_state_policy_a1, train_state_value_a1,
                            # train_state_policy_a2, train_state_value_a2
        )
        force_combined = False # if timestep < 20 else False # inhibits switching until > 20 epochs
        force_reach1, force_reach2 = False, False
        policy_controls = (force_combined, force_reach1, force_reach2)
        runner_state_rraa = (*runner_state_standard_rraa, decomposed_state_rraa, policy_controls)

        # COLLECT TRAJECTORY COMPOSED
        runner_state_rraa, traj_batch_rraa = jax.lax.scan(
            env_step_rraa, runner_state_rraa, None, config["NUM_STEPS"]
        )

        ####################################################################################################################
        # ROLLOUT RAA 1 (COUPLED OR STANDARD INIT)

        # RESET ENV - (COUPLED OR STANDARD INIT)
        rng, _rng = jax.random.split(rng_og)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])

        if config["DEC_INIT_TYPE"] == "standard": 
            obsv_raa1, env_state_raa1 = jax.vmap(env_raa1.reset, in_axes=(0, None))(reset_rng, env_params_raa1)
        
        elif "toinput" in config["DEC_INIT_TYPE"]: 
            rng_raa1, _rng_raa1 = jax.random.split(rng)
            
            # Select first reach2 step in composed rollout for initial decomposed reach1 state 
            if config["DEC_INIT_TYPE"] == "toinput_goal":
                random_index_pre = jax.random.randint(_rng_raa1, shape=(config["NUM_ENVS"],), minval=0, maxval=config["NUM_STEPS"])
                reach2_idx_pre = (traj_batch_rraa.reach2 < 0).argmax(axis=0)
                reach2_idx = jnp.where(jnp.any((traj_batch_rraa.reach2 < 0) == 1, axis=0), reach2_idx_pre, config["NUM_STEPS"])
                random_index = jnp.where(jnp.any(traj_batch_rraa.reach2 < 0, axis=0), reach2_idx, random_index_pre)
                # random_index = jnp.where(jnp.any(traj_batch.reach2 < 0, axis=0), reach2_idx - 10, random_index_pre) # FIXME: before reaching?
            # Select random step in composed rollout for initial decomposed reach1 state
            else:
                random_index = jax.random.randint(_rng_raa1, shape=(config["NUM_ENVS"],), minval=0, maxval=config["NUM_STEPS"])
            # FIXME FIXME HUMANOID: when terminating unhealthy via brax internals, does not filtering by done lead to misassociated trajectories? FIXME FIXME
            # NOTE: I think to input random will do -- less efficient but wont be stuck in bad states at least
            # NOTE: but it seems internal reset will restore whatever trajectory starting pos was, so all roll-outs in this batch will start there 
            # so it would be better to initialize to random set?


            # Multiple random indices
            if "Hopper" in config["EXP_NAME"] or "HalfCheetah" in config["EXP_NAME"] or "Point" in config["EXP_NAME"]:
                traj_batch_observations_full = traj_batch_rraa.obs 
                untrans_traj_batch_observations_full = env_rraa.untransform_obs(traj_batch_observations_full)
                untrans_traj_batch_observations_full = jnp.transpose(untrans_traj_batch_observations_full, axes=(1, 0, 2))
                untrans_traj_batch_observations = jax.vmap(lambda obs, idx: obs[idx])(untrans_traj_batch_observations_full, random_index)
                obsv_raa1, env_state_raa1 = jax.vmap(env_raa1.reset_toinput, in_axes=(0, 0, None))(reset_rng, untrans_traj_batch_observations, env_params_raa1) 

            elif "Humanoid" in config["EXP_NAME"]:

                # Filter Unhealthy Humanoid Trajectories from reset
                unhealthy_traj = jnp.any(jnp.logical_or(
                    traj_batch_rraa.info['torso'][..., 2] < HUMANOID_TORSO_MIN_Z,
                    traj_batch_rraa.info['torso'][..., 2] > HUMANOID_TORSO_MAX_Z
                ), axis=0)
                random_index_healthy = jnp.where(unhealthy_traj, reach2_idx_pre, random_index) 
                # when unhealthy -> set first reach2 index and if no reach2, then init (argmax defaults to 0)

                # FIXME: humanoid._get_obs() needs an action, meaning we should pass reset action too, for now just zeros
                traj_batch_observations_full = traj_batch_rraa.obs 
                untrans_traj_batch_observations_full = env_rraa.untransform_obs(traj_batch_observations_full)
                untrans_traj_batch_observations_full = jnp.transpose(untrans_traj_batch_observations_full, axes=(1, 0, 2))
                untrans_traj_batch_observations = jax.vmap(lambda obs, idx: obs[idx])(untrans_traj_batch_observations_full, random_index_healthy)
                obsv_raa1, env_state_raa1 = jax.vmap(env_raa1.reset_toinput, in_axes=(0, 0, None))(reset_rng, untrans_traj_batch_observations, env_params_raa1) 

            elif "F16" in config["EXP_NAME"]:
                traj_batch_states = traj_batch_rraa.info['state']
                traj_batch_states = jnp.transpose(traj_batch_states, axes=(1, 0, 2))
                reset_states = jax.vmap(lambda obs, idx: obs[idx])(traj_batch_states, random_index)
                obsv_raa1, env_state_raa1 = jax.vmap(env_raa1.reset_env_toinput, in_axes=(0, None))(reset_states, env_params_raa1) 
            
            else:
                raise NotImplementedError("Unknown environment type for toinput reset")
            
        else:
            raise ValueError(f"Unknown init type: {config["DEC_INIT_TYPE"]}")
        
        rng, _rng = jax.random.split(rng)
        runner_state_standard_raa1 = (train_state_policy_rraa, train_state_value_rraa, env_state_raa1, obsv_raa1, _rng)
        
        # RAA DECOMPOSED STATES - 1
        decomposed_state_raa1 = (
            train_state_policy_raa1, train_state_value_raa1, 
            # train_state_policy_raa2, train_state_value_raa2, 
            train_state_policy_a, train_state_value_a,
            # train_state_policy_a1, train_state_value_a1,
            # train_state_policy_a2, train_state_value_a2
        )
        force_reach1, force_reach2 = True, False
        policy_controls_raa1 = (force_combined, force_reach1, force_reach2)
        runner_state_raa1 = (*runner_state_standard_raa1, decomposed_state_raa1, policy_controls_raa1)

        # COLLECT TRAJECTORY DECOMPOSED - 1
        runner_state_raa1, traj_batch_raa1 = jax.lax.scan(
            env_step_raa1, runner_state_raa1, None, config["NUM_STEPS"]
        )

        ####################################################################################################################
        # ROLLOUT RAA 2

        # RESET ENV - (COUPLED OR STANDARD INIT)
        rng, _rng = jax.random.split(rng_og)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        
        if config["DEC_INIT_TYPE"] == "standard":
            obsv_raa2, env_state_raa2 = jax.vmap(env_raa2.reset, in_axes=(0, None))(reset_rng, env_params_raa2)
 
        elif "toinput" in config["DEC_INIT_TYPE"]: 
            rng_raa2, _rng_raa2 = jax.random.split(rng)
            
            # Select first reach1 step in composed rollout for initial decomposed reach2 state 
            if config["DEC_INIT_TYPE"] == "toinput_goal":
                random_index_pre = jax.random.randint(rng_raa2, shape=(config["NUM_ENVS"],), minval=0, maxval=config["NUM_STEPS"])
                reach1_idx_pre = (traj_batch_rraa.reach1 < 0).argmax(axis=0)
                reach1_idx = jnp.where(jnp.any((traj_batch_rraa.reach1 < 0) == 1, axis=0), reach1_idx_pre, config["NUM_STEPS"])
                random_index = jnp.where(jnp.any(traj_batch_rraa.reach1 < 0, axis=0), reach1_idx, random_index_pre)
                # random_index = jnp.where(jnp.any(traj_batch.reach1 < 0, axis=0), reach1_idx - 10, random_index_pre) # FIXME: before reaching?

            # Select random step in composed rollout for initial decomposed reach2 state 
            else:
                random_index = jax.random.randint(rng_raa2, shape=(config["NUM_ENVS"],), minval=0, maxval=config["NUM_STEPS"])
            # random_index = jax.random.randint(_rng_raa2, shape=(untrans_traj_batch_observations_full.shape[0],), minval=0, maxval=untrans_traj_batch_observations_full.shape[1])

            # Multiple random indices
            if "Hopper" in config["EXP_NAME"] or "HalfCheetah" in config["EXP_NAME"] or "Point" in config["EXP_NAME"]:
                traj_batch_observations_full = traj_batch_rraa.obs 
                untrans_traj_batch_observations_full = env_rraa.untransform_obs(traj_batch_observations_full)
                untrans_traj_batch_observations_full = jnp.transpose(untrans_traj_batch_observations_full, axes=(1, 0, 2))
                untrans_traj_batch_observations = jax.vmap(lambda obs, idx: obs[idx])(untrans_traj_batch_observations_full, random_index)
                # obsv_reach_1, env_state_reach_1 = jax.vmap(env_reach_1.reset_toinput, in_axes=(0, 0, None))(reset_rng, untrans_traj_batch_observations, env_params_reach_1) 
                obsv_raa2, env_state_raa2 = jax.vmap(env_raa2.reset_toinput, in_axes=(0, 0, None))(reset_rng, untrans_traj_batch_observations, env_params_raa2) 

            elif "Humanoid" in config["EXP_NAME"]:

                ## Filter Unhealthy Humanoid Trajectories from reset
                unhealthy_traj = jnp.any(jnp.logical_or(
                    traj_batch_rraa.info['torso'][..., 2] < HUMANOID_TORSO_MIN_Z,
                    traj_batch_rraa.info['torso'][..., 2] > HUMANOID_TORSO_MAX_Z
                ), axis=0)
                random_index_healthy = jnp.where(unhealthy_traj, reach1_idx_pre, random_index) # set to init when unhealthy
                # when unhealthy -> set first reach1 index and if no reach1, then init (argmax defaults to 0)

                # FIXME: humanoid._get_obs() needs an action, meaning we should pass reset action too, for now just zeros
                traj_batch_observations_full = traj_batch_rraa.obs 
                untrans_traj_batch_observations_full = env_rraa.untransform_obs(traj_batch_observations_full)
                untrans_traj_batch_observations_full = jnp.transpose(untrans_traj_batch_observations_full, axes=(1, 0, 2))
                untrans_traj_batch_observations = jax.vmap(lambda obs, idx: obs[idx])(untrans_traj_batch_observations_full, random_index_healthy)
                # obsv_reach_1, env_state_reach_1 = jax.vmap(env_reach_1.reset_toinput, in_axes=(0, 0, None))(reset_rng, untrans_traj_batch_observations, env_params_reach_1) 
                obsv_raa2, env_state_raa2 = jax.vmap(env_raa2.reset_toinput, in_axes=(0, 0, None))(reset_rng, untrans_traj_batch_observations, env_params_raa2) 

            elif "F16" in config["EXP_NAME"]:
                traj_batch_states = traj_batch_rraa.info['state']
                traj_batch_states = jnp.transpose(traj_batch_states, axes=(1, 0, 2))
                reset_states = jax.vmap(lambda obs, idx: obs[idx])(traj_batch_states, random_index)
                # obsv_reach_1, env_state_reach_1 = jax.vmap(env_reach_1.reset_env_toinput, in_axes=(0, None))(reset_states, env_params_reach_1) 
                obsv_raa2, env_state_raa2 = jax.vmap(env_raa2.reset_env_toinput, in_axes=(0, None))(reset_states, env_params_raa2) 
        
            else:
                raise NotImplementedError("Unknown environment type for toinput reset")
        else:
            raise ValueError(f"Unknown init type: {config["DEC_INIT_TYPE"]}")

        rng, _rng = jax.random.split(rng)
        runner_state_standard_raa2 = (train_state_policy_rraa, train_state_value_rraa, env_state_raa2, obsv_raa2, _rng)
        # TODO clean this up so that the env_step only uses the following decomposed state 
        
        # SPECIAL DECOMPOSED STATES - RAA 2
        decomposed_state_raa2 = ( 
            train_state_policy_raa2, train_state_value_raa2, 
            train_state_policy_a, train_state_value_a
        )
        force_reach1, force_reach2 = False, True
        policy_controls_reach2 = (force_combined, force_reach1, force_reach2)
        runner_state_raa2 = (*runner_state_standard_raa2, decomposed_state_raa2, policy_controls_reach2)

        # COLLECT TRAJECTORY DECOMPOSED - RAA 2
        runner_state_raa2, traj_batch_raa2 = jax.lax.scan(
            env_step_raa2, runner_state_raa2, None, config["NUM_STEPS"]
        )

        ####################################################################################################################
        # ROLLOUT A
        # NOTE, assuming same avoid value/policy -> for A, sample from (couple to) both raa1 & raa2 batches
        # (50% from each TODO could be smarter, eg. prioritize split by reaching qty)

        # RESET ENV - (COUPLED OR STANDARD INIT)
        rng_avoid, _rng_avoid = jax.random.split(rng_og)
        reset_rng_avoid = jax.random.split(_rng_avoid, config["NUM_ENVS"])
        
        if "toinput" in config["DEC_INIT_TYPE"]: 
            # Select observations from standard rollout to use for initial avoid state 

            # Init to first reached state, if none then random
            if config["DEC_INIT_TYPE"] == "toinput_goal":
                random_index_pre_raa1 = jax.random.randint(_rng_avoid, shape=(config["NUM_ENVS"],), minval=0, maxval=config["NUM_STEPS"])
                reach_idx_raa1 = (traj_batch_raa1.reach < 0).argmax(axis=0)
                random_index_raa1 = jnp.where(jnp.any((traj_batch_raa1.reach < 0), axis=0), reach_idx_raa1, random_index_pre_raa1)
                
                random_index_pre_raa2 = jax.random.randint(_rng_avoid, shape=(config["NUM_ENVS"],), minval=0, maxval=config["NUM_STEPS"])
                reach_idx_raa2 = (traj_batch_raa2.reach < 0).argmax(axis=0)
                random_index_raa2 = jnp.where(jnp.any((traj_batch_raa2.reach < 0), axis=0), reach_idx_raa2, random_index_pre_raa2)

            # Init to first reached state if avoided, if none then random before crash
            elif "safegoal" in config["DEC_INIT_TYPE"]:
                avoid_idx_pre_raa1 = (traj_batch_raa1.avoid > 0).argmax(axis=0)
                avoid_idx_raa1 = jnp.where(jnp.any((traj_batch_raa1.avoid > 0) == 1, axis=0), avoid_idx_pre_raa1, config["NUM_STEPS"])
                avoid_idx_pre_raa2 = (traj_batch_raa2.avoid > 0).argmax(axis=0)
                avoid_idx_raa2 = jnp.where(jnp.any((traj_batch_raa2.avoid > 0) == 1, axis=0), avoid_idx_pre_raa2, config["NUM_STEPS"])

                reach_idx_pre_raa1 = (traj_batch_raa1.reach < 0).argmax(axis=0)
                reach_idx_raa1 = jnp.where(jnp.any((traj_batch_raa1.reach < 0) == 1, axis=0), reach_idx_pre_raa1, config["NUM_STEPS"])
                reach_idx_pre_raa2 = (traj_batch_raa2.reach < 0).argmax(axis=0)
                reach_idx_raa2 = jnp.where(jnp.any((traj_batch_raa2.reach < 0) == 1, axis=0), reach_idx_pre_raa2, config["NUM_STEPS"])

                safe_buffer = 50
                if config["DEC_INIT_TYPE"] == "toinput_safegoal_nearcrash":
                    random_index_precrash_raa1 = jax.random.randint(_rng_avoid, shape=(config["NUM_ENVS"],), minval=avoid_idx_raa1//2, maxval=(3 * avoid_idx_raa1)//4) # sample otw to crashing
                    random_index_precrash_raa2 = jax.random.randint(_rng_avoid, shape=(config["NUM_ENVS"],), minval=avoid_idx_raa2//2, maxval=(3 * avoid_idx_raa2)//4) # sample otw to crashing
                elif config["DEC_INIT_TYPE"] == "toinput_safegoal":
                    random_index_precrash_raa1 = jax.random.randint(_rng_avoid, shape=(config["NUM_ENVS"],), minval=0, maxval=avoid_idx_raa1) # sample before crashing
                    # random_index_precrash = jax.random.randint(_rng_avoid, shape=(config["NUM_ENVS"],), minval=0, maxval=avoid_idx//2) # sample well before crashing
                    random_index_precrash_raa2 = jax.random.randint(_rng_avoid, shape=(config["NUM_ENVS"],), minval=0, maxval=avoid_idx_raa2) # sample before crashing
                else:
                    raise ValueError(f"Unknown init type: {config["DEC_INIT_TYPE"]}")

                random_index_raa1 = jnp.where(jnp.logical_and(jnp.any(traj_batch_raa1.reach < 0, axis=0), # reached
                                                        #  reach_idx < avoid_idx), # reached before crash
                                                         reach_idx_raa1 - safe_buffer < avoid_idx_raa1), # reached swell before crash
                                        reach_idx_raa1, random_index_precrash_raa1)
                
                random_index_raa2 = jnp.where(jnp.logical_and(jnp.any(traj_batch_raa2.reach < 0, axis=0), # reached
                                                        #  reach_idx < avoid_idx), # reached before crash
                                                         reach_idx_raa2 - safe_buffer < avoid_idx_raa2), # reached swell before crash
                                        reach_idx_raa2, random_index_precrash_raa2)
                
            # FIXME FIXME when terminating unhealthy via brax internals, does not filtering by done lead to misassociated trajectories? FIXME FIXME

            # Init to random point along rollout
            elif config["DEC_INIT_TYPE"] == "toinput":
                random_index_raa1 = jax.random.randint(_rng_avoid, shape=(config["NUM_ENVS"],), minval=0, maxval=config["NUM_STEPS"])
                random_index_raa2 = jax.random.randint(_rng_avoid, shape=(config["NUM_ENVS"],), minval=0, maxval=config["NUM_STEPS"])
            else:
                raise ValueError(f"Unknown init type: {config["DEC_INIT_TYPE"]}")

            # Multiple random indices
            if not "F16" in config["EXP_NAME"]:
                untrans_traj_batch_observations_full_raa1 = jnp.transpose(env_raa1.untransform_obs(traj_batch_raa1.obs), axes=(1, 0, 2))
                untrans_traj_batch_observations_raa1 = jax.vmap(lambda obs, idx: obs[idx])(untrans_traj_batch_observations_full_raa1, random_index_raa1)
                untrans_traj_batch_observations_full_raa2 = jnp.transpose(env_raa2.untransform_obs(traj_batch_raa2.obs), axes=(1, 0, 2))
                untrans_traj_batch_observations_raa2 = jax.vmap(lambda obs, idx: obs[idx])(untrans_traj_batch_observations_full_raa2, random_index_raa2)

                # randomly combine half of each untrans_traj_batch_observations
                untrans_traj_batch_observations = jnp.where(jax.random.bernoulli(_rng_avoid, p=0.5, shape=(config["NUM_ENVS"],))[:, None], untrans_traj_batch_observations_raa1, untrans_traj_batch_observations_raa2)

                obsv_a, env_state_a = jax.vmap(env_a.reset_toinput, in_axes=(0, 0, None))(reset_rng_avoid, untrans_traj_batch_observations, env_params_a) 

            # elif "Humanoid" in config["EXP_NAME"]:
                # FIXME: humanoid._get_obs() needs an action, meaning should pass reset action too, for now just zeros

            else:
                traj_batch_states = jnp.transpose(traj_batch_raa1.info['state'], axes=(1, 0, 2))
                reset_states = jax.vmap(lambda obs, idx: obs[idx])(traj_batch_states, random_index)
                obsv_a, env_state_a = jax.vmap(env_a.reset_env_toinput, in_axes=(0, None))(reset_states, env_params_a) 
                
        elif config["DEC_INIT_TYPE"] == "standard":
            obsv_a, env_state_a = jax.vmap(env_a.reset, in_axes=(0, None))(reset_rng_avoid, env_params_a) # NOTE: old standard reset

        elif config["DEC_INIT_TYPE"] == "fullrandom": # FIXME NIKHIL'S OLD METHOD, not used anymore / not the random we usually mean
            obsv_a, env_state_a = jax.vmap(env_a.reset_fullrandom, in_axes=(0, None))(reset_rng_avoid, env_params_a) # NOTE: old standard reset
        
        else:
            raise ValueError(f"Unknown init type: {config["DEC_INIT_TYPE"]}")
        
        rng, _rng = jax.random.split(rng_avoid)
        runner_state_standard_a = (train_state_policy_rraa, train_state_value_rraa, env_state_a, obsv_a, _rng)

        # SPECIAL DECOMPOSED STATES - A
        decomposed_state_a = (train_state_policy_a, train_state_value_a)
        force_avoid = True 
        policy_controls_avoid = (force_combined, force_avoid)
        runner_state_a = (*runner_state_standard_a, decomposed_state_a, policy_controls_avoid)

        # COLLECT TRAJECTORY DECOMPOSED - A
        runner_state_a, traj_batch_a = jax.lax.scan(
            env_step_a, runner_state_a, None, config["NUM_STEPS"]
        )
        
        # ####################################################################################################################
        # # ROLLOUT A1

        # # RESET ENV - (COUPLED OR STANDARD INIT)
        # rng_avoid, _rng_avoid = jax.random.split(rng_og)
        # reset_rng_avoid = jax.random.split(_rng_avoid, config["NUM_ENVS"])
        
        # if "toinput" in config["DEC_INIT_TYPE"]: 
        #     # Select observations from standard rollout to use for initial avoid state 

        #     # Init to first reached state, if none then random
        #     if config["DEC_INIT_TYPE"] == "toinput_goal":
        #         random_index_pre = jax.random.randint(_rng_avoid, shape=(config["NUM_ENVS"],), minval=0, maxval=config["NUM_STEPS"])
        #         reach_idx = (traj_batch_raa1.reach < 0).argmax(axis=0)
        #         random_index = jnp.where(jnp.any((traj_batch_raa1.reach < 0), axis=0), reach_idx, random_index_pre)

        #     # Init to first reached state if avoided, if none then random before crash
        #     elif "safegoal" in config["DEC_INIT_TYPE"]:
        #         avoid_idx_pre = (traj_batch_raa1.avoid > 0).argmax(axis=0)
        #         avoid_idx = jnp.where(jnp.any((traj_batch_raa1.avoid > 0) == 1, axis=0), avoid_idx_pre, config["NUM_STEPS"])

        #         reach_idx_pre = (traj_batch_raa1.reach < 0).argmax(axis=0)
        #         reach_idx = jnp.where(jnp.any((traj_batch_raa1.reach < 0) == 1, axis=0), reach_idx_pre, config["NUM_STEPS"])

        #         safe_buffer = 50
        #         if config["DEC_INIT_TYPE"] == "toinput_safegoal_nearcrash":
        #             random_index_precrash = jax.random.randint(_rng_avoid, shape=(config["NUM_ENVS"],), minval=avoid_idx//2, maxval=(3 * avoid_idx)//4) # sample otw to crashing
        #         elif config["DEC_INIT_TYPE"] == "toinput_safegoal":
        #             random_index_precrash = jax.random.randint(_rng_avoid, shape=(config["NUM_ENVS"],), minval=0, maxval=avoid_idx) # sample before crashing
        #             # random_index_precrash = jax.random.randint(_rng_avoid, shape=(config["NUM_ENVS"],), minval=0, maxval=avoid_idx//2) # sample well before crashing
        #         else:
        #             raise ValueError(f"Unknown init type: {config["DEC_INIT_TYPE"]}")

        #         random_index = jnp.where(jnp.logical_and(jnp.any(traj_batch_raa1.reach < 0, axis=0), # reached
        #                                                 #  reach_idx < avoid_idx), # reached before crash
        #                                                  reach_idx - safe_buffer < avoid_idx), # reached swell before crash
        #                                 reach_idx, random_index_precrash)
        #     # FIXME FIXME when terminating unhealthy via brax internals, does not filtering by done lead to misassociated trajectories? FIXME FIXME

        #     # Init to random point along rollout
        #     elif config["DEC_INIT_TYPE"] == "toinput":
        #         random_index = jax.random.randint(_rng_avoid, shape=(config["NUM_ENVS"],), minval=0, maxval=config["NUM_STEPS"])
        #     else:
        #         raise ValueError(f"Unknown init type: {config["DEC_INIT_TYPE"]}")

        #     # Multiple random indices
        #     if "Hopper" in config["EXP_NAME"] or "Cheetah" in config["EXP_NAME"]  or "Point" in config["EXP_NAME"]:
        #         traj_batch_observations_full = traj_batch_raa1.obs 
        #         untrans_traj_batch_observations_full = env_raa1.untransform_obs(traj_batch_observations_full)
        #         untrans_traj_batch_observations_full = jnp.transpose(untrans_traj_batch_observations_full, axes=(1, 0, 2))
        #         untrans_traj_batch_observations = jax.vmap(lambda obs, idx: obs[idx])(untrans_traj_batch_observations_full, random_index)
        #         obsv_a1, env_state_a1 = jax.vmap(env_a1.reset_toinput, in_axes=(0, 0, None))(reset_rng_avoid, untrans_traj_batch_observations, env_params_a1) 

        #     elif "Humanoid" in config["EXP_NAME"]:
        #         # FIXME: humanoid._get_obs() needs an action, meaning should pass reset action too, for now just zeros
        #         traj_batch_observations_full = traj_batch_raa1.obs 
        #         untrans_traj_batch_observations_full = env_raa1.untransform_obs(traj_batch_observations_full)
        #         untrans_traj_batch_observations_full = jnp.transpose(untrans_traj_batch_observations_full, axes=(1, 0, 2))
        #         untrans_traj_batch_observations = jax.vmap(lambda obs, idx: obs[idx])(untrans_traj_batch_observations_full, random_index)
        #         obsv_a1, env_state_a1 = jax.vmap(env_a1.reset_toinput, in_axes=(0, 0, None))(reset_rng_avoid, untrans_traj_batch_observations, env_params_a1) 

        #     elif "F16" in config["EXP_NAME"]:
        #         traj_batch_states = traj_batch_raa1.info['state']
        #         traj_batch_states = jnp.transpose(traj_batch_states, axes=(1, 0, 2))
        #         reset_states = jax.vmap(lambda obs, idx: obs[idx])(traj_batch_states, random_index)
        #         # obsv_reach_1, env_state_reach_1 = jax.vmap(env_reach_1.reset_env_toinput, in_axes=(0, None))(reset_states, env_params_reach_1) 
        #         obsv_a1, env_state_a1 = jax.vmap(env_a1.reset_env_toinput, in_axes=(0, None))(reset_states, env_params_a1) 
                
        # elif config["DEC_INIT_TYPE"] == "standard":
        #     obsv_a1, env_state_a1 = jax.vmap(env_a1.reset, in_axes=(0, None))(reset_rng_avoid, env_params_a1) # NOTE: old standard reset

        # elif config["DEC_INIT_TYPE"] == "fullrandom": # FIXME NIKHIL'S OLD METHOD, not used anymore / not the random we usually mean
        #     obsv_a1, env_state_a1 = jax.vmap(env_a1.reset_fullrandom, in_axes=(0, None))(reset_rng_avoid, env_params_a1) # NOTE: old standard reset
        
        # else:
        #     raise ValueError(f"Unknown init type: {config["DEC_INIT_TYPE"]}")
        
        # rng, _rng = jax.random.split(rng_avoid)
        # runner_state_standard_a1 = (train_state_policy_rraa, train_state_value_rraa, env_state_a1, obsv_a1, _rng)

        # # SPECIAL DECOMPOSED STATES - A1
        # decomposed_state_a1 = (train_state_policy_a1, train_state_value_a1)
        # force_avoid = True 
        # policy_controls_avoid = (force_combined, force_avoid)
        # runner_state_a1 = (*runner_state_standard_a1, decomposed_state_a1, policy_controls_avoid)

        # # COLLECT TRAJECTORY DECOMPOSED - A1
        # runner_state_a1, traj_batch_a1 = jax.lax.scan(
        #     env_step_a1, runner_state_a1, None, config["NUM_STEPS"]
        # )

        # ####################################################################################################################
        # # ROLLOUT A2

        # # RESET ENV - (COUPLED OR STANDARD INIT)
        # rng_avoid, _rng_avoid = jax.random.split(rng_og)
        # reset_rng_avoid = jax.random.split(_rng_avoid, config["NUM_ENVS"])
        
        # if "toinput" in config["DEC_INIT_TYPE"]: 
        #     # Select observations from standard rollout to use for initial avoid state 

        #     # Init to first reached state, if none then random
        #     if config["DEC_INIT_TYPE"] == "toinput_goal":
        #         random_index_pre = jax.random.randint(_rng_avoid, shape=(config["NUM_ENVS"],), minval=0, maxval=config["NUM_STEPS"])
        #         reach_idx = (traj_batch_raa2.reach < 0).argmax(axis=0)
        #         random_index = jnp.where(jnp.any((traj_batch_raa2.reach < 0), axis=0), reach_idx, random_index_pre)

        #     # Init to first reached state if avoided, if none then random before crash
        #     elif "safegoal" in config["DEC_INIT_TYPE"]:
        #         avoid_idx_pre = (traj_batch_raa2.avoid > 0).argmax(axis=0)
        #         avoid_idx = jnp.where(jnp.any((traj_batch_raa2.avoid > 0) == 1, axis=0), avoid_idx_pre, config["NUM_STEPS"])

        #         reach_idx_pre = (traj_batch_raa2.reach < 0).argmax(axis=0)
        #         reach_idx = jnp.where(jnp.any((traj_batch_raa2.reach < 0) == 1, axis=0), reach_idx_pre, config["NUM_STEPS"])

        #         safe_buffer = 50
        #         if config["DEC_INIT_TYPE"] == "toinput_safegoal_nearcrash":
        #             random_index_precrash = jax.random.randint(_rng_avoid, shape=(config["NUM_ENVS"],), minval=avoid_idx//2, maxval=(3 * avoid_idx)//4) # sample otw to crashing
        #         elif config["DEC_INIT_TYPE"] == "toinput_safegoal":
        #             random_index_precrash = jax.random.randint(_rng_avoid, shape=(config["NUM_ENVS"],), minval=0, maxval=avoid_idx) # sample before crashing
        #             # random_index_precrash = jax.random.randint(_rng_avoid, shape=(config["NUM_ENVS"],), minval=0, maxval=avoid_idx//2) # sample well before crashing
        #         else:
        #             raise ValueError(f"Unknown init type: {config["DEC_INIT_TYPE"]}")

        #         random_index = jnp.where(jnp.logical_and(jnp.any(traj_batch_raa2.reach2 < 0, axis=0), # reached
        #                                                 #  reach_idx < avoid_idx), # reached before crash
        #                                                  reach_idx - safe_buffer < avoid_idx), # reached swell before crash
        #                                 reach_idx, random_index_precrash)
        #     # FIXME FIXME when terminating unhealthy via brax internals, does not filtering by done lead to misassociated trajectories? FIXME FIXME

        #     # Init to random point along rollout
        #     elif config["DEC_INIT_TYPE"] == "toinput":
        #         random_index = jax.random.randint(_rng_avoid, shape=(config["NUM_ENVS"],), minval=0, maxval=config["NUM_STEPS"])
        #     else:
        #         raise ValueError(f"Unknown init type: {config["DEC_INIT_TYPE"]}")

        #     # Multiple random indices
        #     if "Hopper" in config["EXP_NAME"] or "Cheetah" in config["EXP_NAME"]  or "Point" in config["EXP_NAME"]:
        #         traj_batch_observations_full = traj_batch_raa2.obs 
        #         untrans_traj_batch_observations_full = env_raa2.untransform_obs(traj_batch_observations_full)
        #         untrans_traj_batch_observations_full = jnp.transpose(untrans_traj_batch_observations_full, axes=(1, 0, 2))
        #         untrans_traj_batch_observations = jax.vmap(lambda obs, idx: obs[idx])(untrans_traj_batch_observations_full, random_index)
        #         obsv_a2, env_state_a2 = jax.vmap(env_a2.reset_toinput, in_axes=(0, 0, None))(reset_rng_avoid, untrans_traj_batch_observations, env_params_a2) 

        #     elif "Humanoid" in config["EXP_NAME"]:
        #         # FIXME: humanoid._get_obs() needs an action, meaning should pass reset action too, for now just zeros
        #         traj_batch_observations_full = traj_batch_raa2.obs 
        #         untrans_traj_batch_observations_full = env_raa2.untransform_obs(traj_batch_observations_full)
        #         untrans_traj_batch_observations_full = jnp.transpose(untrans_traj_batch_observations_full, axes=(1, 0, 2))
        #         untrans_traj_batch_observations = jax.vmap(lambda obs, idx: obs[idx])(untrans_traj_batch_observations_full, random_index)
        #         obsv_a2, env_state_a2 = jax.vmap(env_a2.reset_toinput, in_axes=(0, 0, None))(reset_rng_avoid, untrans_traj_batch_observations, env_params_a2) 

        #     elif "F16" in config["EXP_NAME"]:
        #         traj_batch_states = traj_batch_raa2.info['state']
        #         traj_batch_states = jnp.transpose(traj_batch_states, axes=(1, 0, 2))
        #         reset_states = jax.vmap(lambda obs, idx: obs[idx])(traj_batch_states, random_index)
        #         # obsv_reach_1, env_state_reach_1 = jax.vmap(env_reach_1.reset_env_toinput, in_axes=(0, None))(reset_states, env_params_reach_1) 
        #         obsv_a2, env_state_a2 = jax.vmap(env_a2.reset_env_toinput, in_axes=(0, None))(reset_states, env_params_a2) 
                
        # elif config["DEC_INIT_TYPE"] == "standard":
        #     obsv_a2, env_state_a2 = jax.vmap(env_a2.reset, in_axes=(0, None))(reset_rng_avoid, env_params_a2) # NOTE: old standard reset

        # elif config["DEC_INIT_TYPE"] == "fullrandom": # FIXME NIKHIL'S OLD METHOD, not used anymore / not the random we usually mean
        #     obsv_a2, env_state_a2 = jax.vmap(env_a2.reset_fullrandom, in_axes=(0, None))(reset_rng_avoid, env_params_a2) # NOTE: old standard reset
        
        # else:
        #     raise ValueError(f"Unknown init type: {config["DEC_INIT_TYPE"]}")
        
        # rng, _rng = jax.random.split(rng_avoid)
        # runner_state_standard_a2 = (train_state_policy_rraa, train_state_value_rraa, env_state_a2, obsv_a2, _rng)

        # # SPECIAL DECOMPOSED STATES - A2
        # decomposed_state_a2 = (train_state_policy_a2, train_state_value_a2)
        # force_avoid = True 
        # policy_controls_avoid = (force_combined, force_avoid)
        # runner_state_a2 = (*runner_state_standard_a2, decomposed_state_a2, policy_controls_avoid)

        # # COLLECT TRAJECTORY DECOMPOSED - A2
        # runner_state_a2, traj_batch_a2 = jax.lax.scan(
        #     env_step_a2, runner_state_a2, None, config["NUM_STEPS"]
        # )

        ####################################################################################################################
        # UPDATE RRAA
        
        # CALCULATE COMPOSED ADVANTAGE
        (train_state_policy_rraa, train_state_value_rraa, env_state_rraa, last_obs, rng,
          decomposed_state, policy_controls) = runner_state_rraa

        last_val_rraa = train_state_value_rraa.apply_fn(train_state_value_rraa.params, last_obs)
        last_val_raa1 = train_state_value_raa1.apply_fn(train_state_value_raa1.params, last_obs)
        last_val_raa2 = train_state_value_raa2.apply_fn(train_state_value_raa2.params, last_obs)

        # DECOMPOSED REACH VALUES ON COMPOSED PPO ACTOR ROLL OUT
        # reach1_append = jnp.concatenate((traj_batch.reach1, jnp.expand_dims(env_state.reach1, axis=1).T))
        # V_reach1_append = jnp.concatenate((traj_batch.value_reach1, jnp.expand_dims(last_val1, axis=1).T))
        # reach2_append = jnp.concatenate((traj_batch.reach2, jnp.expand_dims(env_state.reach2, axis=1).T))
        # V_reach2_append = jnp.concatenate((traj_batch.value_reach2, jnp.expand_dims(last_val2, axis=1).T))
        # V_append = jnp.concatenate((traj_batch.value, jnp.expand_dims(last_val, axis=1).T))
        
        V_rraa_append = jnp.concatenate((traj_batch_rraa.value, jnp.expand_dims(last_val_rraa, axis=1).T))
        r1_append = jnp.concatenate((traj_batch_rraa.reach1, jnp.expand_dims(env_state_rraa.reach1, axis=1).T))
        V_raa1_append = jnp.concatenate((traj_batch_rraa.value_raa1, jnp.expand_dims(last_val_raa1, axis=1).T))
        r2_append = jnp.concatenate((traj_batch_rraa.reach2, jnp.expand_dims(env_state_rraa.reach2, axis=1).T))
        V_raa2_append = jnp.concatenate((traj_batch_rraa.value_raa2, jnp.expand_dims(last_val_raa2, axis=1).T))

        a_append = jnp.concatenate((traj_batch_rraa.avoid, jnp.expand_dims(env_state_rraa.avoid, axis=1).T))
        # a1_append = jnp.concatenate((traj_batch_rraa.avoid1, jnp.expand_dims(env_state_rraa.avoid1, axis=1).T))
        # a2_append = jnp.concatenate((traj_batch_rraa.avoid2, jnp.expand_dims(env_state_rraa.avoid2, axis=1).T))
        # a12_append = jnp.maximum(a1_append, a2_append)

        # SPECIAL BRT TARGET FOR BRRT PROBLEM
        # l_tile_append = jnp.minimum(jnp.maximum(reach1_append, V_reach2_append), jnp.maximum(reach2_append, V_reach1_append))
        l_tilde_rraa = jnp.minimum(jnp.maximum(r1_append, V_raa2_append), jnp.maximum(r2_append, V_raa1_append))

        indexs, done_rraa = calculate_indexs3_rr(ent_gamma[1], traj_batch_rraa.reward, l_tilde_rraa,
                                               jnp.expand_dims(last_val_rraa, axis=1).T) 
        
        # indexs, done = calculate_indexs_rr(ent_gamma[1], traj_batch.reward, l_tile_append,
        #                                        V_append)
        done_rraa = done_rraa[:-1, :]

        advantages_V_rraa, targets_V_rraa = calculate_gae_reachavoid4(ent_gamma[1], 
                                                            config["GAE_LAMBDA"], 
                                                            T_ls=l_tilde_rraa, 
                                                            T_gs=a_append,
                                                            T_Vs=V_rraa_append, 
                                                            done=done_rraa)

        # UPDATE COMPOSED NETWORK
        composed_policy_mask = jnp.where(traj_batch_rraa.policy_taken == 0, 1., 0.) 
        # FIXME FIXME FIXME needs to include all policies now for policy_taken
        update_state = (train_state_policy_rraa, train_state_value_rraa,
                        traj_batch_rraa, advantages_V_rraa, targets_V_rraa, advantages_V_rraa, composed_policy_mask, rng)

        xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent_gamma[0]
        update_state_rraa, loss_info_rraa = jax.lax.scan(
            update_epoch_rraa, update_state_rraa, xs, config["UPDATE_EPOCHS"]
        )
        train_state_policy_rraa = update_state_rraa[0]
        train_state_value_rraa = update_state_rraa[1]
        rng = update_state_rraa[-1]
        ####################################################################################################################
        # UPDATE RAA1

        # CALCULATE DECOMPOSED ADVANTAGES - RAA 1
        (_, _, env_state_raa1, last_obs_raa1, rng, _, _) = runner_state_raa1

        last_val_raa1 = train_state_value_raa1.apply_fn(train_state_value_raa1.params, last_obs_raa1)
        last_val_a1 = train_state_value_a.apply_fn(train_state_value_a.params, last_obs_raa1)
        # last_val_a1 = train_state_value_a1.apply_fn(train_state_value_a1.params, last_obs_raa1)

        r1_append = jnp.concatenate((traj_batch_raa1.reach1, jnp.expand_dims(env_state_raa1.reach1, axis=1).T))
        V_raa1_append = jnp.concatenate((traj_batch_raa1.value_raa1, jnp.expand_dims(last_val_raa1, axis=1).T))
        a1_append = jnp.concatenate((traj_batch_raa1.avoid, jnp.expand_dims(env_state_raa1.avoid, axis=1).T))
        V_a1_append = jnp.concatenate((traj_batch_raa1.value_a, jnp.expand_dims(last_val_a1, axis=1).T))

        l_tilde_raa1 = jnp.maximum(r1_append, V_a1_append)

        indexs, done_raa1 = calculate_indexs3_rr(ent_gamma[1], traj_batch_raa1.reward, l_tilde_raa1,
                                               jnp.expand_dims(last_val_raa1, axis=1).T)

        done_raa1 = done_raa1[:-1, :]

        # new_done = jnp.zeros_like(done)
        # new_done = new_done.at[-1, :].set(1.0) # TODO: check where this last point actually is 
        # done = new_done
        # # FIXME: FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME - This is wrong? but was included in the working RAA code...

        # advantages_V_raa1, targets_V_raa1 = calculate_gae_reach4(ent_gamma[1], config["GAE_LAMBDA"], r1_append, V_raa1_append, done_raa1)

        advantages_V_raa1, targets_V_raa1 = calculate_gae_reachavoid4(ent_gamma[1], config["GAE_LAMBDA"],
                                                            T_ls=l_tilde_raa1,
                                                            T_gs=a1_append,
                                                            T_Vs=V_raa1_append,
                                                            done=done_raa1)

        # UPDATE DECOMPOSED NETWORK - 1
        # dummy_mask = jnp.ones(traj_batch_raa1.reach1.shape)
        composed_policy_mask = jnp.where(traj_batch_raa1.policy_taken == 0, 1., 0.)
        update_state_raa1 = (train_state_policy_raa1, train_state_value_raa1,
                        traj_batch_raa1, advantages_V_raa1, targets_V_raa1, advantages_V_raa1, composed_policy_mask, rng)

        xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent_gamma[0]
        update_state_raa1, loss_info_raa1 = jax.lax.scan(
            update_epoch_raa1, update_state_raa1, xs, config["UPDATE_EPOCHS"]
        )
        train_state_policy_raa1 = update_state_raa1[0]
        train_state_value_raa1 = update_state_raa1[1]
        rng = update_state_raa1[-1]

        ####################################################################################################################
        # UPDATE RAA2

        # CALCULATE DECOMPOSED ADVANTAGES - RAA 2
        (_, _, env_state_raa2, last_obs_raa2, rng, _, _) = runner_state_raa2

        last_val_raa2 = train_state_value_raa2.apply_fn(train_state_value_raa2.params, last_obs_raa2)
        last_val_a2 = train_state_value_a.apply_fn(train_state_value_a.params, last_obs_raa2)
        # last_val_a2 = train_state_value_a2.apply_fn(train_state_value_a2.params, last_obs_raa2)

        r2_append = jnp.concatenate((traj_batch_raa2.reach2, jnp.expand_dims(env_state_raa2.reach2, axis=1).T))
        V_raa2_append = jnp.concatenate((traj_batch_raa2.value_raa2, jnp.expand_dims(last_val_raa2, axis=1).T))
        a2_append = jnp.concatenate((traj_batch_raa2.avoid, jnp.expand_dims(env_state_raa2.avoid, axis=1).T))
        V_a2_append = jnp.concatenate((traj_batch_raa2.value_a, jnp.expand_dims(last_val_a2, axis=1).T))

        l_tilde_raa2 = jnp.maximum(r2_append, V_a2_append)

        indexs, done_raa2 = calculate_indexs3_rr(ent_gamma[1], traj_batch_raa2.reward, l_tilde_raa2,
                                               jnp.expand_dims(last_val_raa2, axis=1).T)

        done_raa2 = done_raa2[:-1, :]

        # advantages_V_raa1, targets_V_raa1 = calculate_gae_reach4(ent_gamma[1], config["GAE_LAMBDA"], r1_append, V_raa1_append, done_raa1)

        advantages_V_raa2, targets_V_raa2 = calculate_gae_reachavoid4(ent_gamma[1], config["GAE_LAMBDA"],
                                                            T_ls=l_tilde_raa2,
                                                            T_gs=a2_append,
                                                            T_Vs=V_raa2_append,
                                                            done=done_raa2)

        # UPDATE DECOMPOSED NETWORK - RAA 2
        # dummy_mask = jnp.ones(traj_batch_raa2.reach2.shape)
        composed_policy_mask = jnp.where(traj_batch_raa2.policy_taken == 0, 1., 0.)
        update_state_raa2 = (train_state_policy_raa2, train_state_value_raa2,
                        traj_batch_raa2, advantages_V_raa2, targets_V_raa2, advantages_V_raa2, composed_policy_mask, rng)

        xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent_gamma[0]
        update_state_raa2, loss_info_raa2 = jax.lax.scan(
            update_epoch_raa2, update_state_raa2, xs, config["UPDATE_EPOCHS"]
        )
        train_state_policy_raa2 = update_state_raa2[0]
        train_state_value_raa2 = update_state_raa2[1]
        rng = update_state_raa2[-1]

        ####################################################################################################################
        # UPDATE A

        # CALCULATE COMPOSED ADVANTAGE - A1
        (_, _, env_state_a, last_obs_a, rng, _, _) = runner_state_a

        last_val_a = train_state_value_a.apply_fn(train_state_value_a.params, last_obs_a)

        a_append = jnp.concatenate((traj_batch_a.avoid, jnp.expand_dims(env_state_a.avoid, axis=1).T)) # avoid function
        V_a_append = jnp.concatenate((traj_batch_a.value, jnp.expand_dims(last_val_a, axis=1).T)) # avoid value function

        indexs, done_a = calculate_indexs3_rr(ent_gamma[1], traj_batch_a.reward, a_append,
                                               jnp.expand_dims(last_val_a, axis=1).T) # NOTE are we totally sure this works, I dont really get og usage,
        done_a = done_a[:-1, :]
        # # Temp override: done is only the last step
        # new_done_a = jnp.zeros_like(done_a)
        # new_done_a = new_done_a.at[-1, :].set(1.0) # TODO: check where this last point actually is
        # done_a = new_done_a

        advantages_V_a, targets_V_a = calculate_gae_avoid4(ent_gamma[1], config["GAE_LAMBDA"],
                                                            T_hs=a_append,
                                                            T_Vhs=V_a_append,
                                                            done=done_a)
        
        # UPDATE DECOMPOSED NETWORK - AVOID
        dummy_mask = jnp.ones(traj_batch_a.avoid.shape)
        update_state_a = (train_state_policy_a, train_state_value_a, 
                           traj_batch_a, advantages_V_a, targets_V_a, advantages_V_a, dummy_mask, rng)
        xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent_gamma[0]
        update_state_a, loss_info_a = jax.lax.scan(
            update_epoch_a, update_state_a, xs, config["UPDATE_EPOCHS"]
        )
        train_state_policy_a = update_state_a[0]
        train_state_value_a = update_state_a[1]
        rng = update_state_a[-1]

        # ####################################################################################################################
        # # UPDATE A1

        # # CALCULATE COMPOSED ADVANTAGE - A1
        # (_, _, env_state_a1, last_obs_a1, rng, _, _) = runner_state_a1

        # last_val_a1 = train_state_value_a1.apply_fn(train_state_value_a1.params, last_obs_a1)

        # a1_append = jnp.concatenate((traj_batch_a1.avoid, jnp.expand_dims(env_state_a1.avoid, axis=1).T)) # avoid function
        # V_a1_append = jnp.concatenate((traj_batch_a1.value, jnp.expand_dims(last_val_a1, axis=1).T)) # avoid value function

        # indexs, done_a1 = calculate_indexs3_rr(ent_gamma[1], traj_batch_a1.reward, a1_append,
        #                                        jnp.expand_dims(last_val_a1, axis=1).T) # NOTE are we totally sure this works, I dont really get og usage,
        # done_a1 = done_a1[:-1, :]

        # # # Temp override: done is only the last step
        # # new_done_a1 = jnp.zeros_like(done_a1)
        # # new_done_a1 = new_done_a1.at[-1, :].set(1.0) # TODO: check where this last point actually is
        # # done_a1 = new_done_a1

        # advantages_V_a1, targets_V_a1 = calculate_gae_avoid4(ent_gamma[1], config["GAE_LAMBDA"],
        #                                                     T_hs=a1_append,
        #                                                     T_Vhs=V_a1_append,
        #                                                     done=done_a1)
        
        # # UPDATE DECOMPOSED NETWORK - AVOID
        # dummy_mask = jnp.ones(traj_batch_a1.avoid.shape)
        # update_state_a1 = (train_state_policy_a1, train_state_value_a1, 
        #                    traj_batch_a1, advantages_V_a1, targets_V_a1, advantages_V_a1, dummy_mask, rng)
        # xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent_gamma[0]
        # update_state_a1, loss_info_a1 = jax.lax.scan(
        #     update_epoch_a1, update_state_a1, xs, config["UPDATE_EPOCHS"]
        # )
        # train_state_policy_a1 = update_state_a1[0]
        # train_state_value_a1 = update_state_a1[1]
        # rng = update_state_a1[-1]

        # ####################################################################################################################
        # # UPDATE A2

        # # CALCULATE COMPOSED ADVANTAGE - A2
        # (_, _, env_state_a2, last_obs_a2, rng, _, _) = runner_state_a2

        # last_val_a2 = train_state_value_a2.apply_fn(train_state_value_a2.params, last_obs_a2)

        # a2_append = jnp.concatenate((traj_batch_a2.avoid, jnp.expand_dims(env_state_a2.avoid, axis=1).T)) # avoid function
        # V_a2_append = jnp.concatenate((traj_batch_a2.value, jnp.expand_dims(last_val_a2, axis=1).T)) # avoid value function

        # indexs, done_a2 = calculate_indexs3_rr(ent_gamma[1], traj_batch_a2.reward, a2_append,
        #                                        jnp.expand_dims(last_val_a2, axis=1).T) # NOTE are we totally sure this works, I dont really get og usage,
        # done_a2 = done_a2[:-1, :]

        # # # Temp override: done is only the last step
        # # new_done_a2 = jnp.zeros_like(done_a2)
        # # new_done_a2 = new_done_a2.at[-1, :].set(1.0) # TODO: check where this last point actually is
        # # done_a2 = new_done_a2

        # advantages_V_a2, targets_V_a2 = calculate_gae_avoid4(ent_gamma[1], config["GAE_LAMBDA"],
        #                                                     T_hs=a2_append,
        #                                                     T_Vhs=V_a2_append,
        #                                                     done=done_a2)

        # # UPDATE DECOMPOSED NETWORK - AVOID
        # dummy_mask = jnp.ones(traj_batch_a2.avoid.shape)
        # update_state_a2 = (train_state_policy_a2, train_state_value_a2,
        #                    traj_batch_a2, advantages_V_a2, targets_V_a2, advantages_V_a2, dummy_mask, rng)
        # xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent_gamma[0]
        # update_state_a2, loss_info_a2 = jax.lax.scan(
        #     update_epoch_a2, update_state_a2, xs, config["UPDATE_EPOCHS"]
        # )
        # train_state_policy_a2 = update_state_a2[0]
        # train_state_value_a2 = update_state_a2[1]
        # rng = update_state_a2[-1]

        ####################################################################################################################
        # Output

        train_state_total_out = (train_state_policy_rraa, train_state_value_rraa,
            train_state_policy_raa1, train_state_value_raa1,
            train_state_policy_raa2, train_state_value_raa2, 
            train_state_policy_a, train_state_value_a,
            # train_state_policy_a1, train_state_value_a1,
            # train_state_policy_a2, train_state_value_a2, 
            rng, timestep)
        
        return (train_state_total_out,
                {"batch_info_rraa": (traj_batch_rraa, targets_V_rraa, done_rraa), "loss_info_rraa": loss_info_rraa,
                 "batch_info_raa1": (traj_batch_raa1, targets_V_raa1, done_raa1), "loss_info_raa1": loss_info_raa1,
                 "batch_info_raa2": (traj_batch_raa2, targets_V_raa2, done_raa2), "loss_info_raa2": loss_info_raa2,
                 "batch_info_a": (traj_batch_a, targets_V_a, done_a), "loss_info_a": loss_info_a,
                #  "batch_info_a1": (traj_batch_a1, targets_V_a1, done_a1), "loss_info_a1": loss_info_a1,
                #  "batch_info_a2": (traj_batch_a2, targets_V_a2, done_a2), "loss_info_a2": loss_info_a2,
                 "reach_gamma": ent_gamma[1], "entropy_weight": ent_gamma[0]})
    
    # INIT JAX WRAPPERS
    update_epoch_rraa = partial(_ppo_vanilla_update, config)
    env_step_rraa = partial(_env_step_rraa, env_rraa, env_params_rraa)
    env_step_raa1 = partial(_env_step_raa, env_raa1, env_params_raa1)
    env_step_raa2 = partial(_env_step_raa, env_raa2, env_params_raa2)
    env_step_a = partial(_env_step_a, env_a, env_params_a)
    # env_step_a1 = partial(_env_step_a, env_a1, env_params_a1)
    # env_step_a2 = partial(_env_step_a, env_a2, env_params_a2)
    training = jax.jit(_train)

    tx = optimizer(config)

    # INIT POLICY NETWORK
    if config["DISCRETE"] == False:
        policy_network_rraa = MoGPolicy_Network( # MoG
        # policy_network_rraa = Policy_Network(
            env_rraa.action_space(env_params_rraa).shape[0], activation=config["ACTIVATION"]
        )
        policy_network_raa1 = MoGPolicy_Network(
            env_raa1.action_space(env_params_raa1).shape[0], activation=config["ACTIVATION"]
        )
        policy_network_raa2 = MoGPolicy_Network(
            env_raa2.action_space(env_params_raa2).shape[0], activation=config["ACTIVATION"]
        )
        policy_network_a = Policy_Network( # should these also be MoG?
            env_a.action_space(env_params_a).shape[0], activation=config["ACTIVATION"]
        )
        # policy_network_a1 = Policy_Network( # should these also be MoG?
        #     env_a1.action_space(env_params_a1).shape[0], activation=config["ACTIVATION"]
        # )
        # policy_network_a2 = Policy_Network(
        #     env_a2.action_space(env_params_a2).shape[0], activation=config["ACTIVATION"]
        # )
    else:
        policy_network_rraa = Policy_Network_Discrete(
            env_rraa.action_space(env_params_rraa).n, activation=config["ACTIVATION"]
        )
        policy_network_raa1 = Policy_Network(
            env_raa1.action_space(env_params_raa1).n, activation=config["ACTIVATION"]
        )
        policy_network_raa2 = Policy_Network(
            env_raa2.action_space(env_params_raa2).n, activation=config["ACTIVATION"]
        )
        policy_network_a = Policy_Network(
            env_a.action_space(env_params_a).n, activation=config["ACTIVATION"]
        )
        # policy_network_a1 = Policy_Network(
        #     env_a1.action_space(env_params_a1).n, activation=config["ACTIVATION"]
        # )
        # policy_network_a2 = Policy_Network(
        #     env_a2.action_space(env_params_a2).n, activation=config["ACTIVATION"]
        # )

    # INIT RRAA Actor and Critic
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(env_rraa.observation_space(env_params_rraa).shape)
    network_params_policy_rraa = policy_network_rraa.init(_rng, init_x)
    train_state_policy_rraa = TrainState.create(
        apply_fn=policy_network_rraa.apply,
        params=network_params_policy_rraa,
        tx=tx,
        mean=jnp.zeros(env_rraa.observation_space(env_params_rraa).shape),
        variance=jnp.zeros(env_rraa.observation_space(env_params_rraa).shape),
        count=1e-4,
    )

    value_network_rraa = Value_Network(activation=config["ACTIVATION"])
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(env_rraa.observation_space(env_params_rraa).shape)
    network_params_rraa = value_network_rraa.init(_rng, init_x)
    train_state_value_rraa = TrainState.create(
        apply_fn=value_network_rraa.apply,
        params=network_params_rraa,
        tx=tx,
        mean=jnp.zeros(env_rraa.observation_space(env_params_rraa).shape),
        variance=jnp.zeros(env_rraa.observation_space(env_params_rraa).shape),
        count=1e-4,
    )

    # INIT RAA Actors and Critics
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(env_raa1.observation_space(env_params_raa1).shape)
    network_params_policy_raa1 = policy_network_raa1.init(_rng, init_x)
    train_state_policy_raa1 = TrainState.create(
        apply_fn=policy_network_raa1.apply,
        params=network_params_policy_raa1,
        tx=tx,
        mean=jnp.zeros(env_raa1.observation_space(env_params_raa1).shape),
        variance=jnp.zeros(env_raa1.observation_space(env_params_raa1).shape),
        count=1e-4,
    )

    value_network_raa1 = Value_Network(activation=config["ACTIVATION"])
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(env_raa1.observation_space(env_params_raa1).shape)
    network_params_raa1 = value_network_raa1.init(_rng, init_x)
    train_state_value_raa1 = TrainState.create(
        apply_fn=value_network_raa1.apply,
        params=network_params_raa1,
        tx=tx,
        mean=jnp.zeros(env_raa1.observation_space(env_params_raa1).shape),
        variance=jnp.zeros(env_raa1.observation_space(env_params_raa1).shape),
        count=1e-4,
    )

    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(env_raa2.observation_space(env_params_raa2).shape)
    network_params_policy_raa2 = policy_network_raa2.init(_rng, init_x)
    train_state_policy_raa2 = TrainState.create(
        apply_fn=policy_network_raa2.apply,
        params=network_params_policy_raa2,
        tx=tx,
        mean=jnp.zeros(env_raa2.observation_space(env_params_raa2).shape),
        variance=jnp.zeros(env_raa2.observation_space(env_params_raa2).shape),
        count=1e-4,
    )

    value_network_raa2 = Value_Network(activation=config["ACTIVATION"])
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(env_raa2.observation_space(env_params_raa2).shape)
    network_params_raa2 = value_network_raa2.init(_rng, init_x)
    train_state_value_raa2 = TrainState.create(
        apply_fn=value_network_raa2.apply,
        params=network_params_raa2,
        tx=tx,
        mean=jnp.zeros(env_raa2.observation_space(env_params_raa2).shape),
        variance=jnp.zeros(env_raa2.observation_space(env_params_raa2).shape),
        count=1e-4,
    )

    # # INIT DECOMPOSED ACTOR AND CRITICS
    # if not config["LOAD_DECOMPOSED"]:
    
    # DECOMPOSED POLICIES
    init_x = jnp.zeros(env_a.observation_space(env_params_a).shape)
    network_params_policy_a = policy_network_a.init(_rng, init_x)
    train_state_policy_a = TrainState.create(
        apply_fn=policy_network_a.apply,
        params=network_params_policy_a,
        tx=tx,
        mean=jnp.zeros(env_a.observation_space(env_params_a).shape),
        variance=jnp.zeros(env_a.observation_space(env_params_a).shape),
        count=1e-4,
    )

    # init_x = jnp.zeros(env_a1.observation_space(env_params_a1).shape)
    # network_params_policy_a1 = policy_network_a1.init(_rng, init_x)
    # train_state_policy_a1 = TrainState.create(
    #     apply_fn=policy_network_a1.apply,
    #     params=network_params_policy_a1,
    #     tx=tx,
    #     mean=jnp.zeros(env_a1.observation_space(env_params_a1).shape),
    #     variance=jnp.zeros(env_a1.observation_space(env_params_a1).shape),
    #     count=1e-4,
    # )

    # init_x = jnp.zeros(env_a2.observation_space(env_params_a2).shape)
    # network_params_policy_a2 = policy_network_a2.init(_rng, init_x)
    # train_state_policy_a2 = TrainState.create(
    #     apply_fn=policy_network_a2.apply,
    #     params=network_params_policy_a2,
    #     tx=tx,
    #     mean=jnp.zeros(env_a2.observation_space(env_params_a2).shape),
    #     variance=jnp.zeros(env_a2.observation_space(env_params_a2).shape),
    #     count=1e-4,
    # )

    # DECOMPOSED VALUE CRITICS
    value_network_a = Value_Network(activation=config["ACTIVATION"])
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(env_a.observation_space(env_params_a).shape)
    network_params_a = value_network_a.init(_rng, init_x)
    train_state_value_a = TrainState.create(
        apply_fn=value_network_a.apply,
        params=network_params_a,
        tx=tx,
        mean=jnp.zeros(env_a.observation_space(env_params_a).shape),
        variance=jnp.zeros(env_a.observation_space(env_params_a).shape),
        count=1e-4,
    )

    # value_network_a1 = Value_Network(activation=config["ACTIVATION"])
    # rng, _rng = jax.random.split(rng)
    # init_x = jnp.zeros(env_a1.observation_space(env_params_a1).shape)
    # network_params_a1 = value_network_a1.init(_rng, init_x)
    # train_state_value_a1 = TrainState.create(
    #     apply_fn=value_network_a1.apply,
    #     params=network_params_a1,
    #     tx=tx,
    #     mean=jnp.zeros(env_a1.observation_space(env_params_a1).shape),
    #     variance=jnp.zeros(env_a1.observation_space(env_params_a1).shape),
    #     count=1e-4,
    # )

    # value_network_a2 = Value_Network(activation=config["ACTIVATION"])
    # rng, _rng = jax.random.split(rng)
    # init_x = jnp.zeros(env_a2.observation_space(env_params_a2).shape)
    # network_params_a2 = value_network_a2.init(_rng, init_x)
    # train_state_value_a2 = TrainState.create(
    #     apply_fn=value_network_a2.apply,
    #     params=network_params_a2,
    #     tx=tx,
    #     mean=jnp.zeros(env_a2.observation_space(env_params_a2).shape),
    #     variance=jnp.zeros(env_a2.observation_space(env_params_a2).shape),
    #     count=1e-4,
    # )

    # # LOAD DECOMPOSED ACTOR AND CRITICS
    # else:
    #     raw_restored = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath('model/{}/{}'.format(
    #         config["LOAD_DEC_DIR"], config["LOAD_DEC_DIR_MODEL"])), target=None)
        
    #     train_state_policy_reach1 = TrainState.create(
    #         apply_fn=policy_network_reach1.apply,
    #         params=raw_restored['policy_reach1_network']['params'],
    #         mean=raw_restored['policy_reach1_network']["mean"],
    #         variance=raw_restored['policy_reach1_network']["variance"],
    #         count=raw_restored['policy_reach1_network']["count"],
    #         tx=tx,
    #     )
    #     train_state_policy_reach2 = TrainState.create(
    #         apply_fn=policy_network_reach2.apply,
    #         params=raw_restored['policy_reach2_network']['params'],
    #         mean=raw_restored['policy_reach2_network']["mean"],
    #         variance=raw_restored['policy_reach2_network']["variance"],
    #         count=raw_restored['policy_reach2_network']["count"],
    #         tx=tx,
    #     )

    #     value_network_reach1 = Value_Network(activation=config["ACTIVATION"])
    #     train_state_value_reach1 = TrainState.create(
    #         apply_fn=value_network_reach1.apply,
    #         params=raw_restored['value_reach1_network']['params'],
    #         mean=raw_restored['value_reach1_network']["mean"],
    #         variance=raw_restored['value_reach1_network']["variance"],
    #         count=raw_restored['value_reach1_network']["count"],
    #         tx=tx,
    #     )
    #     value_network_reach2 = Value_Network(activation=config["ACTIVATION"])
    #     train_state_value_reach2 = TrainState.create(
    #         apply_fn=value_network_reach2.apply,
    #         params=raw_restored['value_reach2_network']['params'],
    #         mean=raw_restored['value_reach2_network']["mean"],
    #         variance=raw_restored['value_reach2_network']["variance"],
    #         count=raw_restored['value_reach2_network']["count"],
    #         tx=tx,
    #     )

    # IF TRAINING DECOMPOSED, USE PPO
    if not config["LOAD_DECOMPOSED"]:
        update_epoch_raa1 = partial(_ppo_vanilla_update, config)
        update_epoch_raa2 = partial(_ppo_vanilla_update, config)
        update_epoch_a = partial(_ppo_vanilla_update, config)
        # update_epoch_a1 = partial(_ppo_vanilla_update, config)
        # update_epoch_a2 = partial(_ppo_vanilla_update, config)

    # IF LOADING PRESOLVED DECOMPOSED, NO TRAINING
    else:
        def _no_update(config, update_state, ent):
            dummy_loss = {
                "actor_loss": 0.0,
                "value_loss": 0.0,
                "entropy_loss": 0.0,
            }
            return update_state, dummy_loss
        update_epoch_raa1 = partial(_no_update, config)
        update_epoch_raa2 = partial(_no_update, config)
        update_epoch_a = partial(_no_update, config)
        # update_epoch_a1 = partial(_no_update, config)
        # update_epoch_a2 = partial(_no_update, config)

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
            training, (train_state_policy_rraa, train_state_value_rraa,
                       train_state_policy_raa1, train_state_value_raa1,
                       train_state_policy_raa2, train_state_value_raa2, 
                       train_state_policy_a, train_state_value_a,
                    #    train_state_policy_a1, train_state_value_a1,
                    #    train_state_policy_a2, train_state_value_a2, 
                       rng, timestep),
            xs, config["STEP_SCAN"]
        )

        (train_state_policy_rraa, train_state_value_rraa, 
            train_state_policy_raa1, train_state_value_raa1, 
            train_state_policy_raa2, train_state_value_raa2, 
            train_state_policy_a, train_state_value_a,
            # train_state_policy_a1, train_state_value_a1,
            # train_state_policy_a2, train_state_value_a2, 
            rng, timestep
        ) = update_state

        loss_info_rraa = result['loss_info_rraa']
        loss_info_raa1 = result['loss_info_raa1']
        loss_info_raa2 = result['loss_info_raa2']
        loss_info_a = result['loss_info_a']
        # loss_info_a1 = result['loss_info_a1']
        # loss_info_a2 = result['loss_info_a2']

        result_traj_rraa = tree_index1(result['batch_info_rraa'], 0)
        result_traj_raa1 = tree_index1(result['batch_info_raa1'], 0)
        result_traj_raa2 = tree_index1(result['batch_info_raa2'], 0)
        result_traj_a = tree_index1(result['batch_info_a1'], 0)
        # result_traj_a1 = tree_index1(result['batch_info_a1'], 0)
        # result_traj_a2 = tree_index1(result['batch_info_a2'], 0)
        
        traj_batch_rraa, targets_V_rraa, done_rraa = result_traj_rraa
        traj_batch_raa1, targets_V_raa1, done_raa1 = result_traj_raa1
        traj_batch_raa2, targets_V_raa2, done_raa2 = result_traj_raa2
        traj_batch_a, targets_V_a, done_a = result_traj_a
        # traj_batch_a1, targets_V_a1, done_a1 = result_traj_a1
        # traj_batch_a2, targets_V_a2, done_a2 = result_traj_a2

        # FIXME for RRAA
        ((reach_1_perc, reach_2_perc, reach_perc),
            (reach_idx_1, reach_idx_2, reach_idx)) = calculate_rraa(traj_batch_rraa)
        raa1_reach_idx, raa1_avoid_idx = calculate_reachalwaysavoid(traj_batch_raa1, idx, type="both")
        raa2_reach_idx, raa2_avoid_idx = calculate_reachalwaysavoid(traj_batch_raa2, idx, type="both")
        a_reach_idx, a_avoid_idx = calculate_reachalwaysavoid(traj_batch_a, idx, type="avoid")
        # a1_reach_idx, a1_avoid_idx = calculate_reachalwaysavoid(traj_batch_a1, idx, type="avoid")
        # a2_reach_idx, a2_avoid_idx = calculate_reachalwaysavoid(traj_batch_a2, idx, type="avoid")

        idx = 0

        # reach_idx = calculate_minimal_reach(traj_batch.reach[:, idx])

        info_rraa = tree_index2(traj_batch_rraa.info, idx)
        info_raa1 = tree_index2(traj_batch_raa1.info, idx)
        info_raa2 = tree_index2(traj_batch_raa2.info, idx)
        info_a = tree_index2(traj_batch_a.info, idx)
        # info_a1 = tree_index2(traj_batch_a1.info, idx)
        # info_a2 = tree_index2(traj_batch_a2.info, idx)
        info['reach_index_1'], info['reach_index_2'] = reach_idx_1[idx], reach_idx_2[idx]
        info_1['reach_index_1'], info_1['reach_index_2'] = reach_idx_1_1[idx], np.array(-1)
        info_2['reach_index_1'], info_2['reach_index_2'] = np.array(-1), reach_idx_2_2[idx]

        if config['EXP_NAME'] == 'WindField' or config['EXP_NAME'] == 'WindFieldFull':
            info['u_air'] = env_params.u_air
            info['v_air'] = env_params.v_air
            info['obs'] = env_params.obstacle

        ## SAVE MODEL CHECKPOINTS
        checkpoints.save_checkpoint(ckpt_dir=os.path.abspath(os.path.join("model", config["DIR"])),
                                    target={"policy_network_rraa":train_state_policy_rraa, 
                                            "value_network_rraa":train_state_value_rraa,
                                            "policy_network_raa1":train_state_policy_raa1, 
                                            "value_network_raa1":train_state_value_raa1,
                                            "policy_network_raa2":train_state_policy_raa2, 
                                            "value_network_raa2":train_state_value_raa2,
                                            "policy_network_a":train_state_policy_a, 
                                            "value_network_a":train_state_value_a,
                                            # "policy_network_a1":train_state_policy_a1, 
                                            # "value_network_a1":train_state_value_a1,
                                            # "policy_network_a2":train_state_policy_a2, 
                                            # "value_network_a2":train_state_value_a2,
                                            },
                                    step=timestep,
                                    overwrite=True, 
                                    keep=2)
        
        if config["SAVE_MILESTONE"] and timestep in config["MILESTONES"]:
            checkpoints.save_checkpoint(ckpt_dir=os.path.abspath(os.path.join("model", config["DIR"])),
                                        target={"policy_network_rraa":train_state_policy_rraa, 
                                            "value_network_rraa":train_state_value_rraa,
                                            "policy_network_raa1":train_state_policy_raa1, 
                                            "value_network_raa1":train_state_value_raa1,
                                            "policy_network_raa2":train_state_policy_raa2, 
                                            "value_network_raa2":train_state_value_raa2,
                                            "policy_network_a":train_state_policy_a, 
                                            "value_network_a":train_state_value_a,
                                            # "policy_network_a1":train_state_policy_a1, 
                                            # "value_network_a1":train_state_value_a1,
                                            # "policy_network_a2":train_state_policy_a2, 
                                            # "value_network_a2":train_state_value_a2,
                                            },
                                        step=timestep,
                                        overwrite=False,
                                        prefix="milestone_",)
        
        if reach_perc > best_score:
            best_score = reach_perc
            checkpoints.save_checkpoint(ckpt_dir=os.path.abspath(os.path.join("model", config["DIR"])),
                                        target={"policy_network_rraa":train_state_policy_rraa, 
                                            "value_network_rraa":train_state_value_rraa,
                                            "policy_network_raa1":train_state_policy_raa1, 
                                            "value_network_raa1":train_state_value_raa1,
                                            "policy_network_raa2":train_state_policy_raa2, 
                                            "value_network_raa2":train_state_value_raa2,
                                            "policy_network_a":train_state_policy_a, 
                                            "value_network_a":train_state_value_a,
                                            # "policy_network_a1":train_state_policy_a1, 
                                            # "value_network_a1":train_state_value_a1,
                                            # "policy_network_a2":train_state_policy_a2, 
                                            # "value_network_a2":train_state_value_a2,
                                            },
                                        step=timestep,
                                        prefix="best_",
                                        overwrite=True,)
        
        # MAKE DIAGNOSTIC PLOTS -- FIXME for RRAA
        policy_decision_sample = traj_batch_rraa.policy_taken[:,idx]
        # fig = plot_contour_RRAA((info, info_1, info_2), timestep, config)
        fig = plot_contour_RRAA((info, info_1, info_2), timestep, config, policy_decision_sample=policy_decision_sample)

        fig2 = plot_policy_decision(policy_decision_sample, timestep, config)

        # plot_target(targets_h[:, idx], traj_batch.value_reach[:, idx], traj_batch.reach1[:, idx], traj_batch.reach2[:, idx],
        #             timestep, traj_batch.energy[0, idx], done[:, idx], config)
        # plot_value_target(targets_V[:, idx], traj_batch.value[:, idx], timestep,
        #                   traj_batch.energy[0, idx], done[:, idx], config)
        t1 = time.time()

        # WRITE TO WANDB -- FIXME for RRAA
        if config["USE_WANDB"]:
            wandb.log({
                    #    "not reaching goal": cnt,
                    "actor_rraa_loss": jnp.mean(loss_info_rraa["actor_loss"]), "value_rraa_loss": jnp.mean(loss_info_rraa["value_loss"]),
                    "actor_raa1_loss": jnp.mean(loss_info_raa1["actor_loss"]), "value_raa1_loss": jnp.mean(loss_info_raa1["value_loss"]),
                    "actor_raa2_loss": jnp.mean(loss_info_raa2["actor_loss"]), "value_raa2_loss": jnp.mean(loss_info_raa2["value_loss"]),
                    "actor_a_loss": jnp.mean(loss_info_a["actor_loss"]), "value_a_loss": jnp.mean(loss_info_a["value_loss"]),
                    #    "entropy_loss": jnp.mean(loss_info["entropy_loss"]),
                    "reach_gamma": result['reach_gamma'][0], "entropy_weight": result['entropy_weight'][0],
                    "(RAA-1) Reached 1 [%]": reach_1_perc_1,
                    "(RAA-1) Crashed [%]": reach_1_perc_1,
                    "(RAA-2) Reached 2 [%]": reach_2_perc_2,
                    "(RAA-2) Crashed [%]": reach_2_perc_2,
                    "(RRAA) Reach-Reached [%]": reach_perc,
                    "(RRAA) Crashed [%]": reach_perc,
                    "Crashed [%]": crash_perc,
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

    debug = False
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
        # config["DIR"]="halfcheetah_rr_resetgoal_reachv0.1_rerun"
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
        # config["NAME"]="halfcheetah_rr_resetgoal_reachv0.1_rerun"

        # config["EXP_NAME"]="HumanoidReachReach"
        # config["DIR"]="humanoid_rr_debug_donefix"
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
        # config["ANNEAL_LR"]=True
        # config["ANNEAL_ENT"]=True
        # config["NAME"]="humanoid_rr_debug_donefix"

        config["EXP_NAME"]="PointReachReachAlwaysAvoid"
        config["DIR"]="point_rraa_debug"
        config["LR"]=3e-4
        config["NUM_ENVS"]=32
        config["NUM_STEPS"]=200
        config["TOTAL_TIMESTEPS"]=10_000_000
        config["STEP_SCAN"]=1
        config["UPDATE_EPOCHS"]=10
        config["NUM_MINIBATCHES"]=8
        config["GAMMA_ENERGY"]=1.0
        config["GAMMA_REACH_INIT"]=0.999
        config["GAMMA_REACH_FINAL"]=0.9999
        config["GAE_LAMBDA"]=0.95
        config["CLIP_EPS"]=0.2
        config["ENT_COEF"]=0.01
        config["VF_COEF"]=0.5
        config["MAX_GRAD_NORM"]=0.5
        config["ACTIVATION"]="tanh"
        config["CUDA_USE"]="0"
        config["ANNEAL_LR"]=True
        config["ANNEAL_ENT"]=True
        config["NAME"]="point_rraa_debug"
    
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
    env_rraa, env_raa1, env_raa2, env_a = envs
    # env_rraa, env_raa1, env_raa2, env_a1, env_a2 = envs
    env_params_rraa, env_params_raa1, env_params_raa2, env_params_a = env_rraa.default_params, env_raa1.default_params, env_raa2.default_params, env_a.default_params
    # env_params_rraa, env_params_raa1, env_params_raa2, env_params_a1, env_params_a2 = env_rraa.default_params, env_raa1.default_params, env_raa2.default_params, env_a1.default_params, env_a2.default_params

    if config['EXP_NAME'] == 'WindField':
        env_params_rraa = env_params_rraa.replace(index=config['SECTION'])
        env_params_raa1 = env_params_raa1.replace(index=config['SECTION'])
        env_params_raa2 = env_params_raa2.replace(index=config['SECTION'])
        env_params_a = env_params_a.replace(index=config['SECTION'])
        # env_params_a1 = env_params_a1.replace(index=config['SECTION'])
        # env_params_a2 = env_params_a2.replace(index=config['SECTION'])
    env_paramss = (env_params_rraa, env_params_raa1, env_params_raa2, env_params_a)
    # env_paramss = (env_params_rraa, env_params_raa1, env_params_raa2, env_params_a1, env_params_a2)

    config["USE_WANDB"] = True #not debug # False for debugging
    if config["USE_WANDB"]:
        wandb.init(project='RRAA-{}-{}'.format(config["EXP_NAME"], config["WANDB_GROUP"]), name=config["NAME"], config=config,
                   entity='braat_brrt')

    config["LOAD_DECOMPOSED"] = False # TODO make arg
    # if config["LOAD_DECOMPOSED"]:
    #     config["LOAD_DEC_DIR"] ="hopper_reachreach_idxsMAX_switchfix_augstate_obsfix_long"
    #     config["LOAD_DEC_DIR_MODEL"] ="checkpoint_859"

    if 'VIDEO_FREQ' not in config.keys():
        if 'Humanoid' in config['EXP_NAME']:
            config['VIDEO_FREQ'] = 200
        else:
            config['VIDEO_FREQ'] = 25

    rng = jax.random.PRNGKey(config["SEED"])
    out = train(envs, env_paramss, config, rng) 
    # NOTE passing multiple envs (composed + decomposed)
    # TODO more elegant use one env w/ diff env_params, but this is safe for now