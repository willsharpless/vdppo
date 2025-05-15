"""
File for Reach Always Avoid (RAA) PPO training.
"""
import sys

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

from rraa_rl.EFPPO.src.rl.EFPPO_utils import _ppo_vanilla_update, _env_step_rr_vanilla, _env_step_r1_vanilla, _env_step_r2_vanilla, _env_step_raa_vanilla, _env_step_a_vanilla, _env_step_raa_vanilla_deterministic, _env_step_a_vanilla_deterministic
from rraa_rl.EFPPO.src.env.env_list import get_env
from rraa_rl.EFPPO.src.model.actorcritic import Policy_Network, Value_Network, Policy_Network_Discrete
from rraa_rl.EFPPO.src.rl.plot_utils import calculate_minimal_reach, calculate_consumption, calculate_reachreach, calculate_reachalwaysavoid, plot_target, plot_value_target, plot_contour, plot_contour_RRAA, plot_policy_decision, calculate_reach_avoid_stats, \
    plot_video_contour_RRAA, calculate_reachavoid
from rraa_rl.EFPPO.src.rl.utils import optimizer, get_BuRd, tree_index1, tree_index2
from rraa_rl.EFPPO.src.rl.gae import (Transition_reach,
                              calculate_gae, calculate_gae2, calculate_gae3,
                              calculate_gae_reach, calculate_gae_reach2, calculate_gae_reach3, calculate_gae_reach4,
                              calculate_indexs, calculate_indexs2, calculate_indexs3, calculate_indexs3_rr, 
                              calculate_gae_avoid4, calculate_gae_reachavoid4)

class TrainState(train_state.TrainState):
    mean: Any
    variance: Any
    count: Any

def train(envs, env_paramss, config, rngs, env_test=None):
    env, env_avoid = envs # COMPOSED (RAA) + 1 DECOMPOSED (A)
    env_params, env_params_avoid = env_paramss

    def _train(train_state_total, ent_gamma):

        train_state_policy, train_state_value, \
        train_state_policy_avoid, train_state_value_avoid, \
        rngs, timestep = train_state_total 

        ##################  Env step: Composed Env ##################
        # RESET ENV
        rng_composed, rng_avoid = rngs
        rng_composed, _rng_composed = jax.random.split(rng_composed)
        reset_rng_composed = jax.random.split(_rng_composed, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng_composed, env_params)
        rng_composed, _rng_composed = jax.random.split(rng_composed)
        runner_state_standard = (train_state_policy, train_state_value, env_state, obsv, _rng_composed)

        # SPECIAL DECOMPOSED STATES
        decomposed_state = (train_state_policy_avoid, train_state_value_avoid)
        force_combined = False #if timestep < 20 else False # ihibits switching until > 20 epochs
        force_avoid = False 
        policy_controls = (force_combined, force_avoid)
        runner_state = (*runner_state_standard, decomposed_state, policy_controls)

        # COLLECT TRAJECTORY COMPOSED
        runner_state, traj_batch = jax.lax.scan(
            env_step, runner_state, None, config["NUM_STEPS"]
        )

        ##################  Env step: Avoid Env ##################

        init_type = "standard" # "fullrandom" # "toinput" # "standard"

        # RESET ENV
        rng_avoid, _rng_avoid = jax.random.split(rng_avoid)
        reset_rng_avoid = jax.random.split(_rng_avoid, config["NUM_ENVS"])

        if init_type == "toinput":   
            # Select random observations from standard rollout to use for initial avoid state 
            traj_batch_observations = traj_batch.obs #traj_batch.obs # avoid function

            untrans_traj_batch_observations = env.untransform_obs(traj_batch_observations)

            untrans_traj_batch_observations = jnp.transpose(untrans_traj_batch_observations, axes=(1, 0, 2))
            rng_avoid, _rng_avoid = jax.random.split(rng_avoid)
            
            # Single random index
            # random_index = jax.random.randint(_rng_avoid, shape=(), minval=0, maxval=untrans_traj_batch_observations.shape[1])
            # untrans_traj_batch_observations = untrans_traj_batch_observations[:, random_index, :]

            # Multiple random indices
            random_index = jax.random.randint(_rng_avoid, shape=(untrans_traj_batch_observations.shape[0],), minval=0, maxval=untrans_traj_batch_observations.shape[1])
            untrans_traj_batch_observations = jax.vmap(lambda obs, idx: obs[idx])(untrans_traj_batch_observations, random_index)

            obsv_avoid, env_state_avoid = jax.vmap(env_avoid.reset_toinput, in_axes=(0, 0, None))(reset_rng_avoid, untrans_traj_batch_observations, env_params_avoid) 
        
        elif init_type == "standard":
            obsv_avoid, env_state_avoid = jax.vmap(env_avoid.reset, in_axes=(0, None))(reset_rng_avoid, env_params_avoid) # NOTE: old standard reset

        elif init_type == "fullrandom":
            obsv_avoid, env_state_avoid = jax.vmap(env_avoid.reset_fullrandom, in_axes=(0, None))(reset_rng_avoid, env_params_avoid) # NOTE: old standard reset
        
        rng_avoid, _rng_avoid = jax.random.split(rng_avoid)
        runner_state_standard_avoid = (train_state_policy, train_state_value, env_state_avoid, obsv_avoid, _rng_avoid)

        # SPECIAL DECOMPOSED STATES - AVOID
        decomposed_state = (train_state_policy_avoid, train_state_value_avoid)
        force_avoid = True 
        policy_controls_avoid = (force_combined, force_avoid)
        runner_state_avoid = (*runner_state_standard_avoid, decomposed_state, policy_controls_avoid)

        # COLLECT TRAJECTORY DECOMPOSED - AVOID
        runner_state_avoid, traj_batch_avoid = jax.lax.scan(
            env_step_avoid, runner_state_avoid, None, config["NUM_STEPS"]
        )

        ################## Compute Advantages: Composed Env ##################

        # CALCULATE COMPOSED ADVANTAGE
        (train_state_policy, train_state_value, env_state, last_obs, rng_composed, decomposed_state, policy_controls) = runner_state  # FIXME: TEMP TEMP

        last_val = train_state_value.apply_fn(train_state_value.params, last_obs)
        last_val_avoid = train_state_value_avoid.apply_fn(train_state_value_avoid.params, last_obs)

        # DECOMPOSED AVOID VALUES ON COMPOSED PPO ACTOR ROLL OUT
        avoid_append = jnp.concatenate((traj_batch.avoid, jnp.expand_dims(env_state.avoid, axis=1).T)) # avoid function
        V_avoid_append = jnp.concatenate((traj_batch.value_avoid, jnp.expand_dims(last_val_avoid, axis=1).T)) # avoid value function
        reach_append = jnp.concatenate((traj_batch.reach, jnp.expand_dims(env_state.reach, axis=1).T)) # reach function l(x)

        V_append = jnp.concatenate((traj_batch.value, jnp.expand_dims(last_val, axis=1).T)) # V_append - whole thing RA value function
        
        l_tilde = jnp.maximum(reach_append, V_avoid_append) # l tilde - max(l(x), V_avoid(x))
        # FIXME: FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME - This is definitely wrong
        indexs, done = calculate_indexs3_rr(ent_gamma[1], traj_batch.reward, l_tilde,
                                               jnp.expand_dims(last_val, axis=1).T) # NOTE are we totally sure this works, I dont really get og usage,
        done =  done[:-1, :] 

        # Temp override: done is only the last step
        new_done = jnp.zeros_like(done)
        new_done = new_done.at[-1, :].set(1.0) # TODO: check where this last point actually is 
        done = new_done
        # FIXME: FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME - This is definitely wrong

        advantages_V, targets_V = calculate_gae_reachavoid4(ent_gamma[1], config["GAE_LAMBDA"], 
                                                            T_ls=l_tilde,
                                                            T_gs=avoid_append, 
                                                            T_Vs=V_append, 
                                                            done=done)

        # UPDATE COMPOSED NETWORK
        composed_policy_mask = jnp.where(traj_batch.policy_taken == 0, 1., 0.)
        update_state = (train_state_policy, train_state_value, 
                        traj_batch, advantages_V, targets_V, advantages_V, composed_policy_mask, rng_composed)
        
        xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent_gamma[0]
        update_state, loss_info = jax.lax.scan(
            update_epoch, update_state, xs, config["UPDATE_EPOCHS"]
        )
        train_state_policy = update_state[0]
        train_state_value = update_state[1]
        rng_composed = update_state[-1]

        ################## Compute Advantages: Avoid Env ##################

        # # CALCULATE COMPOSED ADVANTAGE - AVOID
        (_, _, env_state_avoid, last_obs_avoid, rng_avoid, 
         decomposed_state, policy_controls) = runner_state_avoid
        
        last_val_avoid = train_state_value_avoid.apply_fn(train_state_value_avoid.params, last_obs_avoid)

        avoid_append = jnp.concatenate((traj_batch_avoid.avoid, jnp.expand_dims(env_state_avoid.avoid, axis=1).T)) # avoid function
        V_avoid_append = jnp.concatenate((traj_batch_avoid.value, jnp.expand_dims(last_val_avoid, axis=1).T)) # avoid value function

        # FIXME: FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME - This is definitely wrong
        indexs, done_avoid = calculate_indexs3_rr(ent_gamma[1], traj_batch_avoid.reward, avoid_append,
                                               jnp.expand_dims(last_val_avoid, axis=1).T) # NOTE are we totally sure this works, I dont really get og usage,
        done_avoid = done_avoid[:-1, :]

        # Temp override: done is only the last step
        new_done_avoid = jnp.zeros_like(done_avoid)
        new_done_avoid = new_done_avoid.at[-1, :].set(1.0) # TODO: check where this last point actually is 
        done_avoid = new_done_avoid
        # FIXME: FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME - This is definitely wrong

        advantages_V_avoid, targets_V_avoid = calculate_gae_avoid4(ent_gamma[1], config["GAE_LAMBDA"],
                                                            T_hs=avoid_append,
                                                            T_Vhs=V_avoid_append,
                                                            done=done_avoid)
        
        # UPDATE DECOMPOSED NETWORK - AVOID
        dummy_mask = jnp.ones(traj_batch_avoid.avoid.shape)
        update_state_avoid = (train_state_policy_avoid, train_state_value_avoid, 
                              traj_batch_avoid, advantages_V_avoid, targets_V_avoid, advantages_V_avoid, dummy_mask, rng_avoid)
        xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent_gamma[0]
        update_state_avoid, loss_info_avoid = jax.lax.scan(
            update_epoch_avoid, update_state_avoid, xs, config["UPDATE_EPOCHS"]
        )
        train_state_policy_avoid = update_state_avoid[0]
        train_state_value_avoid = update_state_avoid[1]
        rng_avoid = update_state_avoid[-1]


        ##########################################################################################

        return ((train_state_policy, train_state_value, train_state_policy_avoid, train_state_value_avoid, (rng_composed, rng_avoid), timestep),
                {"batch_info": (traj_batch, targets_V, done), "loss_info": loss_info,
                 "batch_avoid_info": (traj_batch_avoid, targets_V_avoid, done_avoid), "loss_info_avoid": loss_info_avoid,
                 "reach_gamma": ent_gamma[1], "entropy_weight": ent_gamma[0]})
    
    # INIT JAX WRAPPERS
    update_epoch = partial(_ppo_vanilla_update, config)
    env_step = partial(_env_step_raa_vanilla, env, env_params)
    env_step_avoid = partial(_env_step_a_vanilla, env_avoid, env_params_avoid)
    training = jax.jit(_train)

    tx = optimizer(config)

    # INIT POLICY NETWORK
    if config["DISCRETE"] == False:
        policy_network = Policy_Network(
            env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
        )
        policy_network_avoid = Policy_Network(
            env_avoid.action_space(env_params_avoid).shape[0], activation=config["ACTIVATION"]
        )
    else:
        policy_network = Policy_Network_Discrete(
            env.action_space(env_params).n, activation=config["ACTIVATION"]
        )
        policy_network_avoid = Policy_Network(
            env_avoid.action_space(env_params_avoid).n, activation=config["ACTIVATION"]
        )
    rng_composed, rng_avoid = rngs
    rng_composed, _rng_composed = jax.random.split(rng_composed)
    init_x = jnp.zeros(env.observation_space(env_params).shape)
    network_params_policy = policy_network.init(_rng_composed, init_x)
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
    rng_composed, _rng_composed = jax.random.split(rng_composed)
    init_x = jnp.zeros(env.observation_space(env_params).shape)
    network_params = value_network.init(_rng_composed, init_x)
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
        rng_avoid, _rng_avoid = jax.random.split(rng_avoid)
        init_x_avoid = jnp.zeros(env_avoid.observation_space(env_params_avoid).shape)
        network_params_policy_avoid = policy_network_avoid.init(_rng_avoid, init_x_avoid)
        train_state_policy_avoid = TrainState.create(
            apply_fn=policy_network_avoid.apply,
            params=network_params_policy_avoid,
            tx=tx,
            mean=jnp.zeros(env_avoid.observation_space(env_params_avoid).shape),
            variance=jnp.zeros(env_avoid.observation_space(env_params_avoid).shape),
            count=1e-4,
        )

        # DECOMPOSED VALUE CRITICS
        value_network_avoid = Value_Network(activation=config["ACTIVATION"])
        rng_avoid, _rng_avoid = jax.random.split(rng_avoid)
        init_x = jnp.zeros(env_avoid.observation_space(env_params_avoid).shape)
        network_params_avoid = value_network_avoid.init(_rng_avoid, init_x)
        train_state_value_avoid = TrainState.create(
            apply_fn=value_network_avoid.apply,
            params=network_params_avoid,
            tx=tx,
            mean=jnp.zeros(env_avoid.observation_space(env_params_avoid).shape),
            variance=jnp.zeros(env_avoid.observation_space(env_params_avoid).shape),
            count=1e-4,
        )

    # LOAD DECOMPOSED ACTOR AND CRITICS
    else:
        raw_restored = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath('model/{}/{}'.format(
            config["LOAD_DEC_DIR"], config["LOAD_DEC_DIR_MODEL"])), target=None)
        
        train_state_policy_avoid = TrainState.create(
            apply_fn=policy_network_avoid.apply,
            params=raw_restored['policy_avoid_network']['params'],
            mean=raw_restored['policy_avoid_network']["mean"],
            variance=raw_restored['policy_avoid_network']["variance"],
            count=raw_restored['policy_avoid_network']["count"],
            tx=tx,
        )

        value_network_avoid = Value_Network(activation=config["ACTIVATION"])
        train_state_value_avoid = TrainState.create(
            apply_fn=value_network_avoid.apply,
            params=raw_restored['value_avoid_network']['params'],
            mean=raw_restored['value_avoid_network']["mean"],
            variance=raw_restored['value_avoid_network']["variance"],
            count=raw_restored['value_avoid_network']["count"],
            tx=tx,
        )


    # IF TRAINING DECOMPOSED, USE PPO
    if not config["LOAD_DECOMPOSED"]:
        update_epoch_avoid = partial(_ppo_vanilla_update, config)

    # IF LOADING PRESOLVED DECOMPOSED, NO TRAINING
    else:
        def _no_update(config, update_state, ent):
            dummy_loss = {
                "actor_loss": 0.0,
                "value_loss": 0.0,
                "entropy_loss": 0.0,
            }
            return update_state, dummy_loss

        update_epoch_avoid = partial(_no_update, config)

    total_timesteps = config["NUM_UPDATES"] // config["STEP_SCAN"]

    for timestep in range(config["NUM_UPDATES"] // config["STEP_SCAN"]):

        t0 = time.time()

        xs = jnp.zeros((config["STEP_SCAN"], 2))

        if config['ANNEAL_ENT'] == True:
            ent = jnp.ones(config["STEP_SCAN"]) * config["ENT_COEF"] * (total_timesteps - timestep) / total_timesteps
        else:
            ent = jnp.ones(config["STEP_SCAN"]) * config["ENT_COEF"]

        # FIXME: What is happening with gamma_1 and gamma_2?
        gamma_1 = jnp.ones(config["STEP_SCAN"]) * config["GAMMA_REACH_INIT"] + (config['GAMMA_REACH_FINAL'] - config["GAMMA_REACH_INIT"]) * timestep / total_timesteps
        gamma_2 = jnp.ones(config["STEP_SCAN"]) * jnp.minimum(config['GAMMA_REACH_FINAL'], config["GAMMA_REACH_INIT"] +
                              (config['GAMMA_REACH_FINAL'] - config["GAMMA_REACH_INIT"]) * timestep * 2 / total_timesteps)

        xs = xs.at[:, 0].set(ent)
        xs = xs.at[:, 1].set(gamma_2)

        update_state, result = jax.lax.scan(
            training, (train_state_policy, train_state_value,
                       train_state_policy_avoid, train_state_value_avoid,
                       (rng_composed, rng_avoid), timestep),
            xs, config["STEP_SCAN"]
        )

        (train_state_policy, train_state_value, 
            train_state_policy_avoid, train_state_value_avoid, 
            (rng_composed, rng_avoid), timestep) = update_state

        loss_info = result['loss_info']
        loss_info_avoid = result['loss_info_avoid']

        result_traj = tree_index1(result['batch_info'], 0)
        result_traj_avoid = tree_index1(result['batch_avoid_info'], 0)
        
        traj_batch, targets_V, done = result_traj
        traj_batch_avoid, targets_V_avoid, done_avoid = result_traj_avoid

        # FIXME: FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME 
        # TODO: Need to add plot utils function
        idx = 0
        reach_idx, avoid_idx = calculate_reachalwaysavoid(traj_batch, idx, type="both")
        reach_avoidonly_idx, avoid_avoidonly_idx = calculate_reachalwaysavoid(traj_batch_avoid, idx, type="avoid")
        info = tree_index2(traj_batch.info, idx)
        info_avoid = tree_index2(traj_batch_avoid.info, idx)

        cnt_never_reached, cnt_crashed, cnt_crash_after_reach = calculate_reach_avoid_stats(traj_batch)
        (reach_perc, crash_perc, reach_avoid_perc) = calculate_reachavoid(traj_batch)

        info['reach_index'] = reach_idx
        info['avoid_index'] = avoid_idx
        info_avoid['reach_index'] = reach_avoidonly_idx
        info_avoid['avoid_index'] = avoid_avoidonly_idx

        # TODO: Need to add plot utils function
        # FIXME: FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME 
        

        if config['EXP_NAME'] == 'WindField' or config['EXP_NAME'] == 'WindFieldFull':
            info['u_air'] = env_params.u_air
            info['v_air'] = env_params.v_air
            info['obs'] = env_params.obstacle

        checkpoints.save_checkpoint(ckpt_dir=os.path.abspath(os.path.join("model", config["DIR"])),
                                    target={"policy_network":train_state_policy, 
                                            "value_network":train_state_value,
                                            "policy_avoid_network":train_state_policy_avoid, 
                                            "value_avoid_network":train_state_value_avoid,
                                            },
                                    step=timestep,
                                    overwrite=True,
                                    keep=2)

         # TODO: Need to add plot utils function
        fig = plot_contour_RRAA((info, info_avoid), timestep, config)

        policy_decision_sample = traj_batch.policy_taken[:,idx]
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
                    "actor_avoid_loss": jnp.mean(loss_info_avoid["actor_loss"]), "value_avoid_loss": jnp.mean(loss_info_avoid["value_loss"]),
                    "reach_gamma": result['reach_gamma'][0], "entropy_weight": result['entropy_weight'][0],
                    # 'trajectory_sample':wandb.Image(fig),
                    # 'policy_decision_sample':wandb.Image(fig2),
                    "crashed [%]": crash_perc,
                    "reached [%]": reach_perc,
                    "reached_avoid [%]": reach_avoid_perc,
                    "cnt crashed ": cnt_crashed,
                    "cnt not reaching goal ": cnt_never_reached,
                    "cnt crash after reach ": cnt_crash_after_reach,
                        # 'trajectory_sample_R1':wandb.Image(fig1), 'trajectory_sample_R2':wandb.Image(fig2)
                    }, step=timestep)
            
            if "Hopper" in config["EXP_NAME"]:
                wandb.log({
                    'trajectory_sample':wandb.Image(fig),
                    'policy_decision_sample':wandb.Image(fig2),
                })
            
        # Save video of trajectory 
        if "Hopper" in config["EXP_NAME"]:
            video_freq = 5 #25 
            save_video = config["USE_WANDB"] #True 
            if timestep % video_freq == 0 or timestep == total_timesteps - 1: 
                video_frames = plot_video_contour_RRAA((info, info_avoid), timestep, config, save_video=save_video, log_wandb=config["USE_WANDB"])
                # wandb.log({"trajectory_video": wandb.Video(np.array(video_frames), fps=10, format="mp4")})
                
        plt.close("all")
        # print("Earliest Reach {}: {}        {}".format(timestep, cnt, np.mean(consumption)))
        print(f"ITER TIME : {t1-t0:2.1f}s    SUCCESS :  (Crashed)  {100*crash_perc:2.1f}%  (Reached)  {100*reach_perc:2.1f}%  (RAA)  {100*reach_avoid_perc:2.1f}%")
        print("Time {}".format(t1-t0))

        # Add in eval with deterministic checkpoint
        if env_test is not None and timestep % 5 == 0 and "Hopper" in config["EXP_NAME"]:
            rng_composed, _rng_composed = jax.random.split(rng_composed)
            reset_rng_composed = jax.random.split(_rng_composed, config["NUM_ENVS"])# FIXME: Have eval envs use a different seed than train envs
            # FIXME: Is it just running same initial state over and over?
            env_test_raa, env_test_avoid = env_test
            obsv, env_state = jax.vmap(env_test_raa.reset, in_axes=(0, None))(reset_rng_composed, env_params)
            decomposed_state = (train_state_policy_avoid, train_state_value_avoid)
            force_combined = False #if timestep < 20 else False # ihibits switching until > 20 epochs
            force_avoid = False 
            policy_controls = (force_combined, force_avoid)
            rng_composed, _rng_composed = jax.random.split(rng_composed)
            runner_state_standard = (train_state_policy, train_state_value, env_state, obsv, _rng_composed)
            runner_state = (*runner_state_standard, decomposed_state, policy_controls)

            rng_avoid, _rng_avoid = jax.random.split(rng_avoid)
            reset_rng_avoid = jax.random.split(_rng_avoid, config["NUM_ENVS"])
            obsv_avoid, env_state_avoid = jax.vmap(env_test_avoid.reset, in_axes=(0, None))(reset_rng_avoid, env_params_avoid)
            rng_avoid, _rng_avoid = jax.random.split(rng_avoid)
            runner_state_standard_avoid = (train_state_policy, train_state_value, env_state_avoid, obsv_avoid, _rng_avoid)
            runner_state_avoid = (*runner_state_standard_avoid, decomposed_state, policy_controls)

            runner_state, traj_batch_eval = jax.lax.scan(
                partial(_env_step_raa_vanilla, env_test_raa, env_params), runner_state, None, config["NUM_STEPS"]
            )

            runner_state_avoid, traj_batch_avoid_eval = jax.lax.scan(
                partial(_env_step_a_vanilla, env_test_avoid, env_params_avoid), runner_state_avoid, None, config["NUM_STEPS"]
            )

            idx = 0
            reach_idx, avoid_idx = calculate_reachalwaysavoid(traj_batch_eval, idx, type="both")
            reach_avoidonly_idx, avoid_avoidonly_idx = calculate_reachalwaysavoid(traj_batch_avoid_eval, idx, type="avoid")
            info_eval = tree_index2(traj_batch_eval.info, idx)
            info_avoid_eval = tree_index2(traj_batch_avoid_eval.info, idx)
            info_eval['reach_index'] = reach_idx
            info_eval['avoid_index'] = avoid_idx
            info_avoid_eval['reach_index'] = reach_avoidonly_idx
            info_avoid_eval['avoid_index'] = avoid_avoidonly_idx
            fig_eval = plot_contour_RRAA((info_eval, info_avoid_eval), timestep, config)
            cnt_never_reached, cnt_crashed, cnt_crash_after_reach = calculate_reach_avoid_stats(traj_batch_eval)
            (reach_perc, crash_perc, reach_avoid_perc) = calculate_reachavoid(traj_batch_eval)
            if config["USE_WANDB"]:
                wandb.log({
                "eval/not reaching goal": cnt_never_reached,
                "eval/crashed": cnt_crashed,
                "eval/crash after reach": cnt_crash_after_reach,
                "eval/trajectory_sample": wandb.Image(fig_eval),
                "eval/crashed [%]": crash_perc,
                "eval/reached [%]": reach_perc,
                "eval/reached_avoid [%]": reach_avoid_perc,
                }, step=timestep)
                video_frames = plot_video_contour_RRAA((info_eval, info_avoid_eval), timestep, config, save_video=True, prefix="eval/", log_wandb=config["USE_WANDB"])

            plt.close("all")
            # FIXME: Below highly inefficient (rolling out 128 envs with deterministic policy)


            rng_composed, _rng_composed = jax.random.split(rng_composed)
            reset_rng_composed = jax.random.split(_rng_composed, config["NUM_ENVS"])# FIXME: Have eval envs use a different seed than train envs
            # FIXME: Is it just running same initial state over and over?
            env_test_raa, env_test_avoid = env_test
            obsv, env_state = jax.vmap(env_test_raa.reset, in_axes=(0, None))(reset_rng_composed, env_params)
            decomposed_state = (train_state_policy_avoid, train_state_value_avoid)
            force_combined = False #if timestep < 20 else False # ihibits switching until > 20 epochs
            force_avoid = False 
            policy_controls = (force_combined, force_avoid)
            rng_composed, _rng_composed = jax.random.split(rng_composed)
            runner_state_standard = (train_state_policy, train_state_value, env_state, obsv, _rng_composed)
            runner_state = (*runner_state_standard, decomposed_state, policy_controls)

            rng_avoid, _rng_avoid = jax.random.split(rng_avoid)
            reset_rng_avoid = jax.random.split(_rng_avoid, config["NUM_ENVS"])
            obsv_avoid, env_state_avoid = jax.vmap(env_test_avoid.reset, in_axes=(0, None))(reset_rng_avoid, env_params_avoid)
            rng_avoid, _rng_avoid = jax.random.split(rng_avoid)
            runner_state_standard_avoid = (train_state_policy, train_state_value, env_state_avoid, obsv_avoid, _rng_avoid)
            runner_state_avoid = (*runner_state_standard_avoid, decomposed_state, policy_controls)

            runner_state, traj_batch_eval = jax.lax.scan(
                partial(_env_step_raa_vanilla_deterministic, env_test_raa, env_params), runner_state, None, config["NUM_STEPS"]
            )

            runner_state_avoid, traj_batch_avoid_eval = jax.lax.scan(
                partial(_env_step_a_vanilla_deterministic, env_test_avoid, env_params_avoid), runner_state_avoid, None, config["NUM_STEPS"]
            )

            idx = 0
            reach_idx, avoid_idx = calculate_reachalwaysavoid(traj_batch_eval, idx, type="both")
            reach_avoidonly_idx, avoid_avoidonly_idx = calculate_reachalwaysavoid(traj_batch_avoid_eval, idx, type="avoid")
            info_eval = tree_index2(traj_batch_eval.info, idx)
            info_avoid_eval = tree_index2(traj_batch_avoid_eval.info, idx)
            info_eval['reach_index'] = reach_idx
            info_eval['avoid_index'] = avoid_idx
            info_avoid_eval['reach_index'] = reach_avoidonly_idx
            info_avoid_eval['avoid_index'] = avoid_avoidonly_idx
            fig_eval = plot_contour_RRAA((info_eval, info_avoid_eval), timestep, config)
            cnt_never_reached, cnt_crashed, cnt_crash_after_reach = calculate_reach_avoid_stats(traj_batch_eval)
            if config["USE_WANDB"]:
                wandb.log({
                "eval/deter/trajectory_sample": wandb.Image(fig_eval),
                }, step=timestep)
                video_frames = plot_video_contour_RRAA((info_eval, info_avoid_eval), timestep, config, save_video=True, prefix="eval/deter/")
            plt.close("all")


    return

if __name__ == "__main__":
    config = vars(get_args(sys.argv[1:]))

    debug = True
    if debug:
        config["EXP_NAME"]="F16ReachAlwaysAvoid"
        config["DIR"]="F16_raa_PE500_halfsamp2_"
        config["LR"]=3e-4
        config["NUM_ENVS"]=256
        config["NUM_STEPS"]=200
        config["TOTAL_TIMESTEPS"]=100_000_000
        config["STEP_SCAN"]=10
        config["UPDATE_EPOCHS"]=10
        config["NUM_MINIBATCHES"]=64
        config["GAMMA_ENERGY"]=1.0
        config["GAMMA_REACH_INIT"]=0.995
        config["GAMMA_REACH_FINAL"]=0.9995
        config["GAE_LAMBDA"]=0.95
        config["CLIP_EPS"]=0.2
        config["ENT_COEF"]=0.001
        config["VF_COEF"]=2.0
        config["MAX_GRAD_NORM"]=0.5
        config["ACTIVATION"]="tanh"
        config["CUDA_USE"]="0"
        config["ANNEAL_LR"]=True,
        config["ANNEAL_ENT"]=True
        config["NAME"]="F16_raa_PE500_halfsamp2_"
        # config["TEST_MODE"]=True # USES DETERMINISTIC MODELS

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
    env, env_avoid = envs

    env_params = env.default_params
    env_params_avoid = env_avoid.default_params

    if config['EXP_NAME'] == 'WindField':
        env_params = env_params.replace(index=config['SECTION'])
        env_params_avoid = env_params_avoid.replace(index=config['SECTION'])
    env_paramss = (env_params, env_params_avoid)

    from copy import copy
    config_test = copy(config)
    config_test["TEST_MODE"] = True
    env_test = get_env(config_test)

    config["USE_WANDB"] = True #True # False for debugging
    if config["USE_WANDB"]:
        wandb.init(project='EC-EFPPO-{}'.format(config["EXP_NAME"]), name=config["NAME"], config=config,
                   entity='braat_brrt')

    config["LOAD_DECOMPOSED"] = False # TODO make args
    if config["LOAD_DECOMPOSED"]:
        config["LOAD_DEC_DIR"] ="hopper_reachalwaysavoid_idxsMAX_switchfix_augstate"
        config["LOAD_DEC_DIR_MODEL"] ="checkpoint_2303"

    rng_composed = jax.random.PRNGKey(20)
    rng_avoid = jax.random.PRNGKey(20)  # FIXME: Maybe?
    out = train(envs, env_paramss, config, (rng_composed, rng_avoid), env_test=env_test) # TODO assumes same env params (should be tuple if diff)
    # NOTE passing multiple envs (composed + decomposed)
    # TODO more elegant use one env w/ diff env_params, but this is safe for now
