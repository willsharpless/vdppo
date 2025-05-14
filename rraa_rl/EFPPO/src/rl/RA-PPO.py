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

from rraa_rl.EFPPO.src.rl.EFPPO_utils import _ppo_vanilla_update, _env_step_rr_vanilla, _env_step_r1_vanilla, _env_step_r2_vanilla, _env_step_raa_vanilla, _env_step_a_vanilla, _env_step_ra_vanilla, _env_step_ra_vanilla_deterministic
from rraa_rl.EFPPO.src.env.env_list import get_env
from rraa_rl.EFPPO.src.model.actorcritic import Policy_Network, Value_Network, Policy_Network_Discrete, Policy_Network_Learnable_Std, MoGPolicy_Network
from rraa_rl.EFPPO.src.rl.plot_utils import calculate_minimal_reach, calculate_consumption, calculate_reach_avoid_stats, calculate_reachreach, calculate_reachalwaysavoid, plot_target, plot_value_target, plot_contour, plot_contour_RRAA, plot_policy_decision
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

def train(envs, env_paramss, config, rng, env_test=None):
    env = envs
    env_params = env_paramss

    def _train(train_state_total, ent_gamma):

        train_state_policy, train_state_value, rng_og, timestep = train_state_total 

        ##################  Reset Env ##################
        # RESET ENV
        rng, _rng = jax.random.split(rng_og)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state_policy, train_state_value, env_state, obsv, _rng)

        # COLLECT TRAJECTORY
        runner_state, traj_batch = jax.lax.scan(
            env_step, runner_state, None, config["NUM_STEPS"]
        )

        ################## Compute Advantages: Env ##################

        # CALCULATE ADVANTAGE
        (train_state_policy, train_state_value, env_state, last_obs, rng) = runner_state

        last_val = train_state_value.apply_fn(train_state_value.params, last_obs)

        avoid_append = jnp.concatenate((traj_batch.avoid, jnp.expand_dims(env_state.avoid, axis=1).T)) # avoid function
        reach_append = jnp.concatenate((traj_batch.reach, jnp.expand_dims(env_state.reach, axis=1).T)) # reach function l(x)

        V_append = jnp.concatenate((traj_batch.value, jnp.expand_dims(last_val, axis=1).T)) # V_append - whole thing RA value function
        
        l_tilde = jnp.maximum(reach_append, avoid_append) # l tilde - max(l(x), g(x))
        # FIXME ST: Should this just be reach instead of ltilde
        # FIXME ST: FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME - This is definitely wrong
        indexs, done = calculate_indexs3_rr(ent_gamma[1], traj_batch.reward, l_tilde,
                                               jnp.expand_dims(last_val, axis=1).T) # NOTE are we totally sure this works, I dont really get og usage,
        done =  done[:-1, :] 

        # Temp override: done is only the last step
        new_done = jnp.zeros_like(done)
        new_done = new_done.at[-1, :].set(1.0) # TODO: check where this last point actually is 
        done = new_done
        # FIXME: FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME - This is definitely wrong

        advantages_V, targets_V = calculate_gae_reachavoid4(ent_gamma[1], config["GAE_LAMBDA"], 
                                                            T_ls=reach_append,
                                                            T_gs=avoid_append, 
                                                            T_Vs=V_append, 
                                                            done=done)

        # UPDATE COMPOSED NETWORK
        dummy_mask = jnp.ones(traj_batch.avoid.shape)
        update_state = (train_state_policy, train_state_value, 
                        traj_batch, advantages_V, targets_V, advantages_V, dummy_mask, rng)
        
        xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent_gamma[0]
        update_state, loss_info = jax.lax.scan(
            update_epoch, update_state, xs, config["UPDATE_EPOCHS"]
        )
        train_state_policy = update_state[0]
        train_state_value = update_state[1]
        rng = update_state[-1]


        ##########################################################################################

        return ((train_state_policy, train_state_value, rng, timestep),
                {"batch_info": (traj_batch, targets_V, done), "loss_info": loss_info,
                 "reach_gamma": ent_gamma[1], "entropy_weight": ent_gamma[0]})
    
    # INIT JAX WRAPPERS
    update_epoch = partial(_ppo_vanilla_update, config)
    env_step = partial(_env_step_ra_vanilla, env, env_params)
    training = jax.jit(_train)

    tx = optimizer(config)

    # INIT POLICY NETWORK
    if config["DISCRETE"] == False:
        policy_network = MoGPolicy_Network(
            env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
        )
    else:
        policy_network = Policy_Network_Discrete(
            env.action_space(env_params).n, activation=config["ACTIVATION"]
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
            training, (train_state_policy, train_state_value, rng, timestep),
            xs, config["STEP_SCAN"]
        )

        (train_state_policy, train_state_value, rng, timestep) = update_state

        loss_info = result['loss_info']

        result_traj = tree_index1(result['batch_info'], 0)
        
        traj_batch, targets_V, done = result_traj

        idx = 0
        reach_idx = calculate_minimal_reach(traj_batch.reach[:, idx])
        info = tree_index2(traj_batch.info, idx)

        info['reach_index'] = reach_idx

        if config['EXP_NAME'] == 'WindField' or config['EXP_NAME'] == 'WindFieldFull':
            info['u_air'] = env_params.u_air
            info['v_air'] = env_params.v_air
            info['obs'] = env_params.obstacle

        checkpoints.save_checkpoint(ckpt_dir=os.path.abspath(os.path.join("model", config["DIR"])),
                                    target={"policy_network":train_state_policy, 
                                            "value_network":train_state_value,
                                            },
                                    step=timestep,
                                    overwrite=True,
                                    keep=2)

         # TODO: Need to add plot utils function
        fig_contour = plot_contour_RRAA((info, None), timestep, config)
        t1 = time.time()

        if config["USE_WANDB"]:
            wandb.log({
                    #    "not reaching goal": cnt,
                    "actor_loss": jnp.mean(loss_info["actor_loss"]), "value_loss": jnp.mean(loss_info["value_loss"]),
                    "reach_gamma": result['reach_gamma'][0], "entropy_weight": result['entropy_weight'][0],
                    'trajectory_sample':wandb.Image(fig_contour),
                        # 'trajectory_sample_R1':wandb.Image(fig1), 'trajectory_sample_R2':wandb.Image(fig2)
                    }, step=timestep)
        plt.close("all")
        # print("Earliest Reach {}: {}        {}".format(timestep, cnt, np.mean(consumption)))
        print("Time {}".format(t1-t0))
        # Add in eval with deterministic checkpoint
        if env_test is not None and timestep % 5 == 0:
            rng_og = rng
            rng, _rng = jax.random.split(rng_og)
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])# FIXME: Have eval envs use a different seed than train envs
            obsv, env_state = jax.vmap(env_test.reset, in_axes=(0, None))(reset_rng, env_params)
            
            runner_state = (train_state_policy, train_state_value, env_state, obsv, _rng)

            runner_state, traj_batch_eval = jax.lax.scan(
                partial(_env_step_ra_vanilla, env_test, env_params), runner_state, None, config["NUM_STEPS"]
            )

            idx = 0
            reach_idx = calculate_minimal_reach(traj_batch_eval.reach[:, idx])
            info_eval = tree_index2(traj_batch_eval.info, idx)
            info_eval['reach_index'] = reach_idx
            cnt_never_reached, cnt_crashed, cnt_crash_after_reach = calculate_reach_avoid_stats(traj_batch_eval)
            fig_eval = plot_contour_RRAA((info_eval, None), timestep, config)
            wandb.log({
                "eval/not reaching goal": cnt_never_reached,
                "eval/crashed": cnt_crashed,
                "eval/share crash after reach": cnt_crash_after_reach,
                "eval/trajectory_sample": wandb.Image(fig_eval),
            }, step=timestep)
            plt.close("all")
            # FIXME: Below highly inefficient (rolling out 128 envs with deterministic policy)
            rng_og = rng
            rng, _rng = jax.random.split(rng_og)
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])# FIXME: Have eval envs use a different seed than train envs
            # FIXME: Is it just running same initial state over and over?
            obsv, env_state = jax.vmap(env_test.reset, in_axes=(0, None))(reset_rng, env_params)
            
            runner_state = (train_state_policy, train_state_value, env_state, obsv, _rng)

            runner_state, traj_batch_eval = jax.lax.scan(
                partial(_env_step_ra_vanilla_deterministic, env_test, env_params), runner_state, None, config["NUM_STEPS"]
            )

            idx = 0
            reach_idx = calculate_minimal_reach(traj_batch_eval.reach[:, idx])
            info_eval = tree_index2(traj_batch_eval.info, idx)
            info_eval['reach_index'] = reach_idx
            fig_eval = plot_contour_RRAA((info_eval, None), timestep, config)
            wandb.log({
                "eval/deter trajectory_sample": wandb.Image(fig_eval),
            }, step=timestep)
            plt.close("all")

    return

if __name__ == "__main__":
    config = vars(get_args(sys.argv[1:]))

    debug = False
    if debug:
        config["EXP_NAME"]="HopperReachAvoid"
        config["DIR"]="hopper_reachavoid_ceilingwall_debug"
        config["LR"]=3e-4
        config["NUM_ENVS"]=128
        config["NUM_STEPS"]=400
        config["TOTAL_TIMESTEPS"]=500_000_000
        config["STEP_SCAN"]=4
        config["UPDATE_EPOCHS"]=10
        config["NUM_MINIBATCHES"]=32
        config["GAMMA_ENERGY"]=1.0
        config["GAMMA_REACH_INIT"]=0.995
        config["GAMMA_REACH_FINAL"]=0.9995
        config["GAE_LAMBDA"]=0.95
        config["CLIP_EPS"]=0.2
        config["ENT_COEF"]=0.0001
        config["VF_COEF"]=2.0
        config["MAX_GRAD_NORM"]=0.5
        config["ACTIVATION"]="tanh"
        config["CUDA_USE"]="0,1,2,3"
        config["ANNEAL_LR"]=True,
        config["ANNEAL_ENT"]=True
        config["NAME"]="hopper_debug"
        config["TEST_MODE"]=True # USES DETERMINISTIC MODELS

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
    env = envs
    env_params = env.default_params

    from copy import copy
    config_test = copy(config)
    config_test["TEST_MODE"] = True
    env_test = get_env(config_test)


    if config['EXP_NAME'] == 'WindField':
        env_params = env_params.replace(index=config['SECTION'])
    env_paramss = (env_params)

    config["USE_WANDB"] = True #True # False for debugging
    if config["USE_WANDB"]:
        wandb.init(project='EC-EFPPO-{}'.format(config["EXP_NAME"]), name=config["NAME"], config=config,
                   entity='braat_brrt')

    config["LOAD_DECOMPOSED"] = False # TODO make args
    if config["LOAD_DECOMPOSED"]:
        config["LOAD_DEC_DIR"] ="hopper_reachavoid_idxsMAX_switchfix_augstate"
        config["LOAD_DEC_DIR_MODEL"] ="checkpoint_2303"

    rng = jax.random.PRNGKey(20)
    out = train(envs, env_paramss, config, rng, env_test=env_test) # TODO assumes same env params (should be tuple if diff)
    # NOTE passing multiple envs (composed + decomposed)
    # TODO more elegant use one env w/ diff env_params, but this is safe for now
