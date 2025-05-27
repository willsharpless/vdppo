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

from rraa_rl.EFPPO.src.rl.EFPPO_utils import _ppo_vanilla_update, _env_step_r_decomposed, _env_step_rr_decomposed
from rraa_rl.EFPPO.src.env.env_list import get_env
from rraa_rl.EFPPO.src.model.actorcritic import Policy_Network, Value_Network, Policy_Network_Discrete, MoGPolicy_Network
from rraa_rl.EFPPO.src.rl.plot_utils import calculate_minimal_reach, calculate_consumption, calculate_reachreach, plot_target, plot_value_target, plot_contour, plot_contour_RRAA, plot_policy_decision, plot_video_contour_RRAA, calculate_reach
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

        train_state_policy_reach1, train_state_value_reach1, \
            train_state_policy_reach2, train_state_value_reach2, \
            rng_og, timestep = train_state_total
        
        rng, _rng = jax.random.split(rng_og)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv_reach, env_state_reach = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state_policy_reach1, train_state_value_reach1,
                        train_state_policy_reach2, train_state_value_reach2,
                        env_state_reach, obsv_reach, _rng)
        # Collect trajectory
        runner_state, traj_batch = jax.lax.scan(
            env_step_rr, runner_state, None, config["NUM_STEPS"]
        )


        rng, _rng = jax.random.split(rng_og)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv_reach_1, env_state_reach_1 = jax.vmap(env_reach_1.reset, in_axes=(0, None))(reset_rng, env_params_reach_1)
        rng, _rng = jax.random.split(rng)
        runner_state_reach_1 = (train_state_policy_reach1, train_state_value_reach1, env_state_reach_1, obsv_reach_1, _rng)
        
        # Collect trajectory
        runner_state_reach_1, traj_batch_reach1 = jax.lax.scan(
            env_step_reach_1, runner_state_reach_1, None, config["NUM_STEPS"]
        )
    
        ####### Compute advantages: Env #########
        (train_state_policy_reach1, train_state_value_reach1, env_state_reach1, last_obs_reach1, rng) = runner_state_reach_1

        last_val1 = train_state_value_reach1.apply_fn(train_state_value_reach1.params, last_obs_reach1)

        # DECOMPOSED REACH VALUES ON COMPOSED PPO ACTOR ROLL OUT
        reach1_append = jnp.concatenate((traj_batch_reach1.reach, jnp.expand_dims(env_state_reach1.reach, axis=1).T))
        V_reach1_append = jnp.concatenate((traj_batch_reach1.value, jnp.expand_dims(last_val1, axis=1).T))

        indexs, done_1 = calculate_indexs3_rr(ent_gamma[1], traj_batch_reach1.reward, reach1_append,
                                               jnp.expand_dims(last_val1, axis=1).T)
        done_1 = done_1[:-1, :]

        new_done = jnp.zeros_like(done_1)
        new_done = new_done.at[0, :].set(1.0)
        done_1 = new_done

        advantage_V_reach1, targets_V_reach1 = calculate_gae_reach4(ent_gamma[1], config["GAE_LAMBDA"], reach1_append, V_reach1_append, done_1)

        # Update R1 network
        dummy_mask = jnp.ones(traj_batch_reach1.reach.shape)
        update_state_reach1 = (train_state_policy_reach1, train_state_value_reach1,
                               traj_batch_reach1, advantage_V_reach1, targets_V_reach1, advantage_V_reach1, dummy_mask, rng)
        xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent_gamma[0]
        update_state_reach1, loss_info_1 = jax.lax.scan(
            update_epoch_reach1, update_state_reach1, xs, config["UPDATE_EPOCHS"]
        )
        train_state_policy_reach1 = update_state_reach1[0]
        train_state_value_reach1 = update_state_reach1[1]
        rng = update_state_reach1[-1]

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv_reach_2, env_state_reach_2 = jax.vmap(env_reach_2.reset, in_axes=(0, None))(reset_rng, env_params_reach_2)
        rng, _rng = jax.random.split(rng)
        runner_state_reach_2 = (train_state_policy_reach2, train_state_value_reach2, env_state_reach_2, obsv_reach_2, _rng)

        # Collect trajectory
        runner_state_reach_2, traj_batch_reach2 = jax.lax.scan(
            env_step_reach_2, runner_state_reach_2, None, config["NUM_STEPS"]
        )

        ####### Compute advantages: Env #########
        (train_state_policy_reach2, train_state_value_reach2, env_state_reach2, last_obs_reach2, rng) = runner_state_reach_2

        last_val2 = train_state_value_reach2.apply_fn(train_state_value_reach2.params, last_obs_reach2)

        # DECOMPOSED REACH VALUES ON COMPOSED PPO ACTOR ROLL OUT
        reach2_append = jnp.concatenate((traj_batch_reach2.reach, jnp.expand_dims(env_state_reach2.reach, axis=1).T))
        V_reach2_append = jnp.concatenate((traj_batch_reach2.value, jnp.expand_dims(last_val2, axis=1).T))

        indexs, done_2 = calculate_indexs3_rr(ent_gamma[1], traj_batch_reach2.reward, reach2_append,
                                            jnp.expand_dims(last_val2, axis=1).T)
        done_2 = done_2[:-1, :]

        new_done = jnp.zeros_like(done_2)
        new_done = new_done.at[0, :].set(1.0)
        done_2 = new_done

        advantage_V_reach2, targets_V_reach2 = calculate_gae_reach4(ent_gamma[1], config["GAE_LAMBDA"], reach2_append, V_reach2_append, done_1)

        # Update R2 network
        dummy_mask = jnp.ones(traj_batch_reach2.reach.shape)
        update_state_reach2 = (train_state_policy_reach2, train_state_value_reach2,
                            traj_batch_reach2, advantage_V_reach2, targets_V_reach2, advantage_V_reach2, dummy_mask, rng)
        xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent_gamma[0]
        update_state_reach2, loss_info_2 = jax.lax.scan(
            update_epoch_reach2, update_state_reach2, xs, config["UPDATE_EPOCHS"]
        )
        train_state_policy_reach2 = update_state_reach2[0]
        train_state_value_reach2 = update_state_reach2[1]
        rng = update_state_reach2[-1]

        return ((train_state_policy_reach1, train_state_value_reach1, train_state_policy_reach2, train_state_value_reach2, rng, timestep),
                {"batch_info": (traj_batch), "loss_info": None, 
                 "batch_1_info": (traj_batch_reach1, targets_V_reach1, done_1), "loss_info_1": loss_info_1,
                 "batch_2_info": (traj_batch_reach2, targets_V_reach2, done_2), "loss_info_2": loss_info_2,
                 "reach_gamma": ent_gamma[1], "entropy_weight": ent_gamma[0]})
    
    # INIT JAX WRAPPERS
    env_step_rr = partial(_env_step_rr_decomposed, env, env_params)
    env_step_reach_1 = partial(_env_step_r_decomposed, env_reach_1, env_params_reach_1)
    env_step_reach_2 = partial(_env_step_r_decomposed, env_reach_2, env_params_reach_2)
    training = jax.jit(_train)

    tx = optimizer(config)

    # INIT POLICY NETWORK
    if config["DISCRETE"] == False:
        policy_network_reach1 = Policy_Network(
            env_reach_1.action_space(env_params_reach_1).shape[0], activation=config["ACTIVATION"]
        )
        policy_network_reach2 = Policy_Network(
            env_reach_2.action_space(env_params_reach_2).shape[0], activation=config["ACTIVATION"]
        )
    else:
        policy_network_reach1 = Policy_Network(
            env_reach_1.action_space(env_params_reach_1).n, activation=config["ACTIVATION"]
        )
        policy_network_reach2 = Policy_Network(
            env_reach_2.action_space(env_params_reach_2).n, activation=config["ACTIVATION"]
        )

    rng, _rng = jax.random.split(rng)
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
    rng, _rng = jax.random.split(rng)
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


    update_epoch_reach1 = partial(_ppo_vanilla_update, config)
    update_epoch_reach2 = partial(_ppo_vanilla_update, config)


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
            training, (train_state_policy_reach1, train_state_value_reach1,
                       train_state_policy_reach2, train_state_value_reach2, 
                       rng, timestep),
            xs, config["STEP_SCAN"]
        )

        (train_state_policy_reach1, train_state_value_reach1, 
            train_state_policy_reach2, train_state_value_reach2, 
            rng, timestep) = update_state

        loss_info_1 = result['loss_info_1']
        loss_info_2 = result['loss_info_2']

        result_traj = tree_index1(result['batch_info'], 0)
        result_traj_1 = tree_index1(result['batch_1_info'], 0)
        result_traj_2 = tree_index1(result['batch_2_info'], 0)
        
        traj_batch = result_traj
        traj_batch_1, targets_V_1, done_1 = result_traj_1
        traj_batch_2, targets_V_2, done_2 = result_traj_2
        ((reach_1_perc, reach_2_perc, reach_perc),
            (reach_idx_1, reach_idx_2, reach_idx)) = calculate_reachreach(traj_batch)
        reach_1_perc_1, reach_idx_1_1 = calculate_reach(traj_batch_1)
        reach_2_perc_2, reach_idx_2_2 = calculate_reach(traj_batch_2)

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

        if timestep % 5 == 0:
            checkpoints.save_checkpoint(ckpt_dir=os.path.abspath(os.path.join("model", config["DIR"])),
                                        target={"policy_reach1_network":train_state_policy_reach1, 
                                            "value_reach1_network":train_state_value_reach1,
                                                "policy_reach2_network":train_state_policy_reach2, 
                                                "value_reach2_network":train_state_value_reach2,
                                                },
                                        step=timestep,
                                        overwrite=True,
                                        keep=2)
            
        if reach_perc > best_score:
            best_score = reach_perc
            checkpoints.save_checkpoint(ckpt_dir=os.path.abspath(os.path.join("model", config["DIR"])),
                                        target={"policy_reach1_network":train_state_policy_reach1, 
                                            "value_reach1_network":train_state_value_reach1,
                                                "policy_reach2_network":train_state_policy_reach2, 
                                                "value_reach2_network":train_state_value_reach2,
                                                },
                                        step=timestep,
                                        overwrite=True,
                                        prefix="best_",
                                        )
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
                    #    "entropy_loss": jnp.mean(loss_info["entropy_loss"]),
                    "actor_1_loss": jnp.mean(loss_info_1["actor_loss"]), "value_1_loss": jnp.mean(loss_info_1["value_loss"]),
                    "actor_2_loss": jnp.mean(loss_info_2["actor_loss"]), "value_2_loss": jnp.mean(loss_info_2["value_loss"]),
                    "reach_gamma": result['reach_gamma'][0], "entropy_weight": result['entropy_weight'][0],
                    # 'trajectory_sample':wandb.Image(fig),
                    # 'policy_decision_sample':wandb.Image(fig2),
                    "Dec. Reach 1 Success %": reach_1_perc_1,
                    "Dec. Reach 2 Success %": reach_2_perc_2,
                    "Reach-Reach Success %": reach_perc,
                    }, step=timestep)
            
            if "Hopper" in config["EXP_NAME"]:
                wandb.log({"trajectory_sample": wandb.Image(fig),
                            "policy_decision_sample": wandb.Image(fig2),}, step=timestep)
            
            # Save video of trajectory 
            video_freq = 25 
            if timestep % video_freq == 0 or timestep == total_timesteps - 1 and "Hopper" in config["EXP_NAME"]: 
                video_frames = plot_video_contour_RRAA((info, info_1, info_2), timestep, config, save_video=True)

        plt.close("all")
        print(f"ITER TIME : {t1-t0:2.1f}s    SUCCESS : (DEC. R1)  {100*reach_1_perc_1:2.1f}%  (DEC. R2)  {100*reach_2_perc_2:2.1f}%  (COM. RR)  {100*reach_perc:2.1f}%")
        # print("Time {}".format(t1-t0))

    return

if __name__ == "__main__":
    config = vars(get_args(sys.argv[1:]))

    debug = False
    if debug:
        config["EXP_NAME"]="HopperReachReachDecomposed"
        config["DIR"]="hopper_reachreach_debug"
        config["LR"]=3e-4
        config["NUM_ENVS"]=128
        config["NUM_STEPS"]=400
        config["TOTAL_TIMESTEPS"]=50_000_000
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
    env, env_reach_1, env_reach_2 = envs
    env_params = env.default_params
    env_params_reach_1 = env_reach_1.default_params
    env_params_reach_2 = env_reach_2.default_params

    if config['EXP_NAME'] == 'WindField':
        env_params = env_params.replace(index=config['SECTION'])
        env_params_reach_1 = env_params_reach_1.replace(index=config['SECTION'])
        env_params_reach_2 = env_params_reach_2.replace(index=config['SECTION'])
    env_paramss = (env_params, env_params_reach_1, env_params_reach_2)

    config["USE_WANDB"] = not debug # False for debugging
    if config["USE_WANDB"]:
        wandb.init(project='EC-EFPPO-{}'.format(config["EXP_NAME"]), name=config["NAME"], config=config,
                   entity='braat_brrt')

    config["LOAD_DECOMPOSED"] = False # TODO make arg
    if config["LOAD_DECOMPOSED"]:
        config["LOAD_DEC_DIR"] ="hopper_reachreach_idxsMAX_switchfix_augstate_obsfix_long"
        config["LOAD_DEC_DIR_MODEL"] ="checkpoint_859"

    rng = jax.random.PRNGKey(20)
    out = train(envs, env_paramss, config, rng) # TODO assumes same env params (should be tuple if diff)
    # NOTE passing multiple envs (composed + decomposed)
    # TODO more elegant use one env w/ diff env_params, but this is safe for now