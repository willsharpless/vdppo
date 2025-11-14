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

from rraa_rl.src.rl.utils.alg_utils import _ecefppo_update, _env_step, _env_step_adapted_rr, _env_step_adapted_raa
from rraa_rl.src.env.env_list import get_env
from rraa_rl.src.model.actorcritic import Policy_Network, Value_Network, Policy_Network_Discrete
from rraa_rl.src.rl.utils.plot_utils import calculate_minimal_reach, calculate_consumption, plot_target, plot_value_target, plot_contour
from rraa_rl.src.rl.utils.utils import optimizer, get_BuRd, tree_index1, tree_index2
from rraa_rl.src.rl.utils.gae import (Transition_reach,
                              calculate_gae, calculate_gae2, calculate_gae3,
                              calculate_gae_reach, calculate_gae_reach2, calculate_gae_reach3, calculate_gae_reach4,
                              calculate_indexs, calculate_indexs2, calculate_indexs3)

from rraa_rl.src.rl.utils.plot_utils import plot_contour_RRAA, plot_video_contour_RRAA, calculate_reachreach, calculate_reach_avoid_stats, calculate_reachavoid

####### RRAA Change ######
def calculate_reward_cost_rr(traj_batch): 
    reward = jnp.sum(traj_batch.reward, axis=0)
    cost = jnp.sum(traj_batch.energy, axis=0)

    cnt1 = 0 # reach 1 not reached
    cnt2 = 0 # reach 2 not reached
    cnt3 = 0 # reach 1 and 2 not reached
    reach1_idx = ((traj_batch.reach1) < 0).argmax(axis=0)
    reach2_idx = ((traj_batch.reach2) < 0).argmax(axis=0)
    reach3_idx =  ((traj_batch.reach1 < 0) & (traj_batch.reach2 < 0)).argmax(axis=0)

    for i in range(reach1_idx.shape[0]):
        if reach1_idx[i] == 0 and (traj_batch.reach1[0, i] >= 0):
            cnt1 += 1
        
        if reach2_idx[i] == 0 and (traj_batch.reach2[0, i] >= 0):
            cnt2 += 1

        if reach3_idx[i] == 0:
            cnt3 += 1
        
    return jnp.array(reward), jnp.array(cost), cnt1, cnt2, cnt3

def calculate_reward_cost_raa(traj_batch): 
    reach_idx = ((traj_batch.reach) < 0 & (traj_batch.avoid < 0)).argmax(axis=0)
    reward = []
    cost = []
    cnt = 0
    for i in range(reach_idx.shape[0]):
        if reach_idx[i] == 0 and (traj_batch.reach[0, i] >= 0 or traj_batch.avoid[0, i] > 0):
            cnt += 1
        else:
            reward.append(jnp.sum(traj_batch.reward[0: reach_idx[i], i]))
            cost.append(jnp.sum(traj_batch.energy[0: reach_idx[i], i]))
    return jnp.array(reward), jnp.array(cost), cnt
####### RRAA Change ######

class TrainState(train_state.TrainState):
    mean: Any
    variance: Any
    count: Any

def train(env, env_params, config, rng, env_test=None):
    def _train(train_state_total, ent_gamma):

        train_state_policy, train_state_energy, train_state_h, rng = train_state_total

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state_policy, train_state_energy,
                        train_state_h, env_state, obsv, _rng)

        # COLLECT TRAJECTORY
        runner_state, traj_batch = jax.lax.scan(
            env_step, runner_state, None, config["NUM_STEPS"]
        )

        # CALCULATE ADVANTAGE
        (train_state_policy, train_state_energy, train_state_h,
         env_state, last_obs, rng) = runner_state

        last_val = train_state_energy.apply_fn(train_state_energy.params, last_obs)
        last_val_h = train_state_h.apply_fn(train_state_h.params, last_obs)

        ####### RRAA Change ######
        if "ReachReach" in config["EXP_NAME"]:
            env_state_reach = jnp.maximum(env_state.min_reach1, env_state.min_reach2)
            reach_append = jnp.concatenate((traj_batch.reach, jnp.expand_dims(env_state_reach, axis=1).T))
        elif "ReachAlwaysAvoid" in config["EXP_NAME"]:
            reach_append = jnp.concatenate((traj_batch.reach, jnp.expand_dims(env_state.reach, axis=1).T))
        else:
            raise ValueError("Must be RR or RAA problem")
        ####### RRAA Change ######

        V_reach_append = jnp.concatenate((traj_batch.value_reach, jnp.expand_dims(last_val_h, axis=1).T))

        energy_append = jnp.concatenate((traj_batch.energy, jnp.expand_dims(env_state.cost, axis=1).T))
        V_append = jnp.concatenate((traj_batch.value, jnp.expand_dims(last_val, axis=1).T))
        V_total_append = jnp.maximum(V_reach_append, V_append - energy_append)
        g_append = jnp.maximum(reach_append, -energy_append)

        indexs, done = calculate_indexs3(config["GAMMA_ENERGY"], traj_batch.reward, traj_batch.energy, reach_append,
                                               jnp.expand_dims(last_val, axis=1).T, jnp.expand_dims(last_val_h, axis=1).T)
        done = done[:-1, :]

        advantages_h, targets_h = calculate_gae_reach4(ent_gamma[1], config["GAE_LAMBDA"], reach_append, V_reach_append, done)

        advantages_V, targets_V = calculate_gae2(config["GAMMA_ENERGY"], config["GAE_LAMBDA"], traj_batch, done, last_val)

        advantages_total, _ = calculate_gae_reach4(config["GAMMA_REACH_INIT"], config["GAE_LAMBDA"], g_append, V_total_append, done)

        # UPDATE NETWORK
        update_state = (train_state_policy, train_state_energy, train_state_h,
                        traj_batch, advantages_h, targets_h, advantages_V, targets_V, advantages_total, rng)

        xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent_gamma[0]
        update_state, loss_info = jax.lax.scan(
            update_epoch, update_state, xs, config["UPDATE_EPOCHS"]
        )
        train_state_policy = update_state[0]
        train_state_energy = update_state[1]
        train_state_h = update_state[2]
        rng = update_state[-1]

        return ((train_state_policy, train_state_energy, train_state_h, rng),
                {"batch_info": (traj_batch, targets_h, targets_V, done), "loss_info": loss_info,
                 "reach_gamma": ent_gamma[1], "entropy_weight": ent_gamma[0]})

    update_epoch = partial(_ecefppo_update, config)

    ####### RRAA Change ######
    if "ReachReach" in config["EXP_NAME"]:
        env_step = partial(_env_step_adapted_rr, env, env_params)
    elif "ReachAlwaysAvoid" in config["EXP_NAME"]:
        env_step = partial(_env_step_adapted_raa, env, env_params)
    else:
        raise ValueError("Must be RR or RAA problem")
    ####### RRAA Change ######
            
    training = jax.jit(_train)

    tx = optimizer(config)

    # INIT POLICY NETWORK
    if config["DISCRETE"] == False:
        policy_network = Policy_Network(
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

    # INIT VALUE ENERGY NETWORK
    value_network_energy = Value_Network(activation=config["ACTIVATION"])
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(env.observation_space(env_params).shape)
    network_params_energy = value_network_energy.init(_rng, init_x)
    train_state_energy = TrainState.create(
        apply_fn=value_network_energy.apply,
        params=network_params_energy,
        tx=tx,
        mean=jnp.zeros(env.observation_space(env_params).shape),
        variance=jnp.zeros(env.observation_space(env_params).shape),
        count=1e-4,
    )

    # INIT VALUE FIND NETWORK
    value_network_h = Value_Network(activation=config["ACTIVATION"])
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(env.observation_space(env_params).shape)
    network_params_energy = value_network_h.init(_rng, init_x)
    train_state_h = TrainState.create(
        apply_fn=value_network_h.apply,
        params=network_params_energy,
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

        gamma_1 = jnp.ones(config["STEP_SCAN"]) * config["GAMMA_REACH_INIT"] + (config['GAMMA_REACH_FINAL'] - config["GAMMA_REACH_INIT"]) * timestep / total_timesteps
        gamma_2 = jnp.ones(config["STEP_SCAN"]) * jnp.minimum(config['GAMMA_REACH_FINAL'], config["GAMMA_REACH_INIT"] +
                              (config['GAMMA_REACH_FINAL'] - config["GAMMA_REACH_INIT"]) * timestep * 2 / total_timesteps)

        xs = xs.at[:, 0].set(ent)
        xs = xs.at[:, 1].set(gamma_2)

        update_state, result = jax.lax.scan(
            training, (train_state_policy, train_state_energy, train_state_h, rng),
            xs, config["STEP_SCAN"]
        )

        train_state_policy, train_state_energy, train_state_h, rng = update_state

        loss_info = result['loss_info']

        result_traj = tree_index1(result['batch_info'], 0)
        
        traj_batch, targets_h, targets_V, done = result_traj

        consumption, cnt, idx = calculate_consumption(traj_batch)

        idx = 0

        # reach_idx = calculate_minimal_reach(traj_batch.reach[:, idx])

        checkpoints.save_checkpoint(ckpt_dir=os.path.abspath(os.path.join("model", config["DIR"])),
                                    target={"policy_network":train_state_policy, "energy_network":train_state_energy,
                                            "reach_network":train_state_h},
                                    step=timestep,
                                    overwrite=True,
                                    keep=2)
        
        ####### RRAA Change ######

        # info = tree_index2(traj_batch.info, idx)
        # info['init_energy'] = traj_batch.cost[0, idx]
        # info['final_energy'] = traj_batch.cost[reach_idx, idx]
        # info['reach_index'] = reach_idx
        # if config['EXP_NAME'] == 'WindField' or config['EXP_NAME'] == 'WindFieldFull':
        #     info['u_air'] = env_params.u_air
        #     info['v_air'] = env_params.v_air
        #     info['obs'] = env_params.obstacle

        # fig_contour = plot_contour(train_state_energy, train_state_h, train_state_policy, info, timestep, config)
        # plot_target(targets_h[:, idx], traj_batch.value_reach[:, idx], traj_batch.reach[:, idx],
        #             timestep, traj_batch.cost[0, idx], done[:, idx], config)
        # plot_value_target(targets_V[:, idx], traj_batch.value[:, idx], timestep,
        #                   traj_batch.cost[0, idx], done[:, idx], config)
        # t1 = time.time()
        # wandb.log({"not reaching goal": cnt, "average energy consumption": np.mean(consumption),
        #         "actor_loss": jnp.mean(loss_info["actor_loss"]), "entropy_loss": jnp.mean(loss_info["entropy_loss"]),
        #         "energy_loss": jnp.mean(loss_info["energy_loss"]), "reach_loss": jnp.mean(loss_info["reach_loss"]),
        #         "reach_gamma": result['reach_gamma'][0], "entropy_weight": result['entropy_weight'][0],
        #         "trajectory_sample": wandb.Image(fig_contour)}, step=timestep)
        # plt.close("all")
        # print("Earliest Reach {}: {}        {}".format(timestep, cnt, np.mean(consumption)))
        # print("Time {}".format(t1-t0))

        # # Add in eval with deterministic checkpoint
        # if env_test is not None and timestep % 5 == 0:
        #     # FIXME: Is it just running same initial state over and over?
        #     reset_rng = jax.random.split(rng, config["NUM_ENVS"])  # FIXME: Have eval envs use a different seed than train envs
        #     obsv, env_state = jax.vmap(env_test.reset, in_axes=(0, None))(reset_rng, env_params)
        #     runner_state = (train_state_policy, train_state_energy, train_state_h, env_state, obsv, rng)

        #     runner_state, traj_batch_eval = jax.lax.scan(
        #         partial(_env_step, env_test, env_params), runner_state, None, config["NUM_STEPS"]
        #     )

        #     consumption_eval, cnt_eval, _ = calculate_consumption(traj_batch_eval)
        #     reach_idx_eval = calculate_minimal_reach(traj_batch_eval.reach[:, 0])
        #     idx = 0
        #     info_eval = tree_index2(traj_batch_eval.info, idx)
        #     info_eval['init_energy'] = traj_batch.cost[0, idx]
        #     info_eval['final_energy'] = traj_batch.cost[reach_idx, idx]
        #     info_eval['reach_index'] = reach_idx_eval
        #     fig_eval = plot_contour(train_state_energy, train_state_h, train_state_policy, info_eval, timestep, config)
        #     wandb.log({
        #         "eval/not reaching goal": cnt_eval,
        #         "eval/average energy consumption": np.mean(consumption_eval),
        #         "eval/trajectory_sample": wandb.Image(fig_eval),
        #     }, step=timestep)
        #     plt.close("all")

        if "ReachReach" in config["EXP_NAME"]:
            idx = 0 # index to plot
            info = tree_index2(traj_batch.info, idx)
            
            ((reach_1_perc, reach_2_perc, reach_perc),
                (reach_idx_1, reach_idx_2, reach_idx)) = calculate_reachreach(traj_batch, th=0.5)
            info["reach_index_1"] = reach_idx_1[idx]
            info["reach_index_2"] = reach_idx_2[idx]

            fig = plot_contour_RRAA((info, None, None), timestep, config, policy_decision_sample=None)

            # Keep the best performaing model
            best_score = -float(jnp.inf) if timestep == 0 else best_score
            if reach_perc > best_score:
                best_score = reach_perc
                checkpoints.save_checkpoint(ckpt_dir=os.path.abspath(os.path.join("model", config["DIR"])),
                                            target={"policy_network":train_state_policy, "energy_network":train_state_energy,
                                            "reach_network":train_state_h},
                                            step=timestep,
                                            prefix="best_",
                                            overwrite=True,)

            t1 = time.time()

            reward, cost, cnt1, cnt2, cnt3 = calculate_reward_cost_rr(traj_batch)

            wandb.log({
                    "not reaching reach 1": cnt1,
                    "not reaching reach 2": cnt2,
                    "not reaching both": cnt3,
                    "average total return": -jnp.mean(reward),
                    "average cost": jnp.mean(cost),
                    "actor_loss": jnp.mean(loss_info["actor_loss"]), "entropy_loss": jnp.mean(loss_info["entropy_loss"]),
                    "value_loss": jnp.mean(loss_info["reach_loss"]), "cost_loss": jnp.mean(loss_info["energy_loss"]),
                    #    'trajectory_sample':wandb.Image(fig),
                        "Reach 1 Success %": reach_1_perc,
                        "Reach 2 Success %": reach_2_perc,
                        "Reach-Reach Success %": reach_perc,
                    }, step=timestep)
            
            if "F16" not in config["EXP_NAME"]:
                    wandb.log({
                        'trajectory_sample':wandb.Image(fig)
                    }, step=timestep)
            
            # Save video of trajectory 
            video_freq = 25 
            save_video = True 
            if timestep % video_freq == 0 or timestep == total_timesteps - 1: 
                video_frames = plot_video_contour_RRAA((info, None, None), timestep, config, save_video=save_video, log_wandb=True)

            print("Iteration {}: not reach 1 {} not reach 2 {} not reach both {} reward {} cost {}".format(timestep, cnt1, cnt2, cnt3, -jnp.mean(reward), jnp.mean(cost)))
            print("Time {}".format(t1-t0))
        
        elif "ReachAlwaysAvoid" in config["EXP_NAME"]:
            idx = 0 # index to plot
            info = tree_index2(traj_batch.info, idx)

            reach_idx = (traj_batch.reach < 0).argmax(axis=0)[idx]
            avoid_idx = (traj_batch.avoid > 0).argmax(axis=0)[idx]
            info["reach_index"] = reach_idx
            info["avoid_index"] = avoid_idx

            fig = plot_contour_RRAA((info, None), timestep, config)

            cnt_never_reached, cnt_crashed, cnt_crash_after_reach = calculate_reach_avoid_stats(traj_batch)
            ((reach_perc, crash_perc, reach_avoid_perc), 
                (reach_idx_, crash_idx_, reach_and_avoid_idx_), 
                rora_perc, values_mean, values_std
            ) = calculate_reachavoid(traj_batch)

            # write to score file
            with open("model/{}/training_scores.txt".format(config['DIR']), "a") as f:
                f.write("{},{},{},{},{},{},{}\n".format(
                    timestep, 
                    round(values_mean, 6), 
                    round(values_std, 6), 
                    round(crash_perc, 6), 
                    round(reach_perc, 6), 
                    round(rora_perc, 6), 
                    round(reach_avoid_perc, 6)
                ))
            
            # Keep the best performaing model
            best_score = -float(jnp.inf) if timestep == 0 else best_score
            if reach_avoid_perc > best_score:
                best_score = reach_avoid_perc
                checkpoints.save_checkpoint(ckpt_dir=os.path.abspath(os.path.join("model", config["DIR"])),
                                            target={"policy_network":train_state_policy, "energy_network":train_state_energy,
                                            "reach_network":train_state_h},
                                            step=timestep,
                                            prefix="best_",
                                            overwrite=True,)

            t1 = time.time()

            reward, cost, cnt = calculate_reward_cost_raa(traj_batch)

            wandb.log({"not reaching goal": cnt,
                    "average total return": -jnp.mean(reward),
                    "average cost": jnp.mean(cost),
                    "actor_loss": jnp.mean(loss_info["actor_loss"]), "entropy_loss": jnp.mean(loss_info["entropy_loss"]),
                    "value_loss": jnp.mean(loss_info["reach_loss"]), "cost_loss": jnp.mean(loss_info["energy_loss"]),
                    "crashed [%]": crash_perc,
                    #    'trajectory_sample':wandb.Image(fig),
                        "reached [%]": reach_perc,
                        "reached_avoid [%]": reach_avoid_perc,
                        "cnt crashed ": cnt_crashed,
                        "cnt not reaching goal ": cnt_never_reached,
                        "cnt crash after reach ": cnt_crash_after_reach,
                    }, step=timestep)
            
            if "F16" not in config["EXP_NAME"]:
                    wandb.log({
                        'trajectory_sample':wandb.Image(fig),
                    }, step=timestep)
            
            # Save video of trajectory 
            video_freq = 25 
            save_video = True 
            if timestep % video_freq == 0 or timestep == total_timesteps - 1: 
                video_frames = plot_video_contour_RRAA((info, None), timestep + 1, config, save_video=save_video, log_wandb=config["USE_WANDB"])

            print("Iteration {}: not reach {} reward {} cost {}".format(timestep, cnt, -jnp.mean(reward), jnp.mean(cost)))
            print("Time {}".format(t1-t0))
        
        else:
            raise ValueError("Must be RR or RAA problem")
        
        plt.close("all")
        ####### RRAA Change ######

    return

if __name__ == "__main__":
    config = vars(get_args(sys.argv[1:]))

    debug = False
    if debug:
        # config["EXP_NAME"]="HopperReachReach_sum_RCPPO"
        # config["DIR"]="hopper_rr_rcppo_sum_debug"
        # config["LR"]=3e-4
        # config["NUM_ENVS"]=128
        # config["NUM_STEPS"]=400
        # config["TOTAL_TIMESTEPS"]=50_000_000
        # config["STEP_SCAN"]=4
        # config["UPDATE_EPOCHS"]=10
        # config["NUM_MINIBATCHES"]=32
        # config["GAMMA_ENERGY"]=0.99
        # config["GAMMA_REACH_INIT"]=0.995
        # config["GAMMA_REACH_FINAL"]=0.9995
        # config["GAE_LAMBDA"]=0.95
        # config["LAMBDA_REACH"]=0.1
        # config["K_P"]=1.0
        # config["THRESHOLD_CPPO"]=0.
        # config["CLIP_EPS"]=0.2
        # config["ENT_COEF"]=0.0001
        # config["VF_COEF"]=2.0
        # config["MAX_GRAD_NORM"]=0.5
        # config["ACTIVATION"]="tanh"
        # config["CUDA_USE"]="1"
        # config["ANNEAL_LR"]=True
        # config["ANNEAL_ENT"]=True
        # config["NAME"]="hopper_rr_rcppo_sum_debug"

        config["EXP_NAME"]="HopperReachAlwaysAvoid_RCPPO"
        config["DIR"]="hopper_raa_rcppo_debug"
        config["LR"]=3e-4
        config["NUM_ENVS"]=128
        config["NUM_STEPS"]=400
        config["TOTAL_TIMESTEPS"]=50_000_000
        config["STEP_SCAN"]=4
        config["UPDATE_EPOCHS"]=10
        config["NUM_MINIBATCHES"]=32
        config["GAMMA_ENERGY"]=0.99
        config["GAMMA_REACH_INIT"]=0.995
        config["GAMMA_REACH_FINAL"]=0.9995
        config["GAE_LAMBDA"]=0.95
        config["LAMBDA_REACH"]=0.1
        config["K_P"]=1.0
        config["THRESHOLD_CPPO"]=0.
        config["CLIP_EPS"]=0.2
        config["ENT_COEF"]=0.0001
        config["VF_COEF"]=2.0
        config["MAX_GRAD_NORM"]=0.5
        config["ACTIVATION"]="tanh"
        config["CUDA_USE"]="1"
        config["ANNEAL_LR"]=True
        config["ANNEAL_ENT"]=True
        config["NAME"]="hopper_raa_rcppo_debug"

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
        
    with open("model/{}/training_scores.txt".format(config['DIR']), "w") as f:
        f.write("Training Scores for RAA-PPO-{}-{}, started {}\n".format(config["EXP_NAME"], config['NAME'], time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        f.write("epoch,value_mean,crash_percent,reach_percent,rora_percent,raa_percent\n")

    ## Using the same CPPO baseline classes, which in RR case requires the following flags
    if "ReachReach" in config["EXP_NAME"]:
        config["ENV_REWARD_TYPE"] = "accumulated" # reward
        config["ENV_COST_FN"] = "sum" # cost_fn
        config["ENV_COST_TYPE"] = "accumulated" # cost
        config["CPPO_UPDATE_TYPE"] = "mean" # update
        config["USE_STL"] = False # stl 
    
    env = get_env(config)
    from copy import copy
    config_test = copy(config)
    config_test["TEST_MODE"] = True
    env_test = get_env(config_test)
    config["USE_WANDB"] = True 
    if config["USE_WANDB"]: 
        wandb.init(project='RAN-{}'.format(config["EXP_NAME"]), name=config["NAME"], config=config,
                entity='braat_brrt')
    env_params = env.default_params
    if config['EXP_NAME'] == 'WindField':
        env_params = env_params.replace(index=config['SECTION'])
    rng = jax.random.PRNGKey(20)
    out = train(env, env_params, config, rng, env_test=env_test)