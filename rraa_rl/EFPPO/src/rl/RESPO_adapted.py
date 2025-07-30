import os
import sys
import time

import wandb
sys.path.append("/home/mepear_gc")

import jax
import jax.numpy as jnp

from flax.training import train_state
from flax.training import checkpoints
from typing import Any
import matplotlib.pyplot as plt

from arguments import get_args
from functools import partial

from rraa_rl.EFPPO.src.env.env_list import get_env
from rraa_rl.EFPPO.src.model.actorcritic import Policy_Network, Policy_Network_Discrete, Value_Network, ActorCritic_Discrete
from rraa_rl.EFPPO.src.rl.utils import optimizer, get_BuRd, tree_index1, tree_index2
from rraa_rl.EFPPO.src.rl.gae import calculate_gae
from rraa_rl.EFPPO.src.rl.EFPPO_utils import _env_step_rr_respo, _env_step_raa_respo, _respo_update
from rraa_rl.EFPPO.src.rl.plot_utils import plot_contour_RRAA, plot_video_contour_RRAA, calculate_reachreach, calculate_reach_avoid_stats, calculate_reachavoid

def calculate_reward_cost_rr(traj_batch): 
    reward = jnp.sum(traj_batch.reward, axis=0)
    cost = jnp.sum(traj_batch.cost, axis=0)

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
            cost.append(jnp.sum(traj_batch.cost[0: reach_idx[i], i]))
    return jnp.array(reward), jnp.array(cost), cnt

class TrainState(train_state.TrainState):
    lambda_coef: Any

def train(env, env_params, config, rng):
    def _train(update_state, ent):

        train_state_policy, train_state_value, train_state_prob, train_state_cost, rng = update_state

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state_policy, train_state_value, train_state_prob, train_state_cost, env_state, obsv, _rng)

        # COLLECT TRAJECTORY
        runner_state, traj_batch = jax.lax.scan(
            env_step, runner_state, None, config["NUM_STEPS"]
        )

        # CALCULATE ADVANTAGE
        train_state_policy, train_state_value, train_state_prob, train_state_cost, env_state, last_obs, rng = runner_state
        last_val = train_state_value.apply_fn(train_state_value.params, last_obs)
        last_prob = train_state_prob.apply_fn(train_state_prob.params, last_obs)
        last_cost = train_state_cost.apply_fn(train_state_cost.params, last_obs)
        advantages_value, targets_value = calculate_gae(config["GAMMA_ENERGY"], config["GAE_LAMBDA"], traj_batch.value,
                                                        traj_batch.reward, traj_batch.done, last_val)
        advantages_cost, targets_cost = calculate_gae(config["GAMMA_ENERGY"], config["GAE_LAMBDA"], traj_batch.value_cost,
                                                        traj_batch.cost, traj_batch.done, last_cost)


        # UPDATE NETWORK
        update_state = (train_state_policy, train_state_value, train_state_prob, train_state_cost, traj_batch,
                        advantages_value, targets_value, advantages_cost, targets_cost, rng)
        xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent
        update_state, loss_info = jax.lax.scan(
            update_epoch, update_state, xs, config["UPDATE_EPOCHS"]
        )

        return (update_state[0], update_state[1],
                update_state[2], update_state[3], update_state[-1]), {"batch_info": traj_batch, "loss_info": loss_info}

    update_epoch = partial(_respo_update, config)
    if "ReachReach" in config["EXP_NAME"]:
        env_step = partial(_env_step_rr_respo, env, env_params)
    elif "ReachAlwaysAvoid" in config["EXP_NAME"]:
        env_step = partial(_env_step_raa_respo, env, env_params)
    else:
        raise ValueError("Must be RR or RAA problem")
    training = jax.jit(_train)

    tx = optimizer(config)

    # INIT NETWORK
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
        lambda_coef=0.,
    )

    value_network = Value_Network(activation=config["ACTIVATION"])
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(env.observation_space(env_params).shape)
    network_params_value = value_network.init(_rng, init_x)
    train_state_value = TrainState.create(
        apply_fn=value_network.apply,
        params=network_params_value,
        tx=tx,
        lambda_coef=0.,
    )

    value_network_prob = Value_Network(activation=config["ACTIVATION"])
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(env.observation_space(env_params).shape)
    network_params_prob = value_network_prob.init(_rng, init_x)
    train_state_prob = TrainState.create(
        apply_fn=value_network_prob.apply,
        params=network_params_prob,
        tx=tx,
        lambda_coef=0.,
    )

    value_network_cost = Value_Network(activation=config["ACTIVATION"])
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(env.observation_space(env_params).shape)
    network_params_cost = value_network_cost.init(_rng, init_x)
    train_state_cost = TrainState.create(
        apply_fn=value_network_cost.apply,
        params=network_params_cost,
        tx=tx,
        lambda_coef=0.,
    )

    total_timesteps = config["NUM_UPDATES"] // config["STEP_SCAN"]

    for timestep in range(config["NUM_UPDATES"] // config["STEP_SCAN"]):

        t0 = time.time()

        if config['ANNEAL_ENT'] == True:
            xs = jnp.ones(config["STEP_SCAN"]) * config["ENT_COEF"] * (total_timesteps - timestep) / total_timesteps
        else:
            xs = jnp.ones(config["STEP_SCAN"]) * config["ENT_COEF"]

        update_state, result = jax.lax.scan(
            training, (train_state_policy, train_state_value, train_state_prob, train_state_cost, rng), xs, config["STEP_SCAN"]
        )

        train_state_policy, train_state_value, train_state_prob, train_state_cost, rng = update_state

        loss_info = result['loss_info']

        traj_batch = tree_index1(result['batch_info'], 0)


        checkpoints.save_checkpoint(ckpt_dir=os.path.abspath('model/{}/'.format(config["DIR"])),
                                    target={"policy_network": train_state_policy, "value_network": train_state_value,
                                            "prob_network": train_state_prob, "cost_network": train_state_cost},
                                    step=timestep,
                                    overwrite=True,
                                    keep=2)

        t1 = time.time()

        # wandb.log({"reaching goal": jnp.sum(traj_batch.done),
        #            "average total return": -jnp.sum(traj_batch.reward) / jnp.sum(traj_batch.done),
        #            "average cost": jnp.sum(traj_batch.cost) / jnp.sum(traj_batch.done),
        #            "actor_loss": jnp.mean(loss_info["actor_loss"]), "entropy_loss": jnp.mean(loss_info["entropy_loss"]),
        #            "value_loss": jnp.mean(loss_info["value_loss"]), "cost_loss": jnp.mean(loss_info["cost_loss"]),
        #            "prob_loss": jnp.mean(loss_info["prob_loss"]),
        #            "lambda": jnp.mean(loss_info['lambda'])})
        # print("Earliest Reach {}: reach {} reward {} cost {}".format(timestep, jnp.sum(traj_batch.done),
        #                                                              -jnp.sum(traj_batch.reward) / jnp.sum(
        #                                                                  traj_batch.done), jnp.sum(traj_batch.cost)
        #                                                              / jnp.sum(traj_batch.done)))
        # print("Time {}".format(t1-t0))

        if "ReachReach" in config["EXP_NAME"]:
            idx = 0 # index to plot
            info = tree_index2(traj_batch.info, idx)
            
            ((reach_1_perc, reach_2_perc, reach_perc),
                (reach_idx_1, reach_idx_2, reach_idx)) = calculate_reachreach(traj_batch)
            info["reach_index_1"] = reach_idx_1[idx]
            info["reach_index_2"] = reach_idx_2[idx]

            fig = plot_contour_RRAA((info, None, None), timestep, config, policy_decision_sample=None)

            # Keep the best performaing model
            best_score = -float(jnp.inf) if timestep == 0 else best_score
            if reach_perc > best_score:
                best_score = reach_perc
                checkpoints.save_checkpoint(ckpt_dir=os.path.abspath('model/{}/'.format(config["DIR"])),
                                    target={"policy_network": train_state_policy, "value_network": train_state_value,
                                            "prob_network": train_state_prob, "cost_network": train_state_cost},
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
                    "value_loss": jnp.mean(loss_info["value_loss"]), "cost_loss": jnp.mean(loss_info["cost_loss"]),
                    "prob_loss": jnp.mean(loss_info["prob_loss"]),
                    #    'trajectory_sample':wandb.Image(fig),
                        "Reach 1 Success %": reach_1_perc,
                        "Reach 2 Success %": reach_2_perc,
                        "Reach-Reach Success %": reach_perc,
                    }, step=timestep)
            
            if "Hopper" in config["EXP_NAME"] or "HalfCheetah" in config["EXP_NAME"]:
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
            (reach_perc, crash_perc, reach_avoid_perc) = calculate_reachavoid(traj_batch)
            
            # Keep the best performaing model
            best_score = -float(jnp.inf) if timestep == 0 else best_score
            if reach_avoid_perc > best_score:
                best_score = reach_avoid_perc
                checkpoints.save_checkpoint(ckpt_dir=os.path.abspath('model/{}/'.format(config["DIR"])),
                                    target={"policy_network": train_state_policy, "value_network": train_state_value,
                                            "prob_network": train_state_prob, "cost_network": train_state_cost},
                                    step=timestep,
                                    prefix="best_",
                                    overwrite=True,)

            t1 = time.time()

            reward, cost, cnt = calculate_reward_cost_raa(traj_batch)

            wandb.log({"not reaching goal": cnt,
                    "average total return": -jnp.mean(reward),
                    "average cost": jnp.mean(cost),
                    "actor_loss": jnp.mean(loss_info["actor_loss"]), "entropy_loss": jnp.mean(loss_info["entropy_loss"]),
                    "value_loss": jnp.mean(loss_info["value_loss"]), "cost_loss": jnp.mean(loss_info["cost_loss"]),
                    "prob_loss": jnp.mean(loss_info["prob_loss"]),
                    "crashed [%]": crash_perc,
                    #    'trajectory_sample':wandb.Image(fig),
                        "reached [%]": reach_perc,
                        "reached_avoid [%]": reach_avoid_perc,
                        "cnt crashed ": cnt_crashed,
                        "cnt not reaching goal ": cnt_never_reached,
                        "cnt crash after reach ": cnt_crash_after_reach,
                    }, step=timestep)
            
            if "Hopper" in config["EXP_NAME"] or "HalfCheetah" in config["EXP_NAME"]:
                    wandb.log({
                        'trajectory_sample':wandb.Image(fig),
                    }, step=timestep)
            
            # Save video of trajectory 
            video_freq = 25 
            save_video = True 
            if timestep % video_freq == 0 or timestep == total_timesteps - 1: 
                video_frames = plot_video_contour_RRAA((info, None), timestep + 1, config, save_video=save_video, log_wandb=True)

            print("Iteration {}: not reach {} reward {} cost {}".format(timestep, cnt, -jnp.mean(reward), jnp.mean(cost)))
            print("Time {}".format(t1-t0))
        
        else:
            raise ValueError("Must be RR or RAA problem")
        
        plt.close("all")

    return

if __name__ == "__main__":
    config = vars(get_args(sys.argv[1:]))

    debug = False
    if debug:
        # config["EXP_NAME"]="HopperReachReach_sum_RESPO"
        # config["DIR"]="hopper_rr_respo_sum_debug"
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
        # config["NAME"]="hopper_rr_respo_sum_debug"

        config["EXP_NAME"]="HopperReachAlwaysAvoid_RESPO"
        config["DIR"]="hopper_raa_respo_debug_lamPID"
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
        config["LAMBDA_REACH"]=1.
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
        config["NAME"]="hopper_raa_respo_debug_lamPID"

    ## Using the same CPPO baseline classes, which in RR case requires the following flags
    if "ReachReach" in config["EXP_NAME"]:
        config["ENV_REWARD_TYPE"] = "accumulated" # reward
        config["ENV_COST_FN"] = "sum" # cost_fn
        config["ENV_COST_TYPE"] = "accumulated" # cost
        config["CPPO_UPDATE_TYPE"] = "mean" # update
        config["USE_STL"] = False # stl 

    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = int(
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    folder = os.path.exists("model/{}".format(config['DIR']))
    if not folder:
        os.makedirs("model/{}".format(config['DIR']))
        os.makedirs("model/{}/reach".format(config['DIR']))
        os.makedirs("model/{}/value_target".format(config['DIR']))
        os.makedirs("model/{}/traj".format(config['DIR']))
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA_USE']
    env = get_env(config)
    env_params = env.default_params
    env_params = env_params.replace(gamma=config["GAMMA_ENERGY"])
    config["USE_WANDB"] = True 
    if config["USE_WANDB"]:     
        wandb.init(project='RESPO-{}'.format(config["EXP_NAME"]), name=config["NAME"], config=config,
                entity='braat_brrt')
    rng = jax.random.PRNGKey(20)
    out = train(env, env_params, config, rng)