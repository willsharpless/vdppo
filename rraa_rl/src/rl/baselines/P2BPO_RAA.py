"""
File for the P2BPO_RAA algorithm: https://ojs.aaai.org/index.php/AAAI/article/view/30094 
code reference: https://github.com/sumantasunny/P2BPO/blob/main/P2BPO.zip 

Additional config parameters: 
"COST_LIMIT": float, constraint limit for the cost function (default: 0.0)
"PENALTY_PARAM_INIT": float, initial value for the penalty parameter (default: 0.001)
"PENALTY_PARAM_LR": float, learning rate for the penalty parameter (default: 0.2)
"""

import os
import pdb
import sys
import time

import wandb
sys.path.append("/home/mepear_gc")

import jax
import jax.numpy as jnp

from flax.training import train_state
from flax.training import checkpoints
from typing import Any

from rraa_rl.src.rl.utils.arguments import get_args
from functools import partial

from rraa_rl.src.env.env_list import get_env
from rraa_rl.src.model.actorcritic import Policy_Network, Policy_Network_Discrete, Value_Network, ActorCritic_Discrete
from rraa_rl.src.rl.utils.utils import optimizer, get_BuRd, tree_index1, tree_index2
from rraa_rl.src.rl.utils.gae import calculate_gae
from rraa_rl.src.rl.utils.alg_utils import _env_step_cppo_RAA, _p2bpo_update_both #_p2bpo_update_RAA
from rraa_rl.src.rl.utils.plot_utils import plot_contour_RRAA, plot_video_contour_RRAA, calculate_reach_avoid_stats, calculate_reachavoid

####### P2BPO Change ######
import optax
from rraa_rl.src.rl.P2BPO_RR import PenaltyParamModule
####### P2BPO Change ######

def calculate_reward_cost(traj_batch): 
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
    best_score = -float(jnp.inf)

    def _train(update_state, ent):
        ####### P2BPO Change ######
        train_state_policy, train_state_value, train_state_cost, train_state_penalty_params, rng = update_state

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state_policy, train_state_value, train_state_cost, env_state, obsv, _rng)

        # COLLECT TRAJECTORY
        runner_state, traj_batch = jax.lax.scan(
            env_step, runner_state, None, config["NUM_STEPS"]
        )

        ####### RRAA Change ######
        # NOTE: FOR RAA SET DONES TO ONLY THE LAST TIME STEP
        dones = jnp.zeros_like(traj_batch.done)
        dones = dones.at[-1, :].set(1.0)
        ####### RRAA Change ######

        # CALCULATE ADVANTAGE
        train_state_policy, train_state_value, train_state_cost, env_state, last_obs, rng = runner_state
        last_val = train_state_value.apply_fn(train_state_value.params, last_obs)
        last_cost = train_state_cost.apply_fn(train_state_cost.params, last_obs)
        advantages_value, targets_value = calculate_gae(config["GAMMA_ENERGY"], config["GAE_LAMBDA"], traj_batch.value,
                                                        traj_batch.reward, 
                                                        dones, #traj_batch.done, 
                                                        last_val)
        advantages_cost, targets_cost = calculate_gae(1.0, config["GAE_LAMBDA"], traj_batch.value_cost,
                                                        traj_batch.cost, 
                                                        dones, #traj_batch.done, 
                                                        last_cost)

        # UPDATE NETWORK
        ####### P2BPO Change ######
        update_state = (train_state_policy, train_state_value, train_state_cost, train_state_penalty_params, traj_batch,
                        advantages_value, targets_value, advantages_cost, targets_cost, rng)
        xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent
        update_state, loss_info = jax.lax.scan(
            update_epoch, update_state, xs, config["UPDATE_EPOCHS"]
        )

        return (update_state[0], update_state[1],
                update_state[2], update_state[3], ####### P2BPO Change ######
                update_state[-1]), {"batch_info": traj_batch, "loss_info": loss_info}

    ####### RRAA Change ######
    ####### P2BPO Change ######
    update_epoch = partial(_p2bpo_update_both, config) #partial(_p2bpo_update_RAA, config)
    env_step = partial(_env_step_cppo_RAA, env, env_params)
    ####### RRAA Change ######
    training = jax.jit(_train)

    tx = optimizer(config)

    ####### P2BPO Change ######
    # NOTE: hard coded for now, can be changed or configured later! 
    penalty_param_module = PenaltyParamModule(init_value=config["PENALTY_PARAM_INIT"])
    penalty_params = penalty_param_module.init(jax.random.PRNGKey(0))

    # Initialize penalty parameter optimizer 
    penalty_param_optimizer = optax.adam(learning_rate=config["PENALTY_PARAM_LR"])

    # Create penalty parameter train state
    train_state_penalty_params = train_state.TrainState.create(
        apply_fn=penalty_param_module.apply,
        params=penalty_params,
        tx=penalty_param_optimizer,
    )
    ####### P2BPO Change ######

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

        ####### P2BPO Change ######
        update_state, result = jax.lax.scan(
            training, (train_state_policy, train_state_value, train_state_cost, train_state_penalty_params, rng), xs, config["STEP_SCAN"]
        )

        train_state_policy, train_state_value, train_state_cost, train_state_penalty_params, rng = update_state

        ####### P2BPO Change ######
        loss_info = result['loss_info']

        traj_batch = tree_index1(result['batch_info'], 0)


        checkpoints.save_checkpoint(ckpt_dir=os.path.abspath(os.path.join("model", config["DIR"])),
                                    target={"policy_network": train_state_policy, "value_network": train_state_value,
                                            "cost_network": train_state_cost},
                                    step=timestep,
                                    overwrite=True,
                                    keep=2)
        
        ####### RRAA Change ######
        # Perform Logging and Plotting
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

        # Keep the best performing model
        if reach_avoid_perc > best_score:
            best_score = reach_avoid_perc
            checkpoints.save_checkpoint(ckpt_dir=os.path.abspath(os.path.join("model", config["DIR"])),
                                        target={"policy_network": train_state_policy, "value_network": train_state_value,
                                            "cost_network": train_state_cost},
                                        step=timestep,
                                        prefix="best_",
                                        overwrite=True,)

        t1 = time.time()

        reward, cost, cnt = calculate_reward_cost(traj_batch)

        wandb.log({"not reaching goal": cnt,
                   "average total return": -jnp.mean(reward),
                   "average cost": jnp.mean(cost),
                   "actor_loss": jnp.mean(loss_info["actor_loss"]), "entropy_loss": jnp.mean(loss_info["entropy_loss"]),
                   "value_loss": jnp.mean(loss_info["value_loss"]), "cost_loss": jnp.mean(loss_info["cost_loss"]),
                   "crashed [%]": crash_perc,
                #    'trajectory_sample':wandb.Image(fig),
                    "reached [%]": reach_perc,
                    "reached_avoid [%]": reach_avoid_perc,
                    "cnt crashed ": cnt_crashed,
                    "cnt not reaching goal ": cnt_never_reached,
                    "cnt crash after reach ": cnt_crash_after_reach,
                   "lambda": jnp.mean(loss_info['lambda']), 
                   "penalty_param": train_state_penalty_params.apply_fn(train_state_penalty_params.params)},
                   step=timestep)
        
        if "Hopper" in config["EXP_NAME"] or "HalfCheetah" in config["EXP_NAME"]:
            wandb.log({
                'trajectory_sample':wandb.Image(fig),
            })
        
            # Save video of trajectory 
            video_freq = 25 
            save_video = True 
            if timestep % video_freq == 0 or timestep == total_timesteps - 1: 
                video_frames = plot_video_contour_RRAA((info, None), timestep + 1, config, save_video=save_video, log_wandb=config["USE_WANDB"])
        ####### RRAA Change ######

        print("Iteration {}: not reach {} reward {} cost {}".format(timestep, cnt, -jnp.mean(reward), jnp.mean(cost)))
        print("Train state penalty params: RAA: {}".format(train_state_penalty_params.params))
        print("Time {}".format(t1-t0))

    return

if __name__ == "__main__":
    
    config = vars(get_args(sys.argv[1:]))

    ####### P2BPO Change ######
    # FIXME: TODO: Hard coded for now, can be changed or configured later!
    config["COST_LIMIT"] = 0.0
    config["PENALTY_PARAM_INIT"] = 0.001
    config["PENALTY_PARAM_LR"] = 0.2
    
    assert(config["FIX_LAMBDA"] == True) # P2BPO does not use fixed lambda
    assert(config["LAMBDA_REACH"] == 0.0) # P2BPO does not use lambda coefficient

    # config["FIX_PENALTY"] = True # TODO: REMOVE DEBUGGING
    ####### P2BPO Change ######

    config["USE_WANDB"] = True 
    if config["USE_WANDB"]:
        wandb.init(project='RAN-{}'.format(config["EXP_NAME"]), name=config["NAME"], config=config,
                   entity='braat_brrt')

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
        
    with open("model/{}/training_scores.txt".format(config['DIR']), "w") as f:
        f.write("Training Scores for RAA-PPO-{}-{}, started {}\n".format(config["EXP_NAME"], config['NAME'], time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        f.write("epoch,value_mean,crash_percent,reach_percent,rora_percent,raa_percent\n")

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA_USE']
    env = get_env(config)
    env_params = env.default_params
    print(env_params)
    env_params = env_params.replace(gamma=config["GAMMA_ENERGY"])
    wandb.init(project='CPPO-{}'.format(config["EXP_NAME"]), name=config["NAME"], config=config)
    rng = jax.random.PRNGKey(20)
    out = train(env, env_params, config, rng)