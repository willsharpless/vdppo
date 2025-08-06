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

from arguments import get_args
from functools import partial

from rraa_rl.EFPPO.src.env.env_list import get_env
from rraa_rl.EFPPO.src.model.actorcritic import Policy_Network, Policy_Network_Discrete, Value_Network, ActorCritic_Discrete
from rraa_rl.EFPPO.src.rl.utils import optimizer, get_BuRd, tree_index1, tree_index2
from rraa_rl.EFPPO.src.rl.gae import calculate_gae
from rraa_rl.EFPPO.src.rl.EFPPO_utils import _env_step_cppo_RR, _cppo_update_RR
from rraa_rl.EFPPO.src.rl.plot_utils import plot_contour_RRAA, plot_video_contour_RRAA, calculate_reachreach

####### RRAA Change ######
def calculate_reward_cost(traj_batch): 
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
####### RRAA Change ######


class TrainState(train_state.TrainState):
    lambda_coef: Any

def train(env, env_params, config, rng):
    best_score = -float(jnp.inf)
    def _train(update_state, ent):

        train_state_policy, train_state_value, train_state_cost, rng = update_state

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
        # # # FIXME: Check if we want to use these last vlaue dones
        # FIXME: Do we need to flip these done values ? 
        # dones = jnp.zeros_like(traj_batch.done)
        # dones = dones.at[-1, :].set(1.0) 
        ####### RRAA Change ######

        # CALCULATE ADVANTAGE
        train_state_policy, train_state_value, train_state_cost, env_state, last_obs, rng = runner_state
        last_val = train_state_value.apply_fn(train_state_value.params, last_obs)
        last_cost = train_state_cost.apply_fn(train_state_cost.params, last_obs)
        advantages_value, targets_value = calculate_gae(config["GAMMA_ENERGY"], config["GAE_LAMBDA"], traj_batch.value,
                                                        traj_batch.reward, 
                                                        # dones, 
                                                        traj_batch.done, # TODO: CHECK THIS
                                                        last_val)
        advantages_cost, targets_cost = calculate_gae(1.0, config["GAE_LAMBDA"], traj_batch.value_cost,
                                                        traj_batch.cost, 
                                                        # dones, 
                                                        traj_batch.done, # TODO: CHECK THIS
                                                        last_cost)
        
        if config["LOG_BARRIER"]:
            scale = 100000 # should be > max cost, prevents numerical explosion
            logbar_cost = -scale * jnp.log(
                -(jnp.maximum(traj_batch.cost - config["THRESHOLD_CPPO"], 0.)/scale - 1.)
            ) / config["LOG_BARRIER_MU"]  # log barrier
            advantages_cost, targets_cost = calculate_gae(1.0, config["GAE_LAMBDA"], traj_batch.value_cost,
                                                        logbar_cost, traj_batch.done, last_cost)
            # in this case, value_cost is trained on the log barrier cost

        # UPDATE NETWORK
        update_state = (train_state_policy, train_state_value, train_state_cost, traj_batch,
                        advantages_value, targets_value, advantages_cost, targets_cost, rng)
        xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent
        update_state, loss_info = jax.lax.scan(
            update_epoch, update_state, xs, config["UPDATE_EPOCHS"]
        )

        return (update_state[0], update_state[1],
                update_state[2], update_state[-1]), {"batch_info": traj_batch, "loss_info": loss_info}

    ####### RRAA Change ######
    update_epoch = partial(_cppo_update_RR, config)
    env_step = partial(_env_step_cppo_RR, env, env_params)
    ####### RRAA Change ######
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
            training, (train_state_policy, train_state_value, train_state_cost, rng), xs, config["STEP_SCAN"]
        )

        train_state_policy, train_state_value, train_state_cost, rng = update_state

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
        ((reach_1_perc, reach_2_perc, reach_perc),
            (reach_idx_1, reach_idx_2, reach_idx)) = calculate_reachreach(traj_batch)
        info["reach_index_1"] = reach_idx_1[idx]
        info["reach_index_2"] = reach_idx_2[idx]

        fig = plot_contour_RRAA((info, None, None), timestep, config, policy_decision_sample=None)

        # Keep the best performaing model
        if reach_perc > best_score:
            best_score = reach_perc
            checkpoints.save_checkpoint(ckpt_dir=os.path.abspath(os.path.join("model", config["DIR"])),
                                        target={"policy_network": train_state_policy, "value_network": train_state_value,
                                            "cost_network": train_state_cost},
                                        step=timestep,
                                        prefix="best_",
                                        overwrite=True,)

        t1 = time.time()

        reward, cost, cnt1, cnt2, cnt3 = calculate_reward_cost(traj_batch)

        wandb.log({"not reaching reach 1": cnt1,
                   "not reaching reach 2": cnt2,
                   "not reaching both": cnt3,
                   "average total return": -jnp.mean(reward),
                   "average cost": jnp.mean(cost),
                   "actor_loss": jnp.mean(loss_info["actor_loss"]), "entropy_loss": jnp.mean(loss_info["entropy_loss"]),
                   "value_loss": jnp.mean(loss_info["value_loss"]), "cost_loss": jnp.mean(loss_info["cost_loss"]),
                #    'trajectory_sample':wandb.Image(fig),
                    "Reach 1 Success %": reach_1_perc,
                    "Reach 2 Success %": reach_2_perc,
                    "Reach-Reach Success %": reach_perc,
                   "lambda": jnp.mean(loss_info['lambda'])}, step=timestep)
        
        if "F16" not in config["EXP_NAME"]:
            wandb.log({
                'trajectory_sample':wandb.Image(fig)
            }, step=timestep)
        
            # Save video of trajectory 
            video_freq = 5 #25 
            save_video = True 
            if timestep % video_freq == 0 or timestep == total_timesteps - 1: 
                video_frames = plot_video_contour_RRAA((info, None, None), timestep, config, save_video=save_video, log_wandb=config["USE_WANDB"])

        ####### RRAA Change ######
        print("Iteration {}: not reach 1 {} not reach 2 {} not reach both {} reward {} cost {}".format(timestep, cnt1, cnt2, cnt3, -jnp.mean(reward), jnp.mean(cost)))
        print("Time {}".format(t1-t0))

    return

if __name__ == "__main__":
    
    config = vars(get_args(sys.argv[1:]))

    print("LOG_BARRIER,", config["LOG_BARRIER"])

    # FIXME: Manual for now

    # # variant 1
    # config["ENV_REWARD_TYPE"] = "accumulated" # reward
    # config["ENV_COST_FN"] = "max" # cost_fn
    # config["ENV_COST_TYPE"] = "accumulated" # cost
    # config["CPPO_UPDATE_TYPE"] = "min" # update
    # config["USE_STL"] = False # stl 

    # variant 2 - NOTE: BEST - when used with sum rewards: gamma * (r1 + r2) - (prev r1 + prev r2)
    config["ENV_REWARD_TYPE"] = "accumulated" # reward
    config["ENV_COST_FN"] = "sum" # cost_fn
    config["ENV_COST_TYPE"] = "accumulated" # cost
    config["CPPO_UPDATE_TYPE"] = "mean" # update
    config["USE_STL"] = False # stl 

    # # variant 3
    # config["ENV_REWARD_TYPE"] = "instant" # reward
    # config["ENV_COST_FN"] = "sum" # cost_fn
    # config["ENV_COST_TYPE"] = "instant" # cost
    # config["CPPO_UPDATE_TYPE"] = "mean" # update
    # config["USE_STL"] = False # stl 

    if config["EXP_NAME"] == "HopperReachReach_separated_CPPO":
        # Use min target cost accumulation
        config["ENV_REWARD_TYPE"] = "instant" # reward
        config["ENV_COST_FN"] = "sum" # cost_fn
        config["ENV_COST_TYPE"] = "instant" # cost
        config["USE_STL"] = False # stl

        config["CPPO_UPDATE_TYPE"] = "min"

    # HopperReachReach environment specific assertions
    print(config.keys())
    if "HopperReachReach" in config["EXP_NAME"]:
        assert("USE_STL" in config.keys())
        assert("CPPO_UPDATE_TYPE" in config.keys())
        assert("ENV_COST_TYPE" in config.keys())
        assert("ENV_COST_FN" in config.keys())
        assert("ENV_REWARD_TYPE" in config.keys())

        print('\n\n\ENV_COST_TYPE: {}'.format(config["ENV_COST_TYPE"]))
        print('ENV_COST_FN: {}'.format(config["ENV_COST_FN"]))
        print('ENV_REWARD_TYPE: {}'.format(config["ENV_REWARD_TYPE"]))
        print('USE_STL: {}'.format(config["USE_STL"]))
        print('CPPO_UPDATE_TYPE: {}\n\n\n'.format(config["CPPO_UPDATE_TYPE"]))

    config["USE_WANDB"] = True 
    if config["USE_WANDB"]:
        wandb.init(project='EC-EFPPO-{}'.format(config["EXP_NAME"]), name=config["NAME"], config=config,
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
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA_USE']
    env = get_env(config)
    env_params = env.default_params
    env_params = env_params.replace(gamma=config["GAMMA_ENERGY"])
    wandb.init(project='CPPO-{}'.format(config["EXP_NAME"]), name=config["NAME"], config=config)
    rng = jax.random.PRNGKey(20)
    out = train(env, env_params, config, rng)