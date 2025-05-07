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

from rraa_rl.EFPPO.src.rl.EFPPO_utils import _ppo_vanilla_update, _env_step_rr_vanilla
from rraa_rl.EFPPO.src.env.env_list import get_env
from rraa_rl.EFPPO.src.model.actorcritic import Policy_Network, Value_Network, Policy_Network_Discrete
from rraa_rl.EFPPO.src.rl.plot_utils import calculate_minimal_reach, calculate_consumption, calculate_reachreach, plot_target, plot_value_target, plot_contour
from rraa_rl.EFPPO.src.rl.utils import optimizer, get_BuRd, tree_index1, tree_index2
from rraa_rl.EFPPO.src.rl.gae import (Transition_reach,
                              calculate_gae, calculate_gae2, calculate_gae3,
                              calculate_gae_reach, calculate_gae_reach2, calculate_gae_reach3, calculate_gae_reach4,
                              calculate_indexs, calculate_indexs2, calculate_indexs3, calculate_indexs3_rr)

class TrainState(train_state.TrainState):
    mean: Any
    variance: Any
    count: Any

def train(env, env_params, config, rng):

    def _train(train_state_total, ent_gamma):

        train_state_policy, train_state_value, \
            train_state_policy_reach1, train_state_value_reach1, \
            train_state_policy_reach2, train_state_value_reach2, \
            rng, timestep = train_state_total

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        # TODO: start decomposed env_state at terminal of composed
        rng, _rng = jax.random.split(rng)
        runner_state_standard = (train_state_policy, train_state_value, env_state, obsv, _rng)
        
        # SPECIAL DECOMPOSED STATES
        decomposed_state = (train_state_policy_reach1, train_state_value_reach1, train_state_policy_reach2, train_state_value_reach2)

        force_combined = True #if timestep < 20 else False # ihibits switching until > 20 epochs
        # force_combined = True # immediate switching
        force_reach1, force_reach2 = False, False
        policy_controls = (force_combined, force_reach1, force_reach2)
        runner_state = (*runner_state_standard, decomposed_state, policy_controls)

        force_reach1, force_reach2 = True, False
        policy_controls_reach1 = (force_combined, force_reach1, force_reach2)
        runner_state_reach1 = (*runner_state_standard, decomposed_state, policy_controls_reach1)

        force_reach1, force_reach2 = False, True
        policy_controls_reach2 = (force_combined, force_reach1, force_reach2)
        runner_state_reach2 = (*runner_state_standard, decomposed_state, policy_controls_reach2)

        # COLLECT TRAJECTORY COMPOSED
        runner_state, traj_batch = jax.lax.scan(
            env_step, runner_state, None, config["NUM_STEPS"]
        )

        # COLLECT TRAJECTORY DECOMPOSED
        runner_state_reach1, traj_batch_reach1 = jax.lax.scan(
            env_step, runner_state_reach1, None, config["NUM_STEPS"]
        )
        runner_state_reach2, traj_batch_reach2 = jax.lax.scan(
            env_step, runner_state_reach2, None, config["NUM_STEPS"]
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
        l_tile_append = jnp.minimum(jnp.maximum(reach1_append, V_reach2_append), jnp.maximum(reach2_append, V_reach1_append))

        indexs, done = calculate_indexs3_rr(config["GAMMA_ENERGY"], traj_batch.reward, l_tile_append,
                                               jnp.expand_dims(last_val, axis=1).T)
        done = done[:-1, :]

        advantages_V, targets_V = calculate_gae2(config["GAMMA_ENERGY"], config["GAE_LAMBDA"], traj_batch, done, last_val)
        advantages_total, _ = calculate_gae_reach4(config["GAMMA_REACH_INIT"], config["GAE_LAMBDA"], l_tile_append, V_append, done)

        # UPDATE COMPOSED NETWORK
        update_state = (train_state_policy, train_state_value,
                        traj_batch, advantages_V, targets_V, advantages_total, rng)

        xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent_gamma[0]
        update_state, loss_info = jax.lax.scan(
            update_epoch, update_state, xs, config["UPDATE_EPOCHS"]
        )
        train_state_policy = update_state[0]
        train_state_value = update_state[1]
        rng = update_state[-1]

        # CALCULATE DECOMPOSED ADVANTAGES - 1
        (train_state_policy, train_state_value, env_state_1, last_obs_1, rng_1,
          decomposed_state, policy_controls) = runner_state_reach1

        last_val1 = train_state_value_reach1.apply_fn(train_state_value_reach1.params, last_obs_1)
        reach1_append = jnp.concatenate((traj_batch_reach1.reach1, jnp.expand_dims(env_state_1.reach1, axis=1).T))
        V_reach1_append = jnp.concatenate((traj_batch_reach1.value_reach1, jnp.expand_dims(last_val1, axis=1).T))

        indexs, done_1 = calculate_indexs3_rr(config["GAMMA_ENERGY"], traj_batch_reach1.reward, reach1_append,
                                               jnp.expand_dims(last_val1, axis=1).T)
        done_1 = done_1[:-1, :]

        advantages_V_reach1, targets_V_reach1 = calculate_gae2(config["GAMMA_ENERGY"], config["GAE_LAMBDA"], traj_batch_reach1, done_1, last_val1)
        advantages_total_reach1, _ = calculate_gae_reach4(config["GAMMA_REACH_INIT"], config["GAE_LAMBDA"], reach1_append, V_reach1_append, done_1)

        # UPDATE DECOMPOSED NETWORK - 1
        update_state = (train_state_policy_reach1, train_state_value_reach1,
                        traj_batch_reach1, advantages_V_reach1, targets_V_reach1, advantages_total_reach1, rng_1)

        xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent_gamma[0]
        update_state, loss_info_1 = jax.lax.scan(
            update_epoch, update_state, xs, config["UPDATE_EPOCHS"]
        )
        train_state_policy_reach1 = update_state[0]
        train_state_value_reach1 = update_state[1]
        rng_1 = update_state[-1]

        # CALCULATE DECOMPOSED ADVANTAGES - 2
        (train_state_policy, train_state_value, env_state_2, last_obs_2, rng_2,
          decomposed_state, policy_controls) = runner_state_reach1

        last_val2 = train_state_value_reach2.apply_fn(train_state_value_reach2.params, last_obs_2)
        reach2_append = jnp.concatenate((traj_batch_reach2.reach2, jnp.expand_dims(env_state_2.reach2, axis=1).T))
        V_reach2_append = jnp.concatenate((traj_batch_reach2.value_reach2, jnp.expand_dims(last_val2, axis=1).T))

        indexs, done_2 = calculate_indexs3_rr(config["GAMMA_ENERGY"], traj_batch_reach2.reward, reach2_append,
                                               jnp.expand_dims(last_val2, axis=1).T)
        done_2 = done_2[:-1, :]

        advantages_V_reach2, targets_V_reach2 = calculate_gae2(config["GAMMA_ENERGY"], config["GAE_LAMBDA"], traj_batch_reach2, done_2, last_val2)
        advantages_total_reach2, _ = calculate_gae_reach4(config["GAMMA_REACH_INIT"], config["GAE_LAMBDA"], reach2_append, V_reach2_append, done_2)

        # UPDATE DECOMPOSED NETWORK - 2
        update_state = (train_state_policy_reach2, train_state_value_reach2,
                        traj_batch_reach2, advantages_V_reach2, targets_V_reach2, advantages_total_reach2, rng_2)

        xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent_gamma[0]
        update_state, loss_info_2 = jax.lax.scan(
            update_epoch, update_state, xs, config["UPDATE_EPOCHS"]
        )
        train_state_policy_reach2 = update_state[0]
        train_state_value_reach2 = update_state[1]
        rng_2= update_state[-1]

        return ((train_state_policy, train_state_value, train_state_policy_reach1, train_state_value_reach1, train_state_policy_reach2, train_state_value_reach2, rng, timestep),
                {"batch_info": (traj_batch, targets_V, done), "loss_info": loss_info,
                 "batch_1_info": (traj_batch_reach1, targets_V_reach1, done_1), "loss_info_1": loss_info_1,
                 "batch_2_info": (traj_batch_reach2, targets_V_reach2, done_2), "loss_info_2": loss_info_2,
                 "reach_gamma": ent_gamma[1], "entropy_weight": ent_gamma[0]})

    update_epoch = partial(_ppo_vanilla_update, config)
    env_step = partial(_env_step_rr_vanilla, env, env_params)
    training = jax.jit(_train)

    tx = optimizer(config)

    # INIT POLICY NETWORK
    if config["DISCRETE"] == False:
        policy_network = Policy_Network(
            env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
        )
        policy_network_reach1 = Policy_Network(
            env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
        )
        policy_network_reach2 = Policy_Network(
            env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
        )
    else:
        policy_network = Policy_Network_Discrete(
            env.action_space(env_params).n, activation=config["ACTIVATION"]
        )
        policy_network_reach1 = Policy_Network(
            env.action_space(env_params).n, activation=config["ACTIVATION"]
        )
        policy_network_reach2 = Policy_Network(
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

    # DECOMPOSED REACH POLICIES
    network_params_policy_reach1 = policy_network_reach1.init(_rng, init_x)
    train_state_policy_reach1 = TrainState.create(
        apply_fn=policy_network_reach1.apply,
        params=network_params_policy_reach1,
        tx=tx,
        mean=jnp.zeros(env.observation_space(env_params).shape),
        variance=jnp.zeros(env.observation_space(env_params).shape),
        count=1e-4,
    )

    network_params_policy_reach2 = policy_network_reach2.init(_rng, init_x)
    train_state_policy_reach2 = TrainState.create(
        apply_fn=policy_network_reach2.apply,
        params=network_params_policy_reach2,
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

    # DECOMPOSED VALUE CRITICS
    value_network_reach1 = Value_Network(activation=config["ACTIVATION"])
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(env.observation_space(env_params).shape)
    network_params_reach1 = value_network_reach1.init(_rng, init_x)
    train_state_value_reach1 = TrainState.create(
        apply_fn=value_network_reach1.apply,
        params=network_params_reach1,
        tx=tx,
        mean=jnp.zeros(env.observation_space(env_params).shape),
        variance=jnp.zeros(env.observation_space(env_params).shape),
        count=1e-4,
    )

    value_network_reach2 = Value_Network(activation=config["ACTIVATION"])
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(env.observation_space(env_params).shape)
    network_params_reach2 = value_network_reach2.init(_rng, init_x)
    train_state_value_reach2 = TrainState.create(
        apply_fn=value_network_reach2.apply,
        params=network_params_reach2,
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
        
        traj_batch, targets_V, done = result_traj

        cnt_1, cnt_2, reach_idx_1, reach_idx_2 = calculate_reachreach(traj_batch)

        idx = 0

        # reach_idx = calculate_minimal_reach(traj_batch.reach[:, idx])

        info = tree_index2(traj_batch.info, idx)
        # info['init_energy'] = traj_batch.energy[0, idx]
        # info['final_energy'] = traj_batch.energy[reach_idx, idx]
        info['reach_index_1'] = reach_idx_1
        info['reach_index_2'] = reach_idx_2
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
                                    overwrite=True,
                                    keep=2)

        fig = plot_contour(None, None, None, info, timestep, config)
        # plot_target(targets_h[:, idx], traj_batch.value_reach[:, idx], traj_batch.reach1[:, idx], traj_batch.reach2[:, idx],
        #             timestep, traj_batch.energy[0, idx], done[:, idx], config)
        # plot_value_target(targets_V[:, idx], traj_batch.value[:, idx], timestep,
        #                   traj_batch.energy[0, idx], done[:, idx], config)
        t1 = time.time()

        wandb.log({
                #    "not reaching goal": cnt,
                   "actor_loss": jnp.mean(loss_info["actor_loss"]), "value_loss": jnp.mean(loss_info["value_loss"]),
                #    "entropy_loss": jnp.mean(loss_info["entropy_loss"]),
                   "actor_1_loss": jnp.mean(loss_info_1["actor_loss"]), "value_1_loss": jnp.mean(loss_info_1["value_loss"]),
                   "actor_2_loss": jnp.mean(loss_info_2["actor_loss"]), "value_2_loss": jnp.mean(loss_info_2["value_loss"]),
                   "reach_gamma": result['reach_gamma'][0], "entropy_weight": result['entropy_weight'][0],
                   'trajectory_sample':wandb.Image(fig)})
        plt.close("all")
        # print("Earliest Reach {}: {}        {}".format(timestep, cnt, np.mean(consumption)))
        print("Time {}".format(t1-t0))

    return

if __name__ == "__main__":
    config = vars(get_args(sys.argv[1:]))
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
    env = get_env(config)
    wandb.init(project='EC-EFPPO-{}'.format(config["EXP_NAME"]), name=config["NAME"], config=config)
    env_params = env.default_params
    if config['EXP_NAME'] == 'WindField':
        env_params = env_params.replace(index=config['SECTION'])
    rng = jax.random.PRNGKey(20)
    out = train(env, env_params, config, rng)