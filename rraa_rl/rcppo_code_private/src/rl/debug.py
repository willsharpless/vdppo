import sys

sys.path.append("/home/mepear_gc")

import os
import time
import wandb
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from flax.training.train_state import TrainState
from flax.training import checkpoints

from arguments import get_args
from functools import partial

from EFPPO.src.rl.EFPPO_utils import _ecefppo_update, _env_step
from EFPPO.src.env.env_list import get_env
from EFPPO.src.model.actorcritic import Policy_Network, Value_Network, Policy_Network_Discrete
from EFPPO.src.rl.plot_utils import calculate_consumption, plot_target, plot_value_target, plot_contour
from EFPPO.src.rl.utils import optimizer, get_BuRd, tree_index1, tree_index2
from EFPPO.src.rl.gae import (Transition_normal, Transition_reach,
                              calculate_gae, calculate_gae2, calculate_gae3,
                              calculate_gae_reach, calculate_gae_reach2, calculate_gae_reach3,
                              calculate_indexs, calculate_indexs2, calculate_indexs3)


def train(env, env_params, config, rng):
    def plot_consumption(traj_batch, energy_array, config):
        fig, ax = plt.subplots(2, 1, figsize=(6, 12))
        reach = (traj_batch.reach < 0)
        num = energy_array.shape[0]
        reach_idx = reach.argmax(axis=0)
        energy = np.zeros(num)
        for i in range(reach_idx.shape[0]):
            if reach_idx[i] == 0:
                reach_idx = reach_idx.at[i].set(config["NUM_STEPS"])
                energy[i] = np.sum(traj_batch.reward[:, i])
            else:
                energy[i] = np.sum(traj_batch.reward[0: reach_idx[i], i])
        ax[0].plot(energy_array, energy)
        ax[0].plot(energy_array, energy_array)
        ax[0].set_xlabel("Initial Energy Budget")
        ax[0].set_ylabel("Energy Consumption")
        # ax[0].set_ylim((0, 800))
        ax[1].plot(energy_array, reach_idx)
        ax[1].set_ylim((0, 425))
        plt.savefig('model/{}/energy_consumption_random.png'.format(config["DIR"]), dpi=300)
        plt.close()
        return
    def _train(train_state_total, unused):

        train_state_policy, train_state_energy, train_state_h, rng = train_state_total

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        obsv = obsv.at[:, 0].set(jnp.cos(jnp.pi * 0.95))
        obsv = obsv.at[:, 1].set(jnp.sin(jnp.pi * 0.95))
        obsv = obsv.at[:, 2].set(0)
        obsv = obsv.at[:, 3].set((jnp.arange(-200., 600., 20.) - 200.) / 600.)

        env_state = env_state.replace(theta=jnp.ones(config["NUM_ENVS"]) * jnp.pi * 0.95)
        env_state = env_state.replace(theta_dot=jnp.zeros(config["NUM_ENVS"]))
        env_state = env_state.replace(energy=jnp.arange(-200., 600., 20.))

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

        reach_append = jnp.concatenate((traj_batch.reach, jnp.expand_dims(env_state.reach, axis=1).T))
        V_reach_append = jnp.concatenate((traj_batch.value_reach, jnp.expand_dims(last_val_h, axis=1).T))

        energy_append = jnp.concatenate((traj_batch.energy, jnp.expand_dims(env_state.energy, axis=1).T))
        V_append = jnp.concatenate((traj_batch.value, jnp.expand_dims(last_val, axis=1).T))
        V_total_append = jnp.maximum(V_reach_append, V_append - energy_append)
        g_append = jnp.maximum(reach_append, -energy_append)

        indexs, done = calculate_indexs3(config["GAMMA_ENERGY"], traj_batch.reward, traj_batch.energy, reach_append,
                                         jnp.expand_dims(last_val, axis=1).T, jnp.expand_dims(last_val_h, axis=1).T)
        done = done[:-1, :]

        advantages_h, targets_h = calculate_gae_reach3(config['GAMMA_REACH_INIT'], config["GAE_LAMBDA"], reach_append,
                                                       V_reach_append, done)

        advantages_V, targets_V = calculate_gae2(config["GAMMA_ENERGY"], config["GAE_LAMBDA"], traj_batch, done,
                                                 last_val)

        advantages_total, _ = calculate_gae_reach3(config["GAMMA_REACH_INIT"], config["GAE_LAMBDA"], g_append,
                                                   V_total_append, done)

        # UPDATE NETWORK
        update_state = (train_state_policy, train_state_energy, train_state_h,
                        traj_batch, advantages_h, targets_h, advantages_V, targets_V, advantages_total, rng)
        train_state_policy = update_state[0]
        train_state_energy = update_state[1]
        train_state_h = update_state[2]
        rng = update_state[-1]

        return (train_state_policy, train_state_energy, train_state_h, rng), (traj_batch, targets_h, targets_V, done)

    update_epoch = partial(_ecefppo_update, config)
    env_step = partial(_env_step, env, env_params)
    training = jax.jit(_train)

    tx = optimizer(config)

    raw_restored = checkpoints.restore_checkpoint(ckpt_dir='/home/mepear_gc/EFPPO/model/{}/{}'.format(config["DIR"], config["DIR_MODEL"]),
                                                  target=None
                                                  )

    train_state_policy_params = raw_restored['policy_network']['params']
    train_state_energy_params = raw_restored['energy_network']['params']
    train_state_h_params = raw_restored['reach_network']['params']

    # INIT POLICY NETWORK
    policy_network = Policy_Network(
        env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
    )
    rng, _rng = jax.random.split(rng)
    train_state_policy = TrainState.create(
        apply_fn=policy_network.apply,
        params=train_state_policy_params,
        tx=tx,
    )

    # INIT VALUE ENERGY NETWORK
    value_network_energy = Value_Network(activation=config["ACTIVATION"])
    rng, _rng = jax.random.split(rng)
    train_state_energy = TrainState.create(
        apply_fn=value_network_energy.apply,
        params=train_state_energy_params,
        tx=tx,
    )

    # INIT VALUE FIND NETWORK
    value_network_h = Value_Network(activation=config["ACTIVATION"])
    rng, _rng = jax.random.split(rng)
    train_state_h = TrainState.create(
        apply_fn=value_network_h.apply,
        params=train_state_h_params,
        tx=tx,
    )

    t0 = time.time()

    update_state, result = jax.lax.scan(
        training, (train_state_policy, train_state_energy, train_state_h, rng), None, config["STEP_SCAN"]
    )

    result = tree_index1(result, 0)

    traj_batch, targets_h, targets_V, done = result

    plot_consumption(traj_batch, jnp.arange(-200., 400, 15), config)

    for idx in range(config["NUM_ENVS"]):

        info = tree_index2(traj_batch.info, idx)

        info['init_energy'] = traj_batch.energy[0, idx]
        info['final_energy'] = traj_batch.energy[-1, idx]

        plot_contour(train_state_energy, train_state_h, train_state_policy, info, 1000 + 20 * idx, config)
        plot_target(targets_h[:, idx], traj_batch.value_reach[:, idx], traj_batch.reach[:, idx],
                    1000 + 20 * idx, traj_batch.energy[0, idx], done[:, idx], config)
        plot_value_target(targets_V[:, idx], traj_batch.value[:, idx], 1000 + 20 * idx,
                          traj_batch.energy[0, idx], done[:, idx], config)
        t1 = time.time()

        # print("Earliest Reach {}: {}        {}".format(timestep, cnt, np.mean(consumption)))
        print("Time {}".format(t1 - t0))

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
    env = get_env(config)
    env_params = env.default_params
    rng = jax.random.PRNGKey(30)
    out = train(env, env_params, config, rng)