# import seaborn as sns
import os
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import sys
import numpy as np
sys.path.append("/home/mepear_gc")

from flax.training import checkpoints

from EFPPO.src.rl.arguments import get_args
from EFPPO.src.env.env_list import get_env
from EFPPO.src.model.actorcritic import Policy_Network, Value_Network, Policy_Network_Discrete
from EFPPO.src.rl.gae import Transition_reach
from EFPPO.src.rl.plot_utils import plot_contour, calculate_minimal_reach
from EFPPO.src.rl.utils import tree_index2

def test(env, env_params, config, rng):

    def plot_consumption(traj_batch_list, traj_batch_deterministic, value, value_h, energy_array, config):
        fig, ax = plt.subplots(4, 1, figsize=(6, 24))
        zero = jnp.maximum(value_h, value - energy_array) < 0
        zero_idx = zero.argmax()
        energy = np.zeros((len(traj_batch_list), energy_array.shape[0]))
        energy_deterministic = np.zeros(energy_array.shape[0])
        reach_mean = np.zeros((len(traj_batch_list), energy_array.shape[0]))
        for idx in range(len(traj_batch_list)):
            reach = (traj_batch_list[idx].reach < 0)
            reach_idx = reach.argmax(axis=0)
            for i in range(reach_idx.shape[0]):
                if reach_idx[i] == 0:
                    reach_idx = reach_idx.at[i].set(config["NUM_STEPS"])
                    energy[idx, i] = np.sum(traj_batch_list[idx].reward[:, i])
                else:
                    energy[idx, i] = np.sum(traj_batch_list[idx].reward[0: reach_idx[i], i])
            reach_mean[idx] = reach_idx
        reach = (traj_batch_deterministic.reach < 0)
        reach_idx = reach.argmax(axis=0)
        for i in range(reach_idx.shape[0]):
            if reach_idx[i] == 0:
                reach_idx = reach_idx.at[i].set(config["NUM_STEPS"])
                energy_deterministic[i] = np.sum(traj_batch_deterministic.reward[:, i])
            else:
                energy_deterministic[i] = np.sum(traj_batch_deterministic.reward[0: reach_idx[i], i])
        reach_deterministic = reach_idx
        ax[0].plot(energy_array, value - energy_array)
        ax[0].axvline(energy_array[zero_idx], color='red', alpha=0.7)
        ax[0].set_title("Energy Network Value")
        ax[1].plot(energy_array, value_h)
        ax[1].axvline(energy_array[zero_idx], color='red', alpha=0.7)
        ax[1].set_title("Reach Network Value")
        avg = np.mean(energy, axis=0)
        half_std = np.std(energy, axis=0) / 2.0
        lower = [x - y for x, y in zip(avg, half_std)]
        upper = [x + y for x, y in zip(avg, half_std)]
        ax[2].plot(energy_array, np.mean(energy, axis=0), color='blue', label='mean')
        ax[2].fill_between(energy_array, lower, upper, color='blue', alpha=0.2)
        ax[2].plot(energy_array, np.max(energy, axis=0), label='max')
        ax[2].plot(energy_array, np.min(energy, axis=0), label='min')
        ax[2].plot(energy_array, energy_deterministic, color='red', label='deterministic')
        ax[2].plot(energy_array, energy_array)
        ax[2].axvline(energy_array[zero_idx], color='red', alpha=0.7)
        ax[2].set_xlabel("Initial Energy Budget")
        ax[2].set_ylabel("Energy Consumption")
        ax[2].legend()
        avg = np.mean(reach_mean, axis=0)
        half_std = np.std(reach_mean, axis=0) / 2.0
        lower = [x - y for x, y in zip(avg, half_std)]
        upper = [x + y for x, y in zip(avg, half_std)]

        reach_limit = np.mean(reach_mean, axis=0) < config["NUM_STEPS"]
        ax[3].plot(np.ma.array(energy_array, mask=~reach_limit),
                   np.ma.array(np.mean(reach_mean, axis=0), mask=~reach_limit), color='blue', label='mean')
        ax[3].fill_between(energy_array, lower, upper, color='blue', alpha=0.2)

        ax[3].plot(np.ma.array(energy_array, mask=~reach_limit),
                   np.ma.array(np.max(reach_mean, axis=0), mask=~reach_limit), label='max')

        ax[3].plot(np.ma.array(energy_array, mask=~reach_limit),
                   np.ma.array(np.min(reach_mean, axis=0), mask=~reach_limit), label='min')

        reach_limit = reach_deterministic < config["NUM_STEPS"]
        ax[3].plot(np.ma.array(energy_array, mask=~reach_limit),
                   np.ma.array(reach_deterministic, mask=~reach_limit), color='red', label='deterministic')

        # ax[3].set_ylim((0, 1000))
        ax[3].axvline(energy_array[zero_idx], color='red', alpha=0.7)
        ax[3].legend()
        ax[3].set_xlabel("Initial Energy Budget")
        ax[3].set_ylabel("Timesteps Used to Reach Goal")
        plt.savefig('model/{}/energy_consumption.png'.format(config["DIR"]), dpi=300)
        plt.close()
        return

    def plot_control(controls, energy, idx, config):
        fig, ax = plt.subplots(1, 1)
        plt.plot(controls)
        ax.set_title('Control Plot')
        plt.savefig('model/{}/policy/control_{}_{}.png'.format(config["DIR"], energy, idx), dpi=300)
        plt.close()

    def plot_trajectory(info, index, config):
        if config["EXP_NAME"] == "PendulumConstraint":
            fig, ax = plt.subplots(1, 1)
            plt.scatter(info['theta'][0], info['theta_dot'][0], s=10, c='black')
            plt.plot(info['theta'], info['theta_dot'])
            plt.xlim((-np.pi, 3 * np.pi))
            ax.vlines([-2 * np.pi, 0, 2 * np.pi], -8, 8, colors='red')
            ax.set_title("Trajectory Plot Init Energy {:.2f} Final Energy {:.2f}"
                         .format(info['init_energy'], info['final_energy']))
            plt.savefig('model/{}/traj/energy_{:0>4d}.png'.format(config["DIR"], index), dpi=300)
            plt.close()
        elif config["EXP_NAME"] == 'HopperAvoidCeiling':
            plt.figure(figsize=(12, 6))
            fig, ax = plt.subplots(1, 1)
            reach_idx = info['reach_index']
            for i in range(0, reach_idx, 16):
                ax.plot(np.array([info['head_pos'][i, 0], info['jaw_pos'][i, 0]]),
                        np.array([info['head_pos'][i, 1], info['jaw_pos'][i, 1]]), c='r')
                ax.plot(np.array([info['jaw_pos'][i, 0], info['thg_pos'][i, 0]]),
                        np.array([info['jaw_pos'][i, 1], info['thg_pos'][i, 1]]), c='g')
                ax.plot(np.array([info['thg_pos'][i, 0], info['leg_pos'][i, 0]]),
                        np.array([info['thg_pos'][i, 1], info['leg_pos'][i, 1]]), c='b')
                ax.plot(np.array([info['leg_pos'][i, 0], info['foot_front_pos'][i, 0]]),
                        np.array([info['leg_pos'][i, 1], info['foot_front_pos'][i, 1]]), c='b')
                ax.plot(np.array([info['leg_pos'][i, 0], info['foot_back_pos'][i, 0]]),
                        np.array([info['leg_pos'][i, 1], info['foot_back_pos'][i, 1]]), c='m')
            ax.plot(np.array([info['head_pos'][reach_idx, 0], info['jaw_pos'][reach_idx, 0]]),
                    np.array([info['head_pos'][reach_idx, 1], info['jaw_pos'][reach_idx, 1]]), c='r')
            ax.plot(np.array([info['jaw_pos'][reach_idx, 0], info['thg_pos'][reach_idx, 0]]),
                    np.array([info['jaw_pos'][reach_idx, 1], info['thg_pos'][reach_idx, 1]]), c='g')
            ax.plot(np.array([info['thg_pos'][reach_idx, 0], info['leg_pos'][reach_idx, 0]]),
                    np.array([info['thg_pos'][reach_idx, 1], info['leg_pos'][reach_idx, 1]]), c='b')
            ax.plot(np.array([info['leg_pos'][reach_idx, 0], info['foot_front_pos'][reach_idx, 0]]),
                    np.array([info['leg_pos'][reach_idx, 1], info['foot_front_pos'][reach_idx, 1]]), c='b')
            ax.plot(np.array([info['leg_pos'][reach_idx, 0], info['foot_back_pos'][reach_idx, 0]]),
                    np.array([info['leg_pos'][reach_idx, 1], info['foot_back_pos'][reach_idx, 1]]), c='m')
            draw_circle = plt.Circle((2.0, 1.4), 0.1, fill=False)
            draw_rectangle = plt.Rectangle((0.95, 1.3), 0.1, 0.2, facecolor="red", fill=True)
            ax.add_patch(draw_circle)
            ax.add_patch(draw_rectangle)
            ax.set_xlim((-0.5, 2.5))
            ax.set_ylim((0, 1.5))
            ax.set_aspect('equal')
            ax.set_title("Trajectory Plot Init Energy {:.2f} Final Energy {:.2f}"
                         .format(info['init_energy'], info['final_energy']))
            plt.savefig('model/{}/traj/energy_{:0>4d}'.format(config["DIR"], index), dpi=300)
            plt.close('all')

    def _env_step(runner_state, unused):
        (train_state_policy, train_state_energy, train_state_h,
         last_env_state, last_obs, rng) = runner_state

        # SELECT ACTION
        rng, _rng = jax.random.split(rng)
        pi = policy_network.apply(train_state_policy['params'], last_obs)
        value = value_network_energy.apply(train_state_energy['params'], last_obs)
        value_h = value_network_h.apply(train_state_h['params'], last_obs)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)

        # STEP ENV
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(rng_step, last_env_state, action, env_params)
        transition = Transition_reach(
            done, action, value, value_h, reward, last_env_state.energy, log_prob, last_obs, info,
            last_env_state.reach
        )
        runner_state = (train_state_policy, train_state_energy, train_state_h,
                        env_state, obsv, rng)
        return runner_state, transition

    def _env_step_deterministic(runner_state, unused):
        (train_state_policy, train_state_energy, train_state_h,
         last_env_state, last_obs, rng) = runner_state

        # SELECT ACTION
        rng, _rng = jax.random.split(rng)
        pi = policy_network.apply(train_state_policy['params'], last_obs)
        value = value_network_energy.apply(train_state_energy['params'], last_obs)
        value_h = value_network_h.apply(train_state_h['params'], last_obs)
        action = pi.loc
        log_prob = pi.log_prob(action)

        # STEP ENV
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(rng_step, last_env_state, action, env_params)
        transition = Transition_reach(
            done, action, value, value_h, reward, last_env_state.energy, log_prob, last_obs, info,
            last_env_state.reach
        )
        runner_state = (train_state_policy, train_state_energy, train_state_h,
                        env_state, obsv, rng)
        return runner_state, transition

    # INIT POLICY NETWORK
    policy_network = Policy_Network(
        env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
    )

    # INIT VALUE ENERGY NETWORK
    value_network_energy = Value_Network(activation=config["ACTIVATION"])

    # INIT VALUE FIND NETWORK
    value_network_h = Value_Network(activation=config["ACTIVATION"])

    raw_restored = checkpoints.restore_checkpoint(ckpt_dir='/home/mepear_gc/EFPPO/model/{}/{}'.format(config["DIR"], config["DIR_MODEL"]),
                                                  target=None
                                                  )

    train_state_policy = raw_restored['policy_network']
    train_state_energy = raw_restored['energy_network']
    train_state_h = raw_restored['reach_network']

    if config["EXP_NAME"] == 'PendulumConstraint':

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

        traj_batch_list = []

        # COLLECT TRAJECTORY
        for i in range(10):
            new_runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            rng, _rng = jax.random.split(_rng)
            runner_state = (train_state_policy, train_state_energy,
                            train_state_h, env_state, obsv, _rng)
            traj_batch_list.append(traj_batch)

        new_runner_state, traj_batch = jax.lax.scan(
            _env_step_deterministic, runner_state, None, config["NUM_STEPS"]
        )

        value = value_network_energy.apply(train_state_energy['params'], obsv)
        value_h = value_network_h.apply(train_state_h['params'], obsv)
        plot_consumption(traj_batch_list, traj_batch, value, value_h, jnp.arange(-200., 600, 20), config)
        for i in range(40):
            reach_idx = calculate_minimal_reach(traj_batch.reach[:, i])
            info = tree_index2(traj_batch.info, i)
            info['init_energy'] = traj_batch.energy[0, i]
            info['final_energy'] = traj_batch.energy[reach_idx, i]
            info['reach_index'] = reach_idx
            plot_trajectory(info, 20 * i - 200, config)

    elif config["EXP_NAME"] == 'HopperAvoidCeiling':

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        obsv = obsv.at[:, -1].set((jnp.arange(-200., 600., 20) - 400.) / 400.)

        env_state = env_state.replace(energy=jnp.arange(-200., 600., 20))

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state_policy, train_state_energy,
                        train_state_h, env_state, obsv, _rng)

        traj_batch_list = []

        # COLLECT TRAJECTORY
        for i in range(10):
            new_runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            rng, _rng = jax.random.split(_rng)
            runner_state = (train_state_policy, train_state_energy,
                            train_state_h, env_state, obsv, _rng)
            traj_batch_list.append(traj_batch)

        new_runner_state, traj_batch = jax.lax.scan(
            _env_step_deterministic, runner_state, None, config["NUM_STEPS"]
        )

        value = value_network_energy.apply(train_state_energy['params'], obsv)
        value_h = value_network_h.apply(train_state_h['params'], obsv)

        plot_consumption(traj_batch_list, traj_batch, value, value_h, jnp.arange(-200., 600, 20), config)
        for i in range(40):
            reach_idx = calculate_minimal_reach(traj_batch.reach[:, i])
            info = tree_index2(traj_batch.info, i)
            info['init_energy'] = traj_batch.energy[0, i]
            info['final_energy'] = traj_batch.energy[reach_idx, i]
            info['reach_index'] = reach_idx
            plot_trajectory(info, 20 * i - 200, config)

        for i in range(40):
            plot_control(traj_batch.action[:, i, 0], 20 * i - 200, 0, config)
            plot_control(traj_batch.action[:, i, 1], 20 * i - 200, 1, config)
            plot_control(traj_batch.action[:, i, 2], 20 * i - 200, 2, config)
    return

if __name__ == "__main__":
    config = vars(get_args(sys.argv[1:]))
    env = get_env(config)
    env_params = env.default_params
    rng = jax.random.PRNGKey(23)
    folder = os.path.exists("model/{}/traj".format(config['DIR']))
    if not folder:
        os.makedirs("model/{}/traj".format(config['DIR']))
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    test(env, env_params, config, rng)