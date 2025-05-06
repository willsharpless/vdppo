import os
import optax
import jax
import sys
import numpy as np

sys.path.append("/home/mepear_gc")

from functools import partial
from flax.training.train_state import TrainState
from flax.training import checkpoints
import jax.numpy as jnp

import matplotlib.pyplot as plt

from rraa_rl.EFPPO.src.rl.arguments import get_args
from rraa_rl.EFPPO.src.env.env_list import get_env
from rraa_rl.EFPPO.src.model.actorcritic import Policy_Network, Value_Network, ActorCritic_Continuous
from rraa_rl.EFPPO.src.rl.EFPPO_utils import _env_step, _env_step_deterministic, _env_step_test
from rraa_rl.EFPPO.src.rl.root_finding import Bisection
from rraa_rl.EFPPO.src.rl.utils import tree_index1, tree_index2

def calculate_reach_rate(traj_batch):
    reach_idx = ((traj_batch.reach < 0) * (traj_batch.energy > 0)).argmax(axis=0)
    cnt = 0.0
    for i in range(reach_idx.shape[0]):
        if reach_idx[i] == 0 and (((traj_batch.reach < 0) * (traj_batch.energy > 0))[0, i] == False):
            cnt = cnt
        else:
            cnt = cnt + 1

    return cnt / reach_idx.shape[0]

def plot_trajectory_new(info_list, action, index, config):
    plt.figure(figsize=(6, 6), dpi=300)
    fig, ax = plt.subplots(1, 1)
    X, Y = np.meshgrid(np.arange(85) / 256. * 60. - 30., np.arange(85) / 256. * 60. - 30.)
    ax.quiver(X, Y, info_list[0]['u_air'][0:85, 0:85], info_list[0]['v_air'][0:85, 0:85], color='b')
    for info in info_list:
        ax.scatter(info['pos'][0, 0], info['pos'][0, 1], s=10, c='black')
        ax.plot(info['pos'][0:info['reach_index'], 0], info['pos'][0:info['reach_index'], 1])
    draw_circle = plt.Circle((-11.25, -11.25), radius=1.25, fill=False)
    ax.add_patch(draw_circle)
    ax.set_xlim((-35., -5.))
    ax.set_ylim((-35., -5.))
    ax.set_aspect('equal')
    draw_rectangle_1 = plt.Rectangle((23. / 256. * 60. - 30, 27. / 256. * 60. - 30),
                                     12. / 256. * 60., 10. / 256. * 60., facecolor="black", fill=True)

    draw_rectangle_2 = plt.Rectangle((69. / 256. * 60. - 30, 27. / 256. * 60. - 30),
                                     11. / 256. * 60., 10. / 256. * 60., facecolor="black", fill=True)

    draw_rectangle_3 = plt.Rectangle((28. / 256. * 60. - 30, 48. / 256. * 60. - 30),
                                     12. / 256. * 60., 11. / 256. * 60., facecolor="black", fill=True)

    draw_rectangle_4 = plt.Rectangle((69. / 256. * 60. - 30, 47. / 256. * 60. - 30),
                                     10. / 256. * 60., 12. / 256. * 60., facecolor="black", fill=True)

    ax.add_patch(draw_rectangle_1)
    ax.add_patch(draw_rectangle_2)
    ax.add_patch(draw_rectangle_3)
    ax.add_patch(draw_rectangle_4)
    ax.set_title("Trajectory Plot")
    plt.savefig('model/{}/traj/energy_{:0>4d}'.format(config["DIR"], index), dpi=300)
    plt.close("all")
    plt.plot(action[:, 0])
    plt.plot(action[:, 1])
    plt.plot(action[:, 2])
    plt.savefig('model/{}/traj/action_{:0>4d}'.format(config["DIR"], index), dpi=300)

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
    elif config['EXP_NAME'] == 'WindField':
        x_index = config['SECTION'] % 2
        y_index = config['SECTION'] // 2
        obs = info['obs']
        plt.figure(figsize=(6, 6), dpi=300)
        fig, ax = plt.subplots(1, 1)
        X, Y = np.meshgrid(np.linspace(-15, 15, 128), np.linspace(-15, 15, 128))
        ax.contour(X, Y, obs[(0 + 128 * y_index): (128 + 128 * y_index),
                        (0 + 128 * x_index): (128 + 128 * x_index)], colors='k', linewidth=0.5)
        ax.scatter(info['pos'][0, 0], info['pos'][0, 1], s=10, c='black')
        ax.plot(info['pos'][0:info['reach_index'], 0], info['pos'][0:info['reach_index'], 1])
        draw_rectangle = plt.Rectangle((-26. * x_index + 11., -26. * y_index + 11.), 4., 4., fill=True, color='y')
        ax.add_patch(draw_rectangle)
        ax.set_xlim((-20., 20.))
        ax.set_ylim((-20., 20.))
        ax.set_aspect('equal')
        ax.set_title("Trajectory Plot")
        plt.savefig('model/{}/traj/energy_{:0>4d}_{}'.format(config["DIR"], index, info['reach_index'] == config["NUM_STEPS"]), dpi=300)
        plt.close("all")

def test(env, env_params, config, rng):

    env_step = partial(_env_step, env, env_params)

    env_step_baseline = partial(_env_step_test, env, env_params)

    env_step_deterministic = partial(_env_step_deterministic, env, env_params)

    # INIT POLICY NETWORK
    policy_network = Policy_Network(
        env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
    )

    # INIT VALUE ENERGY NETWORK
    value_network_energy = Value_Network(activation=config["ACTIVATION"])

    # INIT VALUE FIND NETWORK
    value_network_h = Value_Network(activation=config["ACTIVATION"])


    raw_restored = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath('model/{}/{}'.format(
        config["DIR"], config["DIR_MODEL"])), target=None)

    raw_restored_baseline_ppo = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath('model/{}_ppo/{}'.format(
        config["DIR"], config["DIR_MODEL"])), target=None)

    raw_restored_baseline_cppo = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath('model/{}_cppo/{}'.format(
        config["DIR"], config["DIR_MODEL"])), target=None)


    train_state_policy = TrainState.create(
        apply_fn=policy_network.apply,
        params=raw_restored['policy_network']['params'],
        tx=optax.sgd(0.01, 0.99),
    )

    train_state_energy = TrainState.create(
        apply_fn=value_network_energy.apply,
        params=raw_restored['energy_network']['params'],
        tx=optax.sgd(0.01, 0.99),
    )

    train_state_h = TrainState.create(
        apply_fn=value_network_h.apply,
        params=raw_restored['reach_network']['params'],
        tx=optax.sgd(0.01, 0.99),
    )

    train_state_baseline_ppo = TrainState.create(
        apply_fn=policy_network.apply,
        params=raw_restored_baseline_ppo['policy_network']['params'],
        tx=optax.sgd(0.01, 0.99),
    )

    train_state_baseline_cppo = TrainState.create(
        apply_fn=policy_network.apply,
        params=raw_restored_baseline_cppo['policy_network']['params'],
        tx=optax.sgd(0.01, 0.99),
    )

    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])

    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

    if config['EXP_NAME'] == 'PendulumConstraint':
        root_finding = Bisection(train_state_energy, train_state_h, 400, 400, obsv, 150, 20)
        result, _ = root_finding.run()
    elif config['EXP_NAME'] == 'HalfCheetahAvoid':
        root_finding = Bisection(train_state_energy, train_state_h, 400, 400, obsv, 160, 20)
        result, _ = root_finding.run()
    elif config["EXP_NAME"] == 'HopperAvoidCeiling':
        root_finding = Bisection(train_state_energy, train_state_h, 400, 400, obsv, 200, 20)
        result, _ = root_finding.run()
    elif config["EXP_NAME"] == 'WindField':
        root_finding = Bisection(train_state_energy, train_state_h, 400, 400, obsv, 150, 20)
        result, _ = root_finding.run()

    print(np.sum((result * 400. + 400.) > 799.))

    obsv = obsv.at[:, -1].set(result)

    env_state = env_state.replace(energy=result * 400. + 400.)

    invalid_init = 0

    if (config['EXP_NAME'] == 'HalfCheetahAvoid') or (config["EXP_NAME"] == 'HopperAvoidCeiling') or (config["EXP_NAME"] == 'WindField'):
        invalid_init = jnp.sum(env_state.avoid == -1)

    rng, _rng = jax.random.split(rng)
    runner_state = (train_state_policy, train_state_energy,
                    train_state_h, env_state, obsv, _rng)

    runner_state_baseline_cppo = (train_state_baseline_cppo, env_state, obsv, _rng)

    runner_state_baseline_ppo = (train_state_baseline_ppo, env_state, obsv, _rng)

    # COLLECT TRAJECTORY
    _, traj_batch = jax.lax.scan(
        env_step, runner_state, None, config["NUM_STEPS"]
    )

    _, traj_batch_baseline_cppo = jax.lax.scan(
        env_step_baseline, runner_state_baseline_cppo, None, config["NUM_STEPS"]
    )

    _, traj_batch_baseline_ppo = jax.lax.scan(
        env_step_baseline, runner_state_baseline_ppo, None, config["NUM_STEPS"]
    )

    reach_idx = (traj_batch.reach < 0).argmax(axis=0)
    energy_consumption = []
    for i in range(reach_idx.shape[0]):
        if reach_idx[i] == 0 and ((traj_batch.reach < 0)[0, i] == False):
            reach_idx = reach_idx.at[i].set(config["NUM_STEPS"])
            print("Total Energy Consumption: {}, Root Finding: {}, Index: {}".format(799., result[i] * 400 + 400, i))
        else:
            print("Total Energy Consumption: {}, Root Finding: {}, Index: {}".format(np.sum(traj_batch.reward[0: reach_idx[i], i]),
                                                                          result[i] * 400 + 400, i))
            energy_consumption.append(np.sum(traj_batch.reward[0: reach_idx[i], i]))


    energy_consumption = np.array(energy_consumption)
    rate = np.zeros(17)
    print("Mean Energy: {}".format(np.mean(np.array(energy_consumption))))
    for i in range(17):
        rate[i] = np.sum(energy_consumption < 50. * i) / config["NUM_ENVS"]
        print(np.sum(energy_consumption < 50. * i) / (config["NUM_ENVS"] - invalid_init))

    reach_idx_1 = (traj_batch_baseline_cppo.reach < 0).argmax(axis=0)
    energy_consumption = np.zeros(config["NUM_ENVS"])
    for i in range(reach_idx_1.shape[0]):
        if reach_idx_1[i] == 0 and ((traj_batch_baseline_cppo.reach < 0)[0, i] == False):
            reach_idx_1 = reach_idx_1.at[i].set(config["NUM_STEPS"])
            energy_consumption[i] = 801.
        else:
            energy_consumption[i] = np.sum(traj_batch_baseline_cppo.reward[0: reach_idx_1[i], i])

    rate = np.zeros(17)
    for i in range(17):
        rate[i] = np.sum(energy_consumption < 50. * i) / config["NUM_ENVS"]
        print(np.sum(energy_consumption < 50. * i) / (config["NUM_ENVS"] - invalid_init))

    reach_idx_2 = (traj_batch_baseline_ppo.reach < 0).argmax(axis=0)
    energy_consumption = np.zeros(config["NUM_ENVS"])
    for i in range(reach_idx_2.shape[0]):
        if reach_idx_2[i] == 0 and ((traj_batch_baseline_ppo.reach < 0)[0, i] == False):
            reach_idx_2 = reach_idx_2.at[i].set(config["NUM_STEPS"])
            energy_consumption[i] = 801.
        else:
            energy_consumption[i] = np.sum(traj_batch_baseline_ppo.reward[0: reach_idx_2[i], i])

    rate = np.zeros(17)
    for i in range(17):
        rate[i] = np.sum(energy_consumption < 50. * i) / config["NUM_ENVS"]
        print(np.sum(energy_consumption < 50. * i) / (config["NUM_ENVS"] - invalid_init))


    # for i in range(reach_idx.shape[0]):
    #     info = tree_index2(traj_batch.info, i)
    #     info['reach_index'] = reach_idx[i]
    #     if config["EXP_NAME"] == 'WindField':
    #         info['u_air'] = env_params.u_air
    #         info['v_air'] = env_params.v_air
    #         info['obs'] = env_params.obstacle
    #         plot_trajectory(info, i, config)


    # action_0 = np.zeros((400, 100))
    # action_1 = np.zeros((400, 100))
    # action_2 = np.zeros((400, 100))
    #
    # for i in range(100):
    #     action_0[:, i] = traj_list[i].action[:, 13, 0]
    #     action_1[:, i] = traj_list[i].action[:, 13, 1]
    #     action_2[:, i] = traj_list[i].action[:, 13, 2]
    #
    #
    # _, traj_batch_0 = jax.lax.scan(
    #     env_step_deterministic, runner_state, None, config["NUM_STEPS"]
    # )
    # rng, _rng = jax.random.split(rng)

    # for i in range(10):
    #
    #     # COLLECT TRAJECTORY
    #     _, traj_batch = jax.lax.scan(
    #         env_step, runner_state, None, config["NUM_STEPS"]
    #     )
    #     rng, _rng = jax.random.split(rng)
    #
    #     traj_list.append(traj_batch)
    #     runner_state = (train_state_policy, train_state_energy,
    #                     train_state_h, env_state, obsv, _rng)

    # last_val = train_state_energy.apply_fn(train_state_energy.params, last_obs)
    # last_val_h = train_state_h.apply_fn(train_state_h.params, last_obs)
    #
    # reach_append = jnp.concatenate((traj_batch.reach, jnp.expand_dims(env_state.reach, axis=1).T))
    # V_reach_append = jnp.concatenate((traj_batch.value_reach, jnp.expand_dims(last_val_h, axis=1).T))
    #
    # energy_append = jnp.concatenate((traj_batch.energy, jnp.expand_dims(env_state.energy, axis=1).T))
    # V_append = jnp.concatenate((traj_batch.value, jnp.expand_dims(last_val, axis=1).T))
    # V_total_append = jnp.maximum(V_reach_append, V_append - energy_append)
    # g_append = jnp.maximum(reach_append, -energy_append)
    #
    # indexs, done = calculate_indexs3(config["GAMMA_ENERGY"], traj_batch.reward, traj_batch.energy, reach_append,
    #                                        jnp.expand_dims(last_val, axis=1).T, jnp.expand_dims(last_val_h, axis=1).T)
    # done = done[:-1, :]
    #
    # advantages_h, targets_h = calculate_gae_reach3(0.99999, 0.95, reach_append, V_reach_append, done)
    #
    # gamma_list = [0.995, 0.999]
    # value = np.zeros((2, 280))
    # value[:, -1] = -300.
    #
    # for i in range(2):
    #     gamma = gamma_list[i]
    #     for j in range(278, -1, -1):
    #         value[i, j] = (1 - gamma) * traj_batch.reach[j, 3] + gamma * value[i, j + 1]
    #
    # for i in range(2):
    #     plt.plot(value[i, :], ls='--', label='{}'.format(gamma_list[i]))
    #
    # plt.plot(traj_batch.reach[:, 3], label='reach')
    # plt.plot(traj_batch.value_reach[:, 3], ls='--', label='value reach')
    # plt.plot(targets_h[:, 3], label='targets')
    # plt.legend()
    # plt.savefig('model/{}/traj/energy_{:0>4d}'.format(config["DIR"], 101), dpi=300)

    # info_list = []
    # for i in range(10):
    #     info = tree_index2(traj_list[i].info, 13)
    #     reach_idx = (traj_list[i].reach < 0).argmax(axis=0)
    #     if reach_idx[13] == 0 and ((traj_list[i].reach < 0)[0, 13] == False):
    #         reach_idx = reach_idx.at[13].set(config["NUM_STEPS"]-1)
    #     info['reach_index'] = reach_idx[13]
    #     info['u_air'] = env_params.u_air
    #     info['v_air'] = env_params.v_air
    #     info_list.append(info)
    #
    # plot_trajectory_new(info_list, traj_batch.action[:, 13], 0, config)

    # plt.plot(np.max(action_0, axis=1), label='max')
    # plt.plot(np.min(action_0, axis=1), label='min')
    # plt.plot(traj_batch_0.action[:, 13, 0], label='mean')
    # plt.savefig('model/{}/traj/energy_{:0>4d}'.format(config["DIR"], 0), dpi=300)
    # plt.legend()
    # plt.close()
    # plt.plot(np.max(action_1, axis=1),  label='max')
    # plt.plot(np.min(action_1, axis=1),  label='min')
    # plt.plot(traj_batch_0.action[:, 13, 1],  label='mean')
    # plt.legend()
    # plt.savefig('model/{}/traj/energy_{:0>4d}'.format(config["DIR"], 1), dpi=300)
    # plt.close()
    # plt.plot(np.max(action_2, axis=1),  label='max')
    # plt.plot(np.min(action_2, axis=1),  label='min')
    # plt.plot(traj_batch_0.action[:, 13, 2],  label='mean')
    # plt.legend()
    # plt.savefig('model/{}/traj/energy_{:0>4d}'.format(config["DIR"], 2), dpi=300)
    # plt.close()

    return

if __name__ == "__main__":
    config = vars(get_args(sys.argv[1:]))
    env = get_env(config)
    env_params = env.default_params
    rng = jax.random.PRNGKey(20)
    folder = os.path.exists("model/{}/traj".format(config['DIR']))
    # if not folder:
    #     os.makedirs("model/{}/traj".format(config['DIR']))
    #
    # folder = os.path.exists("model/{}_baseline_0.1/traj".format(config['DIR']))
    # if not folder:
    #     os.makedirs("model/{}_baseline_0.1/traj".format(config['DIR']))
    #
    # folder = os.path.exists("model/{}_baseline_1.0/traj".format(config['DIR']))
    # if not folder:
    #     os.makedirs("model/{}_baseline_1.0/traj".format(config['DIR']))
    #
    # folder = os.path.exists("model/{}_baseline_10.0/traj".format(config['DIR']))
    # if not folder:
    #     os.makedirs("model/{}_baseline_10.0/traj".format(config['DIR']))

    if config['EXP_NAME'] == 'WindField':
        env_params = env_params.replace(index=config['SECTION'])
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    test(env, env_params, config, rng)