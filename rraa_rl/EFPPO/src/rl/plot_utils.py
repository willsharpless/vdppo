import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm

from rraa_rl.EFPPO.src.rl.utils import get_BuRd

def calculate_consumption(traj_batch):
    reach_idx = (traj_batch.reach < 0).argmax(axis=0)
    energy = []
    cnt = 0
    idx = 0
    for i in range(reach_idx.shape[0]):
        if reach_idx[i] == 0 and traj_batch.reach[0, i] >= 0:
            cnt += 1
            idx = i
        else:
            energy.append(np.sum(traj_batch.reward[0: reach_idx[i], i]))
    return np.array(energy), cnt, idx

def calculate_reachreach(traj_batch, reach_type="both"):
    reach_idx_1 = (traj_batch.reach1 < 0).argmax(axis=0) if reach_type in ["both", "1"] else None
    reach_idx_2 = (traj_batch.reach2 < 0).argmax(axis=0) if reach_type in ["both", "2"] else None
    cnt_1 = 0
    cnt_2 = 0
    idx_1 = 0
    idx_2 = 0
    for i in range(traj_batch.done.shape[0]):
        if reach_type in ["both", "1"] and reach_idx_1[i] == 0 and traj_batch.reach1[0, i] <= 0:
            cnt_1 += 1
            idx_1 = i
        elif reach_type in ["both", "2"] and reach_idx_2[i] == 0 and traj_batch.reach2[0, i] <= 0:
            cnt_2 += 1
            idx_2 = i
    return cnt_1, cnt_2, idx_1, idx_2

def calculate_minimal_reach(reach):
    reach_idx = (reach < 0).argmax()
    if reach_idx == 0 and reach[0] >= 0:
        reach_idx = reach.shape[0]
    return reach_idx


def plot_target(target_reach, value_reach, reach, epoch, init_energy, done, config):
    fig, ax = plt.subplots(1, 1)
    ax.plot(target_reach, label='target')
    ax.plot(value_reach, label='value reach')
    ax.plot(reach, label='reach')
    for i, done_signal in enumerate(done):
        if int(done_signal) == 1:
            ax.axvline(i, color='red', alpha=0.4, zorder=1.5)
    plt.title("init energy {:.2f}".format(init_energy))
    plt.legend()
    plt.savefig('model/{}/target/contour_target_{:0>4d}'.format(config['DIR'], epoch), dpi=300)
    plt.close("all")
    return


def plot_value_target(target_value, value, epoch, init_energy, done, config):
    fig, ax = plt.subplots(1, 1)
    ax.plot(target_value, label='target')
    ax.plot(value, label='value')
    plt.title("init energy {:.2f}".format(init_energy))
    for i, done_signal in enumerate(done):
        if int(done_signal) == 1:
            ax.axvline(i, color='red', alpha=0.4, zorder=1.5)
    plt.legend()
    plt.savefig('model/{}/value_target/contour_target_{:0>4d}'.format(config['DIR'], epoch), dpi=300)
    plt.close("all")
    return


def plot_contour(train_state_energy, train_state_h, train_state_policy, info, epoch, config):

    if config['EXP_NAME'] == 'GridConstraint' or config['EXP_NAME'] == 'GridAvoid':
        num = 800
        xlist = np.linspace(-50., 50., num)
        ylist = np.linspace(0., 400., num)
        X, Y = np.meshgrid(xlist, ylist)
        if config['EXP_NAME'] == 'GridConstraint':
            arr = np.concatenate((np.reshape(np.sin(X / 50. * np.pi), (num, num, 1)), np.reshape(np.cos(X / 50. * np.pi), (num, num, 1)),
                                  np.reshape((Y - 100.) / 100., (num, num, 1))), axis=2)
            arr = np.reshape(arr, (num * num, 3))
        elif config['EXP_NAME'] == 'GridAvoid':
            arr = np.concatenate((np.reshape(np.sin(X / 50. * np.pi), (num, num, 1)), np.reshape(np.cos(X / 50. * np.pi), (num, num, 1)),
                                  np.ones((num, num, 1)), np.reshape((Y - 200.) / 200., (num, num, 1))), axis=2)
            arr = np.reshape(arr, (num * num, 4))
        Z_1 = train_state_energy.apply_fn(train_state_energy.params, arr)
        Z_2 = train_state_h.apply_fn(train_state_h.params, arr)
        Z_3 = train_state_policy.apply_fn(train_state_policy.params, arr).probs
        Z_1 = np.reshape(Z_1, (num, num))
        Z_2 = np.reshape(Z_2, (num, num))
        Z_3 = np.reshape(Z_3, (num, num, 2))
        fig, ax = plt.subplots(1, 1)
        cp = ax.contourf(X, Y, Z_1, cmap=get_BuRd())
        fig.colorbar(cp)
        if config['EXP_NAME'] == 'GridAvoid':
            ax.axvline(-30, color='red', alpha=0.7,)
            ax.axvline(-25, color='green', alpha=0.4, linewidth=10)
        ax.set_title('Contour_Plot')
        plt.savefig('./model/{}/value/contour_value_{:0>4d}'.format(config['DIR'], epoch), dpi=300)
        plt.close("all")
        fig, ax = plt.subplots(1, 1)
        cp = ax.contourf(X, Y, Z_2, norm=CenteredNorm(), cmap=get_BuRd())
        fig.colorbar(cp)
        if config['EXP_NAME'] == 'GridAvoid':
            ax.axvline(-30, color='red', alpha=0.7,)
            ax.axvline(-25, color='green', alpha=0.4, linewidth=10)
        ax.set_title('Contour_Plot')
        plt.savefig('./model/{}/reach/contour_reach_{:0>4d}'.format(config['DIR'], epoch), dpi=300)
        plt.close("all")
        fig, ax = plt.subplots(1, 1)
        cp = ax.contourf(X, Y, np.maximum(Z_2, Z_1 - Y), norm=CenteredNorm(), cmap=get_BuRd())
        fig.colorbar(cp)
        if config['EXP_NAME'] == 'GridAvoid':
            ax.axvline(-30, color='red', alpha=0.7,)
            ax.axvline(-25, color='green', alpha=0.4, linewidth=10)
        ax.set_title('Contour_Plot')
        plt.savefig('./model/{}/total/contour_total_{:0>4d}'.format(config['DIR'], epoch), dpi=300)
        plt.close("all")
        fig, ax = plt.subplots(1, 1)
        cp = ax.contourf(X, Y, Z_3[:, :, 1], cmap=get_BuRd())
        fig.colorbar(cp)
        if config['EXP_NAME'] == 'GridAvoid':
            ax.axvline(-30, color='red', alpha=0.7,)
            ax.axvline(-25, color='green', alpha=0.4, linewidth=10)
        ax.set_title('Contour_Plot')
        plt.savefig('./model/{}/policy/contour_policy_{:0>4d}'.format(config['DIR'], epoch), dpi=300)
        plt.close("all")
    elif config['EXP_NAME'] == 'PendulumConstraint':
        num = 200
        xlist = np.linspace(-3 * np.pi, 3 * np.pi, num)
        ylist = np.linspace(-8., 8., num)
        X, Y = np.meshgrid(xlist, ylist)
        arr_1 = np.concatenate((np.reshape(np.cos(X), (num, num, 1)),
                              np.reshape(np.sin(X), (num, num, 1)),
                              np.reshape(Y, (num, num, 1)), np.ones((num, num, 1)) * 150.), axis=2)
        arr_2 = np.concatenate((np.reshape(np.cos(X), (num, num, 1)),
                              np.reshape(np.sin(X), (num, num, 1)),
                              np.reshape(Y, (num, num, 1)), np.ones((num, num, 1)) * 150.), axis=2)
        arr_1 = ((np.reshape(arr_1, (num * num, 4)) - train_state_policy.mean)
                 / np.sqrt(train_state_policy.variance + 1e-8))
        arr_2 = ((np.reshape(arr_2, (num * num, 4)) - train_state_policy.mean)
                 / np.sqrt(train_state_policy.variance + 1e-8))
        Z_1 = train_state_energy.apply_fn(train_state_energy.params, arr_1)
        Z_2 = train_state_h.apply_fn(train_state_h.params, arr_2)
        Z_1 = np.reshape(Z_1, (num, num))
        Z_2 = np.reshape(Z_2, (num, num))
        fig, ax = plt.subplots(1, 1)
        Z_1 = np.clip(Z_1, 0, np.inf)
        cp = ax.contourf(X, Y, Z_1, cmap=get_BuRd())
        fig.colorbar(cp)
        plt.xlim((-3 * np.pi, 3 * np.pi))
        ax.vlines([-2 * np.pi, 0, 2 * np.pi], -8, 8, colors='red')
        ax.set_title('Contour_Plot')
        plt.savefig('model/{}/value/contour_value_{:0>4d}'.format(config["DIR"], epoch), dpi=300)
        plt.close("all")
        fig, ax = plt.subplots(1, 1)
        cp = ax.contourf(X, Y, Z_2, norm=CenteredNorm(), cmap=get_BuRd())
        fig.colorbar(cp)
        plt.scatter(info['theta'][0], info['theta_dot'][0], s=10, c='black')
        plt.plot(info['theta'], info['theta_dot'], 'b-')
        plt.xlim((-3 * np.pi, 3 * np.pi))
        ax.vlines([-2 * np.pi, 0, 2 * np.pi], -8, 8, colors='red')
        ax.set_title('Contour_Plot Init {:.2f} Final {:.2f}'.format(info['init_energy'], info['final_energy']))
        plt.savefig('model/{}/reach/contour_reach_{:0>4d}'.format(config["DIR"], epoch), dpi=300)
        plt.close("all")
        fig, ax = plt.subplots(1, 1)
        cp = ax.contourf(X, Y, np.maximum(Z_1 - 140., Z_2), norm=CenteredNorm(), cmap=get_BuRd())
        fig.colorbar(cp)
        plt.scatter(info['theta'][0], info['theta_dot'][0], s=10, c='black')
        plt.plot(info['theta'], info['theta_dot'], 'b-')
        plt.xlim((-3 * np.pi, 3 * np.pi))
        ax.vlines([-2 * np.pi, 0, 2 * np.pi], -8, 8, colors='red')
        ax.set_title('Contour_Plot Init {:.2f} Final {:.2f}'.format(info['init_energy'], info['final_energy']))
        plt.savefig('model/{}/total/contour_total_{:0>4d}'.format(config["DIR"], epoch), dpi=300)
        plt.close("all")
    elif config['EXP_NAME'] == 'HopperReach':
        plt.figure(figsize=(12, 6))
        fig, ax = plt.subplots(1, 1)
        for i in range(0, 400, 16):
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
        draw_circle = plt.Circle((2.0, 1.4), 0.1, fill=False)
        ax.add_patch(draw_circle)
        ax.set_xlim((-0.5, 2.5))
        ax.set_ylim((0, 1.5))
        ax.set_aspect('equal')
        plt.savefig('model/{}/reach/trajectory_{:0>4d}'.format(config["DIR"], epoch), dpi=300)
        plt.close("all")
    elif config['EXP_NAME'] == 'HopperAvoid':
        plt.figure(figsize=(12, 6))
        fig, ax = plt.subplots(1, 1)
        for i in range(0, 400, 16):
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
        draw_circle = plt.Circle((2.0, 1.4), 0.1, fill=False)
        ax.add_patch(draw_circle)
        ax.set_xlim((-0.5, 2.5))
        ax.set_ylim((0, 1.5))
        ax.set_aspect('equal')
        plt.savefig('model/{}/reach/trajectory_{:0>4d}'.format(config["DIR"], epoch), dpi=300)
        plt.close("all")
    elif config['EXP_NAME'] == 'HopperAvoidCeiling' or config['EXP_NAME'] == 'HopperAvoidCeilingWall':
        plt.figure(figsize=(12, 6))
        fig, ax = plt.subplots(1, 1)
        reach_idx = info['reach_index']
        for i in range(0,reach_idx, 16):
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
        if config['EXP_NAME'] == 'HopperAvoidCeilingWall':
            draw_rectangle2 = plt.Rectangle((2.35, -0.1), 1.6, 0.2, facecolor="red", fill=True)
            ax.add_patch(draw_rectangle2)
        ax.add_patch(draw_circle)
        ax.add_patch(draw_rectangle)
        ax.set_xlim((-0.5, 2.5))
        ax.set_ylim((0, 1.5))
        ax.set_aspect('equal')
        ax.set_title("Trajectory Plot Init Energy {:.2f} Final Energy {:.2f}"
                     .format(info['init_energy'], info['final_energy']))
        plt.savefig('model/{}/reach/trajectory_{:0>4d}'.format(config["DIR"], epoch), dpi=300)
        plt.close("all")
    elif config['EXP_NAME'] == 'HopperReachReach':
        plt.figure(figsize=(12, 6))
        fig, ax = plt.subplots(1, 1)
        reach_idx_1 = info['reach_index_1']
        reach_idx_2 = info['reach_index_2']
        full_len = info['head_pos'].shape[0]
        draw_circle = plt.Circle((2.0, 1.4), 0.1, edgecolor="green", linewidth=2, fill=False)
        draw_circle2 = plt.Circle((-2.0, 1.4), 0.1, edgecolor="blue", linewidth=2, fill=False)
        ax.add_patch(draw_circle)
        ax.add_patch(draw_circle2)
        for i in range(0, full_len, 16):
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
        if reach_idx_1 > 0:
            i = reach_idx_1
            ax.plot(np.array([info['head_pos'][i, 0], info['jaw_pos'][i, 0]]),
                     np.array([info['head_pos'][i, 1], info['jaw_pos'][i, 1]]), c='g', linewidth=4)
            ax.plot(np.array([info['jaw_pos'][i, 0], info['thg_pos'][i, 0]]),
                     np.array([info['jaw_pos'][i, 1], info['thg_pos'][i, 1]]), c='g', linewidth=4)
            ax.plot(np.array([info['thg_pos'][i, 0], info['leg_pos'][i, 0]]),
                     np.array([info['thg_pos'][i, 1], info['leg_pos'][i, 1]]), c='g', linewidth=4)
            ax.plot(np.array([info['leg_pos'][i, 0], info['foot_front_pos'][i, 0]]),
                     np.array([info['leg_pos'][i, 1], info['foot_front_pos'][i, 1]]), c='g', linewidth=4)
            ax.plot(np.array([info['leg_pos'][i, 0], info['foot_back_pos'][i, 0]]),
                     np.array([info['leg_pos'][i, 1], info['foot_back_pos'][i, 1]]), c='g', linewidth=4)
        if reach_idx_2 > 0:
            i = reach_idx_2
            ax.plot(np.array([info['head_pos'][i, 0], info['jaw_pos'][i, 0]]),
                     np.array([info['head_pos'][i, 1], info['jaw_pos'][i, 1]]), c='b', linewidth=4)
            ax.plot(np.array([info['jaw_pos'][i, 0], info['thg_pos'][i, 0]]),
                     np.array([info['jaw_pos'][i, 1], info['thg_pos'][i, 1]]), c='b', linewidth=4)
            ax.plot(np.array([info['thg_pos'][i, 0], info['leg_pos'][i, 0]]),
                     np.array([info['thg_pos'][i, 1], info['leg_pos'][i, 1]]), c='b', linewidth=4)
            ax.plot(np.array([info['leg_pos'][i, 0], info['foot_front_pos'][i, 0]]),
                     np.array([info['leg_pos'][i, 1], info['foot_front_pos'][i, 1]]), c='b', linewidth=4)
            ax.plot(np.array([info['leg_pos'][i, 0], info['foot_back_pos'][i, 0]]),
                     np.array([info['leg_pos'][i, 1], info['foot_back_pos'][i, 1]]), c='b', linewidth=4)
        ax.set_xlim((-2.5, 2.5))
        ax.set_ylim((0, 1.6))
        ax.set_aspect('equal')
        ax.set_title("Reach Reach ?")
        plt.savefig('model/{}/reach/trajectory_{:0>4d}'.format(config["DIR"], epoch), dpi=300)
        # plt.close("all")
        return fig
    elif config['EXP_NAME'] == 'HalfCheetahAvoid':
        plt.figure(figsize=(12, 6))
        fig, ax = plt.subplots(1, 1)
        reach_idx = info['reach_index']
        for i in range(0,reach_idx, 16):
            ax.plot(np.array([info['head_pos'][i, 0], info['neck_pos'][i, 0]]),
                     np.array([info['head_pos'][i, 1], info['neck_pos'][i, 1]]), c='r')
            ax.plot(np.array([info['neck_pos'][i, 0], info['back_pos'][i, 0]]),
                     np.array([info['neck_pos'][i, 1], info['back_pos'][i, 1]]), c='g')
            ax.plot(np.array([info['neck_pos'][i, 0], info['front_thigh_pos'][i, 0]]),
                     np.array([info['neck_pos'][i, 1], info['front_thigh_pos'][i, 1]]), c='m')
            ax.plot(np.array([info['front_thigh_pos'][i, 0], info['front_shin_pos'][i, 0]]),
                     np.array([info['front_thigh_pos'][i, 1], info['front_shin_pos'][i, 1]]), c='g')
            ax.plot(np.array([info['front_foot_pos'][i, 0], info['front_shin_pos'][i, 0]]),
                     np.array([info['front_foot_pos'][i, 1], info['front_shin_pos'][i, 1]]), c='r')
            ax.plot(np.array([info['back_pos'][i, 0], info['back_thigh_pos'][i, 0]]),
                     np.array([info['back_pos'][i, 1], info['back_thigh_pos'][i, 1]]), c='m')
            ax.plot(np.array([info['back_thigh_pos'][i, 0], info['back_shin_pos'][i, 0]]),
                     np.array([info['back_thigh_pos'][i, 1], info['back_shin_pos'][i, 1]]), c='g')
            ax.plot(np.array([info['back_foot_pos'][i, 0], info['back_shin_pos'][i, 0]]),
                     np.array([info['back_foot_pos'][i, 1], info['back_shin_pos'][i, 1]]), c='r')
        ax.plot(np.array([info['head_pos'][reach_idx, 0], info['neck_pos'][reach_idx, 0]]),
                np.array([info['head_pos'][reach_idx, 1], info['neck_pos'][reach_idx, 1]]), c='r')
        ax.plot(np.array([info['neck_pos'][reach_idx, 0], info['back_pos'][reach_idx, 0]]),
                np.array([info['neck_pos'][reach_idx, 1], info['back_pos'][reach_idx, 1]]), c='g')
        ax.plot(np.array([info['neck_pos'][reach_idx, 0], info['front_thigh_pos'][reach_idx, 0]]),
                np.array([info['neck_pos'][reach_idx, 1], info['front_thigh_pos'][reach_idx, 1]]), c='m')
        ax.plot(np.array([info['front_thigh_pos'][reach_idx, 0], info['front_shin_pos'][reach_idx, 0]]),
                np.array([info['front_thigh_pos'][reach_idx, 1], info['front_shin_pos'][reach_idx, 1]]), c='g')
        ax.plot(np.array([info['front_foot_pos'][reach_idx, 0], info['front_shin_pos'][reach_idx, 0]]),
                np.array([info['front_foot_pos'][reach_idx, 1], info['front_shin_pos'][reach_idx, 1]]), c='r')
        ax.plot(np.array([info['back_pos'][reach_idx, 0], info['back_thigh_pos'][reach_idx, 0]]),
                np.array([info['back_pos'][reach_idx, 1], info['back_thigh_pos'][reach_idx, 1]]), c='m')
        ax.plot(np.array([info['back_thigh_pos'][reach_idx, 0], info['back_shin_pos'][reach_idx, 0]]),
                np.array([info['back_thigh_pos'][reach_idx, 1], info['back_shin_pos'][reach_idx, 1]]), c='g')
        ax.plot(np.array([info['back_foot_pos'][reach_idx, 0], info['back_shin_pos'][reach_idx, 0]]),
                np.array([info['back_foot_pos'][reach_idx, 1], info['back_shin_pos'][reach_idx, 1]]), c='r')
        draw_circle = plt.Circle((5.0, 0.), 0.2, fill=False)
        draw_rectangle = plt.Rectangle((2.45, -0.7), 0.1, 0.25, facecolor="red", fill=True)
        ax.add_patch(draw_circle)
        ax.add_patch(draw_rectangle)
        ax.set_xlim((-0.5, 5.5))
        ax.set_ylim((-0.7, 1.3))
        ax.set_aspect('equal')
        ax.set_title("Trajectory Plot Init Energy {:.2f} Final Energy {:.2f}"
                     .format(info['init_energy'], info['final_energy']))
        plt.savefig('model/{}/reach/trajectory_{:0>4d}'.format(config["DIR"], epoch), dpi=300)
        plt.close("all")
    elif config['EXP_NAME'] == 'WindField':
        x_index = config['SECTION'] % 2
        y_index = config['SECTION'] // 2
        obs = np.where(info['obs'] == 1, np.nan, 1)
        plt.figure(figsize=(6, 6), dpi=300)
        fig, ax = plt.subplots(1, 1)
        X, Y = np.meshgrid(np.linspace(-15, 15, 128), np.linspace(-15, 15, 128))
        ax.quiver(X, Y, obs[(0 + 128 * y_index): (128 + 128 * y_index),
                        (0 + 128 * x_index): (128 + 128 * x_index)] *
                  info['u_air'][(0 + 128 * y_index): (128 + 128 * y_index),
                        (0 + 128 * x_index): (128 + 128 * x_index)],
                  obs[(0 + 128 * y_index): (128 + 128 * y_index),
                        (0 + 128 * x_index): (128 + 128 * x_index)] *
                  info['v_air'][(0 + 128 * y_index): (128 + 128 * y_index),
                        (0 + 128 * x_index): (128 + 128 * x_index)], color='b')
        ax.contour(X, Y, info['obs'][(0 + 128 * y_index): (128 + 128 * y_index),
                        (0 + 128 * x_index): (128 + 128 * x_index)], colors='k', linewidths=0.5)
        ax.scatter(info['pos'][0, 0], info['pos'][0, 1], s=10, c='black')
        ax.plot(info['pos'][0:info['reach_index'], 0], info['pos'][0:info['reach_index'], 1])
        draw_rectangle = plt.Rectangle((-26. * x_index + 11., -26. * y_index + 11.), 4., 4., fill=True, color='y')
        ax.add_patch(draw_rectangle)
        ax.set_xlim((-20., 20.))
        ax.set_ylim((-20., 20.))
        ax.set_aspect('equal')
        ax.set_title("Trajectory Plot Init Energy {:.2f} Final Energy {:.2f}"
                     .format(info['init_energy'], info['final_energy']))
        plt.savefig('model/{}/reach/trajectory_{:0>4d}'.format(config["DIR"], epoch), dpi=300)
        plt.close("all")
        num = 128
        arr_1 = np.zeros((num, num, 14))
        arr_1[:, :, 0] = X / 15.
        arr_1[:, :, 1] = Y / 15.
        arr_1[:, :, -2] = 1.
        arr_1 = np.reshape(arr_1, (num * num, 14))
        Z_1 = train_state_h.apply_fn(train_state_h.params, arr_1)
        Z_1 = np.reshape(Z_1, (num, num))
        fig, ax = plt.subplots(1, 1)
        cp = ax.contourf(X, Y, Z_1, norm=CenteredNorm(), cmap=get_BuRd())
        ax.contour(X, Y, info['obs'][(0 + 128 * y_index): (128 + 128 * y_index),
                        (0 + 128 * x_index): (128 + 128 * x_index)], colors='k', linewidths=0.5)
        fig.colorbar(cp)
        draw_rectangle = plt.Rectangle((-26. * x_index + 11., -26. * y_index + 11.), 4., 4., fill=False, color='y')
        ax.add_patch(draw_rectangle)
        ax.set_xlim((-20., 20.))
        ax.set_ylim((-20., 20.))
        ax.set_title('Contour_Plot')
        plt.savefig('model/{}/total/contour_total_{:0>4d}_0'.format(config["DIR"], epoch), dpi=300)
        plt.close("all")
        num = 128
        arr_1 = np.zeros((num, num, 14))
        arr_1[:, :, 0] = X / 15.
        arr_1[:, :, 1] = Y / 15.
        arr_1[:, :, -2] = -1.
        arr_1 = np.reshape(arr_1, (num * num, 14))
        Z_1 = train_state_h.apply_fn(train_state_h.params, arr_1)
        Z_1 = np.reshape(Z_1, (num, num))
        fig, ax = plt.subplots(1, 1)
        cp = ax.contourf(X, Y, Z_1, cmap=get_BuRd())
        ax.contour(X, Y, info['obs'][(0 + 128 * y_index): (128 + 128 * y_index),
                        (0 + 128 * x_index): (128 + 128 * x_index)], colors='k', linewidths=0.5)
        fig.colorbar(cp)
        draw_rectangle = plt.Rectangle((-26. * x_index + 11., -26. * y_index + 11.), 4., 4., fill=False, color='y')
        ax.add_patch(draw_rectangle)
        ax.set_xlim((-20., 20.))
        ax.set_ylim((-20., 20.))
        ax.set_title('Contour_Plot')
        plt.savefig('model/{}/total/contour_total_{:0>4d}_1'.format(config["DIR"], epoch), dpi=300)
        plt.close("all")
    elif config['EXP_NAME'] == 'F16Avoid':
        reach_idx = info['reach_index']
        plt.figure(figsize=(24, 6))
        fig, ax = plt.subplots(1, 1)
        ax.scatter(info['pos_y'][0], info['pos_x'][0], s=10, c='black')
        ax.plot(info['pos_y'][0:reach_idx], info['pos_x'][0:reach_idx])
        draw_rectangle_1 = plt.Rectangle((-200., 475.), 200., 50., facecolor="red", fill=True)
        ax.add_patch(draw_rectangle_1)
        draw_rectangle_2 = plt.Rectangle((0., 975.), 200., 50., facecolor="red", fill=True)
        ax.add_patch(draw_rectangle_2)
        draw_rectangle_3 = plt.Rectangle((-200., 1475.), 200., 50., facecolor="red", fill=True)
        ax.add_patch(draw_rectangle_3)
        draw_rectangle_4 = plt.Rectangle((-200., 1975.), 400., 25., facecolor="yellow", fill=True)
        ax.add_patch(draw_rectangle_4)
        ax.set_xlim((200., -200.))
        ax.set_ylim((0., 2000.))
        ax.set_xlabel("Position East")
        ax.set_ylabel("Position North")
        ax.set_aspect('equal')
        ax.set_title("Trajectory Plot Init Energy {:.2f} Final Energy {:.2f}"
                     .format(info['init_energy'], info['final_energy']))
        plt.savefig('model/{}/reach/trajectory_{:0>4d}_0'.format(config["DIR"], epoch), dpi=300)
        plt.close("all")
        plt.figure(figsize=(12, 6))
        fig, ax = plt.subplots(1, 1)
        ax.scatter(info['pos_x'][0], info['height'][0], s=10, c='black')
        ax.plot(info['pos_x'][0:reach_idx], info['height'][0:reach_idx])
        ax.set_xlim((0., 2000.))
        ax.set_ylim((0., 1100.))
        ax.set_xlabel("Position North")
        ax.set_ylabel("Height")
        ax.set_aspect('equal')
        ax.set_title("Trajectory Plot Init Energy {:.2f} Final Energy {:.2f}"
                     .format(info['init_energy'], info['final_energy']))
        plt.savefig('model/{}/reach/trajectory_{:0>4d}_1'.format(config["DIR"], epoch), dpi=300)
        plt.close("all")
        plt.figure(figsize=(30, 10))
        fig, ax = plt.subplots(3, 2, layout='constrained')
        ax[0, 0].plot(info['state'][:, 0], label='velocity')
        ax[0, 0].legend()
        ax[0, 1].plot(info['state'][:, 1], label='alpha')
        ax[0, 1].plot(info['state'][:, 2], label='beta')
        ax[0, 1].legend()
        ax[1, 0].plot(info['state'][:, 3], label='phi')
        ax[1, 0].plot(info['state'][:, 4], label='theta')
        ax[1, 0].plot(info['state'][:, 5], label='psi')
        ax[1, 0].legend()
        ax[1, 1].plot(info['state'][:, 6], label='P')
        ax[1, 1].plot(info['state'][:, 7], label='Q')
        ax[1, 1].plot(info['state'][:, 8], label='R')
        ax[1, 1].legend()
        ax[2, 0].plot(info['state'][:, -3], label='NZ_INT')
        ax[2, 0].plot(info['state'][:, -2], label='PS_INT')
        ax[2, 0].plot(info['state'][:, -1], label='NYR_INT')
        ax[2, 0].legend()
        plt.savefig('model/{}/state_traj/trajectory_{:0>4d}'.format(config["DIR"], epoch), dpi=300)
        plt.close("all")
    elif config['EXP_NAME'] == 'PendulumConstraintBaseline':
        fig, ax = plt.subplots(1, 1)
        plt.scatter(info['theta'][0], info['theta_dot'][0], s=10, c='black')
        plt.plot(info['theta'], info['theta_dot'], 'b-')
        plt.xlim((-3 * np.pi, 3 * np.pi))
        ax.vlines([-2 * np.pi, 0, 2 * np.pi], -8, 8, colors='red')
        ax.set_title('Trajectory_Plot')
        plt.savefig('model/{}/reach/traj_reach_{:0>4d}'.format(config["DIR"], epoch), dpi=300)
        plt.close("all")
    elif config['EXP_NAME'] == 'HopperAvoidCeilingBaseline':
        plt.figure(figsize=(12, 6))
        fig, ax = plt.subplots(1, 1)
        reach_idx = info['reach_index']
        for i in range(0,reach_idx, 16):
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
        ax.set_title("Trajectory Plot")
        plt.savefig('model/{}/reach/trajectory_{:0>4d}'.format(config["DIR"], epoch), dpi=300)
        plt.close("all")


def plot_contour_RRAA(multi_info, epoch, config):

    if config['EXP_NAME'] == 'HopperReachReach':

        info, info_1, info_2 = multi_info
        plt.figure(figsize=(12, 18))
        fig, axes = plt.subplots(3, 1)

        def draw_hopper_rr(info, title, ax, target_type="both"):
            reach_idx_1 = info['reach_index_1']
            reach_idx_2 = info['reach_index_2']
            full_len = info['head_pos'].shape[0]
            draw_circle = plt.Circle((2.0, 1.4), 0.1, edgecolor="green", linewidth=2, fill=False)
            draw_circle2 = plt.Circle((-2.0, 1.4), 0.1, edgecolor="blue", linewidth=2, fill=False)
            
            if target_type == "both":
                ax.add_patch(draw_circle)
                ax.add_patch(draw_circle2)
            elif target_type == "1":
                ax.add_patch(draw_circle)
            elif target_type == "2":
                ax.add_patch(draw_circle2)
            
            for i in range(0, full_len, 16):
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
            if reach_idx_1 > 0 and (target_type == "both" or target_type == "1"):
                i = reach_idx_1
                ax.plot(np.array([info['head_pos'][i, 0], info['jaw_pos'][i, 0]]),
                        np.array([info['head_pos'][i, 1], info['jaw_pos'][i, 1]]), c='g', linewidth=4)
                ax.plot(np.array([info['jaw_pos'][i, 0], info['thg_pos'][i, 0]]),
                        np.array([info['jaw_pos'][i, 1], info['thg_pos'][i, 1]]), c='g', linewidth=4)
                ax.plot(np.array([info['thg_pos'][i, 0], info['leg_pos'][i, 0]]),
                        np.array([info['thg_pos'][i, 1], info['leg_pos'][i, 1]]), c='g', linewidth=4)
                ax.plot(np.array([info['leg_pos'][i, 0], info['foot_front_pos'][i, 0]]),
                        np.array([info['leg_pos'][i, 1], info['foot_front_pos'][i, 1]]), c='g', linewidth=4)
                ax.plot(np.array([info['leg_pos'][i, 0], info['foot_back_pos'][i, 0]]),
                        np.array([info['leg_pos'][i, 1], info['foot_back_pos'][i, 1]]), c='g', linewidth=4)
            if reach_idx_2 > 0 and (target_type == "both" or target_type == "2"):
                i = reach_idx_2
                ax.plot(np.array([info['head_pos'][i, 0], info['jaw_pos'][i, 0]]),
                        np.array([info['head_pos'][i, 1], info['jaw_pos'][i, 1]]), c='b', linewidth=4)
                ax.plot(np.array([info['jaw_pos'][i, 0], info['thg_pos'][i, 0]]),
                        np.array([info['jaw_pos'][i, 1], info['thg_pos'][i, 1]]), c='b', linewidth=4)
                ax.plot(np.array([info['thg_pos'][i, 0], info['leg_pos'][i, 0]]),
                        np.array([info['thg_pos'][i, 1], info['leg_pos'][i, 1]]), c='b', linewidth=4)
                ax.plot(np.array([info['leg_pos'][i, 0], info['foot_front_pos'][i, 0]]),
                        np.array([info['leg_pos'][i, 1], info['foot_front_pos'][i, 1]]), c='b', linewidth=4)
                ax.plot(np.array([info['leg_pos'][i, 0], info['foot_back_pos'][i, 0]]),
                        np.array([info['leg_pos'][i, 1], info['foot_back_pos'][i, 1]]), c='b', linewidth=4)
            ax.set_xlim((-2.5, 2.5))
            ax.set_ylim((0, 1.6))
            ax.set_aspect('equal')
            
            ax.set_title(title)
        
        draw_hopper_rr(info, "Reach Reach", axes[0], target_type="both")
        draw_hopper_rr(info_1, "Reach 1", axes[1], target_type="1")
        draw_hopper_rr(info_2, "Reach 2", axes[2], target_type="2")

        plt.savefig('model/{}/reach/trajectory_{:0>4d}'.format(config["DIR"], epoch), dpi=300)
        return fig
        # plt.close("all")