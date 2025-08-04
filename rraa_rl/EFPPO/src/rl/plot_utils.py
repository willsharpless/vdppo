import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import CenteredNorm
from mpl_toolkits.mplot3d import Axes3D

from rraa_rl.EFPPO.src.rl.utils import get_BuRd
from PIL import Image
import imageio
import wandb 
from time import time 

from rraa_rl.EFPPO.src.env.reach_avoid.safety_gym_RR import PointReachReach, PointReach1, PointReach2
# from rraa_rl.EFPPO.src.env.reach_avoid.safety_gym_RAA import PointReachAvoid, PointAvoidOnly
from rraa_rl.EFPPO.src.env.reach_avoid.half_cheetah_RAA import HalfCheetahReachAvoid, HalfCheetahAvoidOnly
from rraa_rl.EFPPO.src.env.reach_avoid.half_cheetah_RR import HalfCheetahReachReach, HalfCheetahReach1, HalfCheetahReach2
from rraa_rl.EFPPO.src.env.reach_avoid.humanoid_RR import HumanoidReachReach, HUMANOID_TARGET_RIGHT, HUMANOID_TARGET_LEFT, HUMANOID_TARGET_RADIUS
from rraa_rl.EFPPO.src.env.reach_avoid.humanoid_RAA import HumanoidReachAvoid, HUMANOID_RAA_TARGET, HUMANOID_RAA_TARGET_RADIUS, HUMANOID_RAA_BOX_RADIUS, HUMANOID_RAA_FLOOR_HEIGHT
from jax import jit

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

def calculate_reach_avoid_stats(traj_batch):
    reach_idx = (traj_batch.reach < 0).argmax(axis=0)
    cnt_never_reached = 0
    cnt_crash = 0
    cnt_crash_after_reach = 0
    for i in range(reach_idx.shape[0]):
        if reach_idx[i] == 0 and traj_batch.reach[0, i] >= 0:
            cnt_never_reached += 1
        else:
            if np.any(traj_batch.avoid[reach_idx[i]+1:, i] > 0):
                cnt_crash_after_reach += 1 
    for i in range(traj_batch.avoid.shape[1]):
        if np.any(traj_batch.avoid[:, i] > 0):
            cnt_crash += 1
    share_crash_after_reach = cnt_crash_after_reach / (traj_batch.avoid.shape[1] - cnt_never_reached) if (traj_batch.avoid.shape[1] - cnt_never_reached) > 0 else 0
    return cnt_never_reached, cnt_crash, share_crash_after_reach

def calculate_reachavoid(traj_batch, th=0, to_first_done=False):

    # compute 
    reach_idx = (traj_batch.reach < (0 + th)).argmax(axis=0)
    crash_idx = (traj_batch.avoid > 0).argmax(axis=0)
    reach_idx = np.where(np.any((traj_batch.reach < (0 + th)) == 1, axis=0), reach_idx, np.inf)
    crash_idx = np.where(np.any((traj_batch.avoid > 0) == 1, axis=0), crash_idx, np.inf)

    if to_first_done:
        first_done = np.where(np.any(traj_batch.done, axis=0), traj_batch.done.argmax(axis=0), traj_batch.done.shape[0])
        reach_idx = np.where(first_done >= reach_idx, reach_idx, np.inf)
        crash_idx = np.where(first_done >= crash_idx, crash_idx, np.inf)

    # Find indices where reach < inf and avoid = inf
    reach_and_avoid_idx = np.where(crash_idx == np.inf, reach_idx, np.inf)

    reach_perc = ((reach_idx < np.inf).sum() / reach_idx.__len__()).item()
    crash_perc = ((crash_idx < np.inf).sum() / crash_idx.__len__()).item()
    reach_avoid_perc = ((reach_and_avoid_idx < np.inf).sum() / reach_and_avoid_idx.__len__()).item()
    return (reach_perc, crash_perc, reach_avoid_perc)

def calculate_reachreach(traj_batch, reach_type="both", th=0, to_first_done=False):

    # Compute first reaching idx
    reach_idx_1 = (traj_batch.reach1 < (0 + th)).argmax(axis=0) if reach_type in ["both", "1"] else None
    reach_idx_2 = (traj_batch.reach2 < (0 + th)).argmax(axis=0) if reach_type in ["both", "2"] else None
    reach_idx_1 = np.where(np.any((traj_batch.reach1 < (0 + th)) == 1, axis=0), reach_idx_1, np.inf) if reach_type in ["both", "1"] else None
    reach_idx_2 = np.where(np.any((traj_batch.reach2 < (0 + th)) == 1, axis=0), reach_idx_2, np.inf) if reach_type in ["both", "2"] else None
    
    if to_first_done:
        first_done = np.where(np.any(traj_batch.done, axis=0), traj_batch.done.argmax(axis=0), traj_batch.done.shape[0])
        reach_idx_1 = np.where(first_done >= reach_idx_1, reach_idx_1, np.inf) if reach_type in ["both", "1"] else None
        reach_idx_2 = np.where(first_done >= reach_idx_2, reach_idx_2, np.inf) if reach_type in ["both", "2"] else None

    reach_idx = np.maximum(reach_idx_1, reach_idx_2) if reach_type in ["both"] else None

    # Compute Percentage
    reach_1_perc = ((reach_idx_1 < np.inf).sum() / reach_idx_1.__len__()).item() if reach_type in ["both", "1"] else None
    reach_2_perc = ((reach_idx_2 < np.inf).sum() / reach_idx_2.__len__()).item() if reach_type in ["both", "2"] else None
    reach_perc = ((reach_idx < np.inf).sum() / reach_idx.__len__()).item() if reach_type in ["both"] else None

    reach_percs = (reach_1_perc, reach_2_perc, reach_perc)
    reach_idxs = (reach_idx_1, reach_idx_2, reach_idx)
    return reach_percs, reach_idxs


def calculate_reach(traj_batch):
    reach_idx = (traj_batch.reach < 0).argmax(axis=0)
    reach_idx = np.where(np.any((traj_batch.reach < 0) == 1, axis=0), reach_idx, np.inf)
    reach_perc = ((reach_idx < np.inf).sum() / reach_idx.__len__()).item()
    return reach_perc, reach_idx


# def calculate_reachalwaysavoid_old(traj_batch, idx, type="both"): 
#     assert(type in ["both", "avoid" ])

#     # First Avoid violation index
#     all_avoid_idx = (traj_batch.avoid < 0).argmax(axis=0) # FIXME: Nikhil wrote this when sleep deprived
#     avoid_idx = all_avoid_idx[idx]
#     reach_idx = None 
    
#     if type == "both": 
#         # First Reach Index
#         all_reach_idx = (traj_batch.reach < 0).argmax(axis=0)
#         reach_idx = all_reach_idx[idx]
#     return reach_idx, avoid_idx 

def calculate_reachalwaysavoid(traj_batch, idx, type="both"): 
    assert(type in ["both", "avoid" ])

    # First Avoid violation index
    crash_idx = (traj_batch.avoid > 0).argmax(axis=0)
    crash_idx = np.where(np.any((traj_batch.avoid > 0) == 1, axis=0), crash_idx, -1)
    avoid_idx = crash_idx[idx]
    reach_idx = None 
    
    if type == "both": 
        # First Reach Index
        reach_idx = (traj_batch.reach < 0).argmax(axis=0)
        reach_idx = np.where(np.any((traj_batch.reach < 0) == 1, axis=0), reach_idx, -1)
        reach_idx = reach_idx[idx]
    return reach_idx, avoid_idx 

def calculate_minimal_reach(reach):
    reach_idx = (reach < 0).argmax()
    if reach_idx == 0 and reach[0] >= 0:
        reach_idx = reach.shape[0]
    return reach_idx

def plot_policy_decision(policy_decision_sample, epoch, config):
    fig, ax = plt.subplots(1, 1)
    ax.plot(policy_decision_sample, label='policy #')
    plt.title("Policy Decision over Trajectory")
    plt.xlabel("Trajectory Step")
    plt.ylim((-0.1, 2.1))
    plt.legend()
    plt.savefig('model/{}/policy/policy_decision_{:0>4d}'.format(config['DIR'], epoch), dpi=300)
    return fig

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
    elif config['EXP_NAME'] == 'HopperAvoidCeiling' or config['EXP_NAME'] == 'HopperAvoidCeilingWall' or config['EXP_NAME'] == 'HopperAvoidCeilingWallEnergy':
        plt.figure(figsize=(12, 6))
        fig, ax = plt.subplots(1, 1)
        reach_idx = info['reach_index']
        indices = np.linspace(0, info['head_pos'].shape[0] - 1, 11, dtype=int)
        for step_n, i in enumerate(indices):
            alpha = (step_n + 1) / 11
            ax.plot(np.array([info['head_pos'][i, 0], info['jaw_pos'][i, 0]]),
                     np.array([info['head_pos'][i, 1], info['jaw_pos'][i, 1]]), c='r', alpha=alpha)
            ax.plot(np.array([info['jaw_pos'][i, 0], info['thg_pos'][i, 0]]),
                     np.array([info['jaw_pos'][i, 1], info['thg_pos'][i, 1]]), c='g', alpha=alpha)
            ax.plot(np.array([info['thg_pos'][i, 0], info['leg_pos'][i, 0]]),
                     np.array([info['thg_pos'][i, 1], info['leg_pos'][i, 1]]), c='b', alpha=alpha)
            ax.plot(np.array([info['leg_pos'][i, 0], info['foot_front_pos'][i, 0]]),
                     np.array([info['leg_pos'][i, 1], info['foot_front_pos'][i, 1]]), c='b', alpha=alpha)
            ax.plot(np.array([info['leg_pos'][i, 0], info['foot_back_pos'][i, 0]]),
                     np.array([info['leg_pos'][i, 1], info['foot_back_pos'][i, 1]]), c='m', alpha=alpha)
        draw_circle = plt.Circle((2.0, 1.4), 0.1, fill=False)
        draw_rectangle = plt.Rectangle((0.95, 1.3), 0.1, 0.2, facecolor="red", fill=True)
        if 'HopperAvoidCeilingWall' in config['EXP_NAME']:
            draw_rectangle2 = plt.Rectangle((2.1, -0.1), 0.2, 1.6, facecolor="red", fill=True)
            ax.add_patch(draw_rectangle2)
        ax.add_patch(draw_circle)
        ax.add_patch(draw_rectangle)
        ax.set_xlim((-0.5, 2.5))
        ax.set_ylim((0, 1.5))
        ax.set_aspect('equal')
        ax.set_title("Trajectory Plot Init Energy {:.2f} Final Energy {:.2f}"
                     .format(info['init_energy'], info['final_energy']))
        plt.savefig('model/{}/reach/trajectory_{:0>4d}'.format(config["DIR"], epoch), dpi=300)
        # plt.close("all")
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
    
    return fig


def plot_contour_RRAA(multi_info, epoch, config, policy_decision_sample=None):

    if 'Hopper' in config['EXP_NAME'] and 'ReachReach' in config["EXP_NAME"]: 

        info, info_1, info_2 = multi_info

        if policy_decision_sample is None:
            plt.figure(figsize=(7, 5), constrained_layout=True)
            fig, axes = plt.subplots(3, 1)
        else:
            plt.figure(figsize=(7, 7), constrained_layout=True)
            fig, axes = plt.subplots(4, 1)

        def draw_hopper_rr(info, title, ax, target_type="both", plot_until="success", plot_freq=16):
            reach_idx_1 = info['reach_index_1']
            reach_idx_2 = info['reach_index_2']
            full_len = info['head_pos'].shape[0]
            draw_circle = plt.Circle((2.0, 1.4), 0.1, edgecolor="green", linewidth=2, fill=False)
            # draw_circle2 = plt.Circle((-2.0, 1.4), 0.1, edgecolor="blue", linewidth=2, fill=False)
            draw_circle2 = plt.Circle((0., 1.4), 0.1, edgecolor="blue", linewidth=2, fill=False)
            
            if plot_until == "success":
                full_len = np.maximum(reach_idx_1, reach_idx_2)
                full_len = info['head_pos'].shape[0] if full_len.item() == np.inf else int(full_len.item())
            else:
                full_len = info['head_pos'].shape[0]
            reach_idx_1 = int(reach_idx_1.item()) if reach_idx_1.item() != np.inf else -1
            reach_idx_2 = int(reach_idx_2.item()) if reach_idx_2.item() != np.inf else -1

            if target_type == "both":
                ax.add_patch(draw_circle)
                ax.add_patch(draw_circle2)
            elif target_type == "R1":
                ax.add_patch(draw_circle)
            elif target_type == "R2":
                ax.add_patch(draw_circle2)
            
            def draw_body(ax, info, i, alpha, color_mode="normal"):
                if color_mode == "R1":
                    c1, c2, c3, c4, c5 = 'g', 'g', 'g', 'g', 'g'
                    linewidth=3
                elif color_mode == "R2":
                    c1, c2, c3, c4, c5 = 'b', 'b', 'b', 'b', 'b'
                    linewidth=3
                else:
                    c1, c2, c3, c4, c5 = 'r', 'g', 'b', 'b', 'm'
                    linewidth=1
                ax.plot(np.array([info['head_pos'][i, 0], info['jaw_pos'][i, 0]]),
                        np.array([info['head_pos'][i, 1], info['jaw_pos'][i, 1]]), c=c1, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['jaw_pos'][i, 0], info['thg_pos'][i, 0]]),
                        np.array([info['jaw_pos'][i, 1], info['thg_pos'][i, 1]]), c=c2, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['thg_pos'][i, 0], info['leg_pos'][i, 0]]),
                        np.array([info['thg_pos'][i, 1], info['leg_pos'][i, 1]]), c=c3, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['leg_pos'][i, 0], info['foot_front_pos'][i, 0]]),
                        np.array([info['leg_pos'][i, 1], info['foot_front_pos'][i, 1]]), c=c4, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['leg_pos'][i, 0], info['foot_back_pos'][i, 0]]),
                        np.array([info['leg_pos'][i, 1], info['foot_back_pos'][i, 1]]), c=c5, alpha=alpha, linewidth=linewidth)
            
            for i in range(0, full_len, plot_freq):
                alpha = 0.3 + 0.3 * (i/full_len)
                draw_body(ax, info, i, alpha)

            if reach_idx_1 > -1 and (target_type == "both" or target_type == "R1"):
                draw_body(ax, info, reach_idx_1, 0.9, color_mode = "R1")

            if reach_idx_2 > -1 and (target_type == "both" or target_type == "R2"):
                draw_body(ax, info, reach_idx_2, 0.9, color_mode = "R2")

            # ax.set_xlim((-2.5, 2.5))
            ax.set_xlim((-0.5, 2.5))
            ax.set_ylim((-0.1, 1.6))
            # ax.set_aspect('equal')
            
            ax.set_title(title)
        
        draw_hopper_rr(info, "Reach Reach", axes[0], target_type="both")
        if config['EXP_NAME'] == 'HopperReachReach'  or config["EXP_NAME"] == 'HopperReachReachDecomposed':
            draw_hopper_rr(info_1, "Reach 1", axes[1], target_type="R1")
            draw_hopper_rr(info_2, "Reach 2", axes[2], target_type="R2")

        if policy_decision_sample is not None:
            axes[3].plot(policy_decision_sample, label='policy #')
            axes[3].set_title("Policy Decision over Trajectory")
            axes[3].set_xlabel("Trajectory Step")
            axes[3].set_ylim((-0.1, 2.1))
            # axes[3].set_box_aspect(1.6 / 3.0)
            axes[3].legend()

        plt.savefig('model/{}/reach/trajectory_{:0>4d}'.format(config["DIR"], epoch), dpi=300)
        return fig
    
        
    elif 'Hopper' in config['EXP_NAME'] and 'Avoid' in config["EXP_NAME"]: 
        
        info, info_avoid = multi_info 
        plt.figure(figsize=(12, 6*2))
        fig, axes = plt.subplots(2, 1)

        def draw_hopper_raa(info, title, ax):
            reach_idx = info.get('reach_index')
            avoid_idx = info.get('avoid_index')
            full_len = info['head_pos'].shape[0]

            # Plot Reach  
            draw_circle = plt.Circle((2.0, 1.4), 0.1, fill=False)

            # Plot Avoid
            draw_rectangle = plt.Rectangle((0.95, 1.3), 0.1, 0.3, facecolor="red", fill=True)
            draw_rectangle2 = plt.Rectangle((2.1, -0.1), 0.4, 1.7, facecolor="red", fill=True)
            draw_rectangle3 = plt.Rectangle((-2., 0.), 4.5, 0.5, facecolor="red", fill=True)
            draw_rectangle4 = plt.Rectangle((-0.5, -0.1), 0.5, 1.7, facecolor="red", fill=True)

            ax.add_patch(draw_circle)
            ax.add_patch(draw_rectangle)
            ax.add_patch(draw_rectangle2)
            ax.add_patch(draw_rectangle3)
            ax.add_patch(draw_rectangle4)

            indices = np.linspace(0, full_len, 11, dtype=int)
            for step_n, i in enumerate(indices):
                alpha = (step_n + 1) / 11 
                # Plot Hopper Body 
                ax.plot(np.array([info['head_pos'][i, 0], info['jaw_pos'][i, 0]]),
                        np.array([info['head_pos'][i, 1], info['jaw_pos'][i, 1]]), c='r', alpha=alpha)
                ax.plot(np.array([info['jaw_pos'][i, 0], info['thg_pos'][i, 0]]),
                        np.array([info['jaw_pos'][i, 1], info['thg_pos'][i, 1]]), c='g', alpha=alpha)
                ax.plot(np.array([info['thg_pos'][i, 0], info['leg_pos'][i, 0]]),
                        np.array([info['thg_pos'][i, 1], info['leg_pos'][i, 1]]), c='b', alpha=alpha)
                ax.plot(np.array([info['leg_pos'][i, 0], info['foot_front_pos'][i, 0]]),
                        np.array([info['leg_pos'][i, 1], info['foot_front_pos'][i, 1]]), c='b', alpha=alpha)
                ax.plot(np.array([info['leg_pos'][i, 0], info['foot_back_pos'][i, 0]]),
                        np.array([info['leg_pos'][i, 1], info['foot_back_pos'][i, 1]]), c='m', alpha=alpha)
                
            # Plot First Reach in Green 
            if reach_idx is not None and reach_idx > -1:
                ax.plot(np.array([info['head_pos'][reach_idx, 0], info['jaw_pos'][reach_idx, 0]]),
                        np.array([info['head_pos'][reach_idx, 1], info['jaw_pos'][reach_idx, 1]]), c='g', linewidth=4)
                ax.plot(np.array([info['jaw_pos'][reach_idx, 0], info['thg_pos'][reach_idx, 0]]),
                        np.array([info['jaw_pos'][reach_idx, 1], info['thg_pos'][reach_idx, 1]]), c='g', linewidth=4)
                ax.plot(np.array([info['thg_pos'][reach_idx, 0], info['leg_pos'][reach_idx, 0]]),
                        np.array([info['thg_pos'][reach_idx, 1], info['leg_pos'][reach_idx, 1]]), c='g', linewidth=4)
                ax.plot(np.array([info['leg_pos'][reach_idx, 0], info['foot_front_pos'][reach_idx, 0]]),
                        np.array([info['leg_pos'][reach_idx, 1], info['foot_front_pos'][reach_idx, 1]]), c='g', linewidth=4)
                ax.plot(np.array([info['leg_pos'][reach_idx, 0], info['foot_back_pos'][reach_idx, 0]]),
                        np.array([info['leg_pos'][reach_idx, 1], info['foot_back_pos'][reach_idx, 1]]), c='g', linewidth=4)
                
            # Plot Avoid Violation in Red
            if avoid_idx is not None and avoid_idx > -1: 
                ax.plot(np.array([info['head_pos'][avoid_idx, 0], info['jaw_pos'][avoid_idx, 0]]),
                        np.array([info['head_pos'][avoid_idx, 1], info['jaw_pos'][avoid_idx, 1]]), c='r', linewidth=4)
                ax.plot(np.array([info['jaw_pos'][avoid_idx, 0], info['thg_pos'][avoid_idx, 0]]),
                        np.array([info['jaw_pos'][avoid_idx, 1], info['thg_pos'][avoid_idx, 1]]), c='r', linewidth=4)
                ax.plot(np.array([info['thg_pos'][avoid_idx, 0], info['leg_pos'][avoid_idx, 0]]),
                        np.array([info['thg_pos'][avoid_idx, 1], info['leg_pos'][avoid_idx, 1]]), c='r', linewidth=4)
                ax.plot(np.array([info['leg_pos'][avoid_idx, 0], info['foot_front_pos'][avoid_idx, 0]]),
                        np.array([info['leg_pos'][avoid_idx, 1], info['foot_front_pos'][avoid_idx, 1]]), c='r', linewidth=4)
                ax.plot(np.array([info['leg_pos'][avoid_idx, 0], info['foot_back_pos'][avoid_idx, 0]]),
                        np.array([info['leg_pos'][avoid_idx, 1], info['foot_back_pos'][avoid_idx, 1]]), c='r', linewidth=4) 
                
            ax.set_xlim((-0.5, 2.5))
            ax.set_ylim((0, 1.5))
            ax.set_aspect('equal')

            ax.set_title(title)

        # Draw Reach Avoid and Avoid Only 
        draw_hopper_raa(info, "Reach Avoid", axes[0])
        if config['EXP_NAME'] == 'HopperReachAlwaysAvoid':
            draw_hopper_raa(info_avoid, "Avoid Only", axes[1])

        plt.savefig('model/{}/reach/trajectory_{:0>4d}'.format(config["DIR"], epoch), dpi=300)
        return fig
    
    elif 'HalfCheetah' in config['EXP_NAME'] and 'Avoid' in config['EXP_NAME']: 
        
        info, info_avoid = multi_info 
        plt.figure(figsize=(12, 6*2))
        fig, axes = plt.subplots(2, 1)
        axes_upperx = 8.5
        axes_lowerx = -0.5
        axes_uppery = 1.3
        axes_lowery = -0.7

        def draw_cheetah_raa(info, title, ax):
            reach_idx = info.get('reach_index')
            avoid_idx = info.get('avoid_index')
            full_len = info['head_pos'].shape[0]

            # Plot Targets and Obstacles
            x = np.linspace(axes_lowerx, axes_upperx, 400)
            y = np.linspace(axes_lowery, axes_uppery, 400)
            X, Y = np.meshgrid(x, y)
            positions = np.stack([X, Y], axis=-1)  # shape (400, 400, 2)
            model = HalfCheetahReachAvoid()
            is_reach_np = jit(model.is_reach)
            is_avoid_np = jit(model.is_avoid)
            reach_values = np.array(is_reach_np(positions))
            avoid_values = np.array(is_avoid_np((positions, positions, positions, positions, positions, positions, positions, positions, positions)))
            if reach_idx is not None:
                ax.contourf(X, Y, np.maximum(reach_values, avoid_values), alpha=0.3, levels=20)
            else:
                ax.contourf(X, Y, avoid_values, alpha=0.3, levels=20)
            if reach_idx is not None:
                ax.contourf(X, Y, reach_values, levels=[reach_values.min(), 0], colors=['green'], alpha=0.4)
            ax.contourf(X, Y, avoid_values, levels=[0, avoid_values.max()], colors=['red'], alpha=0.4)

            def draw_body(ax, info, i, alpha, color_mode="normal"):
                
                if color_mode == "R":
                    c1, c2, c3, c4, c5, c6, c7, c8 = 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g'
                    linewidth=3
                elif color_mode == "A":
                    c1, c2, c3, c4, c5, c6, c7, c8 = 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r'
                    linewidth=3
                else:
                    c1, c2, c3, c4, c5, c6, c7, c8 = 'r', 'g', 'm', 'g', 'r', 'm', 'g', 'r'
                    linewidth=1

                # Plot Cheetah Body 
                ax.plot(np.array([info['head_pos'][i, 0], info['neck_pos'][i, 0]]),
                        np.array([info['head_pos'][i, 1], info['neck_pos'][i, 1]]), c=c1, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['neck_pos'][i, 0], info['back_pos'][i, 0]]),
                        np.array([info['neck_pos'][i, 1], info['back_pos'][i, 1]]), c=c2, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['neck_pos'][i, 0], info['front_thigh_pos'][i, 0]]),
                        np.array([info['neck_pos'][i, 1], info['front_thigh_pos'][i, 1]]), c=c3, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['front_thigh_pos'][i, 0], info['front_shin_pos'][i, 0]]),
                        np.array([info['front_thigh_pos'][i, 1], info['front_shin_pos'][i, 1]]), c=c4, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['front_foot_pos'][i, 0], info['front_shin_pos'][i, 0]]),
                        np.array([info['front_foot_pos'][i, 1], info['front_shin_pos'][i, 1]]), c=c5, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['back_pos'][i, 0], info['back_thigh_pos'][i, 0]]),
                        np.array([info['back_pos'][i, 1], info['back_thigh_pos'][i, 1]]), c=c6, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['back_thigh_pos'][i, 0], info['back_shin_pos'][i, 0]]),
                        np.array([info['back_thigh_pos'][i, 1], info['back_shin_pos'][i, 1]]), c=c7, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['back_foot_pos'][i, 0], info['back_shin_pos'][i, 0]]),
                        np.array([info['back_foot_pos'][i, 1], info['back_shin_pos'][i, 1]]), c=c8, alpha=alpha, linewidth=linewidth)
                
            indices = np.linspace(0, full_len, 11, dtype=int)
            for step_n, i in enumerate(indices):
                alpha = (step_n + 1) / 11

                reach_val = is_reach_np(info['head_pos'][i])
                avoid_val = is_avoid_np((info['head_pos'][i], info['neck_pos'][i], info['back_pos'][i],
                    info['front_thigh_pos'][i], info['front_shin_pos'][i], info['front_foot_pos'][i], 
                    info['back_thigh_pos'][i], info['back_shin_pos'][i], info['back_foot_pos'][i]))

                if avoid_val > 0.:
                    color_mode = "A"
                elif reach_idx is not None and reach_val < 0.:
                    color_mode = "R"
                else:
                    color_mode = "normal"
                draw_body(ax, info, i, alpha, color_mode=color_mode)

            if reach_idx is not None and reach_idx > -1:
                draw_body(ax, info, reach_idx, alpha, color_mode="R")
            if avoid_idx is not None and avoid_idx > -1:
                draw_body(ax, info, avoid_idx, alpha, color_mode="A")

            ax.set_xlim((axes_lowerx, axes_upperx))
            ax.set_ylim((axes_lowery, axes_uppery))
            ax.set_aspect('equal')

            ax.set_title(title)

        # Draw Reach Avoid and Avoid Only 
        draw_cheetah_raa(info, "Reach Avoid", axes[0])
        if config['EXP_NAME'] == 'HalfCheetahReachAlwaysAvoid':
            draw_cheetah_raa(info_avoid, "Avoid Only", axes[1])

        plt.savefig('model/{}/reach/trajectory_{:0>4d}'.format(config["DIR"], epoch), dpi=300)
        return fig

    elif 'HalfCheetah' in config['EXP_NAME'] and 'ReachReach' in config['EXP_NAME']: 
        
        info, info_1, info_2 = multi_info
        plt.figure(figsize=(12, 6*2))
        fig, axes = plt.subplots(3, 1)
        axes_upperx = 5.5
        axes_lowerx = -5.5
        axes_uppery = 1.3
        axes_lowery = -0.7

        def draw_cheetah_rr(info, title, ax, mode="both"):
            reach1_idx = info.get('reach1_index')
            reach2_idx = info.get('reach2_index')
            full_len = info['head_pos'].shape[0]

            # Plot Targets and Obstacles
            x = np.linspace(axes_lowerx, axes_upperx, 400)
            y = np.linspace(axes_lowery, axes_uppery, 400)
            X, Y = np.meshgrid(x, y)
            positions = np.stack([X, Y], axis=-1)  # shape (400, 400, 2)
            model = HalfCheetahReachReach()
            is_reach1_np = jit(model.is_reach1)
            is_reach2_np = jit(model.is_reach2)
            reach1_values = np.array(is_reach1_np((positions, positions, positions, positions, positions, positions, positions, positions, positions), (0., 0., 0.)))
            reach2_values = np.array(is_reach2_np((positions, positions, positions, positions, positions, positions, positions, positions, positions), (0., 0., 0.)))
            if mode=="both":
                ax.contourf(X, Y, np.maximum(reach1_values, reach2_values), alpha=0.3, levels=20)
                ax.contourf(X, Y, reach1_values, levels=[reach1_values.min(), 0], colors=['green'], alpha=0.4)
                ax.contourf(X, Y, reach2_values, levels=[reach2_values.min(), 0], colors=['blue'], alpha=0.4)
            elif mode=="reach1":
                ax.contourf(X, Y, reach1_values, alpha=0.3, levels=20)
                ax.contourf(X, Y, reach1_values, levels=[reach1_values.min(), 0], colors=['green'], alpha=0.4)
            else:
                ax.contourf(X, Y, reach2_values, alpha=0.3, levels=20)
                ax.contourf(X, Y, reach2_values, levels=[reach2_values.min(), 0], colors=['blue'], alpha=0.4)

            def draw_body(ax, info, i, alpha, color_mode="normal"):
                
                if color_mode == "reach1":
                    c1, c2, c3, c4, c5, c6, c7, c8 = 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g'
                    linewidth=3
                elif color_mode == "reach2":
                    c1, c2, c3, c4, c5, c6, c7, c8 = 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b'
                    linewidth=3
                else:
                    c1, c2, c3, c4, c5, c6, c7, c8 = 'r', 'g', 'm', 'g', 'r', 'm', 'g', 'r'
                    linewidth=1

                # Plot Cheetah Body 
                ax.plot(np.array([info['head_pos'][i, 0], info['neck_pos'][i, 0]]),
                        np.array([info['head_pos'][i, 1], info['neck_pos'][i, 1]]), c=c1, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['neck_pos'][i, 0], info['back_pos'][i, 0]]),
                        np.array([info['neck_pos'][i, 1], info['back_pos'][i, 1]]), c=c2, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['neck_pos'][i, 0], info['front_thigh_pos'][i, 0]]),
                        np.array([info['neck_pos'][i, 1], info['front_thigh_pos'][i, 1]]), c=c3, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['front_thigh_pos'][i, 0], info['front_shin_pos'][i, 0]]),
                        np.array([info['front_thigh_pos'][i, 1], info['front_shin_pos'][i, 1]]), c=c4, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['front_foot_pos'][i, 0], info['front_shin_pos'][i, 0]]),
                        np.array([info['front_foot_pos'][i, 1], info['front_shin_pos'][i, 1]]), c=c5, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['back_pos'][i, 0], info['back_thigh_pos'][i, 0]]),
                        np.array([info['back_pos'][i, 1], info['back_thigh_pos'][i, 1]]), c=c6, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['back_thigh_pos'][i, 0], info['back_shin_pos'][i, 0]]),
                        np.array([info['back_thigh_pos'][i, 1], info['back_shin_pos'][i, 1]]), c=c7, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['back_foot_pos'][i, 0], info['back_shin_pos'][i, 0]]),
                        np.array([info['back_foot_pos'][i, 1], info['back_shin_pos'][i, 1]]), c=c8, alpha=alpha, linewidth=linewidth)
                
            indices = np.linspace(0, full_len, 11, dtype=int)
            for step_n, i in enumerate(indices):
                alpha = (step_n + 1) / 11

                reach1_val = is_reach1_np((info['head_pos'][i], info['neck_pos'][i], info['back_pos'][i],
                    info['front_thigh_pos'][i], info['front_shin_pos'][i], info['front_foot_pos'][i], 
                    info['back_thigh_pos'][i], info['back_shin_pos'][i], info['back_foot_pos'][i]), (0., 0., 0.))
                reach2_val = is_reach2_np((info['head_pos'][i], info['neck_pos'][i], info['back_pos'][i],
                    info['front_thigh_pos'][i], info['front_shin_pos'][i], info['front_foot_pos'][i], 
                    info['back_thigh_pos'][i], info['back_shin_pos'][i], info['back_foot_pos'][i]), (0., 0., 0.))

                if reach1_val < 0.:
                    color_mode = "reach1"
                elif reach2_val < 0.:
                    color_mode = "reach2"
                else:
                    color_mode = "normal"
                draw_body(ax, info, i, alpha, color_mode=color_mode)

            if reach1_idx is not None and reach1_idx > -1:
                draw_body(ax, info, reach1_idx, alpha, color_mode="reach1")
            if reach2_idx is not None and reach2_idx > -1:
                draw_body(ax, info, reach2_idx, alpha, color_mode="reach2")

            ax.set_xlim((axes_lowerx, axes_upperx))
            ax.set_ylim((axes_lowery, axes_uppery))
            ax.set_aspect('equal')

            ax.set_title(title)

        # Draw Reach Avoid and Avoid Only 
        draw_cheetah_rr(info, "Reach Reach", axes[0])
        if config['EXP_NAME'] == 'HalfCheetahReachReach':
            draw_cheetah_rr(info_1, "Reach 1", axes[1], mode="reach1")
            draw_cheetah_rr(info_2, "Reach 2", axes[2], mode="reach2")

        plt.savefig('model/{}/reach/trajectory_{:0>4d}'.format(config["DIR"], epoch), dpi=300)
        return fig

    elif 'Humanoid' in config['EXP_NAME'] and 'ReachReach' in config['EXP_NAME']:

        info, info_1, info_2 = multi_info
        fig = plt.figure(figsize=(20, 6))
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')
        axes = [ax1, ax2, ax3]
        axes_upperx = 3.5
        axes_lowerx = -0.5
        axes_uppery = 0.5
        axes_lowery = -3.5
        axes_upperz = 1.5
        axes_lowerz = -0.1

        def draw_humanoid_rr(info, title, ax, mode="both"):
            reach1_idx = info.get('reach1_index')
            reach2_idx = info.get('reach2_index')
            full_len = info['head_pos'].shape[0]

            # Plot Targets and Obstacles
            if mode == "both" or mode == "reach1":
                add_sphere(ax, HUMANOID_TARGET_RIGHT, radius=HUMANOID_TARGET_RADIUS, resolution=30, alpha=0.4, color='green')
            if mode == "both" or mode == "reach2":
                add_sphere(ax, HUMANOID_TARGET_LEFT, radius=HUMANOID_TARGET_RADIUS, resolution=30, alpha=0.4, color='blue')

            model = HumanoidReachReach()
            is_reach1_np = jit(model.is_reach1)
            is_reach2_np = jit(model.is_reach2)

            def draw_body_3d(ax, info, i, alpha, color_mode="normal"):
                if color_mode == "reach1":
                    c1, c2, c3, c4, c5, c6 = ['g'] * 6
                    linewidth = 3
                elif color_mode == "reach2":
                    c1, c2, c3, c4, c5, c6 = ['b'] * 6
                    linewidth = 3
                else:
                    # head to shoulders, arms, hips to knees/feet
                    c1, c2, c3, c4, c5, c6 = 'k', 'r', 'm', 'g', 'c', 'b'
                    linewidth = 1

                def line(p1, p2, color):
                    ax.plot(
                        [p1[0], p2[0]],
                        [p1[1], p2[1]],
                        [p1[2], p2[2]],
                        c=color, alpha=alpha, linewidth=linewidth
                    )

                # Joint locations
                head = info["head_pos"][i]
                torso = info["torso"][i]
                lwaist = info["lwaist"][i]
                pelvis = info["pelvis"][i]
                
                left_upper_arm = info["left_upper_arm"][i]
                right_upper_arm = info["right_upper_arm"][i]
                left_lower_arm = info["left_lower_arm"][i]
                right_lower_arm = info["right_lower_arm"][i]
                left_hand = info["left_hand"][i]
                right_hand = info["right_hand"][i]

                left_thigh = info["left_thigh"][i]
                right_thigh = info["right_thigh"][i]
                left_shin = info["left_shin"][i]
                right_shin = info["right_shin"][i]
                left_foot = info["left_foot"][i]
                right_foot = info["right_foot"][i]

                # Draw abdomen
                line(head, torso, c6)
                line(torso, lwaist, c2)
                line(lwaist, pelvis, c3)

                # Draw arms
                line(torso, left_upper_arm, c4)
                line(torso, right_upper_arm, c4)
                line(left_upper_arm, left_lower_arm, c5)
                line(left_lower_arm, left_hand, c1)
                line(right_upper_arm, right_lower_arm, c5)
                line(right_lower_arm, right_hand, c1)

                # Draw legs
                line(pelvis, left_thigh, c4)
                line(pelvis, right_thigh, c4)
                line(left_thigh, left_shin, c5)
                line(left_shin, left_foot, c1)
                line(right_thigh, right_shin, c5)
                line(right_shin, right_foot, c1)    
            
            indices = np.linspace(0, full_len, 11, dtype=int)
            for step_n, i in enumerate(indices):
                alpha = (step_n + 1) / 11

                step_poses = {k: info[k][i] for k in info.keys() if not k in ['reach_index_1', 'reach_index_2']}

                reach1_val = is_reach1_np(step_poses)
                reach2_val = is_reach2_np(step_poses)

                if reach1_val < 0.:
                    color_mode = "reach1"
                elif reach2_val < 0.:
                    color_mode = "reach2"
                else:
                    color_mode = "normal"
                draw_body_3d(ax, info, i, alpha, color_mode=color_mode)

            if reach1_idx is not None and reach1_idx > -1:
                draw_body_3d(ax, info, reach1_idx, alpha, color_mode="reach1")
            if reach2_idx is not None and reach2_idx > -1:
                draw_body_3d(ax, info, reach2_idx, alpha, color_mode="reach2")

            ax.set_xlim((axes_lowerx, axes_upperx))
            ax.set_ylim((axes_lowery, axes_uppery))
            ax.set_zlim((axes_lowerz, axes_upperz))
            ax.set_aspect('equal')
            ax.set_title(title)
            ax.view_init(elev=10, azim=-45)

        # Draw Reach Avoid and Avoid Only 
        draw_humanoid_rr(info, "Reach Reach", axes[0])
        if config['EXP_NAME'] == 'HumanoidReachReach':
            draw_humanoid_rr(info_1, "Reach 1", axes[1], mode="reach1")
            draw_humanoid_rr(info_2, "Reach 2", axes[2], mode="reach2")

        plt.savefig('model/{}/reach/trajectory_{:0>4d}'.format(config["DIR"], epoch), dpi=300)
        return fig
    
    elif 'Humanoid' in config['EXP_NAME'] and 'Avoid' in config['EXP_NAME']:

        info, info_avoid = multi_info
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        axes = [ax1, ax2]
        axes_upperx = HUMANOID_RAA_BOX_RADIUS
        axes_lowerx = -(HUMANOID_RAA_BOX_RADIUS)
        axes_uppery = HUMANOID_RAA_BOX_RADIUS
        axes_lowery = -(HUMANOID_RAA_BOX_RADIUS)
        axes_upperz = 1.5
        axes_lowerz = -0.1

        def draw_humanoid_raa(info, title, ax, mode="both"):
            reach_idx = info.get('reach_index')
            avoid_idx = info.get('avoid_index')
            full_len = info['head_pos'].shape[0]

            # Plot Targets and Obstacles
            add_box_3d(ax, center=np.array([0., 0., HUMANOID_RAA_FLOOR_HEIGHT/2.]), size=2*np.array([HUMANOID_RAA_BOX_RADIUS, HUMANOID_RAA_BOX_RADIUS, HUMANOID_RAA_FLOOR_HEIGHT]), alpha=0.05) # floor
            add_box_3d(ax, center=np.array([HUMANOID_RAA_BOX_RADIUS + 0.05, 0., 0.5]), size=2*np.array([0.1, HUMANOID_RAA_BOX_RADIUS-0.1, 0.5])) # wall
            add_box_3d(ax, center=np.array([-(HUMANOID_RAA_BOX_RADIUS + 0.05), 0., 0.5]), size=2*np.array([0.1, HUMANOID_RAA_BOX_RADIUS-0.1, 0.5])) # wall
            add_box_3d(ax, center=np.array([0., HUMANOID_RAA_BOX_RADIUS + 0.05, 0.5]), size=2*np.array([HUMANOID_RAA_BOX_RADIUS-0.1, 0.1, 0.5])) # wall
            add_box_3d(ax, center=np.array([0., -(HUMANOID_RAA_BOX_RADIUS + 0.05), 0.5]), size=2*np.array([HUMANOID_RAA_BOX_RADIUS-0.1, 0.1, 0.5])) # wall
            if mode == "both":
                add_cylinder(ax, HUMANOID_RAA_TARGET, radius=HUMANOID_RAA_TARGET_RADIUS, height=2., resolution=10, alpha=0.4, color='green') # target

            model = HumanoidReachAvoid()
            is_reach_np = jit(model.is_reach)
            is_avoid_np = jit(model.is_avoid)

            def draw_body_3d(ax, info, i, alpha, color_mode="normal"):
                if color_mode == "R":
                    c1, c2, c3, c4, c5, c6 = ['g'] * 6
                    linewidth = 3
                elif color_mode == "A":
                    c1, c2, c3, c4, c5, c6 = ['r'] * 6
                    linewidth = 3
                else:
                    # head to shoulders, arms, hips to knees/feet
                    c1, c2, c3, c4, c5, c6 = 'k', 'r', 'm', 'g', 'c', 'b'
                    linewidth = 1

                def line(p1, p2, color):
                    ax.plot(
                        [p1[0], p2[0]],
                        [p1[1], p2[1]],
                        [p1[2], p2[2]],
                        c=color, alpha=alpha, linewidth=linewidth
                    )

                # Joint locations
                head = info["head_pos"][i]
                torso = info["torso"][i]
                lwaist = info["lwaist"][i]
                pelvis = info["pelvis"][i]
                
                left_upper_arm = info["left_upper_arm"][i]
                right_upper_arm = info["right_upper_arm"][i]
                left_lower_arm = info["left_lower_arm"][i]
                right_lower_arm = info["right_lower_arm"][i]
                left_hand = info["left_hand"][i]
                right_hand = info["right_hand"][i]

                left_thigh = info["left_thigh"][i]
                right_thigh = info["right_thigh"][i]
                left_shin = info["left_shin"][i]
                right_shin = info["right_shin"][i]
                left_foot = info["left_foot"][i]
                right_foot = info["right_foot"][i]

                # Draw abdomen
                line(head, torso, c6)
                line(torso, lwaist, c2)
                line(lwaist, pelvis, c3)

                # Draw arms
                line(torso, left_upper_arm, c4)
                line(torso, right_upper_arm, c4)
                line(left_upper_arm, left_lower_arm, c5)
                line(left_lower_arm, left_hand, c1)
                line(right_upper_arm, right_lower_arm, c5)
                line(right_lower_arm, right_hand, c1)

                # Draw legs
                line(pelvis, left_thigh, c4)
                line(pelvis, right_thigh, c4)
                line(left_thigh, left_shin, c5)
                line(left_shin, left_foot, c1)
                line(right_thigh, right_shin, c5)
                line(right_shin, right_foot, c1)    
            
            indices = np.linspace(0, full_len, 11, dtype=int)
            for step_n, i in enumerate(indices):
                alpha = (step_n + 1) / 11

                step_poses = {k: info[k][i] for k in info.keys() if not k in ['reach_index', 'avoid_index']}

                reach_val = is_reach_np(step_poses)
                avoid_val = is_avoid_np(step_poses)

                if avoid_val > 0.:
                    color_mode = "A"
                elif reach_idx is not None and reach_val < 0.:
                    color_mode = "R"
                else:
                    color_mode = "normal"
                draw_body_3d(ax, info, i, alpha, color_mode=color_mode)

            if reach_idx is not None and reach_idx > -1:
                draw_body_3d(ax, info, reach_idx, alpha, color_mode="R")
            if avoid_idx is not None and avoid_idx > -1:
                draw_body_3d(ax, info, avoid_idx, alpha, color_mode="A")

            ax.set_xlim((axes_lowerx, axes_upperx))
            ax.set_ylim((axes_lowery, axes_uppery))
            ax.set_zlim((axes_lowerz, axes_upperz))
            ax.set_aspect('equal')
            ax.set_title(title)
            ax.view_init(elev=45, azim=-45)

        # Draw Reach Avoid and Avoid Only 
        draw_humanoid_raa(info, "Reach Avoid", axes[0])
        if config['EXP_NAME'] == 'HumanoidReachAlwaysAvoid':
            draw_humanoid_raa(info_avoid, "Avoid Only", axes[1], mode="avoid")

        plt.savefig('model/{}/reach/trajectory_{:0>4d}'.format(config["DIR"], epoch), dpi=300)
        return fig
    
    elif 'F16' in config['EXP_NAME']:
        
        if 'ReachReach' in config['EXP_NAME']:
            info, info_1, info_2 = multi_info
        else:
            info, info_avoid = multi_info

        # reach_idx = info['reach_index'] #FIXME just plotting entire trajectory
        plt.figure(figsize=(24, 6))
        fig, ax = plt.subplots(1, 1)
        ax.scatter(info['pos_y'][0], info['pos_x'][0], s=10, c='black')
        ax.plot(info['pos_y'], info['pos_x'])

        #FIXME not drawing any obstacles/targets yet
        # draw_rectangle_1 = plt.Rectangle((-200., 475.), 200., 50., facecolor="red", fill=True)
        # ax.add_patch(draw_rectangle_1)
        # draw_rectangle_2 = plt.Rectangle((0., 975.), 200., 50., facecolor="red", fill=True)
        # ax.add_patch(draw_rectangle_2)
        # draw_rectangle_3 = plt.Rectangle((-200., 1475.), 200., 50., facecolor="red", fill=True)
        # ax.add_patch(draw_rectangle_3)
        # draw_rectangle_4 = plt.Rectangle((-200., 1975.), 400., 25., facecolor="yellow", fill=True)
        # ax.add_patch(draw_rectangle_4)

        ax.set_xlim((200., -200.))
        ax.set_ylim((0., 2000.))
        ax.set_xlabel("Position East")
        ax.set_ylabel("Position North")
        ax.set_aspect('equal')
        # ax.set_title("Trajectory Plot Init Energy {:.2f} Final Energy {:.2f}"
        #              .format(info['init_energy'], info['final_energy']))
        plt.savefig('model/{}/reach/trajectory_{:0>4d}_0'.format(config["DIR"], epoch), dpi=300)
        if config["USE_WANDB"]: 
            wandb.log({"trajectory_view1": wandb.Image(fig)}, step=epoch)
        plt.close("all")

        plt.figure(figsize=(12, 6))
        fig, ax = plt.subplots(1, 1)
        if 'ReachReach' in config['EXP_NAME']:
            radius = 150
            draw_circle = plt.Circle((1200., 850), radius, facecolor="green", fill=True, alpha = 0.4)
            ax.add_patch(draw_circle)
            draw_circle = plt.Circle((1200., 350), radius, facecolor="blue", fill=True, alpha = 0.4)
            ax.add_patch(draw_circle)
        else:
            draw_rectangle_1 = plt.Rectangle((1250., -0.1), 500., 1200., facecolor="green", fill=True, alpha = 0.4)
            ax.add_patch(draw_rectangle_1)
            draw_rectangle_2 = plt.Rectangle((2000 - 25, -0.1), 100., 1200., facecolor="red", fill=True, alpha = 0.4)
            ax.add_patch(draw_rectangle_2)
            draw_rectangle_2 = plt.Rectangle((-0.1, -0.1), 2000.1, 2., facecolor="red", fill=True, alpha = 0.4)
            ax.add_patch(draw_rectangle_2)
        ax.scatter(info['pos_x'][0], info['height'][0], s=10, c='black')
        ax.plot(info['pos_x'], info['height'])        
        ax.set_xlim((0., 2000.))
        ax.set_ylim((0., 1100.))
        ax.set_xlabel("Position North")
        ax.set_ylabel("Height")
        ax.set_aspect('equal')
        # ax.set_title("Trajectory Plot Init Energy {:.2f} Final Energy {:.2f}"
        #              .format(info['init_energy'], info['final_energy']))
        plt.savefig('model/{}/reach/trajectory_{:0>4d}_1'.format(config["DIR"], epoch), dpi=300)
        if config["USE_WANDB"]: 
            wandb.log({"trajectory_view2": wandb.Image(fig)}, step=epoch)
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
        plt.savefig('model/{}/reach/trajectory_{:0>4d}_2'.format(config["DIR"], epoch), dpi=300)
        if config["USE_WANDB"]: 
            wandb.log({"trajectory_view3": wandb.Image(fig)}, step=epoch)
        plt.close("all")
        return None
    
    elif 'Point' in config['EXP_NAME'] and 'ReachReach' in config['EXP_NAME']: 
        
        info, info_1, info_2 = multi_info
        plt.figure(figsize=(12, 6*2))
        fig, axes = plt.subplots(3, 1)
        axes_upperx = 3.
        axes_lowerx = -3.
        axes_uppery = 3.
        axes_lowery = -3.

        def draw_point_rr(info, title, ax, mode="both"):
            reach1_idx = info.get('reach1_index')
            reach2_idx = info.get('reach2_index')
            full_len = info['x'].shape[0]

            # Plot Targets and Obstacles
            x = np.linspace(axes_lowerx, axes_upperx, 400)
            y = np.linspace(axes_lowery, axes_uppery, 400)
            X, Y = np.meshgrid(x, y)
            positions = np.stack([X, Y], axis=-1)  # shape (400, 400, 2)
            model = PointReachReach()
            is_reach1_np = jit(model.is_reach1)
            is_reach2_np = jit(model.is_reach2)
            reach1_values = np.array(is_reach1_np(positions))
            reach2_values = np.array(is_reach2_np(positions))
            if mode=="both":
                ax.contourf(X, Y, np.maximum(reach1_values, reach2_values), alpha=0.3, levels=20)
                ax.contourf(X, Y, reach1_values, levels=[reach1_values.min(), 0], colors=['green'], alpha=0.4)
                ax.contourf(X, Y, reach2_values, levels=[reach2_values.min(), 0], colors=['blue'], alpha=0.4)
            elif mode=="reach1":
                ax.contourf(X, Y, reach1_values, alpha=0.3, levels=20)
                ax.contourf(X, Y, reach1_values, levels=[reach1_values.min(), 0], colors=['green'], alpha=0.4)
            else:
                ax.contourf(X, Y, reach2_values, alpha=0.3, levels=20)
                ax.contourf(X, Y, reach2_values, levels=[reach2_values.min(), 0], colors=['blue'], alpha=0.4)

            def draw_body(ax, info, i, alpha, color_mode="normal"):
                
                if color_mode == "reach1":
                    c = 'g'
                    linewidth=3
                elif color_mode == "reach2":
                    c = 'b'
                    linewidth=3
                else:
                    c = 'k'
                    linewidth=1

                # ax.scatter(np.array([info['x'][i], info['y'][i]]), c=c, alpha=alpha, linewidth=linewidth)
                ax.scatter(info['x'][i], info['y'][i], color=c, alpha=alpha)

            indices = np.linspace(0, full_len, 11, dtype=int)
            for step_n, i in enumerate(indices):
                alpha = (step_n + 1) / 11

                reach1_val = is_reach1_np(np.array([info['x'][i].item(), info['y'][i].item()]))
                reach2_val = is_reach2_np(np.array([info['x'][i].item(), info['y'][i].item()]))

                if reach1_val < 0.:
                    color_mode = "reach1"
                elif reach2_val < 0.:
                    color_mode = "reach2"
                else:
                    color_mode = "normal"
                draw_body(ax, info, i, alpha, color_mode=color_mode)

            if reach1_idx is not None and reach1_idx > -1:
                draw_body(ax, info, reach1_idx, alpha, color_mode="reach1")
            if reach2_idx is not None and reach2_idx > -1:
                draw_body(ax, info, reach2_idx, alpha, color_mode="reach2")

            ax.set_xlim((axes_lowerx, axes_upperx))
            ax.set_ylim((axes_lowery, axes_uppery))
            ax.set_aspect('equal')

            ax.set_title(title)

        # Draw Reach Avoid and Avoid Only 
        draw_point_rr(info, "Reach Reach", axes[0])
        if config['EXP_NAME'] == 'PointReachReach':
            draw_point_rr(info_1, "Reach 1", axes[1], mode="reach1")
            draw_point_rr(info_2, "Reach 2", axes[2], mode="reach2")

        plt.savefig('model/{}/reach/trajectory_{:0>4d}'.format(config["DIR"], epoch), dpi=300)
        return fig
    
def plot_video_contour_RRAA(multi_info, epoch, config, save_video=False, prefix="", log_wandb=True):
    start_time = time()
    frames = None
    if 'Hopper' in config['EXP_NAME'] and 'Avoid' in config['EXP_NAME']: 
        
        info, info_avoid = multi_info 

        def draw_hopper_raa(step, info, title, ax):

            reach_idx = info.get('reach_index')
            avoid_idx = info.get('avoid_index')
            full_len = info['head_pos'].shape[0]

            # Plot Reach  
            draw_circle = plt.Circle((2.0, 1.4), 0.1, fill=False)

            # Plot Avoid
            draw_rectangle = plt.Rectangle((0.95, 1.3), 0.1, 0.3, facecolor="red", fill=True)
            draw_rectangle2 = plt.Rectangle((2.1, -0.1), 0.4, 1.7, facecolor="red", fill=True)
            draw_rectangle3 = plt.Rectangle((-2., 0.), 4.5, 0.5, facecolor="red", fill=True)
            draw_rectangle4 = plt.Rectangle((-0.5, -0.1), 0.5, 1.7, facecolor="red", fill=True)

            ax.add_patch(draw_circle)
            ax.add_patch(draw_rectangle)
            ax.add_patch(draw_rectangle2)
            ax.add_patch(draw_rectangle3)
            ax.add_patch(draw_rectangle4)

            # Plot Hopper Body 
            ax.plot(np.array([info['head_pos'][step, 0], info['jaw_pos'][step, 0]]),
                    np.array([info['head_pos'][step, 1], info['jaw_pos'][step, 1]]), c='r')
            ax.plot(np.array([info['jaw_pos'][step, 0], info['thg_pos'][step, 0]]),
                    np.array([info['jaw_pos'][step, 1], info['thg_pos'][step, 1]]), c='g')
            ax.plot(np.array([info['thg_pos'][step, 0], info['leg_pos'][step, 0]]),
                    np.array([info['thg_pos'][step, 1], info['leg_pos'][step, 1]]), c='b')
            ax.plot(np.array([info['leg_pos'][step, 0], info['foot_front_pos'][step, 0]]),
                    np.array([info['leg_pos'][step, 1], info['foot_front_pos'][step, 1]]), c='b')
            ax.plot(np.array([info['leg_pos'][step, 0], info['foot_back_pos'][step, 0]]),
                    np.array([info['leg_pos'][step, 1], info['foot_back_pos'][step, 1]]), c='m')
                
            # Plot First Reach in Green 
            if reach_idx is not None and reach_idx > -1:
                ax.plot(np.array([info['head_pos'][reach_idx, 0], info['jaw_pos'][reach_idx, 0]]),
                        np.array([info['head_pos'][reach_idx, 1], info['jaw_pos'][reach_idx, 1]]), c='g', linewidth=4)
                ax.plot(np.array([info['jaw_pos'][reach_idx, 0], info['thg_pos'][reach_idx, 0]]),
                        np.array([info['jaw_pos'][reach_idx, 1], info['thg_pos'][reach_idx, 1]]), c='g', linewidth=4)
                ax.plot(np.array([info['thg_pos'][reach_idx, 0], info['leg_pos'][reach_idx, 0]]),
                        np.array([info['thg_pos'][reach_idx, 1], info['leg_pos'][reach_idx, 1]]), c='g', linewidth=4)
                ax.plot(np.array([info['leg_pos'][reach_idx, 0], info['foot_front_pos'][reach_idx, 0]]),
                        np.array([info['leg_pos'][reach_idx, 1], info['foot_front_pos'][reach_idx, 1]]), c='g', linewidth=4)
                ax.plot(np.array([info['leg_pos'][reach_idx, 0], info['foot_back_pos'][reach_idx, 0]]),
                        np.array([info['leg_pos'][reach_idx, 1], info['foot_back_pos'][reach_idx, 1]]), c='g', linewidth=4)
                
            # Plot Avoid Violation in Red
            if avoid_idx is not None and avoid_idx > -1: 
                ax.plot(np.array([info['head_pos'][avoid_idx, 0], info['jaw_pos'][avoid_idx, 0]]),
                        np.array([info['head_pos'][avoid_idx, 1], info['jaw_pos'][avoid_idx, 1]]), c='r', linewidth=4)
                ax.plot(np.array([info['jaw_pos'][avoid_idx, 0], info['thg_pos'][avoid_idx, 0]]),
                        np.array([info['jaw_pos'][avoid_idx, 1], info['thg_pos'][avoid_idx, 1]]), c='r', linewidth=4)
                ax.plot(np.array([info['thg_pos'][avoid_idx, 0], info['leg_pos'][avoid_idx, 0]]),
                        np.array([info['thg_pos'][avoid_idx, 1], info['leg_pos'][avoid_idx, 1]]), c='r', linewidth=4)
                ax.plot(np.array([info['leg_pos'][avoid_idx, 0], info['foot_front_pos'][avoid_idx, 0]]),
                        np.array([info['leg_pos'][avoid_idx, 1], info['foot_front_pos'][avoid_idx, 1]]), c='r', linewidth=4)
                ax.plot(np.array([info['leg_pos'][avoid_idx, 0], info['foot_back_pos'][avoid_idx, 0]]),
                        np.array([info['leg_pos'][avoid_idx, 1], info['foot_back_pos'][avoid_idx, 1]]), c='r', linewidth=4) 
                
            ax.set_xlim((-0.5, 2.5))
            ax.set_ylim((0, 1.6))
            ax.set_aspect('equal')

            ax.set_title(title)
        

        frames = []
        full_len = info['head_pos'].shape[0]
        num_frames = full_len//2
        indices = np.linspace(0, full_len, num_frames, dtype=int)
        for step_n in indices: 
            # plt.figure(figsize=(12, 6*2))
            fig, axes = plt.subplots(2, 1, figsize=(4, 4), dpi=100)  # Smaller and square figure
            draw_hopper_raa(step_n, info, "Reach Avoid", axes[0])
            if config['EXP_NAME'] == 'HopperReachAlwaysAvoid':
                draw_hopper_raa(step_n, info_avoid, "Avoid Only", axes[1])
            
            # Render the figure to an image (smaller size)
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # RGBA (4 channels)
            frames.append(frame)
            
            plt.close(fig)
            plt.close("all")
    
    elif 'Hopper' in config['EXP_NAME'] and 'ReachReach' in config['EXP_NAME']: 
        info, info_1, info_2 = multi_info

        def draw_hopper_rr(step, info, reach_idx_1, reach_idx_2, title, ax, target_type="both"):

            # reach_idx_1 = info['reach_index_1']
            # reach_idx_2 = info['reach_index_2']
            draw_circle = plt.Circle((2.0, 1.4), 0.1, edgecolor="green", linewidth=2, fill=False)
            # draw_circle2 = plt.Circle((-2.0, 1.4), 0.1, edgecolor="blue", linewidth=2, fill=False)
            draw_circle2 = plt.Circle((0., 1.4), 0.1, edgecolor="blue", linewidth=2, fill=False)

            if target_type == "both":
                ax.add_patch(draw_circle)
                ax.add_patch(draw_circle2)
            elif target_type == "R1":
                ax.add_patch(draw_circle)
            elif target_type == "R2":
                ax.add_patch(draw_circle2)
            
            def draw_body(ax, info, i, alpha, color_mode="normal"):
                if color_mode == "R1":
                    c1, c2, c3, c4, c5 = 'g', 'g', 'g', 'g', 'g'
                    linewidth=3
                elif color_mode == "R2":
                    c1, c2, c3, c4, c5 = 'b', 'b', 'b', 'b', 'b'
                    linewidth=3
                else:
                    c1, c2, c3, c4, c5 = 'r', 'g', 'b', 'b', 'm'
                    linewidth=1
                ax.plot(np.array([info['head_pos'][i, 0], info['jaw_pos'][i, 0]]),
                        np.array([info['head_pos'][i, 1], info['jaw_pos'][i, 1]]), c=c1, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['jaw_pos'][i, 0], info['thg_pos'][i, 0]]),
                        np.array([info['jaw_pos'][i, 1], info['thg_pos'][i, 1]]), c=c2, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['thg_pos'][i, 0], info['leg_pos'][i, 0]]),
                        np.array([info['thg_pos'][i, 1], info['leg_pos'][i, 1]]), c=c3, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['leg_pos'][i, 0], info['foot_front_pos'][i, 0]]),
                        np.array([info['leg_pos'][i, 1], info['foot_front_pos'][i, 1]]), c=c4, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['leg_pos'][i, 0], info['foot_back_pos'][i, 0]]),
                        np.array([info['leg_pos'][i, 1], info['foot_back_pos'][i, 1]]), c=c5, alpha=alpha, linewidth=linewidth)
            
            draw_body(ax, info, step, 0.9)

            if step >= reach_idx_1 and reach_idx_1 > -1 and (target_type == "both" or target_type == "R1"):
                draw_body(ax, info, reach_idx_1, 0.9, color_mode = "R1")

            if step >= reach_idx_2 and reach_idx_2 > -1 and (target_type == "both" or target_type == "R2"):
                draw_body(ax, info, reach_idx_2, 0.9, color_mode = "R2")

            # ax.set_xlim((-2.5, 2.5))
            ax.set_xlim((-0.5, 2.5))
            ax.set_ylim((0., 1.6))
            ax.set_aspect('equal')
            
            ax.set_title(title)

        # DEFINE VIDEO LENGTH TO DUAL-REACHING OR FULL TRAJ
        reach_idx_1 = info['reach_index_1']
        reach_idx_2 = info['reach_index_2']

        full_len = np.maximum(reach_idx_1, reach_idx_2)
        full_len = info['head_pos'].shape[0] if full_len.item() == np.inf else int(full_len.item())
        # full_len = info['head_pos'].shape[0]
        reach_idx_1 = int(reach_idx_1.item()) if reach_idx_1.item() != np.inf else -1
        reach_idx_2 = int(reach_idx_2.item()) if reach_idx_2.item() != np.inf else -1
        
        frames = []
        num_frames = full_len//2
        indices = np.linspace(0, full_len, num_frames, dtype=int)
        if config['EXP_NAME'] == 'HopperReachReach':
            reach_idx_1_reach1 = info_1['reach_index_1'].item()
            reach_idx_2_reach2 = info_2['reach_index_2'].item()
            
        for step_n in indices: 

            fig, axes = plt.subplots(3, 1, figsize=(6, 4), dpi=100)

            draw_hopper_rr(step_n, info, reach_idx_1, reach_idx_2, "Reach Reach", axes[0], target_type="both")

            if config['EXP_NAME'] == 'HopperReachReach':
                # AFTER DECOMPOSED REACH, DRAW LAST POINT
                if step_n >= reach_idx_1_reach1:
                    reach_idx_1_reach1 = step_n
                if step_n >= reach_idx_2_reach2:
                    reach_idx_2_reach2 = step_n

                draw_hopper_rr(step_n, info_1, reach_idx_1_reach1, -1, "Reach 1", axes[1], target_type="R1")
                draw_hopper_rr(step_n, info_2, -1, reach_idx_2_reach2, "Reach 2", axes[2], target_type="R2")
                
            # Render the figure to an image (smaller size)
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # RGBA (4 channels)
            frames.append(frame)
            
            plt.close(fig)
            plt.close("all")

    elif 'HalfCheetah' in config['EXP_NAME'] and 'Avoid' in config['EXP_NAME']:
        
        info, info_avoid = multi_info 

        axes_upperx = 8.5
        axes_lowerx = -0.5
        axes_uppery = 1.3
        axes_lowery = -0.7

        # Reward percomputation (targets & obstacles)
        x = np.linspace(axes_lowerx, axes_upperx, 400)
        y = np.linspace(axes_lowery, axes_uppery, 400)
        X, Y = np.meshgrid(x, y)
        positions = np.stack([X, Y], axis=-1)  # shape (400, 400, 2)
        model = HalfCheetahReachAvoid()
        is_reach_np = jit(model.is_reach)
        is_avoid_np = jit(model.is_avoid)
        reach_values = np.array(is_reach_np(positions))
        avoid_values = np.array(is_avoid_np((positions, positions, positions, positions, positions, positions, positions, positions, positions)))

        def draw_cheetah_raa(step, info, title, ax):
            reach_idx = info.get('reach_index')
            avoid_idx = info.get('avoid_index')

            # Plot Reach  
            # draw_target = plt.Rectangle((3.25, -0.7), 0.5, 5., fill=False)
            # draw_rectangle = plt.Rectangle((2., -0.7), 1., 0.25, facecolor="red", fill=True)
            # draw_rectangle2 = plt.Rectangle((4., -0.7), 1., 0.25, facecolor="red", fill=True)
            # ax.add_patch(draw_target)
            # ax.add_patch(draw_rectangle)
            # ax.add_patch(draw_rectangle2)
            if reach_idx is not None:
                ax.contourf(X, Y, np.maximum(reach_values, avoid_values), alpha=0.3, levels=20)
            else:
                ax.contourf(X, Y, avoid_values, alpha=0.3, levels=20)
            if reach_idx is not None:
                ax.contourf(X, Y, reach_values, levels=[reach_values.min(), 0], colors=['green'], alpha=0.4)
            ax.contourf(X, Y, avoid_values, levels=[0, avoid_values.max()], colors=['red'], alpha=0.4)
            # indices = np.linspace(0, full_len, 11, dtype=int)
            # for step_n, i in enumerate(indices):

            def draw_body(ax, info, i, alpha, color_mode="normal"):
                
                if color_mode == "R":
                    c1, c2, c3, c4, c5, c6, c7, c8 = 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g'
                    linewidth=3
                elif color_mode == "A":
                    c1, c2, c3, c4, c5, c6, c7, c8 = 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r'
                    linewidth=3
                else:
                    c1, c2, c3, c4, c5, c6, c7, c8 = 'r', 'g', 'm', 'g', 'r', 'm', 'g', 'r'
                    linewidth=1

                # Plot Cheetah Body 
                ax.plot(np.array([info['head_pos'][i, 0], info['neck_pos'][i, 0]]),
                        np.array([info['head_pos'][i, 1], info['neck_pos'][i, 1]]), c=c1, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['neck_pos'][i, 0], info['back_pos'][i, 0]]),
                        np.array([info['neck_pos'][i, 1], info['back_pos'][i, 1]]), c=c2, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['neck_pos'][i, 0], info['front_thigh_pos'][i, 0]]),
                        np.array([info['neck_pos'][i, 1], info['front_thigh_pos'][i, 1]]), c=c3, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['front_thigh_pos'][i, 0], info['front_shin_pos'][i, 0]]),
                        np.array([info['front_thigh_pos'][i, 1], info['front_shin_pos'][i, 1]]), c=c4, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['front_foot_pos'][i, 0], info['front_shin_pos'][i, 0]]),
                        np.array([info['front_foot_pos'][i, 1], info['front_shin_pos'][i, 1]]), c=c5, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['back_pos'][i, 0], info['back_thigh_pos'][i, 0]]),
                        np.array([info['back_pos'][i, 1], info['back_thigh_pos'][i, 1]]), c=c6, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['back_thigh_pos'][i, 0], info['back_shin_pos'][i, 0]]),
                        np.array([info['back_thigh_pos'][i, 1], info['back_shin_pos'][i, 1]]), c=c7, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['back_foot_pos'][i, 0], info['back_shin_pos'][i, 0]]),
                        np.array([info['back_foot_pos'][i, 1], info['back_shin_pos'][i, 1]]), c=c8, alpha=alpha, linewidth=linewidth)

            reach_val = is_reach_np(info['head_pos'][step])
            avoid_val = is_avoid_np((info['head_pos'][step], info['neck_pos'][step], info['back_pos'][step],
                info['front_thigh_pos'][step], info['front_shin_pos'][step], info['front_foot_pos'][step], 
                info['back_thigh_pos'][step], info['back_shin_pos'][step], info['back_foot_pos'][step]))

            if avoid_val > 0.:
                color_mode = "A"
            elif reach_idx is not None and reach_val < 0.:
                color_mode = "R"
            else:
                color_mode = "normal"

            draw_body(ax, info, step, 0.9, color_mode=color_mode)

            if reach_idx is not None and step >= reach_idx and reach_idx > -1:
                draw_body(ax, info, reach_idx, 0.5, color_mode = "R")

            if avoid_idx is not None and step >= avoid_idx and avoid_idx > -1:
                draw_body(ax, info, avoid_idx, 0.5, color_mode = "A")
            
            ax.set_xlim((axes_lowerx, axes_upperx))
            ax.set_ylim((axes_lowery, axes_uppery))
            ax.set_aspect('equal')

            ax.set_title(title)

        frames = []
        full_len = info['head_pos'].shape[0]
        num_frames = full_len//2
        indices = np.linspace(0, full_len, num_frames, dtype=int)
        for step_n in indices: 
            # plt.figure(figsize=(12, 6*2))
            fig, axes = plt.subplots(2, 1, figsize=(4, 4), dpi=100)  # Smaller and square figure
            draw_cheetah_raa(step_n, info, "Reach Avoid", axes[0])
            if config['EXP_NAME'] == 'HalfCheetahReachAlwaysAvoid':
                draw_cheetah_raa(step_n, info_avoid, "Avoid Only", axes[1])
            
            # Render the figure to an image (smaller size)
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # RGBA (4 channels)
            frames.append(frame)
            
            plt.close(fig)
            plt.close("all")

    elif 'HalfCheetah' in config['EXP_NAME'] and 'ReachReach' in config['EXP_NAME']:
        
        info, info_1, info_2 = multi_info

        axes_upperx = 5.5
        axes_lowerx = -5.5
        axes_uppery = 1.3
        axes_lowery = -0.7

        # Reward percomputation (targets & obstacles)
        x = np.linspace(axes_lowerx, axes_upperx, 400)
        y = np.linspace(axes_lowery, axes_uppery, 400)
        X, Y = np.meshgrid(x, y)
        positions = np.stack([X, Y], axis=-1)  # shape (400, 400, 2)
        model = HalfCheetahReachReach()
        is_reach1_np = jit(model.is_reach1)
        is_reach2_np = jit(model.is_reach2)
        reach1_values = np.array(is_reach1_np((positions, positions, positions, positions, positions, positions, positions, positions, positions), (0., 0., 0.)))
        reach2_values = np.array(is_reach2_np((positions, positions, positions, positions, positions, positions, positions, positions, positions), (0., 0., 0.)))
    
        def draw_cheetah_rr(step, info, reach1_idx, reach2_idx, title, ax, mode="both"):

            # Plot Reach  
            if mode=="both":
                ax.contourf(X, Y, np.maximum(reach1_values, reach2_values), alpha=0.3, levels=20)
                ax.contourf(X, Y, reach1_values, levels=[reach1_values.min(), 0], colors=['green'], alpha=0.4)
                ax.contourf(X, Y, reach2_values, levels=[reach2_values.min(), 0], colors=['blue'], alpha=0.4)
            elif mode=="reach1":
                ax.contourf(X, Y, reach1_values, alpha=0.3, levels=20)
                ax.contourf(X, Y, reach1_values, levels=[reach1_values.min(), 0], colors=['green'], alpha=0.4)
            else:
                ax.contourf(X, Y, reach2_values, alpha=0.3, levels=20)
                ax.contourf(X, Y, reach2_values, levels=[reach2_values.min(), 0], colors=['blue'], alpha=0.4)

            def draw_body(ax, info, i, alpha, color_mode="normal"):
                
                if color_mode == "reach1":
                    c1, c2, c3, c4, c5, c6, c7, c8 = 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g'
                    linewidth=3
                elif color_mode == "reach2":
                    c1, c2, c3, c4, c5, c6, c7, c8 = 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b'
                    linewidth=3
                else:
                    c1, c2, c3, c4, c5, c6, c7, c8 = 'r', 'g', 'm', 'g', 'r', 'm', 'g', 'r'
                    linewidth=1

                # Plot Cheetah Body 
                ax.plot(np.array([info['head_pos'][i, 0], info['neck_pos'][i, 0]]),
                        np.array([info['head_pos'][i, 1], info['neck_pos'][i, 1]]), c=c1, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['neck_pos'][i, 0], info['back_pos'][i, 0]]),
                        np.array([info['neck_pos'][i, 1], info['back_pos'][i, 1]]), c=c2, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['neck_pos'][i, 0], info['front_thigh_pos'][i, 0]]),
                        np.array([info['neck_pos'][i, 1], info['front_thigh_pos'][i, 1]]), c=c3, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['front_thigh_pos'][i, 0], info['front_shin_pos'][i, 0]]),
                        np.array([info['front_thigh_pos'][i, 1], info['front_shin_pos'][i, 1]]), c=c4, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['front_foot_pos'][i, 0], info['front_shin_pos'][i, 0]]),
                        np.array([info['front_foot_pos'][i, 1], info['front_shin_pos'][i, 1]]), c=c5, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['back_pos'][i, 0], info['back_thigh_pos'][i, 0]]),
                        np.array([info['back_pos'][i, 1], info['back_thigh_pos'][i, 1]]), c=c6, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['back_thigh_pos'][i, 0], info['back_shin_pos'][i, 0]]),
                        np.array([info['back_thigh_pos'][i, 1], info['back_shin_pos'][i, 1]]), c=c7, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['back_foot_pos'][i, 0], info['back_shin_pos'][i, 0]]),
                        np.array([info['back_foot_pos'][i, 1], info['back_shin_pos'][i, 1]]), c=c8, alpha=alpha, linewidth=linewidth)
                
            reach1_val = is_reach1_np((info['head_pos'][step], info['neck_pos'][step], info['back_pos'][step],
                info['front_thigh_pos'][step], info['front_shin_pos'][step], info['front_foot_pos'][step], 
                info['back_thigh_pos'][step], info['back_shin_pos'][step], info['back_foot_pos'][step]), (0., 0., 0.))
            reach2_val = is_reach2_np((info['head_pos'][step], info['neck_pos'][step], info['back_pos'][step],
                info['front_thigh_pos'][step], info['front_shin_pos'][step], info['front_foot_pos'][step], 
                info['back_thigh_pos'][step], info['back_shin_pos'][step], info['back_foot_pos'][step]), (0., 0., 0.))

            if reach1_val < 0.:
                color_mode = "reach1"
            elif reach2_val < 0.:
                color_mode = "reach2"
            else:
                color_mode = "normal"

            draw_body(ax, info, step, 0.9, color_mode=color_mode)

            if reach1_idx is not None and step >= reach1_idx and reach1_idx > -1:
                draw_body(ax, info, reach1_idx, 0.5, color_mode="reach1")

            if reach2_idx is not None and step >= reach2_idx and reach2_idx > -1:
                draw_body(ax, info, reach2_idx, 0.5, color_mode="reach2")
            
            ax.set_xlim((axes_lowerx, axes_upperx))
            ax.set_ylim((axes_lowery, axes_uppery))
            ax.set_aspect('equal')

            ax.set_title(title)

        # DEFINE VIDEO LENGTH TO DUAL-REACHING OR FULL TRAJ
        reach_idx_1 = info['reach_index_1']
        reach_idx_2 = info['reach_index_2']

        full_len = np.maximum(reach_idx_1, reach_idx_2)
        full_len = info['head_pos'].shape[0] if full_len.item() == np.inf else int(full_len.item())
        # full_len = info['head_pos'].shape[0]
        reach_idx_1 = int(reach_idx_1.item()) if reach_idx_1.item() != np.inf else -1
        reach_idx_2 = int(reach_idx_2.item()) if reach_idx_2.item() != np.inf else -1
        
        frames = []
        num_frames = full_len//2
        indices = np.linspace(0, full_len, num_frames, dtype=int)
        if config['EXP_NAME'] == 'HalfCheetahReachReach':
            reach_idx_1_reach1 = info_1['reach_index_1'].item()
            reach_idx_2_reach2 = info_2['reach_index_2'].item()
            
        for step_n in indices: 

            fig, axes = plt.subplots(3, 1, figsize=(6, 4), dpi=100)

            draw_cheetah_rr(step_n, info, reach_idx_1, reach_idx_2, "Reach Reach", axes[0], mode="both")

            if config['EXP_NAME'] == 'HalfCheetahReachReach':
                # AFTER DECOMPOSED REACH, DRAW LAST POINT
                if step_n >= reach_idx_1_reach1:
                    reach_idx_1_reach1 = step_n
                if step_n >= reach_idx_2_reach2:
                    reach_idx_2_reach2 = step_n

                draw_cheetah_rr(step_n, info_1, reach_idx_1_reach1, -1, "Reach 1", axes[1], mode="reach1")
                draw_cheetah_rr(step_n, info_2, -1, reach_idx_2_reach2, "Reach 2", axes[2], mode="reach2")
                
            # Render the figure to an image (smaller size)
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # RGBA (4 channels)
            frames.append(frame)
            
            plt.close(fig)
            plt.close("all")

    elif 'Humanoid' in config['EXP_NAME'] and 'ReachReach' in config['EXP_NAME']:
        
        info, info_1, info_2 = multi_info

        axes_upperx = 3.5
        axes_lowerx = -0.5
        axes_uppery = 0.5
        axes_lowery = -3.5
        axes_upperz = 1.5
        axes_lowerz = -0.1

        def draw_humanoid_rr(step, info, reach1_idx, reach2_idx, title, ax, mode="both"):

            # Plot Targets and Obstacles
            if mode == "both" or mode == "reach1":
                add_sphere(ax, HUMANOID_TARGET_RIGHT, radius=HUMANOID_TARGET_RADIUS, resolution=10, alpha=0.4, color='green')
            if mode == "both" or mode == "reach2":
                add_sphere(ax, HUMANOID_TARGET_LEFT, radius=HUMANOID_TARGET_RADIUS, resolution=10, alpha=0.4, color='blue')

            model = HumanoidReachReach()
            is_reach1_np = jit(model.is_reach1)
            is_reach2_np = jit(model.is_reach2)

            def draw_body_3d(ax, info, i, alpha, color_mode="normal"):
                if color_mode == "reach1":
                    c1, c2, c3, c4, c5, c6 = ['g'] * 6
                    linewidth = 3
                elif color_mode == "reach2":
                    c1, c2, c3, c4, c5, c6 = ['b'] * 6
                    linewidth = 3
                else:
                    # head to shoulders, arms, hips to knees/feet
                    c1, c2, c3, c4, c5, c6 = 'k', 'r', 'm', 'g', 'c', 'b'
                    linewidth = 1

                def line(p1, p2, color):
                    ax.plot(
                        [p1[0], p2[0]],
                        [p1[1], p2[1]],
                        [p1[2], p2[2]],
                        c=color, alpha=alpha, linewidth=linewidth
                    )

                # Joint locations
                head = info["head_pos"][i]
                torso = info["torso"][i]
                lwaist = info["lwaist"][i]
                pelvis = info["pelvis"][i]
                
                left_upper_arm = info["left_upper_arm"][i]
                right_upper_arm = info["right_upper_arm"][i]
                left_lower_arm = info["left_lower_arm"][i]
                right_lower_arm = info["right_lower_arm"][i]
                left_hand = info["left_hand"][i]
                right_hand = info["right_hand"][i]

                left_thigh = info["left_thigh"][i]
                right_thigh = info["right_thigh"][i]
                left_shin = info["left_shin"][i]
                right_shin = info["right_shin"][i]
                left_foot = info["left_foot"][i]
                right_foot = info["right_foot"][i]

                # Draw abdomen
                line(head, torso, c6)
                line(torso, lwaist, c2)
                line(lwaist, pelvis, c3)

                # Draw arms
                line(torso, left_upper_arm, c4)
                line(torso, right_upper_arm, c4)
                line(left_upper_arm, left_lower_arm, c5)
                line(left_lower_arm, left_hand, c1)
                line(right_upper_arm, right_lower_arm, c5)
                line(right_lower_arm, right_hand, c1)

                # Draw legs
                line(pelvis, left_thigh, c4)
                line(pelvis, right_thigh, c4)
                line(left_thigh, left_shin, c5)
                line(left_shin, left_foot, c1)
                line(right_thigh, right_shin, c5)
                line(right_shin, right_foot, c1) 
            
            step_poses = {k: info[k][step] for k in info.keys() if not k in ['reach_index_1', 'reach_index_2']}
            reach1_val = is_reach1_np(step_poses)
            reach2_val = is_reach2_np(step_poses)

            if reach1_val < 0.:
                color_mode = "reach1"
            elif reach2_val < 0.:
                color_mode = "reach2"
            else:
                color_mode = "normal"

            draw_body_3d(ax, info, step, 0.9, color_mode=color_mode)

            if reach1_idx is not None and step >= reach1_idx and reach1_idx > -1:
                draw_body_3d(ax, info, reach1_idx, 0.5, color_mode="reach1")

            if reach2_idx is not None and step >= reach2_idx and reach2_idx > -1:
                draw_body_3d(ax, info, reach2_idx, 0.5, color_mode="reach2")
            
            ax.set_xlim((axes_lowerx, axes_upperx))
            ax.set_ylim((axes_lowery, axes_uppery))
            ax.set_zlim((axes_lowerz, axes_upperz))
            ax.set_aspect('equal')
            ax.set_title(title)
            ax.view_init(elev=10, azim=-45)

        # DEFINE VIDEO LENGTH TO DUAL-REACHING OR FULL TRAJ
        reach_idx_1 = info['reach_index_1']
        reach_idx_2 = info['reach_index_2']

        full_len = np.maximum(reach_idx_1, reach_idx_2)
        full_len = info['head_pos'].shape[0] if full_len.item() == np.inf else int(full_len.item())
        reach_idx_1 = int(reach_idx_1.item()) if reach_idx_1.item() != np.inf else -1
        reach_idx_2 = int(reach_idx_2.item()) if reach_idx_2.item() != np.inf else -1
        
        frames = []
        num_frames = full_len//4
        indices = np.linspace(0, full_len, num_frames, dtype=int)
        if config['EXP_NAME'] == 'HumanoidReachReach':
            reach_idx_1_reach1 = info_1['reach_index_1'].item()
            reach_idx_2_reach2 = info_2['reach_index_2'].item()
            
        for step_n in indices: 

            fig = plt.figure(figsize=(20, 6), dpi=50)
            ax1 = fig.add_subplot(131, projection='3d')
            ax2 = fig.add_subplot(132, projection='3d')
            ax3 = fig.add_subplot(133, projection='3d')
            axes = [ax1, ax2, ax3]

            draw_humanoid_rr(step_n, info, reach_idx_1, reach_idx_2, "Reach Reach", axes[0], mode="both")

            if config['EXP_NAME'] == 'HumanoidReachReach':
                # AFTER DECOMPOSED REACH, DRAW LAST POINT
                if step_n >= reach_idx_1_reach1:
                    reach_idx_1_reach1 = step_n
                if step_n >= reach_idx_2_reach2:
                    reach_idx_2_reach2 = step_n

                draw_humanoid_rr(step_n, info_1, reach_idx_1_reach1, -1, "Reach 1", axes[1], mode="reach1")
                draw_humanoid_rr(step_n, info_2, -1, reach_idx_2_reach2, "Reach 2", axes[2], mode="reach2")
                
            # Render the figure to an image (smaller size)
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # RGBA (4 channels)
            frames.append(frame)
            
            plt.close(fig)
            plt.close("all")

    elif 'Humanoid' in config['EXP_NAME'] and 'Avoid' in config['EXP_NAME']:

        info, info_avoid = multi_info

        axes_upperx = HUMANOID_RAA_BOX_RADIUS
        axes_lowerx = -(HUMANOID_RAA_BOX_RADIUS)
        axes_uppery = HUMANOID_RAA_BOX_RADIUS
        axes_lowery = -(HUMANOID_RAA_BOX_RADIUS)
        axes_upperz = 1.5
        axes_lowerz = -0.1

        def draw_humanoid_raa(step, info, title, ax, mode="both"):
            reach_idx = info.get('reach_index')
            avoid_idx = info.get('avoid_index')

            # Plot Targets and Obstacles
            add_box_3d(ax, center=np.array([0., 0., HUMANOID_RAA_FLOOR_HEIGHT/2.]), size=2*np.array([HUMANOID_RAA_BOX_RADIUS, HUMANOID_RAA_BOX_RADIUS, HUMANOID_RAA_FLOOR_HEIGHT]), alpha=0.05) # floor
            add_box_3d(ax, center=np.array([HUMANOID_RAA_BOX_RADIUS + 0.05, 0., 0.5]), size=2*np.array([0.1, HUMANOID_RAA_BOX_RADIUS-0.1, 0.5])) # wall
            add_box_3d(ax, center=np.array([-(HUMANOID_RAA_BOX_RADIUS + 0.05), 0., 0.5]), size=2*np.array([0.1, HUMANOID_RAA_BOX_RADIUS-0.1, 0.5])) # wall
            add_box_3d(ax, center=np.array([0., HUMANOID_RAA_BOX_RADIUS + 0.05, 0.5]), size=2*np.array([HUMANOID_RAA_BOX_RADIUS-0.1, 0.1, 0.5])) # wall
            add_box_3d(ax, center=np.array([0., -(HUMANOID_RAA_BOX_RADIUS + 0.05), 0.5]), size=2*np.array([HUMANOID_RAA_BOX_RADIUS-0.1, 0.1, 0.5])) # wall
            if mode == "both":
                add_cylinder(ax, HUMANOID_RAA_TARGET, radius=HUMANOID_RAA_TARGET_RADIUS, height=2., resolution=10, alpha=0.4, color='green') # target

            model = HumanoidReachAvoid()
            is_reach_np = jit(model.is_reach)
            is_avoid_np = jit(model.is_avoid)

            def draw_body_3d(ax, info, i, alpha, color_mode="normal"):
                if color_mode == "R":
                    c1, c2, c3, c4, c5, c6 = ['g'] * 6
                    linewidth = 3
                elif color_mode == "A":
                    c1, c2, c3, c4, c5, c6 = ['r'] * 6
                    linewidth = 3
                else:
                    # head to shoulders, arms, hips to knees/feet
                    c1, c2, c3, c4, c5, c6 = 'k', 'r', 'm', 'g', 'c', 'b'
                    linewidth = 1

                def line(p1, p2, color):
                    ax.plot(
                        [p1[0], p2[0]],
                        [p1[1], p2[1]],
                        [p1[2], p2[2]],
                        c=color, alpha=alpha, linewidth=linewidth
                    )

                # Joint locations
                head = info["head_pos"][i]
                torso = info["torso"][i]
                lwaist = info["lwaist"][i]
                pelvis = info["pelvis"][i]
                
                left_upper_arm = info["left_upper_arm"][i]
                right_upper_arm = info["right_upper_arm"][i]
                left_lower_arm = info["left_lower_arm"][i]
                right_lower_arm = info["right_lower_arm"][i]
                left_hand = info["left_hand"][i]
                right_hand = info["right_hand"][i]

                left_thigh = info["left_thigh"][i]
                right_thigh = info["right_thigh"][i]
                left_shin = info["left_shin"][i]
                right_shin = info["right_shin"][i]
                left_foot = info["left_foot"][i]
                right_foot = info["right_foot"][i]

                # Draw abdomen
                line(head, torso, c6)
                line(torso, lwaist, c2)
                line(lwaist, pelvis, c3)

                # Draw arms
                line(torso, left_upper_arm, c4)
                line(torso, right_upper_arm, c4)
                line(left_upper_arm, left_lower_arm, c5)
                line(left_lower_arm, left_hand, c1)
                line(right_upper_arm, right_lower_arm, c5)
                line(right_lower_arm, right_hand, c1)

                # Draw legs
                line(pelvis, left_thigh, c4)
                line(pelvis, right_thigh, c4)
                line(left_thigh, left_shin, c5)
                line(left_shin, left_foot, c1)
                line(right_thigh, right_shin, c5)
                line(right_shin, right_foot, c1)    
            
            step_poses = {k: info[k][step] for k in info.keys() if not k in ['reach_index', 'avoid_index']}
            reach_val = is_reach_np(step_poses)
            avoid_val = is_avoid_np(step_poses)

            if avoid_val > 0.:
                color_mode = "A"
            elif reach_idx is not None and reach_val < 0.:
                color_mode = "R"
            else:
                color_mode = "normal"

            draw_body_3d(ax, info, step, 0.9, color_mode=color_mode)

            if reach_idx is not None and step >= reach_idx and reach_idx > -1:
                draw_body_3d(ax, info, reach_idx, 0.5, color_mode = "R")

            if avoid_idx is not None and step >= avoid_idx and avoid_idx > -1:
                draw_body_3d(ax, info, avoid_idx, 0.5, color_mode = "A")

            ax.set_xlim((axes_lowerx, axes_upperx))
            ax.set_ylim((axes_lowery, axes_uppery))
            ax.set_zlim((axes_lowerz, axes_upperz))
            ax.set_aspect('equal')
            ax.set_title(title)
            ax.view_init(elev=45, azim=-45)

        frames = []
        full_len = info['head_pos'].shape[0]
        num_frames = full_len//4
        indices = np.linspace(0, full_len, num_frames, dtype=int)
        for step_n in indices: 

            fig = plt.figure(figsize=(12, 6))
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122, projection='3d')
            axes = [ax1, ax2]

            draw_humanoid_raa(step_n, info, "Reach Avoid", axes[0], mode="both")
            if config['EXP_NAME'] == 'HumanoidReachAlwaysAvoid':
                draw_humanoid_raa(step_n, info_avoid, "Avoid Only", axes[1], mode="avoid")
            
            # Render the figure to an image (smaller size)
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # RGBA (4 channels)
            frames.append(frame)
            
            plt.close(fig)
            plt.close("all")

    elif 'Point' in config['EXP_NAME'] and 'ReachReach' in config['EXP_NAME']:
        
        info, info_1, info_2 = multi_info

        axes_upperx = 3.
        axes_lowerx = -3.
        axes_uppery = 3.
        axes_lowery = -3.

        # Reward percomputation (targets & obstacles)
        x = np.linspace(axes_lowerx, axes_upperx, 400)
        y = np.linspace(axes_lowery, axes_uppery, 400)
        X, Y = np.meshgrid(x, y)
        positions = np.stack([X, Y], axis=-1)  # shape (400, 400, 2)
        model = PointReachReach()
        is_reach1_np = jit(model.is_reach1)
        is_reach2_np = jit(model.is_reach2)
        reach1_values = np.array(is_reach1_np(positions))
        reach2_values = np.array(is_reach2_np(positions))
        
        def draw_point_rr(step, info, reach1_idx, reach2_idx, title, ax, mode="both"):

            # Plot Reach  
            if mode=="both":
                ax.contourf(X, Y, np.maximum(reach1_values, reach2_values), alpha=0.3, levels=20)
                ax.contourf(X, Y, reach1_values, levels=[reach1_values.min(), 0], colors=['green'], alpha=0.4)
                ax.contourf(X, Y, reach2_values, levels=[reach2_values.min(), 0], colors=['blue'], alpha=0.4)
            elif mode=="reach1":
                ax.contourf(X, Y, reach1_values, alpha=0.3, levels=20)
                ax.contourf(X, Y, reach1_values, levels=[reach1_values.min(), 0], colors=['green'], alpha=0.4)
            else:
                ax.contourf(X, Y, reach2_values, alpha=0.3, levels=20)
                ax.contourf(X, Y, reach2_values, levels=[reach2_values.min(), 0], colors=['blue'], alpha=0.4)

            def draw_body(ax, info, i, alpha, color_mode="normal"):
                
                if color_mode == "reach1":
                    c = 'g'
                    linewidth=3
                elif color_mode == "reach2":
                    c = 'b'
                    linewidth=3
                else:
                    c = 'k'
                    linewidth=1

                # ax.plot(np.array([info['x'][i], info['y'][i]]), c=c, alpha=alpha, linewidth=linewidth)
                ax.scatter(info['x'][i], info['y'][i], color=c, alpha=alpha)

            reach1_val = is_reach1_np(np.array([info['x'][step].item(), info['y'][step].item()]))
            reach2_val = is_reach2_np(np.array([info['x'][step].item(), info['y'][step].item()]))

            if reach1_val < 0.:
                color_mode = "reach1"
            elif reach2_val < 0.:
                color_mode = "reach2"
            else:
                color_mode = "normal"

            draw_body(ax, info, step, 0.9, color_mode=color_mode)

            if reach1_idx is not None and step >= reach1_idx and reach1_idx > -1:
                draw_body(ax, info, reach1_idx, 0.5, color_mode="reach1")

            if reach2_idx is not None and step >= reach2_idx and reach2_idx > -1:
                draw_body(ax, info, reach2_idx, 0.5, color_mode="reach2")
            
            ax.set_xlim((axes_lowerx, axes_upperx))
            ax.set_ylim((axes_lowery, axes_uppery))
            ax.set_aspect('equal')

            ax.set_title(title)

        # DEFINE VIDEO LENGTH TO DUAL-REACHING OR FULL TRAJ
        reach_idx_1 = info['reach_index_1']
        reach_idx_2 = info['reach_index_2']

        full_len = np.maximum(reach_idx_1, reach_idx_2)
        full_len = info['x'].shape[0] if full_len.item() == np.inf else int(full_len.item())
        # full_len = info['head_pos'].shape[0]
        reach_idx_1 = int(reach_idx_1.item()) if reach_idx_1.item() != np.inf else -1
        reach_idx_2 = int(reach_idx_2.item()) if reach_idx_2.item() != np.inf else -1
        
        frames = []
        num_frames = full_len//2
        indices = np.linspace(0, full_len, num_frames, dtype=int)
        if config['EXP_NAME'] == 'PointReachReach':
            reach_idx_1_reach1 = info_1['reach_index_1'].item()
            reach_idx_2_reach2 = info_2['reach_index_2'].item()
            
        for step_n in indices: 

            fig, axes = plt.subplots(3, 1, figsize=(6, 4), dpi=100)

            draw_point_rr(step_n, info, reach_idx_1, reach_idx_2, "Reach Reach", axes[0], mode="both")

            if config['EXP_NAME'] == 'PointReachReach':
                # AFTER DECOMPOSED REACH, DRAW LAST POINT
                if step_n >= reach_idx_1_reach1:
                    reach_idx_1_reach1 = step_n
                if step_n >= reach_idx_2_reach2:
                    reach_idx_2_reach2 = step_n

                draw_point_rr(step_n, info_1, reach_idx_1_reach1, -1, "Reach 1", axes[1], mode="reach1")
                draw_point_rr(step_n, info_2, -1, reach_idx_2_reach2, "Reach 2", axes[2], mode="reach2")
                
            # Render the figure to an image (smaller size)
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # RGBA (4 channels)
            frames.append(frame)
            
            plt.close(fig)
            plt.close("all")
        
    # Save frames as a video using PIL - don't do this in general
    
    if frames is not None: 
        frames = [Image.fromarray(frame) for frame in frames]
        if save_video: 
            # video_path = 'model/{}/reach/trajectory_{:0>4d}.mp4'.format(config["DIR"], epoch)
            # frames[0].save(video_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
            # mod prefix from / to _
            prefix_underscore = prefix.replace("/", "_")
            video_path = 'model/{}/reach/trajectory_{}{:0>4d}.mp4'.format(config["DIR"], prefix_underscore, epoch)
            print("\n\nSaving video to: ", video_path)
            imageio.mimsave(video_path, frames, fps=30)
            if log_wandb: 
                wandb_name = f"{prefix}trajectory video"
                print("Logging video to wandb: ", wandb_name)
                try:
                    wandb.log({wandb_name: wandb.Video(video_path, format="mp4")}, step=epoch)
                except: 
                    print("Error logging video to wandb")
                    
        end_time = time()
        print("Time taken to plot and push video: ", end_time - start_time)
        return frames

def add_sphere(ax, center, radius=1.0, resolution=30, color='green', alpha=0.5):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, shade=True)

    lw=0.2
    theta = np.linspace(0, 2 * np.pi, resolution)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = np.full_like(x, center[2])
    ax.plot3D(x, y, z, color='k', linewidth=lw, alpha=0.8)

    x = np.full_like(theta, center[0])
    y = center[1] + radius * np.cos(theta)
    z = center[2] + radius * np.sin(theta)
    ax.plot3D(x, y, z, color='k', linewidth=lw, alpha=0.8)

def add_cylinder(ax, center, radius=1.0, height=1.0, resolution=30, color='green', alpha=0.5):
    u = np.linspace(0, 2 * np.pi, resolution)
    z = np.linspace(0, height, resolution)
    U, Z = np.meshgrid(u, z)
    X = center[0] + radius * np.cos(U)
    Y = center[1] + radius * np.sin(U)
    Z = center[2] + Z  # lift base to z = center[2]
    
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0, shade=True)

def add_box_3d(ax, center, size, color='red', alpha=0.1):
    """
    Draws a 3D axis-aligned box given center and size (width, height, depth).
    eg. add_box_3d(ax, center=np.array([1000., 525, 550]), size=np.array([2000., 50, 1100]))
    """
    cx, cy, cz = center
    sx, sy, sz = size[0] / 2, size[1] / 2, size[2] / 2

    # Define the 8 corners of the box
    points = np.array([[cx - sx, cy - sy, cz - sz],
                       [cx + sx, cy - sy, cz - sz],
                       [cx + sx, cy + sy, cz - sz],
                       [cx - sx, cy + sy, cz - sz],
                       [cx - sx, cy - sy, cz + sz],
                       [cx + sx, cy - sy, cz + sz],
                       [cx + sx, cy + sy, cz + sz],
                       [cx - sx, cy + sy, cz + sz]])

    # Define 6 box faces using the indices
    faces = [[points[j] for j in [0,1,2,3]],
             [points[j] for j in [4,5,6,7]],
             [points[j] for j in [0,1,5,4]],
             [points[j] for j in [2,3,7,6]],
             [points[j] for j in [1,2,6,5]],
             [points[j] for j in [4,7,3,0]]]

    box = Poly3DCollection(faces, facecolors=color, linewidths=0.3, edgecolors='k', alpha=alpha)
    ax.add_collection3d(box)

# def is_reach(head_pos):
#     radius, target_pos = 0.25, np.array([3.5, 0.0])
#     reach_value = np.sqrt((head_pos[..., 0] - target_pos[0]) ** 2) - radius

#     # has_reached_goal = reach_value < 0
#     # reach_value = np.where(has_reached_goal, -3., reach_value)

#     return reach_value

# def is_avoid(front_foot_pos, back_foot_pos):
#     radius, box_halfwidth = 0.05, 0.5

#     avoid_box_1_front = -(np.maximum((radius/box_halfwidth) * np.fabs(front_foot_pos[..., 0] - 2.5), front_foot_pos[..., 1] + 0.5) - radius)
#     avoid_box_1_back = -(np.maximum((radius/box_halfwidth) * np.fabs(back_foot_pos[..., 0] - 2.5), back_foot_pos[..., 1] + 0.5) - radius)
#     avoid_box_2_front = -(np.maximum((radius/box_halfwidth) * np.fabs(front_foot_pos[..., 0] - 4.5), front_foot_pos[..., 1] + 0.5) - radius)
#     avoid_box_2_back = -(np.maximum((radius/box_halfwidth) * np.fabs(back_foot_pos[..., 0] - 4.5), back_foot_pos[..., 1] + 0.5) - radius)
    
#     avoid_value = np.maximum(
#         np.maximum(avoid_box_1_front, avoid_box_1_back),
#         np.maximum(avoid_box_2_front, avoid_box_2_back)
#     )
#     # avoid_value = np.where(avoid_value > 0, 10., avoid_value)

#     return 10. * avoid_value