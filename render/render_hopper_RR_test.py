import os
import sys
import trimesh
from copy import deepcopy 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.transform import Rotation as R
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D 
plt.rcParams["text.usetex"] = False
import jax.numpy as jnp
from rraa_rl.EFPPO.src.rl.arguments import get_args
from rraa_rl.EFPPO.src.env.env_list import get_env
import imageio

from brax.io import html

def load_traj(file_path):
    with jnp.load(file_path, allow_pickle=False) as traj_data:
        traj_batch = {key: traj_data[key] for key in traj_data.files}
    return traj_batch

# def _draw_f16_scene(position, euler_angles, setting_type, mesh, scale, trail=None, title=None, title2='Body', set_follow_cam=True):
#     fig = plt.figure(dpi=200)
#     gs = gridspec.GridSpec(1, 3)
#     ax = fig.add_subplot(gs[0, :2], projection='3d')

#     # Main scene
#     draw_targets(ax, setting_type=setting_type)
#     plot_mesh(ax, mesh, pos=position, euler_angles=euler_angles, scale=scale, set_follow_cam=set_follow_cam)

#     # Add trajectory trail (if given)
#     if trail is not None and len(trail) > 1:
#         trail = np.array(trail)
#         ax.plot3D(trail[:, 0], trail[:, 1], trail[:, 2], linestyle='--', color='black', linewidth=0.5, alpha=0.7)

#     # CONFIGURE PLOT
#     if setting_type == 'RR':
#         ax.set_xlim(0, 1600)
#         ax.set_ylim(-400, 400)
#         ax.set_zlim(100, 1000)
#         ax.set_xticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600])
#         ax.set_xticklabels(['0', '', '', '', '', '', '', '', '1600'])
#         ax.set_yticks([-400, -200, 0, 200, 400])
#         ax.set_yticklabels(['-400', '', '0', '', '400'])
#         ax.set_zticks([100, 300, 500, 700, 900, 1100])
#         ax.set_zticklabels(['100', '', '', '', '', '1100'])
#     elif setting_type == 'RAA':
#         ax.set_xlim(0, 2100)
#         ax.set_ylim(-550, 550)
#         ax.set_zlim(-50, 1000)
#         ax.set_xticks([0, 400, 800, 1200, 1600, 2000])
#         ax.set_xticklabels(['0', '', '', '', '', '2000'])
#         ax.set_yticks([-500, -250, 0, 250, 500])
#         ax.set_yticklabels(['-500', '', '0', '', '500'])
#         ax.set_zticks([0, 200, 400, 600, 800, 1100])
#         ax.set_zticklabels(['0', '', '', '', '', '1100'])

#     ax.set_xlabel("Position North")
#     ax.set_ylabel("Position East")
#     ax.set_zlabel("Altitude")
#     ax.zaxis.set_label_coords(-0.5, 0.)
#     ax.set_aspect('equal')

#     bg_color = "#e6ecf2"
#     for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
#         axis.set_pane_color(plt.matplotlib.colors.to_rgba(bg_color))
#         axis._axinfo["grid"]['color'] = (1, 1, 1, 1)

#     # Close-up view
#     ax2 = fig.add_subplot(gs[0, 2], projection='3d')
#     plot_mesh(ax2, mesh, pos=position, euler_angles=euler_angles, scale=scale, alpha=1.0, set_follow_cam=set_follow_cam)
#     draw_targets(ax2, setting_type=setting_type, alpha=0.2, draw_obstacles=False)
#     ax2.set_xticklabels([])
#     ax2.set_yticklabels([])
#     ax2.set_zticklabels([])
#     span = 30 * 2
#     ax2.set_xlim(position[0] - span/2, position[0] + span/2)
#     ax2.set_ylim(position[1] - span/2, position[1] + span/2)
#     ax2.set_zlim(position[2] - span/2, position[2] + span/2)

#     for axis in [ax2.xaxis, ax2.yaxis, ax2.zaxis]:
#         axis.set_pane_color(plt.matplotlib.colors.to_rgba(bg_color))
#         axis._axinfo["grid"]['color'] = (1, 1, 1, 1)

#     ax2.set_title(title2)
#     ax2.set_aspect('equal')

#     # FINAL TOUCHES
#     plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
#     plt.subplots_adjust(wspace=0.5)
#     plt.suptitle(title, fontsize=20, y=0.95)

#     return fig

# def render_f16_trajectory_gif(position_traj, angles_traj, setting_type, mesh, scale, output_gif_path, temp_dir="frames", dpi=200, config=None):
#     os.makedirs(temp_dir, exist_ok=True)
#     frame_paths = []
#     print(f"Rendering F16 trajectory GIF to {output_gif_path}...")

#     if config is not None and config['ALG'] == 'DOHJPPO':
#         title = r'$\mathtt{F16}$ — RR — $\mathbf{DO\text{-}HJ\text{-}PPO}$'
#     elif config is not None and 'CPPO' in config['ALG']:
#         title = r'$\mathtt{F16}$ — RR — $\mathbf{C\text{-}PPO}$'
#     elif config is not None and 'DSTL' in config['ALG']:
#         title = r'$\mathtt{F16}$ — RR — $\mathbf{D\text{-}STL}$'

#     for t in range(len(position_traj)):
#         print(f"Rendering frame {t+1}/{len(position_traj)}")
#         trail = position_traj[:t+1]  # slice up to current time
#         fig = _draw_f16_scene(position_traj[t], angles_traj[t], setting_type, mesh, scale, trail=trail, title=title)
#         frame_path = os.path.join(temp_dir, f"frame_{t:04d}.png")
#         fig.savefig(frame_path, dpi=dpi)
#         plt.close(fig)
#         frame_paths.append(frame_path)

#     # Create GIF
#     images = [imageio.v3.imread(fp) for fp in frame_paths]
#     imageio.mimsave(output_gif_path, images, duration=0.05)

#     # Optional cleanup
#     for fp in frame_paths:
#         os.remove(fp)
#     os.rmdir(temp_dir)

# def render_f16_trajectory_png(position_traj, angles_traj, setting_type, mesh, scale, output_png_path, dpi=200, config=None):

#     if config is not None and config['ALG'] == 'DOHJPPO':
#         title = r'$\mathtt{F16}$ — RR — $\mathbf{DO\text{-}HJ\text{-}PPO}$'
#     elif config is not None and 'CPPO' in config['ALG']:
#         title = r'$\mathtt{F16}$ — RR — $\mathbf{C\text{-}PPO}$'
#     elif config is not None and 'DSTL' in config['ALG']:
#         title = r'$\mathtt{F16}$ — RR — $\mathbf{D\text{-}STL}$'

#     # Draw final scene with trail
#     t = len(position_traj) - 1
#     trail = position_traj[:t+1]  # slice up to current time
#     fig = _draw_f16_scene(position_traj[t], angles_traj[t], setting_type, mesh, scale, 
#                           trail=trail, title=title, title2='Final Body', set_follow_cam=False)
    
#     # Draw intermediate scenes on fig
#     ax = fig.axes[0]
#     snapshot_sample_indices = [0, 25, 50, 100, 150]
#     # snapshot_sample_indices = [0, 25, 50, 100]
#     for t in snapshot_sample_indices:
#         plot_mesh(ax, mesh, pos=position_traj[t], euler_angles=angles_traj[t], scale=scale, set_follow_cam=False)
        
#     fig.savefig(output_png_path, dpi=dpi)
#     plt.close(fig)

if __name__ == "__main__":

    ## INIT
    draw_gif = False
    sample_index = 0
    config = vars(get_args(sys.argv[1:]))
    config["EXP_NAME"]="HopperReachReach"
    config["PROBLEM_TYPE"]="RR"
    config["ALG"]="DOHJPPO"
    fig_file_name = f"render/gifs/hopper_RR_trajectory_render_{config['ALG']}_{sample_index}_view0_{np.random.randint(100000):2d}" #FIXME randint for touch bug
    traj_batch = load_traj(f"model/eval_all_figs/Hopper_RR_061925/traj_sample/traj_{config['ALG']}.npz")

    envs = get_env(config)
    env = envs[0]

    # TEST STATE
    # position = np.array([400.0, 0.0, 600.0])  # North, East, Alt
    # euler_angles = (np.deg2rad(0), np.deg2rad(0), np.deg2rad(0))  # pitch, yaw, roll

    step_index = 0
    # state_sample_ti = traj_batch['state'][sample_index, step_index, :]
    # position = np.array([state_sample_ti[env._env.PN], state_sample_ti[env._env.PE], state_sample_ti[env._env.H]])  # North, East, Alt
    # euler_angles = (state_sample_ti[env._env.THETA], state_sample_ti[env._env.PSI], state_sample_ti[env._env.PHI])  # pitch, yaw, roll

    test_obs = traj_batch['obs'][step_index, sample_index, :]
    test_obs[1] = test_obs[1] - 1.25
    qpos = test_obs[:6]
    qvel = test_obs[6:12]
    pipeline_state = env._env._env.env.env.pipeline_init(qpos, qvel)
    
    html_str = html.render(
        sys=env._env._env.env.env.sys,
        states=[pipeline_state],
        height=480,  # optional
        # colab=False  # set True if using Colab
    )
    with open("render/hopper_render_test.html", "w") as f:
        f.write(html_str)

    pipeline_states = []
    for step_i in range(traj_batch['obs'].shape[1]):
        test_obs = traj_batch['obs'][step_i, sample_index, :]
        test_obs[1] = test_obs[1] - 1.25
        qpos = test_obs[:6]
        qvel = test_obs[6:12]
        pipeline_state = env._env._env.env.env.pipeline_init(qpos, qvel)
        pipeline_states.append(pipeline_state)

    html_str = html.render(
        sys=env._env._env.env.env.sys,
        states=pipeline_states,
        height=480,  # optional
        # colab=False  # set True if using Colab
    )
    with open("render/hopper_render_traj_test.html", "w") as f:
        f.write(html_str)

    # render_f16_trajectory_png(
    #     position_traj=traj_batch['state'],
    #     angles_traj=traj_batch['state'],
    #     setting_type=config['PROBLEM_TYPE'],
    #     output_png_path=fig_file_name+".png",
    #     dpi=200,
    #     config=config
    # )

    # if draw_gif:
    #     render_f16_trajectory_gif(
    #         position_traj=traj_batch['state'],
    #         angles_traj=traj_batch['state'],
    #         setting_type=config['PROBLEM_TYPE'],
    #         output_png_path=fig_file_name+".gif",
    #         temp_dir="render/gifs/frames",
    #         dpi=200,
    #         config=config
    #     )

    