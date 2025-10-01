import sys
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.transform import Rotation as R
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D 
plt.rcParams["text.usetex"] = False
import jax.numpy as jnp
from rraa_rl.src.rl.utils.arguments import get_args
from rraa_rl.src.env.env_list import get_env

def load_mesh(path):
    """
    Load a .glb or .gltf file using trimesh and return a unified mesh.
    """
    scene_or_mesh = trimesh.load(path)
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = scene_or_mesh.dump(concatenate=True)
    else:
        mesh = scene_or_mesh
    return mesh

def set_follow_camera(ax, pos, look_dir=np.array([1, 0, 0]), up=np.array([0, 0, 1]), dist=100.0):
    """
    Centers and orients the 3D plot to follow the jet from behind.
    
    Parameters:
    - ax: 3D axis
    - pos: jet position (np.array [x, y, z])
    - look_dir: direction jet is facing (unit vector in world space)
    - up: up direction (approximate, for consistent vertical framing)
    - dist: how far back the camera should be from the jet
    """
    # Offset to follow from behind
    camera_pos = pos - look_dir * dist - up * (dist * 0.3)

    # Center the plot around the jet
    span = dist * 2
    ax.set_xlim(pos[0] - span/2, pos[0] + span/2)
    ax.set_ylim(pos[1] - span/2, pos[1] + span/2)
    ax.set_zlim(pos[2] - span/2, pos[2] + span/2)
    # ax.set_box_aspect([1, 1, 1])

    # Compute azim and elev from camera vector (optional)
    rel = pos - camera_pos
    azim = np.degrees(np.arctan2(rel[1], rel[0]))  # horizontal
    elev = np.degrees(np.arctan2(rel[2], np.linalg.norm(rel[:2])))  # vertical

    ax.view_init(elev=elev, azim=azim-50)

def plot_mesh(ax, mesh, pos=(0, 0, 0), euler_angles=(0, 0, 0), scale=1.0, set_follow_cam=False, alpha=0.8):
    """
    Plots a 3D mesh in a matplotlib 3D plot with rotation and translation.
    """
    # R_mat = R.from_euler('xyz', euler_angles).as_matrix()
    # vertices = (mesh.vertices @ R_mat.T) * scale + pos

    # Apply a -90 degree pitch to bring it from Z-forward to X-forward
    correction = R.from_euler('y', -np.pi / 2) * R.from_euler('z', -np.pi / 2)
    pose_rot = R.from_euler('xyz', euler_angles)

    # Combine correction * trajectory rotation
    R_total = correction * pose_rot
    R_mat = R_total.as_matrix()

    look_dir = R_mat @ np.array([0, 0, 1])  # jet’s forward vector in world space
    if set_follow_cam: set_follow_camera(ax, pos, look_dir=look_dir, dist=100)

    # vertices = (mesh.vertices @ R_mat.T) * scale + pos
    # vertices = ((mesh.vertices * scale) @ R_mat.T) + pos
    center = mesh.vertices.mean(axis=0)
    centered_vertices = (mesh.vertices - center) * scale
    vertices = (centered_vertices @ R_mat.T) + pos
    faces = mesh.faces

    for face in faces:
        triangle = vertices[face]
        poly = Poly3DCollection([triangle], color='darkgray', edgecolor='k', alpha=alpha, linewidth=0.1)
        ax.add_collection3d(poly)

def draw_targets(ax, setting_type='RR',alpha=0.5, draw_obstacles=True):
    """
    Draws target points in the 3D plot.
    
    Parameters:
    - ax: 3D axis
    - targets: list of target positions (np.array [x, y, z])
    - color: color of the target points
    - radius: size of the target points
    """

    if setting_type == 'RR':
        radius = 150
        targets = [np.array([1200., 0, 850]), np.array([1200., 0, 350])]
        for ti, target in enumerate(targets):
            # ax.scatter(*target, color='green', s=radius**2, alpha=0.5)
            color = 'green' if ti == 0 else 'blue'
            add_sphere(ax, target, radius=radius, resolution=30, alpha=alpha/1.5, color=color)

    elif setting_type == 'RAA':
        radius = 150

        # OBSTACLEs
        if draw_obstacles:
            add_box_3d(ax, center=np.array([2050., 0, 550]), size=np.array([100., 1000, 1100]), color='red', alpha=alpha/4) # forward fence
            add_box_3d(ax, center=np.array([1000., 0, -25]), size=np.array([2000., 1000, 50]), color='red', alpha=alpha/4) # ground
            add_box_3d(ax, center=np.array([1000., -525, 550]), size=np.array([2000., 50, 1100]), color='red', alpha=alpha/4) # corridor fence

        # TARGET
        add_box_3d(ax, center=np.array([1500., 0, 550]), size=np.array([500., 1000, 1100]), color='green', alpha=alpha/2)

        # if config['EXP_NAME'] == 'F16ReachReach':
        #     radius = 150
        #     draw_circle = plt.Circle((1200., 850), radius, facecolor="green", fill=True, alpha = 0.4)
        #     ax.add_patch(draw_circle)
        #     draw_circle = plt.Circle((1200., 350), radius, facecolor="green", fill=True, alpha = 0.4)
        #     ax.add_patch(draw_circle)
        # else:
        #     draw_rectangle_1 = plt.Rectangle((1250., -0.1), 500., 1200., facecolor="green", fill=True, alpha = 0.4)
        #     ax.add_patch(draw_rectangle_1)
        #     draw_rectangle_2 = plt.Rectangle((2000 - 25, -0.1), 100., 1200., facecolor="red", fill=True, alpha = 0.4)
        #     ax.add_patch(draw_rectangle_2)
        #     draw_rectangle_2 = plt.Rectangle((-0.1, -0.1), 2000.1, 2., facecolor="red", fill=True, alpha = 0.4)
        #     ax.add_patch(draw_rectangle_2)

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

def add_box_3d(ax, center, size, color='red', alpha=0.2):
    """
    Draws a 3D axis-aligned box given center and size (width, height, depth).
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

def load_traj(file_path):
    with jnp.load(file_path, allow_pickle=False) as traj_data:
        traj_batch = {key: traj_data[key] for key in traj_data.files}
    return traj_batch

if __name__ == "__main__":

    ## INIT
    config = vars(get_args(sys.argv[1:]))
    config["EXP_NAME"]="F16ReachReach"
    config["ALG"]="DOHJPPO"
    config["PROBLEM_TYPE"]="RR"
    setting_type=config['PROBLEM_TYPE']
    fig_file_name = f"render/figs/f16_{setting_type}_render_test_load.png"
    traj_batch = load_traj("model/eval_all_figs/F16_RR_052825_2/traj_sample/traj_DOHJPPO.npz")

    path_to_glb = "render/f16-c_falcon.glb"  # Replace with your .glb or .gltf path
    mesh = load_mesh(path_to_glb)
    true_length = 50.0  # meters
    model_length = mesh.extents[0]
    scale = true_length / model_length

    if config["ALG"]=="DOHJPPO":
        envs = get_env(config)
        env, _, _ = envs
    else:
        env = get_env(config)

    sample_index = 0
    step_index = 0

    # TEST STATE
    # position = np.array([400.0, 0.0, 600.0])  # North, East, Alt
    # euler_angles = (np.deg2rad(0), np.deg2rad(0), np.deg2rad(0))  # pitch, yaw, roll

    state_sample_ti = traj_batch['state'][sample_index, step_index, :]

    position = np.array([state_sample_ti[env._env.PN], state_sample_ti[env._env.PE], state_sample_ti[env._env.H]])  # North, East, Alt
    euler_angles = (state_sample_ti[env._env.THETA], state_sample_ti[env._env.PSI], state_sample_ti[env._env.PHI])  # pitch, yaw, roll

    fig = plt.figure()
    # ax = fig.add_subplot(121, projection='3d')
    gs = gridspec.GridSpec(1, 3)
    ax = fig.add_subplot(gs[0, :2], projection='3d')
    # ax.view_init(elev=30, azim=60)

    draw_targets(ax, setting_type=setting_type)
    plot_mesh(ax, mesh, pos=position, euler_angles=euler_angles, scale=scale, set_follow_cam=True)

    # CONFIGURE PLOT
    if setting_type == 'RR':
        # ax.set_title("F-16 Reach-Reach")
        ax.set_xlim(0, 1600)
        ax.set_ylim(-400, 400)
        ax.set_zlim(100, 1000)
        ax.set_xticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600], labels=['0', '', '', '', '', '', '', '', '1600'])
        ax.set_yticks([-400, 300, -200, -100, 0, 100, 200, 300, 400], labels=['-400', '', '', '', '0', '', '', '', '400'])
        ax.set_zticks([100, 300, 500, 700, 900, 1100], labels=['100', '', '', '', '', '1100'])
    elif setting_type == 'RAA':
        ax.set_xlim(0, 2100)
        ax.set_ylim(-550, 550)
        ax.set_zlim(-50, 1000)
        # ax.set_title("F-16 Reach-Always-Avoid")
        ax.set_xticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000], labels=['0', '', '', '', '', '', '', '', '', '', '2000'])
        ax.set_yticks([-500, -250, 0, 250, 500], labels=['-500', '', '0', '', '500'])
        ax.set_zticks([0, 200, 400, 600, 800, 1100], labels=['0', '', '', '', '', '1100'])
    ax.set_xlabel(r"Position North")
    ax.set_ylabel(r"Position East")
    ax.set_zlabel(r"Altitude")
    ax.zaxis.set_label_coords(0.5, -0.1)
    ax.set_aspect('equal')

    bg_color = "#e6ecf2"
    ax.xaxis.set_pane_color(plt.matplotlib.colors.to_rgba(bg_color))
    ax.yaxis.set_pane_color(plt.matplotlib.colors.to_rgba(bg_color))
    ax.zaxis.set_pane_color(plt.matplotlib.colors.to_rgba(bg_color))
    ax.xaxis._axinfo["grid"]['color'] =  (1, 1, 1, 1.)  # white, with alpha
    ax.yaxis._axinfo["grid"]['color'] =  (1, 1, 1, 1.)
    ax.zaxis._axinfo["grid"]['color'] =  (1, 1, 1, 1.)

    ## CLOSE UP PLOT

    ax = fig.add_subplot(gs[0, 2], projection='3d')
    # ax.view_init(elev=30, azim=60)

    plot_mesh(ax, mesh, pos=position, euler_angles=euler_angles, scale=scale, alpha=1.0, set_follow_cam=True)
    draw_targets(ax, setting_type=setting_type, alpha=0.2, draw_obstacles=False)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    span = 30 * 2
    ax.set_xlim(position[0] - span/2, position[0] + span/2)
    ax.set_ylim(position[1] - span/2, position[1] + span/2)
    ax.set_zlim(position[2] - span/2, position[2] + span/2)

    bg_color = "#e6ecf2"
    ax.xaxis.set_pane_color(plt.matplotlib.colors.to_rgba(bg_color))
    ax.yaxis.set_pane_color(plt.matplotlib.colors.to_rgba(bg_color))
    ax.zaxis.set_pane_color(plt.matplotlib.colors.to_rgba(bg_color))
    ax.xaxis._axinfo["grid"]['color'] =  (1, 1, 1, 1)  # white, with alpha
    ax.yaxis._axinfo["grid"]['color'] =  (1, 1, 1, 1)
    ax.zaxis._axinfo["grid"]['color'] =  (1, 1, 1, 1)

    ax.set_title('Body')
    ax.set_aspect('equal')

    ## FINAL TOUCHES ##
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.subplots_adjust(wspace=0.5)
    plt.suptitle(r'$\mathtt{F16}$ — RR — $\mathbf{DO\text{-}HJ\text{-}PPO}$', fontsize=20, y=0.95)
    plt.savefig(fig_file_name, dpi=500)
