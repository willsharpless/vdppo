import os
import sys
import re
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
from brax.v1.io import html as html1

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

def inject_custom_script(html_str):
    pattern = r'<script type="module">.*?</script>'

    custom_script = """
    <script type="module">
    import * as THREE from 'three';
    import { Viewer } from 'viewer';

    const domElement = document.getElementById("brax-viewer");
    const viewer = new Viewer(domElement, system);

    viewer.renderer.setPixelRatio(2);
    viewer.renderer.setSize(viewer.renderer.domElement.width, viewer.renderer.domElement.height, false);
    
    function lockCameraView() {
      if (!viewer || !viewer.camera || !viewer.controls) return;

      // Custom position and focus point
      viewer.camera.position.set(0, 2.5, 0.8);
      viewer.controls.target.set(0, 0, 0.8);
      viewer.controls.update();

      // Disable interactivity
      viewer.controls.enabled = false;

      // Re-apply every frame to prevent Brax from resetting
      requestAnimationFrame(lockCameraView);
    }

    const interval = setInterval(() => {

        // Hide lil-gui controls
        if (viewer.gui) {
          document.querySelectorAll('.lil-gui').forEach(el => {
            el.style.display = 'none';
          });
          clearInterval(interval);
        }

        // Lock camera view
        if (viewer.scene && viewer.camera && viewer.controls) {
          lockCameraView();  // Start the persistent enforcement loop
          clearInterval(interval);
        }

        // Enforce animator settings
        const controllers = [...document.querySelectorAll("div.controller")];
        let followTargetSet = false;
        let timeScaleSet = false;

        controllers.forEach(div => {
          const label = div.querySelector(".name");
          if (!label) return;
          const labelText = label.textContent.trim();

          // Follow Target (checkbox)
          if (labelText === "Follow Target") {
            const checkbox = div.querySelector("input[type='checkbox']");
            if (checkbox && checkbox.checked) {
              checkbox.click();  // uncheck it
              followTargetSet = true;
            }
          }

          // timeScale (input numer)
          if (labelText === "timeScale") {
            const numberInput = div.querySelector("input[type='number']");
            if (numberInput && parseFloat(numberInput.value) !== 0.1) {
              numberInput.value = 0.1;
              numberInput.dispatchEvent(new Event("input", { bubbles: true }));
              numberInput.dispatchEvent(new Event("change", { bubbles: true }));
            }
          }
        });
        clearInterval(interval);
        
        // Set checker texture on Brax ground mesh
        if (viewer.scene && viewer.renderer) {
        const world = viewer.scene.getObjectByName('world');
        if (world && world.children.length > 0) {

            const ground = world.children[0];
            ground.scale.set(100, 100, 100);
            ground.visible = true;
            ground.receiveShadow = true;

            const size = 512;
            const squares = 256;
            const canvas = document.createElement('canvas');
            canvas.width = canvas.height = size;
            const ctx = canvas.getContext('2d');
            const squareSize = size / squares;

            for (let y = 0; y < squares; y++) {
            for (let x = 0; x < squares; x++) {
                ctx.fillStyle = (x + y) % 2 === 0 ? '#ffffff' : '#cccccc';
                ctx.fillRect(x * squareSize, y * squareSize, squareSize, squareSize);
            }
            }

            const texture = new THREE.CanvasTexture(canvas);
            texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
            texture.repeat.set(10, 10);
            texture.minFilter = THREE.LinearFilter;
            texture.magFilter = THREE.NearestFilter;
            texture.anisotropy = viewer.renderer.capabilities.getMaxAnisotropy();
            texture.encoding = THREE.sRGBEncoding;
            texture.needsUpdate = true;
            viewer.renderer.outputEncoding = THREE.sRGBEncoding;

            ground.material = new THREE.MeshStandardMaterial({
            map: texture,
            color: 0xffffff,
            metalness: 0.0,
            roughness: 1.0,
            side: THREE.DoubleSide,
            });
        }

        // Sphere 1: Blue
        const blueSphere = new THREE.Mesh(
          new THREE.SphereGeometry(0.1, 32, 32),
          new THREE.MeshStandardMaterial({
            color: 0x0000ff,         // Blue
            transparent: true,
            opacity: 0.6,
            metalness: 0.0,
            roughness: 0.5,
          })
        );
        blueSphere.position.set(-1.0, 0.0, 1.4);
        blueSphere.castShadow = true;
        blueSphere.receiveShadow = true;
        viewer.scene.add(blueSphere);

        // Sphere 2: Green
        const greenSphere = new THREE.Mesh(
          new THREE.SphereGeometry(0.1, 32, 32),
          new THREE.MeshStandardMaterial({
            color: 0x00ff00,         // Green
            transparent: true,
            opacity: 0.6,
            metalness: 0.0,
            roughness: 0.5,
          })
        );
        greenSphere.position.set(1.0, 0.0, 1.4);
        greenSphere.castShadow = true;
        greenSphere.receiveShadow = true;
        viewer.scene.add(greenSphere);

        clearInterval(interval);
        }
    }, 100);
    </script>
""".strip()

    modified_html = re.sub(pattern, custom_script, html_str, flags=re.DOTALL)
    return modified_html

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

    ## Render single step

    # step_index = 0
    # test_obs = traj_batch['obs'][step_index, sample_index, :]
    # test_obs[1] = test_obs[1] - 1.25
    # qpos = test_obs[:6]
    # qvel = test_obs[6:12]
    # pipeline_state = env._env._env.env.env.pipeline_init(qpos, qvel)
    
    # html_str = html.render(
    #     sys=env._env._env.env.env.sys,
    #     states=[pipeline_state],
    #     height=720,  # optional
    # )

    ## Render trajectory snapshot

    renderer = html1.Renderer(env._env._env.env.env.sys)

    reached_both = traj_batch['has_reached_1'][:, sample_index] * traj_batch['has_reached_2'][:, sample_index]
    first_reached_both = np.where(reached_both)[0] if np.any(reached_both) else traj_batch['obs'].shape[0] - 1
    interval = first_reached_both // 5 if first_reached_both > 5 else 1

    default_color = env.sys.link_color
    default_r, default_g, default_b = default_color[0][0], default_color[0][1], default_color[0][2]

    for step_i in range(traj_batch['obs'].shape[0]):
        
        at_interval = step_i % interval == 0 or step_i == first_reached_both
        if not at_interval or traj_batch['has_reached_1'][step_i, sample_index] or traj_batch['has_reached_2'][step_i, sample_index]:
            continue

        test_obs = traj_batch['obs'][step_i, sample_index, :]
        test_obs[1] = test_obs[1] - 1.25
        qpos = test_obs[:6]
        qvel = test_obs[6:12]
        pipeline_state = env._env._env.env.env.pipeline_init(qpos, qvel)
        # pipeline_states.append(pipeline_state)
        
        alpha_range = 0.2  # range for alpha values
        # alpha_i = 1.0 - alpha_range * (step_i / first_reached_both) # decreasing
        alpha_i = (1 - alpha_range) + alpha_range * (step_i / first_reached_both) # increasing

        color = (default_r, default_g, default_b, alpha_i)  # blue with decreasing alpha
        if traj_batch['has_reached_1'][step_i, sample_index]: 
            color = (0.0, 1.0, 0.0, 1.0)  # green with full alpha
        elif traj_batch['has_reached_2'][step_i, sample_index]:
            color = (0.0, 0.0, 1.0, 1.0)  # red with full alpha

        renderer.add(pipeline_state, color=color)

        if traj_batch['has_reached_1'][step_i, sample_index] and traj_batch['has_reached_2'][step_i, sample_index]:
            break

    html_str = html1.render(renderer.scene)
    html_str = inject_custom_script(html_str)

    with open("render/hopper_render_test.html", "w") as f:
        f.write(html_str)

    if draw_gif:
        
        pipeline_states = []
        for step_i in range(traj_batch['obs'].shape[0]):

            test_obs = traj_batch['obs'][step_i, sample_index, :]
            test_obs[1] = test_obs[1] - 1.25
            qpos = test_obs[:6]
            qvel = test_obs[6:12]
            pipeline_state = env._env._env.env.env.pipeline_init(qpos, qvel)
            pipeline_states.append(pipeline_state)

            if traj_batch['has_reached_1'][step_i, sample_index] and traj_batch['has_reached_2'][step_i, sample_index]:
                break

        html_str = html.render(
            sys=env._env._env.env.env.sys,
            states=pipeline_states,
            height=720,  # optional
        )
        html_str = inject_custom_script(html_str)
        with open("render/hopper_render_anim_RR.html", "w") as f:
            f.write(html_str)
