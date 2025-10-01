import os
import sys
import re
import json
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
from rraa_rl.src.rl.utils.arguments import get_args
from rraa_rl.src.env.env_list import get_env
import imageio

from brax.io import html
# from brax.v1.io import html as html1

def load_traj(file_path):
    with jnp.load(file_path, allow_pickle=False) as traj_data:
        traj_batch = {key: traj_data[key] for key in traj_data.files}
    return traj_batch

HTML_ANIMATION_SCRIPT = """
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

        // Target
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
        
        // Obstacles
        const redWall = new THREE.Mesh(
          new THREE.PlaneGeometry(5, 5),  // width (y) × height (z)
          new THREE.MeshStandardMaterial({
            color: 0xff0000,
            transparent: true,
            opacity: 0.5,
            side: THREE.DoubleSide,
            metalness: 0.0,
            roughness: 0.5,
          })
        );
        redWall.rotation.y = Math.PI / 2;  // rotate to face x-direction
        redWall.position.set(1.1, 0.0, 3.);  // center at x=2.2, z=2.5 (mid height)
        redWall.receiveShadow = true;
        viewer.scene.add(redWall);

        const redWall2 = new THREE.Mesh(
          new THREE.PlaneGeometry(5, 5),  // width (y) × height (z)
          new THREE.MeshStandardMaterial({
            color: 0xff0000,
            transparent: true,
            opacity: 0.5,
            side: THREE.DoubleSide,
            metalness: 0.0,
            roughness: 0.5,
          })
        );
        redWall2.rotation.y = Math.PI / 2;  // rotate to face x-direction
        redWall2.position.set(-1.1, 0.0, 3.);  // center at x=2.2, z=2.5 (mid height)
        redWall2.receiveShadow = true;
        viewer.scene.add(redWall2);

        const redFloor = new THREE.Mesh(
          new THREE.PlaneGeometry(2.2, 5),  // width (y) × height (z)
          new THREE.MeshStandardMaterial({
            color: 0xff0000,
            transparent: true,
            opacity: 0.5,
            side: THREE.DoubleSide,
            metalness: 0.0,
            roughness: 0.5,
          })
        );
        redFloor.rotation.y = Math.PI;  // rotate to face x-direction
        redFloor.position.set(0.0, 0.0, 0.5);  // center at x=2.2, z=2.5 (mid height)
        redFloor.receiveShadow = true;
        viewer.scene.add(redFloor);

        const redBox = new THREE.Mesh(
        new THREE.BoxGeometry(0.1, 1, 0.5),  // size = 2 × radius = 0.2
        new THREE.MeshStandardMaterial({
            color: 0xff0000,
            transparent: true,
            opacity: 0.5,
            metalness: 0.0,
            roughness: 0.5,
          })
        );
        redBox.position.set(0.0, 0.0, 1.6);
        redBox.castShadow = true;
        redBox.receiveShadow = true;
        viewer.scene.add(redBox);

        clearInterval(interval);
        }
    }, 100);
    </script>
""".strip()

def inject_custom_script(html_str, custom_script):
    pattern = r'<script type="module">.*?</script>'

    modified_html = re.sub(pattern, custom_script, html_str, flags=re.DOTALL)
    return modified_html

if __name__ == "__main__":

    ## INIT
    draw_gif = False
    sample_index = 4
    config = vars(get_args(sys.argv[1:]))
    config["EXP_NAME"]="HopperReachReach"
    config["PROBLEM_TYPE"]="RAA"
    config["ALG"]="DOHJPPO" # DOHJPPO, CPPO, DSTL
    fig_file_name = f"render/gifs/hopper_RAA_trajectory_render_{config['ALG']}_{sample_index}_view0_{np.random.randint(100000):2d}" #FIXME randint for touch bug
    traj_batch = load_traj(f"model/eval_all_figs/Hopper_RAA_061925/traj_sample/traj_{config['ALG']}.npz")

    envs = get_env(config)
    env = envs[0]

    ## Compute important points

    reached = traj_batch['reach'][:, sample_index] < 0.
    crashed = traj_batch['avoid'][:, sample_index] > 0.
    first_reached = np.where(reached)[0][0] if np.any(reached) else traj_batch['obs'].shape[0] - 1
    first_crashed = np.where(crashed)[0][0] if np.any(crashed) else traj_batch['obs'].shape[0] - 1
    final_index = (traj_batch['obs'].shape[0] - 1)
    final_index = np.min([first_reached + 200, final_index]) if first_crashed == final_index else first_crashed

    ## Render single step

    # step_index = 0
    # test_obs = traj_batch['obs'][step_index, sample_index, :]
    # test_obs[1] = test_obs[1] - 1.25
    # qpos = test_obs[:6]
    # qvel = test_obs[6:12]
    # init_pipeline_state = env._env._env.env.env.pipeline_init(qpos, qvel)
    
    # html_str = html.render(
    #     sys=env._env._env.env.env.sys,
    #     states=[init_pipeline_state],
    #     height=720,  # optional
    # )

    ## Render trajectory snapshot

    # renderer = html1.Renderer(env._env._env.env.env.sys)

    num_bodies_approx = 10
    interval = final_index // num_bodies_approx
    interval = 1 # FIXME DEBUG

    # default_color = env.sys.link_color
    # default_r, default_g, default_b = default_color[0][0], default_color[0][1], default_color[0][2]
    
    # qpos_list, qvel_list = [], []
    pipeline_states = []
    steps_saved = []
    reached_list, crashed_list, alpha_list = [], [], []
    for step_i in range(traj_batch['obs'].shape[0]):
        
        at_interval = step_i % interval == 0
        if not (at_interval or step_i in [first_reached, first_crashed, final_index]) and step_i != 1:
            continue
        if step_i == 0: # BUG: bad rendering, plotting step 1 instead
            continue
        steps_saved.append(step_i)

        test_obs = traj_batch['obs'][step_i, sample_index, :]
        test_obs[1] = test_obs[1] - 1.25
        qpos = test_obs[:6]
        qvel = test_obs[6:12]

        pipeline_state = env._env._env.env.env.pipeline_init(qpos, qvel)
        pipeline_states.append(pipeline_state)

        z_vals = pipeline_state.x.pos[:, 2]  # z is at index 2
        if (z_vals < 0).any():
            print(f"Frame {step_i}: Negative Z detected in body positions: {z_vals}")
        
        ## alpha and color
        alpha_range = 0.8  # range for alpha values
        # alpha_i = 1.0 - alpha_range * (step_i / first_reached_both) # decreasing
        alpha_i = (1 - alpha_range) + alpha_range * (step_i / final_index) # increasing
        alpha_list.append(alpha_i)

        # color = (default_r, default_g, default_b, alpha_i)  # blue with decreasing alpha
        if step_i == first_reached and not step_i == traj_batch['obs'].shape[0] - 1: 
            # color = (0.0, 1.0, 0.0, 1.0)  # green with full alpha
            reached_list.append(1)
        else:
            reached_list.append(0)
          
        if step_i == first_crashed and not step_i == traj_batch['obs'].shape[0] - 1:
            crashed_list.append(1)
            # color = (0.0, 0.0, 1.0, 1.0)  # red with full alpha
        else:
            crashed_list.append(0)

        # renderer.add(pipeline_state, color=color)
        if step_i == final_index:
            break

    os.makedirs(f"render/hopper_{config['PROBLEM_TYPE']}_snapshots_{config['ALG']}_seed{sample_index}", exist_ok=True)
    for i, state_i in enumerate(pipeline_states):
        html_str = html.render(
            sys=env._env._env.env.env.sys,
            states=[state_i],
            height=720,  # optional
        )
        html_str = inject_custom_script(html_str, HTML_ANIMATION_SCRIPT)
        with open(f"render/hopper_{config['PROBLEM_TYPE']}_snapshots_{config['ALG']}_seed{sample_index}/snap_{i}.html", "w") as f:
            f.write(html_str)

    # html_str = html.render(
    #     sys=env._env._env.env.env.sys,
    #     states=[init_pipeline_state],
    #     height=720,
    # )
    # Serialize trajectory as JSON (for injection into HTML)
    
    # traj_json = json.dumps([
    #     {'qpos': qpos, 'qvel': qvel}
    #     for qpos, qvel in zip(qpos_list, qvel_list)
    # ])
    
    # qpos_json = json.dumps([q.tolist() for q in qpos_list])
    # qvel_json = json.dumps([v.tolist() for v in qvel_list])

    # poses_json = json.dumps([state.x.pos.tolist() for state in pipeline_states])
    # rotes_json = json.dumps([state.x.rot.tolist() for state in pipeline_states])

    # reached_json = json.dumps([
    #     {'reached1': reached1, 'reached2': reached2}
    #     for reached1, reached2 in zip(reached_1_list, reached_2_list)
    # ])

    # alpha_json = json.dumps([
    #     {'alpha': alpha}
    #     for alpha in alpha_list
    # ])

    # html_str = html1.render(renderer.scene)

    ## Inject trajectory data into script

    # Inject custom <script type="module"> block
    # html_snapshot_script = f"""
    # <script type="module">
    #   import * as THREE from 'three';
    #   import {{ Viewer }} from 'viewer';

    #   const domElement = document.getElementById("brax-viewer");
    #   const viewer = new Viewer(domElement, system);

    #   const poses = {poses_json};
    #   const rotes = {rotes_json};
    #   const reachs = {reached_json};
    #   const alphas = {alpha_json};

    #   const interval = setInterval(() => {{
    #     if (!viewer.scene || !viewer.renderer || !viewer.camera || !viewer.controls) return;

    #     // Step 1: Locate original body parts by filtering visible mesh objects
    #     const originalBodyMeshes = viewer.scene.children.filter(obj =>
    #       obj.type === 'Mesh' && obj.name !== 'ground' && obj.visible
    #     );
    #     console.log('originalBodyMeshes', originalBodyMeshes)

    #     // Step 2: Create transparent clones for each timestep
    #     for (let t = 0; t < poses.length; t++) {{
    #       const pose_t = poses[t];
    #       const rot_t = rotes[t];

    #       for (let j = 0; j < pose_t.length; j++) {{
    #         if (!originalBodyMeshes[j]) continue;

    #         const clone = originalBodyMeshes[j].clone();
    #         clone.position.fromArray(pose_t[j]);
    #         clone.quaternion.fromArray(rot_t[j]);

    #         // Set transparent material
    #         clone.material = clone.material.clone();
    #         clone.material.transparent = true;
    #         clone.material.opacity = alphas[t];  // adjust for fading effect

    #         // Optional: prevent z-fighting / sorting issues
    #         clone.renderOrder = t;

    #         viewer.scene.add(clone);
    #       }}
    #     }}

    #     clearInterval(interval);
    #   }}, 100);
    # </script>
    # """

    # Inject script into the HTML before closing </body>
    # html_with_traj = html_str.replace("</body>", custom_script + "\n</body>")

    # html_str = inject_custom_script(html_str, html_snapshot_script)

    # with open("render/hopper_snapshot_test.html", "w") as f:
    #     f.write(html_str)

    if draw_gif:
        
        pipeline_states = []
        for step_i in range(traj_batch['obs'].shape[0]):

            test_obs = traj_batch['obs'][step_i, sample_index, :]
            test_obs[1] = test_obs[1] - 1.25
            qpos = test_obs[:6]
            qvel = test_obs[6:12]
            pipeline_state = env._env._env.env.env.pipeline_init(qpos, qvel)
            pipeline_states.append(pipeline_state)

            if step_i == final_index:
                break

        html_str = html.render(
            sys=env._env._env.env.env.sys,
            states=pipeline_states,
            height=720,  # optional
        )
        html_str = inject_custom_script(html_str, HTML_ANIMATION_SCRIPT)
        with open("render/hopper_render_anim_RAA.html", "w") as f:
            f.write(html_str)
