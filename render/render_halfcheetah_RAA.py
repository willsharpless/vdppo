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
from rraa_rl.EFPPO.src.rl.arguments import get_args
from rraa_rl.EFPPO.src.env.env_list import get_env
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
      viewer.camera.position.set(3.5, 6.5, 0.8);
      viewer.controls.target.set(3.5, 0, 0.8);
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
        const greenTarget = new THREE.Mesh(
        new THREE.BoxGeometry(0.5, 1, 8),  // size = 2 × radius = 0.2
        new THREE.MeshStandardMaterial({
            color: 0x00ff00,
            transparent: true,
            opacity: 0.4,
            metalness: 0.0,
            roughness: 0.5,
          })
        );
        greenTarget.position.set(5.5, 0.0, 0.0);
        greenTarget.castShadow = true;
        greenTarget.receiveShadow = true;
        viewer.scene.add(greenTarget);

        // Obstacles
        const redBox = new THREE.Mesh(
        new THREE.BoxGeometry(0.1, 1, 1),  // size = 2 × radius = 0.2
        new THREE.MeshStandardMaterial({
            color: 0xff0000,
            transparent: true,
            opacity: 0.5,
            metalness: 0.0,
            roughness: 0.5,
          })
        );
        redBox.position.set(-0.2, 0.0, 0.5);
        redBox.castShadow = true;
        redBox.receiveShadow = true;
        viewer.scene.add(redBox);

        const redFloor1 = new THREE.Mesh(
        new THREE.BoxGeometry(0.5, 1, 0.05),  // size = 2 × radius = 0.2
        new THREE.MeshStandardMaterial({
            color: 0xff0000,
            transparent: true,
            opacity: 0.5,
            metalness: 0.0,
            roughness: 0.5,
          })
        );
        redFloor1.position.set(4.5, 0.0, 0.0);
        redFloor1.castShadow = true;
        redFloor1.receiveShadow = true;
        viewer.scene.add(redFloor1);

        const redFloor2 = new THREE.Mesh(
        new THREE.BoxGeometry(0.5, 1, 0.05),  // size = 2 × radius = 0.2
        new THREE.MeshStandardMaterial({
            color: 0xff0000,
            transparent: true,
            opacity: 0.5,
            metalness: 0.0,
            roughness: 0.5,
          })
        );
        redFloor2.position.set(6.5, 0.0, 0.0);
        redFloor2.castShadow = true;
        redFloor2.receiveShadow = true;
        viewer.scene.add(redFloor2);

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
    draw_gif = True
    draw_snapshots = False
    config = vars(get_args(sys.argv[1:]))
    config["EXP_NAME"]="HalfCheetahReachReach"
    config["PROBLEM_TYPE"]="RAA"
    config["ALG"]="DOHJPPO" # DOHJPPO, CPPO, DSTL
    traj_batch = load_traj(f"eval/eval_all_080725/HalfCheetah_RAA/traj_sample/traj_{config['ALG']}.npz")

    envs = get_env(config)
    env = envs[0]

    sample_range = [2,3,4,5,6,7,8,9]
    # sample_range = np.arange(traj_batch['obs'].shape[1])
    # samples = 10
    # sample_range = np.arange(samples)

    for sample_index in sample_range:
      
      print(f"Rendering sample {sample_index} of {sample_range}...")
      fig_file_name = f"render/gifs/halfcheetah_RAA_trajectory_render_{config['ALG']}_seed{sample_index}_{np.random.randint(100000):2d}" #FIXME randint for touch bug

      ## Compute important points

      reached = traj_batch['reach'][:, sample_index] < 0.
      crashed = traj_batch['avoid'][:, sample_index] > 0.
      first_reached = np.where(reached)[0][0] if np.any(reached) else traj_batch['obs'].shape[0] - 1
      first_crashed = np.where(crashed)[0][0] if np.any(crashed) else traj_batch['obs'].shape[0] - 1
      final_index = (traj_batch['obs'].shape[0] - 1)
      final_index = np.min([first_reached + 200, final_index]) if first_crashed == final_index else first_crashed

      ## Render trajectory snapshot

      # renderer = html1.Renderer(env._env._env.env.env.sys)
      if draw_snapshots:
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
            test_obs = env.untransform_obs(test_obs) # untransform obs to original scale
            qpos = test_obs[:9]
            qvel = test_obs[9:18]

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

        os.makedirs(f"render/halfcheetah_{config['PROBLEM_TYPE']}_snapshots_{config['ALG']}_seed{sample_index}", exist_ok=True)
        for i, state_i in enumerate(pipeline_states):
            html_str = html.render(
                sys=env._env._env.env.env.sys,
                states=[state_i],
                height=720,  # optional
            )
            html_str = inject_custom_script(html_str, HTML_ANIMATION_SCRIPT)
            with open(f"render/halfcheetah_{config['PROBLEM_TYPE']}_snapshots_{config['ALG']}_seed{sample_index}/snap_{i}.html", "w") as f:
                f.write(html_str)

      if draw_gif:
          
          pipeline_states = []
          for step_i in range(traj_batch['obs'].shape[0]):

              test_obs = traj_batch['obs'][step_i, sample_index, :]
              test_obs = env.untransform_obs(test_obs) # untransform obs to original scale
              qpos = test_obs[:9]
              qvel = test_obs[9:18]
              pipeline_state = env._env._env.env.env.pipeline_init(qpos, qvel)
              pipeline_states.append(pipeline_state)

              if step_i == final_index:
                  break

          html_str = html.render(
              sys=env._env._env.env.env.sys,
              states=pipeline_states,
              height=900,  # optional
          )
          html_str = inject_custom_script(html_str, HTML_ANIMATION_SCRIPT)
          with open(f"render/halfcheetah_render_anim_RAA_seed{sample_index}.html", "w") as f:
              f.write(html_str)
