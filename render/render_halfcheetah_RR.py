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
      viewer.camera.position.set(0, 15., 0.8);
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
        blueSphere.position.set(-5.1, 0.0, 1.7);
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
        greenSphere.position.set(5.1, 0.0, 1.7);
        greenSphere.castShadow = true;
        greenSphere.receiveShadow = true;
        viewer.scene.add(greenSphere);

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
    # sample_index = 0
    samples = 10
    config = vars(get_args(sys.argv[1:]))
    config["EXP_NAME"]="HalfCheetahReachReach"
    config["PROBLEM_TYPE"]="RR"
    config["ALG"]="DOHJPPO_s" # DOHJPPO, CPPOvI, DSTL (non HJ -> PROBLEM_TYPE = RR)

    if config["PROBLEM_TYPE"] == "RR":
      # traj_batch = load_traj(f"eval/eval_all_080725/HalfCheetah_RR/traj_sample/traj_{config['ALG']}.npz")
      traj_batch = load_traj(f"eval/eval_all/HalfCheetah_RR/traj_sample/traj_{config['ALG']}.npz")
    elif config["PROBLEM_TYPE"] == "R1":
      traj_batch = load_traj(f"eval/eval_all/HalfCheetah_RR/traj_sample/traj_{config['ALG']}_reach1.npz")
    elif config["PROBLEM_TYPE"] == "R2":
      traj_batch = load_traj(f"eval/eval_all/HalfCheetah_RR/traj_sample/traj_{config['ALG']}_reach2.npz")
    else:
      raise ValueError(f"Unknown problem type: {config['PROBLEM_TYPE']}")

    config["RENDER_DIR"] = f"render/html/halfcheetah_{config['PROBLEM_TYPE']}_{config['ALG']}"

    envs = get_env(config)
    if config["PROBLEM_TYPE"] == "RR":
      env = envs[0]
    elif config["PROBLEM_TYPE"] == "R1":
      env = envs[1]
    elif config["PROBLEM_TYPE"] == "R2":
      env = envs[2]
    else:
      raise ValueError(f"Unknown problem type: {config['PROBLEM_TYPE']}")

    sample_range = [0,1,2]
    # sample_range = np.arange(traj_batch['obs'].shape[1])
    # sample_range = np.arange(samples)

    for sample_index in sample_range:
      
      print(f"Rendering sample {sample_index} of {sample_range}...")
      fig_file_name = f"render/gifs/halfcheetah_RR_trajectory_render_{config['ALG']}_{sample_index}_view0_{np.random.randint(100000):2d}" #FIXME randint for touch bug

      ## Compute important points

      if config["PROBLEM_TYPE"] == "RR":
        reached_1 = traj_batch['reach1'][:, sample_index] < 0.
        reached_2 = traj_batch['reach2'][:, sample_index] < 0.
      elif config["PROBLEM_TYPE"] == "R1":
        reached_1 = traj_batch['reach1'][:, sample_index] < 0.
        reached_2 = jnp.zeros_like(reached_1, dtype=jnp.bool_)
      elif config["PROBLEM_TYPE"] == "R2":
        reached_2 = traj_batch['reach2'][:, sample_index] < 0.
        reached_1 = jnp.zeros_like(reached_2, dtype=jnp.bool_)
      else:
        raise ValueError(f"Unknown problem type: {config['PROBLEM_TYPE']}")

      first_reached_1 = np.where(reached_1)[0][0] if np.any(reached_1) else traj_batch['obs'].shape[0] - 1
      first_reached_2 = np.where(reached_2)[0][0] if np.any(reached_2) else traj_batch['obs'].shape[0] - 1
      first_reached_both = np.max([first_reached_1, first_reached_2])

      ## Render trajectory snapshot

      if draw_snapshots:

        num_bodies_approx = 10
        interval = first_reached_both // num_bodies_approx if first_reached_both > num_bodies_approx else 1

        pipeline_states = []
        steps_saved = []
        reached_1_list, reached_2_list, alpha_list = [], [], []
        for step_i in range(traj_batch['obs'].shape[0]):
            
            at_interval = step_i % interval == 0
            if not (at_interval or step_i in [first_reached_1, first_reached_2, first_reached_both]) and step_i != 1:
                continue
            if step_i == 0: # bad rendering
                continue
            steps_saved.append(step_i)

            test_obs = traj_batch['obs'][step_i, sample_index, :]
            test_obs = env.untransform_obs(test_obs)
            # test_obs[1] = test_obs[1] - 1.25
            qpos = test_obs[:9]
            qvel = test_obs[9:18]

            pipeline_state = env._env._env.env.env.pipeline_init(qpos, qvel)
            pipeline_states.append(pipeline_state)
            
            ## alpha and color
            alpha_range = 0.8  # range for alpha values
            # alpha_i = 1.0 - alpha_range * (step_i / first_reached_both) # decreasing
            alpha_i = (1 - alpha_range) + alpha_range * (step_i / first_reached_both) # increasing
            alpha_list.append(alpha_i)

            # color = (default_r, default_g, default_b, alpha_i)  # blue with decreasing alpha
            if step_i == first_reached_1: 
                # color = (0.0, 1.0, 0.0, 1.0)  # green with full alpha
                reached_1_list.append(1)
            else:
                reached_1_list.append(0)
              
            if step_i == first_reached_2:
                reached_2_list.append(1)
                # color = (0.0, 0.0, 1.0, 1.0)  # red with full alpha
            else:
                reached_2_list.append(0)

            # renderer.add(pipeline_state, color=color)
            if step_i == first_reached_both:
                break

        os.makedirs(f"{config['RENDER_DIR']}_snapshots_{config['ALG']}_seed{sample_index}", exist_ok=True)
        for i, state_i in enumerate(pipeline_states):
            html_str = html.render(
                sys=env._env._env.env.env.sys,
                states=[state_i],
                height=720,  # optional
            )
            html_str = inject_custom_script(html_str, HTML_ANIMATION_SCRIPT)
            with open(f"{config['RENDER_DIR']}_snapshots_{config['ALG']}_seed{sample_index}/snap_{i}.html", "w") as f:
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

              if step_i == first_reached_both:
                  break

          html_str = html.render(
              sys=env._env._env.env.env.sys,
              states=pipeline_states,
              height=900,  # optional
          )
          html_str = inject_custom_script(html_str, HTML_ANIMATION_SCRIPT)
          with open(f"{config['RENDER_DIR']}_anim_seed{sample_index}.html", "w") as f:
              f.write(html_str)
