import ipdb
from collections import defaultdict
import mujoco
import imageio
import matplotlib.pyplot as plt
import pickle
from typing import Protocol

import jax
from PIL import Image, ImageDraw, ImageFont
import jax.random as jr
import numpy as np
import tqdm
import yaml
from attrs import define
from cyclopts import Parameter
from loguru import logger

import wandb
from rraa_rl.cfg_utils import Cfg
from rraa_rl.collector import Collector, RolloutOutput, extract_info_from_rollout
from rraa_rl.envs.scene import ManipScene, SceneBaseMinState
from rraa_rl.lcrl_mappo import LCRLMAPPOAgent
from rraa_rl.rollout_temporal_analysis import evaluate_ltl_finite
from rraa_rl.rollout_utils import extract_rollouts_eval
from rraa_rl.run import Run
from rraa_rl.src.env.general_task.env import Env, StateWithTemporalNode
from rraa_rl.src.get_agent_cfg import get_vd_agent_cfg
from rraa_rl.vd_mappo import VDMAPPOAgent

def main():
    cfg = ManipScene.Cfg()
    env = ManipScene(cfg)

    n_envs_train = 16

    key_collector = jr.PRNGKey(0)
    collector = Collector.create(
        key=key_collector,
        env=env,
        cfg=Collector.Cfg(n_envs=n_envs_train, auto_reset=False),
    )
    agent_cfg = get_vd_agent_cfg("manip-scene")
    agent_cfg.actor_shared_trunk = True
    agent = VDMAPPOAgent.create(12345, agent_cfg, env)

    rollout_T = 100
    rollout: RolloutOutput
    collector, Tb_rollout, _ = agent.collect_batch(collector, rollout_T)
    Tb_rollout = jax.device_get(Tb_rollout)
    bT_rollout = Tb_rollout.switch01()

    # Extract each rollout
    trajs = extract_rollouts_eval(bT_rollout)

    T_max = 0

    for ii, traj in enumerate(trajs):
        T = len(traj.term)
        T_max = max(T_max, T)

    # Count how many predicates are 1 at each time step
    pred_dict = defaultdict(lambda: np.zeros((T_max,)))

    # Save a trajectory idx where each predicate is true at some point. 
    pred_true_dict = defaultdict(list)

    for ii, traj in enumerate(trajs):
        T_preds: dict[str, np.ndarray] = traj.predicates_next
        T = len(traj.term)
        for k, v in T_preds.items():
            for kk in range(T):
                if v[kk] > 0.5:
                    pred_dict[k][kk] += 1
            
            if np.any(v[:T] > 0.5):
                pred_true_dict[k].append(ii)
            
    pred_names = list(pred_dict.keys())

    nrow = len(pred_names)
    figsize = np.array([8, 2 * nrow])
    fig, axes = plt.subplots(nrow, 1, figsize=figsize, layout="constrained", sharex=True)
    for ii, ax in enumerate(axes):
        pname = pred_names[ii]
        ax.plot(pred_dict[pname])
        ax.set_title(pname)
    axes[-1].set_xlabel("Step")
    fig.savefig("manip_pred_count.pdf", bbox_inches="tight")

    # For each predicate, render the trajectory where it is true at some point.
    # Show:
    # - The current time step on the top left
    # - Status of each predicate on the top right of the video.
    env.base.mj_model.vis.global_.offwidth  = 1280
    env.base.mj_model.vis.global_.offheight = 720

    d = mujoco.MjData(env.base.mj_model)
    width, height = 1280, 720
    renderer = mujoco.Renderer(model=env.base.mj_model, height=height, width=width)
    camera = "front"

    for pname, traj_idxs in pred_true_dict.items():
        traj_idx = traj_idxs[0]
        traj = trajs[traj_idx]
        T = len(traj.term)

        T_state_now: StateWithTemporalNode[SceneBaseMinState] = traj.state_now
        T_qpos = T_state_now.base.qpos
        T_qvel = T_state_now.base.qvel
        T_preds: dict[str, np.ndarray] = traj.predicates_next

        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except IOError:
            font = ImageFont.load_default()

        vid_frames = []
        for kk in range(T):
            d.qpos[:] = T_qpos[kk]
            d.qvel[:] = T_qvel[kk]
            mujoco.mj_forward(env.base.mj_model, d)

            renderer.update_scene(data=d, camera=camera)
            img = renderer.render()

            # Convert to PIL Image for drawing text
            pil_img = Image.fromarray(img)
            draw = ImageDraw.Draw(pil_img)

            # Draw timestep on top left
            timestep_text = f"t = {kk}"
            draw.text((10, 10), timestep_text, fill=(255, 255, 255), font=font)

            # Draw predicate status on top right
            y_offset = 10
            for pred_name, pred_vals in T_preds.items():
                status = "✓" if pred_vals[kk] > 0.5 else "✗"
                color = (0, 255, 0) if pred_vals[kk] > 0.5 else (255, 0, 0)
                pred_text = f"{pred_name}: {status}"
                # Get text bounding box to right-align
                bbox = draw.textbbox((0, 0), pred_text, font=font)
                text_width = bbox[2] - bbox[0]
                draw.text((width - text_width - 10, y_offset), pred_text, fill=color, font=font)
                y_offset += 30

            vid_frames.append(np.array(pil_img))

        # Save the video frames as a video file
        video_path = f"manip_pred_{pname}.mp4"
        imageio.mimwrite(video_path, vid_frames, fps=30)
        logger.info(f"Saved video for predicate '{pname}' at {video_path}")


            

if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()