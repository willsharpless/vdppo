import pathlib

import cyclopts
import ipdb
import jax
import jax.random as jr
import mujoco
import numpy as np
import tqdm
from loguru import logger
from PIL import Image

from vdppo.training.collector import Collector
from vdppo.env.scene import ManipScene
from vdppo.training.load_ckpt import load_ckpt
from vdppo.training.rollout_temporal_analysis import evaluate_ltl_finite
from vdppo.training.rollout_utils import extract_rollouts_eval
from vdppo.callbacks.scene_cbs import animate_scene_traj
from vdppo.agents.vdppo import VDPPOAgent

app = cyclopts.App()


@app.default()
def main(run_path: pathlib.Path, n_env: int = 16, step: int | None = None):
    run, agent, env, cfg_dict = load_ckpt(run_path, step)

    env: ManipScene
    env.base.cfg.p_reset_data_clean = 0.0
    env.base.cfg.p_reset_data_noisy = 0.0

    if hasattr(env.base, "n_envs"):
        env.base.n_envs = agent.cfg.n_envs_train

    collector = Collector.create(
        key=jr.PRNGKey(1234),
        env=env,
        cfg=Collector.Cfg(n_envs=n_env, auto_reset=False, ignore_trunc=True),
    )
    b_state0 = env.get_eval_states(collector.cfg.n_envs, root_only=True)

    collect_opts = {}
    if isinstance(agent, VDPPOAgent):
        collect_opts["temporal_transitions"] = True

    Tb_rollout, info_collect = agent.collect_eval_with_states(collector, b_state0, env.eval_T, **collect_opts)
    Tb_rollout = jax.device_get(Tb_rollout)
    bT_rollout = Tb_rollout.switch01()

    # Extract each rollout
    b_trajs = extract_rollouts_eval(bT_rollout)

    # Animate all trajectories.
    anim_dir = run_path / "eval"
    anim_dir.mkdir(exist_ok=True)

    b_dag_values = []
    for ii, traj in enumerate(b_trajs):
        # Compute whether the DAG values.
        dag_value = evaluate_ltl_finite(env, traj.predicates_next, which=np)[env.dag_root]
        b_dag_values.append(dag_value)
    b_dag_values = np.array(b_dag_values)
    logger.info(
        "b_dag_values | min: {} | mean: {} | max: {}", b_dag_values.min(), b_dag_values.mean(), b_dag_values.max()
    )
    b_success = b_dag_values >= 0.1
    logger.info("Success rate: {}/{} ({:.2%})", b_success.sum(), len(b_success), b_success.mean())

    pbar = tqdm.tqdm(b_trajs)
    for ii, traj in enumerate(pbar):
        T_state: ManipScene.MinState = traj.state_now
        T_minstate = T_state.base
        T_predicates = traj.predicates_next

        T_labels = [
            f"Temporal {t_node_idx} ({env.temporal_node_names[t_node_idx]})" for t_node_idx in T_state.temporal_node_idx
        ]

        anim_path = anim_dir / f"eval_animation_{ii:02}.mp4"
        animate_scene_traj(anim_path, env.base, T_minstate, T_preds=T_predicates, T_labels=T_labels)

    # Prompt the user for two numbers. The first is the trajectory index. The second is the time step.
    # Save a high resolution screenshot of the environment at that time step.
    width = 3840
    height = 2160
    m = env.base.mj_model
    m.vis.global_.offwidth = width
    m.vis.global_.offheight = height
    renderer = mujoco.Renderer(model=m, height=height, width=width)
    camera = "front"

    d = mujoco.MjData(m)
    while True:
        user_inp = input("Enter trajectory index and time step (e.g., '0 10'), or 'q' to quit: ")
        if user_inp.lower() == "q":
            break

        try:
            traj_idx, time_step = map(int, user_inp.split())
            traj = b_trajs[traj_idx]
            T_state: ManipScene.MinState = traj.state_now
            T_minstate = T_state.base

            # Render high resolution image
            screenshot_path = anim_dir / f"screenshot_traj{traj_idx}_step{time_step}.png"

            d.qpos[:] = T_minstate.qpos[time_step]
            d.qvel[:] = T_minstate.qvel[time_step]
            mujoco.mj_forward(m, d)

            renderer.update_scene(data=d, camera=camera)
            img = renderer.render()
            pil_img = Image.fromarray(img)
            pil_img.save(screenshot_path)

            logger.info("Saved screenshot to {}".format(screenshot_path))
        except Exception as e:
            logger.error("Error processing input: {}".format(e))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
