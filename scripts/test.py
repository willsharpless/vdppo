import pathlib
import pickle
from typing import Protocol

import cyclopts
import flax
import ipdb
import jax
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import yaml
from attrs import define
from loguru import logger
from matplotlib.colors import to_rgba

import wandb
from rraa_rl import herd_os_cbs
from rraa_rl.collector import Collector, RolloutOutput, extract_info_from_rollout
from rraa_rl.distribution import tfd
from rraa_rl.gridworld_cbs import save_animation_blit
from rraa_rl.rollout_temporal_analysis import evaluate_ltl_finite, evaluate_triggers
from rraa_rl.rollout_utils import extract_rollouts_eval
from rraa_rl.run import Run
from rraa_rl.src.env.general_task.env import StateWithTemporalNode
from rraa_rl.src.env.general_task.get_env import get_env_and_cbs
from rraa_rl.src.env.general_task.gridworld import GridworldMA, GridworldMAState
from rraa_rl.src.env.general_task.herd_os import HerdOs
from rraa_rl.trainer import Trainer
from rraa_rl.vd_mappo import VDMAPPOAgent

app = cyclopts.App()


@app.default()
def main(run_path: pathlib.Path, n_env: int = 256):
    # Load the configs.
    yaml_path = run_path / "config.yaml"
    with open(yaml_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    run = Run.fromdict(cfg_dict["run"])
    env_name = run.env_name
    agent_name = run.agent_name

    env: GridworldMA
    env, _, _ = get_env_and_cbs(env_name, agent_name=agent_name)

    agent_cfg = VDMAPPOAgent.Cfg.fromdict(cfg_dict["agent"])
    agent = VDMAPPOAgent.create(123, agent_cfg, env)

    ckpts_path = run_path / "ckpts"
    latest_ckpt = sorted(ckpts_path.glob("params_*.pkl"))
    assert latest_ckpt, f"No checkpoints found in {ckpts_path}"

    load_path = latest_ckpt[-1]
    logger.info(f"Restoring from {load_path}")

    with load_path.open("rb") as f:
        load_dict = pickle.load(f)

    agent: VDMAPPOAgent = flax.serialization.from_state_dict(agent, load_dict["agent"])

    logger.debug("Constructing collector_eval...")
    collector = Collector.create(
        key=jr.PRNGKey(1234),
        env=env,
        cfg=Collector.Cfg(n_envs=n_env, auto_reset=False, ignore_trunc=True),
    )
    b_state0 = env.get_eval_states(collector.cfg.n_envs, root_only=True)

    collect_opts = {}
    if isinstance(agent, VDMAPPOAgent):
        collect_opts["temporal_transitions"] = True

    Tb_rollout, info_collect = agent.collect_eval_with_states(collector, b_state0, env.eval_T, **collect_opts)
    Tb_rollout = jax.device_get(Tb_rollout)
    bT_rollout = Tb_rollout.switch01()

    # Extract each rollout
    b_trajs = extract_rollouts_eval(bT_rollout)

    b_values = []
    for ii, traj in enumerate(b_trajs):
        debug = (ii == 64)
        dag_value = evaluate_ltl_finite(env, traj.predicates_next, which=np)[env.dag_root]
        b_values.append(dag_value)
    b_values = np.array(b_values)
    b_is_satisfied = b_values > 0.1

    p_satisified = np.mean(b_is_satisfied)

    logger.info(f"Probability of satisfying the root node: {p_satisified:.3f}")

    if p_satisified == 1.0:
        return

    idx = np.argmin(b_is_satisfied)
    # Find out why it's not 100%

    traj_bad = b_trajs[idx]

    # The grid cells are 1x1
    agent_radius = 0.2

    fig, ax = plt.subplots()
    env.setup_ax(ax)
    circs = []
    for agent_idx in range(env.n_agents):
        circ = plt.Circle((0, 0), agent_radius, facecolor="C1", edgecolor="none")
        ax.add_patch(circ)
        circs.append(circ)

    kk_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        color="white",
        fontsize=8,
        bbox=dict(facecolor="black", alpha=0.5, pad=2),
    )
    misc_text = ax.text(
        0.98,
        0.02,
        "",
        transform=ax.transAxes,
        verticalalignment="bottom",
        horizontalalignment="right",
        color="white",
        fontsize=8,
        bbox=dict(facecolor="black", alpha=0.5, pad=2),
    )
    fig.tight_layout()
    traj = traj_bad

    colors_discrete = plt.get_cmap("tab10", env.n_temporal_nodes).colors
    color_alive = to_rgba("C0", 0.0)
    color_dead = np.array(to_rgba("C0"))
    T_max = traj.shape[0]

    def update_fn(kk: int):
        kk_text.set_text(f"Step {kk: 3}")
        (T,) = traj.shape
        T_state: StateWithTemporalNode[GridworldMAState] = traj.state_now
        T_pos = T_state.base.pos[:, :, :2]

        T_state_next: StateWithTemporalNode[GridworldMAState] = traj.state_next
        T_pos_next = T_state_next.base.pos[:, :, :2]

        is_dead = kk >= T
        t_idx = min(kk, T - 1)

        temporal_node_idx = T_state.temporal_node_idx[t_idx]

        for agent_idx, circ in enumerate(circs):
            if is_dead:
                pos = T_pos_next[-1, agent_idx, :]
            else:
                pos = T_pos[t_idx, agent_idx, :]
            circ.center = pos

            circ.set_facecolor(colors_discrete[temporal_node_idx])

            if kk < T:
                circ.set_edgecolor(color_alive)
            else:
                circ.set_edgecolor(color_dead)

        misc_text.set_text(f"Temporal {temporal_node_idx}")

    plot_dir = run_path / "eval_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    anim_path = plot_dir / f"eval.mp4"
    animated_artists = circs + [kk_text, misc_text]
    save_animation_blit(fig, animated_artists, anim_path, T_max + 1, update_fn)

    ipdb.set_trace()


# info_satisfaction = {"Eval/Satisfy/Root": float(np.mean(np.array(dag_values) > 0.1))}


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
