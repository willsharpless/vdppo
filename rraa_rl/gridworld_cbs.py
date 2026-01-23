import time

import einops as ei
import imageio.v2 as imageio
import imageio.v3 as iio
import ipdb
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from flax import struct
from loguru import logger
from lovely_histogram import plot_histogram
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import EllipseCollection
from matplotlib.colors import CenteredNorm, to_rgba

from rraa_rl.collector import RolloutOutput
from rraa_rl.jax_utils import jax_vmap, rep_vmap
from rraa_rl.lcrl.lcrl_wrapper import LCRLWrapper
from rraa_rl.src.env.general_task.env import AugObs
from rraa_rl.src.env.general_task.gridworld import GridworldMA, GridworldMABase, GridworldMAState
from rraa_rl.src.rl.utils.utils import get_BuRd_smooth
from rraa_rl.trainer import CallbackProps
from rraa_rl.vd_mappo import PPOData, VDMAPPOAgent


def animate_eval_trajs(p: CallbackProps):
    if isinstance(p.agent, VDMAPPOAgent):
        animate_eval_trajs_vd(p)
    else:
        animate_eval_trajs_base(p)


def animate_eval_trajs_vd(p: CallbackProps):
    plots_dir = p.run.plots_dir
    env: GridworldMA = p.env
    cfg = env.base.cfg

    n_traj_anim = 6

    n_temporal_nodes = env.n_temporal_nodes

    bT_test_rollouts = p.bT_test_rollouts

    bT_states: list[GridworldMA.State] = [traj.state_now for traj in bT_test_rollouts]
    b_temporal_idx = np.array([T_state.temporal_node_idx[0] for T_state in bT_states])

    temporal_node_count = np.array([np.sum(b_temporal_idx == ii) for ii in range(n_temporal_nodes)])
    offsets = np.array([0, *np.cumsum(temporal_node_count)])

    T_max = 0
    batch_idxs: dict[tuple[int, int], int] = {}
    for ii in range(n_traj_anim):
        for jj in range(n_temporal_nodes):
            batch_idx = ii + offsets[jj]
            batch_idxs[ii, jj] = batch_idx
            traj = bT_test_rollouts[batch_idx]
            T_max = max(T_max, len(traj.term))

    ncol = n_traj_anim
    nrow = n_temporal_nodes

    colors_temporal_node = plt.get_cmap("tab20", n_temporal_nodes).colors

    color_alive = to_rgba("C0", 0.0)
    color_dead = np.array(to_rgba("C0"))

    figsize = 0.9 * np.array([3 * ncol, 3 * nrow])
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, dpi=80, squeeze=False, layout="none")

    # The grid cells are 1x1
    agent_radius = 0.2

    agent_collections: dict[tuple[int, int], list[plt.Circle]] = {}
    for ii in range(n_traj_anim):
        for jj in range(n_temporal_nodes):
            ax = axes[jj, ii]
            env.base.setup_ax(ax)

            node_idx = env.temporal_nodes[jj]
            node = env.dag_nodes[node_idx]
            node_name = type(node).__name__[3:]

            # Set the ylabel.
            if ii == 0:
                ax.set_ylabel(f"Node {jj}\n({node_name}) %{node_idx}", fontsize=10)

            # ax.set_title(f"Node {jj} ({node_name})")

            circs = []
            for agent_idx in range(env.n_agents):
                circ = plt.Circle((0, 0), agent_radius, facecolor="C1", edgecolor="none")
                ax.add_patch(circ)
                circs.append(circ)
            agent_collections[(ii, jj)] = circs

    all_circs = [v for values_list in agent_collections.values() for v in values_list]

    kk_text = axes[0, 0].text(
        0.02,
        0.98,
        "",
        transform=axes[0, 0].transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        color="white",
        fontsize=8,
        bbox=dict(facecolor="black", alpha=0.5, pad=2),
    )

    fig.tight_layout()

    for circ in all_circs:
        circ.set_animated(True)
    kk_text.set_animated(True)

    fig.canvas.draw()
    bg = fig.canvas.copy_from_bbox(fig.bbox)

    plot_dir = plots_dir / "eval_trajs_anim"
    plot_dir.mkdir(parents=True, exist_ok=True)
    anim_path = plot_dir / f"eval_trajs_step{p.train_step}.mp4"

    with iio.imopen(anim_path, "w", plugin="pyav") as writer:
        writer.init_video_stream("libx264", fps=30)

        pbar = tqdm.trange(T_max, unit="frame", desc="Generating eval trajs animation")
        for kk in pbar:
            # restore background
            fig.canvas.restore_region(bg)

            # update artists (your existing update body, but do NOT return artists)
            kk_text.set_text(f"Step {kk: 3}")
            for ii in range(n_traj_anim):
                for jj in range(n_temporal_nodes):

                    batch_idx = batch_idxs[ii, jj]
                    traj = bT_test_rollouts[batch_idx]
                    (T,) = traj.shape
                    T_state: GridworldMA.State = traj.state_now
                    T_herder_pos = T_state.base.pos[:, :, :2]

                    t_idx = min(kk, T - 1)

                    circs = agent_collections[(ii, jj)]
                    for agent_idx, circ in enumerate(circs):
                        pos = T_herder_pos[t_idx, agent_idx, :]
                        circ.center = pos

                        temporal_node_idx = T_state.temporal_node_idx[t_idx]
                        circ.set_facecolor(colors_temporal_node[temporal_node_idx])

                        if kk < T:
                            circ.set_edgecolor(color_alive)
                        else:
                            circ.set_edgecolor(color_dead)

            # draw only animated artists onto the restored background
            for a in all_circs:
                a.axes.draw_artist(a)
            kk_text.axes.draw_artist(kk_text)

            # IMPORTANT: update the buffer
            fig.canvas.blit(fig.bbox)

            # Grab frame (ensure contiguous uint8 RGB)
            frame_rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame_rgb = np.ascontiguousarray(frame_rgba[..., :3])

            # If the frame dimensions are not divisible by 16, then pad until they are.
            h, w, _ = frame_rgb.shape
            h_pad = (16 - (h % 16)) % 16
            w_pad = (16 - (w % 16)) % 16
            if h_pad > 0 or w_pad > 0:
                frame_rgb = np.pad(
                    frame_rgb,
                    ((0, h_pad), (0, w_pad), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

            writer.write_frame(frame_rgb)

    plt.close(fig)


def animate_eval_trajs_base(p: CallbackProps):
    plots_dir = p.run.plots_dir
    env: LCRLWrapper = p.env
    cfg = env.base.cfg

    n_traj_anim = 6

    bT_test_rollouts = p.bT_test_rollouts

    bT_states: list[LCRLWrapper.State] = [traj.state_now for traj in bT_test_rollouts]

    T_max = 0
    batch_idxs: dict[int, int] = {}
    for ii in range(n_traj_anim):
        traj = bT_test_rollouts[ii]
        T_max = max(T_max, len(traj.term))

    ncol = n_traj_anim

    color_alive = to_rgba("C0", 0.0)
    color_dead = np.array(to_rgba("C0"))

    figsize = 0.9 * np.array([3 * ncol, 3])
    fig, axes = plt.subplots(1, ncol, figsize=figsize, dpi=80, layout="none")

    # The grid cells are 1x1
    agent_radius = 0.2

    agent_collections: dict[tuple[int, int], list[plt.Circle]] = {}
    for ii in range(n_traj_anim):
        ax = axes[ii]
        env.setup_ax(ax)

        # ax.set_title(f"Node {jj} ({node_name})")

        circs = []
        for agent_idx in range(env.n_agents):
            circ = plt.Circle((0, 0), agent_radius, facecolor="C1", edgecolor="none")
            ax.add_patch(circ)
            circs.append(circ)
        agent_collections[ii] = circs

    all_circs = [v for values_list in agent_collections.values() for v in values_list]

    kk_text = axes[0].text(
        0.02,
        0.98,
        "",
        transform=axes[0].transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        color="white",
        fontsize=8,
        bbox=dict(facecolor="black", alpha=0.5, pad=2),
    )

    fig.tight_layout()

    for circ in all_circs:
        circ.set_animated(True)
    kk_text.set_animated(True)

    fig.canvas.draw()
    bg = fig.canvas.copy_from_bbox(fig.bbox)

    plot_dir = plots_dir / "eval_trajs_anim"
    plot_dir.mkdir(parents=True, exist_ok=True)
    anim_path = plot_dir / f"eval_trajs_step{p.train_step}.mp4"

    with iio.imopen(anim_path, "w", plugin="pyav") as writer:
        writer.init_video_stream("libx264", fps=30)

        pbar = tqdm.trange(T_max, unit="frame", desc="Generating eval trajs animation")
        for kk in pbar:
            # restore background
            fig.canvas.restore_region(bg)

            # update artists (your existing update body, but do NOT return artists)
            kk_text.set_text(f"Step {kk: 3}")
            for ii in range(n_traj_anim):
                traj = bT_test_rollouts[ii]
                (T,) = traj.shape
                T_state: GridworldMA.State = traj.state_now
                T_herder_pos = T_state.base.pos[:, :, :2]

                t_idx = min(kk, T - 1)

                circs = agent_collections[ii]
                for agent_idx, circ in enumerate(circs):
                    pos = T_herder_pos[t_idx, agent_idx, :]
                    circ.center = pos

                    if kk < T:
                        circ.set_edgecolor(color_alive)
                    else:
                        circ.set_edgecolor(color_dead)

            # draw only animated artists onto the restored background
            for a in all_circs:
                a.axes.draw_artist(a)
            kk_text.axes.draw_artist(kk_text)

            # IMPORTANT: update the buffer
            fig.canvas.blit(fig.bbox)

            # Grab frame (ensure contiguous uint8 RGB)
            frame_rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame_rgb = np.ascontiguousarray(frame_rgba[..., :3])

            # If the frame dimensions are not divisible by 16, then pad until they are.
            h, w, _ = frame_rgb.shape
            h_pad = (16 - (h % 16)) % 16
            w_pad = (16 - (w % 16)) % 16
            if h_pad > 0 or w_pad > 0:
                frame_rgb = np.pad(
                    frame_rgb,
                    ((0, h_pad), (0, w_pad), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

            writer.write_frame(frame_rgb)

    plt.close(fig)
