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
from rraa_rl.src.env.general_task.env import AugObs
from rraa_rl.src.env.general_task.herd_os import HerdOs
from rraa_rl.src.rl.utils.utils import get_BuRd_smooth
from rraa_rl.trainer import CallbackProps
from rraa_rl.vd_mappo import PPOData, VDMAPPOAgent


class VizValues(struct.PyTreeNode):
    @staticmethod
    def create():
        return VizValues()

    @jax.jit
    def get_value(self, agent: VDMAPPOAgent):
        env = agent.env
        cfg = env.cfg.base
        halfsize = cfg.halfsize

        n_x = 65
        n_y = 65
        b_x = jnp.linspace(-halfsize[0], halfsize[0], num=n_x)
        b_y = jnp.linspace(-halfsize[1], halfsize[1], num=n_y)
        bb_X, bb_Y = jnp.meshgrid(b_x, b_y)

        bb_pos = jnp.stack([bb_X, bb_Y], axis=-1)

        key = jax.random.PRNGKey(0)
        bb_key = ei.rearrange(jax.random.split(key, num=n_x * n_y), "(x y) ... -> x y ...", x=n_x, y=n_y)
        bb_state = jax_vmap(env.reset, rep=2)(bb_key)

        with jdc.copy_and_mutate(bb_state) as bb_state:
            bb_state.base.herder_state = bb_state.base.herder_state.at[:, :, 0, :2].set(bb_pos)
            bb_state.base.herder_state = bb_state.base.herder_state.at[:, :, 0, 2:4].set(0.0)
            bb_state.temporal_node_idx = bb_state.temporal_node_idx.at[:].set(0)

        bb_predicates = jax_vmap(env.get_predicates, rep=2)(bb_state)

        bb_obs: AugObs = jax_vmap(env.get_obs, rep=2)(bb_state)
        bbt_V = agent.network.select("critic")(bb_obs)

        return bb_X, bb_Y, bbt_V, bb_predicates, bb_obs.combine()

    def __call__(self, p: CallbackProps):
        bb_X, bb_Y, bbt_V, bb_predicates, bb_obs = jax.device_get(self.get_value(p.agent))

        n_predicates = len(bb_predicates)

        env = p.env
        n_temporal_nodes = env.n_temporal_nodes
        nrow = 2
        ncol = max(n_temporal_nodes, n_predicates)

        figsize = np.array([4 * ncol, 3 * nrow])
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize, layout="constrained", squeeze=False)

        # cmap = get_BuRd_trunc().reversed()
        cmap = get_BuRd_smooth().reversed()

        # On the first row, plot the predicates.
        for ii, ax in enumerate(axes[0, :n_predicates]):
            ax: plt.Axes
            env.base.setup_ax(ax)
            pred_name = list(bb_predicates.keys())[ii]

            ax.set_title(f"Predicate: {pred_name}")

            im = ax.contourf(bb_X, bb_Y, bb_predicates[pred_name], levels=50, cmap=cmap, vmin=-1, vmax=1)
            fig.colorbar(im, ax=ax)

        # On the second row, plot the value functions.
        for ii, ax in enumerate(axes[1, :n_temporal_nodes]):
            ax: plt.Axes
            env.base.setup_ax(ax)

            node_idx = env.temporal_nodes[ii]
            node = env.dag_nodes[node_idx]
            node_name = type(node).__name__
            ax.set_title(f"Node {ii} ({node_name})")

            im = ax.contourf(bb_X, bb_Y, bbt_V[:, :, ii], levels=50, cmap=cmap, norm=CenteredNorm())
            fig.colorbar(im, ax=ax)

        plot_dir = p.run.plots_dir / "V"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig_path = plot_dir / f"V_step{p.train_step}.jpg"
        fig.savefig(fig_path, bbox_inches="tight", dpi=500)
        plt.close(fig)

        if p.train_step == 0:
            # Only plot a contourf of the observations on the first step.
            obs_names = env.get_obs_names()

            n_obs = bb_obs.shape[-1]
            ncol = 6
            nrow = int(np.ceil(n_obs / ncol))
            figsize = np.array([4 * ncol, 3 * nrow])
            fig, axes = plt.subplots(nrow, ncol, figsize=figsize, layout="constrained", squeeze=False)
            axes = axes.flatten()
            for ii, ax in enumerate(axes[:n_obs]):
                ax: plt.Axes
                env.base.setup_ax(ax)
                ax.set_title(f"{ii:02}: {obs_names[ii]}")

                im = ax.contourf(bb_X, bb_Y, bb_obs[:, :, ii], levels=50)
                fig.colorbar(im, ax=ax)

            plot_dir = p.run.plots_dir
            fig_path = plot_dir / "obs_contours.pdf"
            fig.savefig(fig_path, bbox_inches="tight")
            plt.close(fig)


def plot_eval_trajs(p: CallbackProps):
    plots_dir = p.run.plots_dir
    env = p.env

    if env.n_agents > 1:
        return

    temporal_values_dict = p.temporal_values_dict

    n_temporal_nodes = env.n_temporal_nodes
    ncol = n_temporal_nodes

    bT_states: list[HerdOs.State] = [traj.state_now for traj in p.bT_test_rollouts]
    b_temporal_idx = np.array([T_state.temporal_node_idx[0] for T_state in bT_states])

    # Count how many trajectories each temporal node has.
    num_temporal_nodes_in_batch = np.array([np.sum(b_temporal_idx == ii) for ii in range(n_temporal_nodes)])

    figsize = np.array([4 * ncol, 3])
    fig, axes = plt.subplots(1, ncol, figsize=figsize, layout="constrained")

    if ncol == 1:
        axes = [axes]

    start_idx = 0
    for ii, ax in enumerate(axes):
        env.base.setup_ax(ax)

        end_idx = start_idx + num_temporal_nodes_in_batch[ii]

        node_idx = env.temporal_nodes[ii]
        node = env.dag_nodes[node_idx]

        temporal_node_value = temporal_values_dict[ii]
        n_satisfy = np.sum(temporal_node_value >= 0.1)
        n_total = len(temporal_node_value)

        node_name = type(node).__name__
        ax.set_title(
            f"Node {ii} ({node_name}) | {num_temporal_nodes_in_batch[ii]} trajs | "
            f"{n_satisfy}/{n_total} ({n_satisfy / n_total:.1%})",
            fontsize="small",
        )

        for traj in p.bT_test_rollouts[start_idx:end_idx]:
            (T,) = traj.shape

            T_state: HerdOs.State = traj.state_now
            T_herder_pos = T_state.base.herder_state[:, 0, :2]
            assert T_herder_pos.shape == (T, 2)

            ax.plot(T_herder_pos[:, 0], T_herder_pos[:, 1], color="C1", alpha=0.2, lw=0.5)
            # Start point.
            ax.plot(
                T_herder_pos[0, 0],
                T_herder_pos[0, 1],
                marker="s",
                color="C1",
                ms=1.5,
            )
            # Plot end point.
            ax.plot(
                T_herder_pos[-1, 0],
                T_herder_pos[-1, 1],
                marker="o",
                color="C0",
                ms=1.5,
            )

        start_idx = end_idx

    plot_dir = plots_dir / "eval_trajs"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plot_dir / f"eval_trajs_step{p.train_step}.jpg"
    fig.savefig(fig_path, bbox_inches="tight", dpi=500)
    plt.close(fig)


def animate_eval_trajs(p: CallbackProps):
    env: HerdOs = p.env
    if env.n_agents == 1:
        animate_eval_trajs_single_agent(p)
    else:
        animate_eval_trajs_multi_agent(p)


def animate_eval_trajs_single_agent(p: CallbackProps):
    plots_dir = p.run.plots_dir
    env = p.env

    n_temporal_nodes = env.n_temporal_nodes
    ncol = n_temporal_nodes

    bT_states: list[HerdOs.State] = [traj.state_now for traj in p.bT_test_rollouts]
    b_temporal_idx = np.array([T_state.temporal_node_idx[0] for T_state in bT_states])

    T_max = max(traj.shape[0] for traj in p.bT_test_rollouts)

    # Count how many trajectories each temporal node has.
    temporal_node_count = np.array([np.sum(b_temporal_idx == ii) for ii in range(n_temporal_nodes)])

    figsize = np.array([4 * ncol, 3])
    fig, axes = plt.subplots(1, ncol, figsize=figsize, dpi=200, layout="none")
    if ncol == 1:
        axes = [axes]

    cfg = env.base.cfg

    # Use facecolor to indicate the current temporal node.
    colors_temporal_node = [f"C{ii}" for ii in range(n_temporal_nodes)]

    # Use edgecolor to indicate alive vs dead.
    color_alive = to_rgba("C0", 0.0)
    color_dead = np.array(to_rgba("C0"))

    circ_collections = []
    start_idxs, end_idxs = [], []
    start_idx = 0
    for ii, ax in enumerate(axes):
        env.base.setup_ax(ax)

        n_traj = temporal_node_count[ii]
        end_idx = start_idx + n_traj
        start_idxs.append(start_idx)
        end_idxs.append(end_idx)

        node_idx = env.temporal_nodes[ii]
        node = env.dag_nodes[node_idx]
        node_name = type(node).__name__
        ax.set_title(f"Node {ii} ({node_name}) | {temporal_node_count[ii]} trajs")

        start_idx = end_idx

        ec = EllipseCollection(
            widths=np.full(n_traj, cfg.agent_radius * 2),
            heights=np.full(n_traj, cfg.agent_radius * 2),
            angles=np.zeros(n_traj),
            units="xy",
            offsets=np.zeros((n_traj, 2)),
            transOffset=ax.transData,
            facecolors="C1",
            edgecolors="none",
            animated=True,
        )
        ax.add_collection(ec)
        circ_collections.append(ec)

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
    fig.tight_layout()

    # Mark animated artists
    for ec in circ_collections:
        ec.set_animated(True)
    kk_text.set_animated(True)

    # Prime the renderer + background
    fig.canvas.draw()
    bg = fig.canvas.copy_from_bbox(fig.bbox)

    plot_dir = plots_dir / "eval_trajs_anim"
    plot_dir.mkdir(parents=True, exist_ok=True)
    anim_path = plot_dir / f"eval_trajs_step{p.train_step}.mp4"

    writer = imageio.get_writer(
        anim_path,
        fps=30,
        codec="libx264",
        format="ffmpeg",
        ffmpeg_params=["-preset", "ultrafast", "-crf", "23"],
    )

    pbar = tqdm.trange(T_max, unit="frame", desc="Generating eval trajs animation")
    try:
        for kk in pbar:
            # Restore background
            fig.canvas.restore_region(bg)

            # Update artists (do NOT return anything)
            kk_text.set_text(f"Step {kk: 3}")

            for ii, ax in enumerate(axes):
                start_idx = start_idxs[ii]
                end_idx = end_idxs[ii]

                trajs = p.bT_test_rollouts[start_idx:end_idx]
                n_traj = end_idx - start_idx

                offsets = np.zeros((n_traj, 2))
                facecolors = []
                edgecolors = []

                for jj, traj in enumerate(trajs):
                    (T,) = traj.shape
                    T_state: HerdOs.State = traj.state_now
                    T_herder_pos = T_state.base.herder_state[:, 0, :2]

                    t_idx = min(kk, T - 1)
                    offsets[jj, :] = T_herder_pos[t_idx, :]

                    temporal_node_idx = T_state.temporal_node_idx[t_idx]
                    facecolors.append(colors_temporal_node[temporal_node_idx])

                    if kk < T:
                        edgecolors.append(color_alive)
                    else:
                        edgecolors.append(color_dead)

                circ_collections[ii].set_offsets(offsets)
                circ_collections[ii].set_facecolor(facecolors)
                circ_collections[ii].set_edgecolor(edgecolors)

            # Draw only animated artists
            for ec in circ_collections:
                ec.axes.draw_artist(ec)
            kk_text.axes.draw_artist(kk_text)

            # Blit + grab frame
            fig.canvas.blit(fig.bbox)
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

            writer.append_data(frame_rgb)

    finally:
        writer.close()
        plt.close(fig)


def animate_eval_trajs_multi_agent(p: CallbackProps):
    plots_dir = p.run.plots_dir
    env: HerdOs = p.env
    cfg = env.base.cfg

    n_traj_anim = 8

    n_temporal_nodes = env.n_temporal_nodes

    bT_test_rollouts = p.bT_test_rollouts

    bT_states: list[HerdOs.State] = [traj.state_now for traj in bT_test_rollouts]
    b_temporal_idx = np.array([T_state.temporal_node_idx[0] for T_state in bT_states])

    temporal_node_count = np.array([np.sum(b_temporal_idx == ii) for ii in range(n_temporal_nodes)])
    offsets = np.array([0, *np.cumsum(temporal_node_count)])

    # T_max = max(traj.shape[0] for traj in bT_test_rollouts)
    T_max = 0
    batch_idxs: dict[tuple[int, int], int] = {}
    for ii in range(n_traj_anim):
        for jj in range(n_temporal_nodes):
            batch_idx = ii + offsets[jj]
            batch_idxs[ii, jj] = batch_idx
            traj = bT_test_rollouts[batch_idx]
            T_max = max(T_max, len(traj.term))
    # -----------------------------

    # ncol = n_temporal_nodes
    # nrow = n_traj_anim
    ncol = n_traj_anim
    nrow = n_temporal_nodes

    # Use facecolor to indicate the current temporal node.
    colors_temporal_node = ["C0", "C1", "C2", "C4", "C5", "C6"]  # C3 is grey.

    # Use edgecolor to indicate alive vs dead.
    color_alive = to_rgba("C0", 0.0)
    color_dead = np.array(to_rgba("C0"))

    figsize = 0.9 * np.array([4 * ncol, 3 * nrow])
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, dpi=80, squeeze=False, layout="none")

    agent_collections: dict[tuple[int, int], list[plt.Circle]] = {}
    herds: dict[tuple[int, int], list[plt.Circle]] = {}
    for ii in range(n_traj_anim):
        for jj in range(n_temporal_nodes):
            ax = axes[jj, ii]
            env.base.setup_ax(ax)

            node_idx = env.temporal_nodes[jj]
            node = env.dag_nodes[node_idx]
            node_name = type(node).__name__
            ax.set_title(f"Node {jj} ({node_name})")

            circs = []
            for agent_idx in range(env.n_agents):
                circ = plt.Circle((0, 0), cfg.agent_radius, facecolor="C1", edgecolor="none")
                ax.add_patch(circ)
                circs.append(circ)
            agent_collections[(ii, jj)] = circs

            circs = []
            for herd_idx in range(env.cfg.base.n_herd):
                circ = plt.Circle((0, 0), cfg.agent_radius, facecolor="C3", edgecolor="none")
                ax.add_patch(circ)
                circs.append(circ)
            herds[(ii, jj)] = circs

    all_circs = [v for values_list in agent_collections.values() for v in values_list]
    all_circs += [v for values_list in herds.values() for v in values_list]

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

    herd_vel_texts = {}
    if "dyn/herd_vel" in bT_test_rollouts[0].info:
        for ii in range(n_traj_anim):
            for jj in range(n_temporal_nodes):
                # bottom right.
                herd_vel_texts[ii, jj] = axes[jj, ii].text(
                    0.98,
                    0.02,
                    "",
                    transform=axes[jj, ii].transAxes,
                    verticalalignment="bottom",
                    horizontalalignment="right",
                    color="white",
                    fontsize=8,
                    bbox=dict(facecolor="black", alpha=0.5, pad=2),
                )

    # fig.canvas.draw()  # compute constrained layout once
    # fig.set_layout_engine("none")  # freeze layout for animation
    fig.tight_layout()

    for circ in all_circs:
        circ.set_animated(True)
    for text in herd_vel_texts.values():
        text.set_animated(True)
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
                    T_state: HerdOs.State = traj.state_now
                    T_herder_pos = T_state.base.herder_state[:, :, :2]

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

                    circs = herds[(ii, jj)]
                    for herd_idx, circ in enumerate(circs):
                        pos = T_state.base.herd_state[t_idx, herd_idx, :2]
                        circ.center = pos

                    if len(herd_vel_texts) > 0:
                        # (n_herd, 2)
                        herd_vel = traj.info["dyn/herd_vel"][t_idx]
                        # (n_herd,)
                        herd_speeds = np.linalg.norm(herd_vel, axis=-1)
                        herd_speed_str = ", ".join([f"{s:.2f}" for s in herd_speeds])
                        herd_vel_texts[ii, jj].set_text(f"Herd: {herd_speed_str}")

            # draw only animated artists onto the restored background
            for a in all_circs:
                a.axes.draw_artist(a)
            for text in herd_vel_texts.values():
                text.axes.draw_artist(text)
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

            # writer.append_data(frame_rgb)
            writer.write_frame(frame_rgb)

    # finally:
    #     writer.close()
    plt.close(fig)


class PlotRootTrajPreds(struct.PyTreeNode):
    """At the top, plot the reach vals. At the bottom, plot the temporal_node_idx."""

    @staticmethod
    def create():
        return PlotRootTrajPreds()

    @jax.jit
    def get_reach_values(self, agent: VDMAPPOAgent, bT_test_rollout: RolloutOutput):
        bT_obs_next = bT_test_rollout.obs_next
        bT_predicates_next = bT_test_rollout.predicates_next
        bTt_reach_val = rep_vmap(agent.get_t_reach_val, rep=2)(bT_obs_next, bT_predicates_next)
        return bTt_reach_val

    def __call__(self, p: CallbackProps):
        bT_test_rollout = p.bT_test_rollout
        bTt_reach_vals = jax.device_get(self.get_reach_values(p.agent, bT_test_rollout))

        trajs: list[RolloutOutput] = p.bT_test_rollouts
        batch_size = len(trajs)

        max_n_plot = 8
        n_plot = min(batch_size, max_n_plot)

        env = p.env
        n_temporal_nodes = env.n_temporal_nodes

        nrow = n_temporal_nodes + 1

        figsize = np.array([10, 2 * nrow])
        fig, axes = plt.subplots(nrow, figsize=figsize, layout="constrained")

        for bb in range(n_plot):
            # Don't plot invalid timesteps.
            traj_len = len(trajs[bb].term)

            for ii, ax in enumerate(axes[:n_temporal_nodes]):
                T_reach_vals = bTt_reach_vals[bb, :traj_len, ii]
                ax.plot(T_reach_vals)

            T_state: HerdOs.State = trajs[bb].state_now
            T_temporal_node_idx = T_state.temporal_node_idx

            ax = axes[-1]
            ax.plot(T_temporal_node_idx)

        # Label.
        for ii, ax in enumerate(axes[:n_temporal_nodes]):
            node_idx = env.temporal_nodes[ii]
            node = env.dag_nodes[node_idx]
            node_name = type(node).__name__
            ax.set_ylabel(f"Node {ii} ({node_name})")

        axes[-1].set_ylabel("temporal_node_idx")
        axes[-1].set_xlabel("Steps")
        axes[-1].set_ylim(-0.5, n_temporal_nodes - 0.5)

        plot_dir = p.run.plots_dir / "root_traj_preds"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig_path = plot_dir / f"root_traj_preds_step{p.train_step}.jpg"
        fig.savefig(fig_path, bbox_inches="tight", dpi=500)
        plt.close(fig)


def viz_collect_data(p: CallbackProps):
    train_step = p.train_step

    # Only plot every 5_000 steps.
    if train_step % 5_000 != 0:
        return

    env = p.env
    if env.n_agents > 1:
        return

    agent = p.agent
    b_data: PPOData = jax.device_get(p.info_update["debug/b_data"])
    Tb_rollout: RolloutOutput = jax.device_get(p.Tb_rollout)

    env = p.env
    env_base = env.base
    cfg_base = env_base.cfg
    cfg_agent = agent.cfg

    # Find rollouts where the target is larger than 2, figure out why...
    Tb_state: HerdOs.State = Tb_rollout.state_now
    bT_A, bT_Q, bT_temporal_idx = agent.compute_A_Q(Tb_rollout, debug=True)

    b_Q = b_data.Q
    n_temporal_nodes = env.n_temporal_nodes
    # ---------------------------------------------------------------
    b_state: HerdOs.State = b_data.state
    b_pos = b_state.base.herder_state[:, 0, :2]

    b_temporal_idx = b_state.temporal_node_idx

    cmap = get_BuRd_smooth().reversed()

    ncol = n_temporal_nodes
    figsize = np.array([4 * ncol, 3])
    fig, axes = plt.subplots(1, ncol, figsize=figsize, layout="constrained")
    if ncol == 1:
        axes = [axes]

    for ii, ax in enumerate(axes):
        env.base.setup_ax(ax)

        b_isthis = b_temporal_idx == ii

        dag_id = env.temporal_nodes[ii]
        node = env.dag_nodes[dag_id]
        node_name = type(node).__name__

        c_pos = b_pos[b_isthis]
        c_Q = b_Q[b_isthis]

        # Visualize the herder positions colored by Q-values
        norm = CenteredNorm()
        sc = ax.scatter(c_pos[:, 0], c_pos[:, 1], c=c_Q, cmap=cmap, s=5, norm=norm)
        ax.set_title(
            "{} %{} | Q∈[{:.2f}, {:.2f}] | mean={:.2f}".format(node_name, dag_id, np.min(c_Q), np.max(c_Q), c_Q.mean()),
            fontsize="small",
        )
        fig.colorbar(sc, ax=ax)

    plot_dir = p.run.plots_dir / "collect_data"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plot_dir / f"collect_data_step{p.train_step}.jpg"
    fig.savefig(fig_path, bbox_inches="tight", dpi=500)
    plt.close(fig)


def viz_obs_histogram(p: CallbackProps):
    train_step = p.train_step

    # Only plot every 5_000 steps.
    if train_step % 5_000 != 0:
        return

    agent = p.agent
    b_data: PPOData = jax.device_get(p.info_update["debug/b_data"])
    Tb_rollout: RolloutOutput = jax.device_get(p.Tb_rollout)

    # Visualize the observation distribution on a histogram.
    b_obs_: AugObs = b_data.obs
    b_obs = b_obs_.combine(which=np)

    batch_size, n_obs = b_obs.shape
    nrow = n_obs
    figsize = np.array([8, 1.5 * nrow])
    fig, axes = plt.subplots(nrow, figsize=figsize, layout="constrained")
    if nrow == 1:
        axes = [axes]
    for ii, ax in enumerate(axes):
        obs_dim = np.asarray(b_obs[:, ii])
        plot_histogram(obs_dim, ax=ax, center="mean")
        ax.set_ylabel(f"{ii}")

    plot_dir = p.run.plots_dir / "obs_hist"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plot_dir / f"obs_hist_step{p.train_step}.jpg"
    fig.savefig(fig_path, bbox_inches="tight", dpi=500)
    plt.close(fig)
