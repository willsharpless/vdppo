import functools as ft

import einops as ei
import ipdb
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
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
from valtr.reachability import DAGReachAvoid

from rraa_rl.collector import RolloutOutput
from rraa_rl.gae import BellmanMax, BellmanMaxMin, BellmanMin, gae_generalized
from rraa_rl.jax_utils import jax_vmap
from rraa_rl.src.env.general_task.herd_os import HerdOs
from rraa_rl.src.rl.utils.utils import get_BuRd, get_BuRd_smooth, get_BuRd_trunc
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

        bb_predicates = jax_vmap(env.get_predicates, rep=2)(bb_state)

        bb_obs = jax_vmap(env.get_obs, rep=2)(bb_state)

        bbt_V = agent.network.select("critic")(bb_obs)
        return bb_X, bb_Y, bbt_V, bb_predicates, bb_obs

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
            n_obs = bb_obs.shape[-1]
            ncol = 6
            nrow = int(np.ceil(n_obs / ncol))
            figsize = np.array([4 * ncol, 3 * nrow])
            fig, axes = plt.subplots(nrow, ncol, figsize=figsize, layout="constrained", squeeze=False)
            axes = axes.flatten()
            for ii, ax in enumerate(axes[:n_obs]):
                ax: plt.Axes
                env.base.setup_ax(ax)
                ax.set_title(f"Obs dim {ii}")

                im = ax.contourf(bb_X, bb_Y, bb_obs[:, :, ii], levels=50)
                fig.colorbar(im, ax=ax)

            plot_dir = p.run.plots_dir
            fig_path = plot_dir / "obs_contours.pdf"
            fig.savefig(fig_path, bbox_inches="tight")
            plt.close(fig)


def plot_eval_trajs(p: CallbackProps):
    plots_dir = p.run.plots_dir
    env = p.env

    temporal_values_dict = p.temporal_values_dict

    n_temporal_nodes = env.n_temporal_nodes
    ncol = n_temporal_nodes

    bT_states: list[HerdOs.State] = [traj.state_now for traj in p.bT_test_rollouts]
    b_temporal_idx = np.array([T_state.temporal_node_idx[0] for T_state in bT_states])

    # Count how many trajectories each temporal node has.
    n_temporal_nodes = np.array([np.sum(b_temporal_idx == ii) for ii in range(n_temporal_nodes)])

    figsize = np.array([4 * ncol, 3])
    fig, axes = plt.subplots(1, ncol, figsize=figsize, layout="constrained")

    if ncol == 1:
        axes = [axes]

    start_idx = 0
    for ii, ax in enumerate(axes):
        env.base.setup_ax(ax)

        end_idx = start_idx + n_temporal_nodes[ii]

        node_idx = env.temporal_nodes[ii]
        node = env.dag_nodes[node_idx]

        temporal_node_value = temporal_values_dict[ii]
        n_satisfy = np.sum(temporal_node_value >= 0.1)
        n_total = len(temporal_node_value)

        node_name = type(node).__name__
        ax.set_title(
            f"Node {ii} ({node_name}) | {n_temporal_nodes[ii]} trajs | "
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
    plots_dir = p.run.plots_dir
    env = p.env

    n_temporal_nodes = env.n_temporal_nodes
    ncol = n_temporal_nodes

    bT_states: list[HerdOs.State] = [traj.state_now for traj in p.bT_test_rollouts]
    b_temporal_idx = np.array([T_state.temporal_node_idx[0] for T_state in bT_states])

    T_max = max(traj.shape[0] for traj in p.bT_test_rollouts)

    # Count how many trajectories each temporal node has.
    n_temporal_nodes = np.array([np.sum(b_temporal_idx == ii) for ii in range(n_temporal_nodes)])

    figsize = np.array([4 * ncol, 3])
    fig, axes = plt.subplots(1, ncol, figsize=figsize)
    if ncol == 1:
        axes = [axes]

    cfg = env.base.cfg

    color_alive = "C1"
    color_dead = "C0"
    color_alive = np.array(to_rgba(color_alive))
    color_dead = np.array(to_rgba(color_dead))

    circ_collections = []
    start_idxs, end_idxs = [], []
    start_idx = 0
    for ii, ax in enumerate(axes):
        env.base.setup_ax(ax)

        n_traj = n_temporal_nodes[ii]
        end_idx = start_idx + n_traj
        start_idxs.append(start_idx)
        end_idxs.append(end_idx)

        node_idx = env.temporal_nodes[ii]
        node = env.dag_nodes[node_idx]
        node_name = type(node).__name__
        ax.set_title(f"Node {ii} ({node_name}) | {n_temporal_nodes[ii]} trajs")

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

    def init():
        return circ_collections + [kk_text]

    def update(kk: int):
        kk_text.set_text(f"Step {kk: 3}")
        for ii, ax in enumerate(axes):
            start_idx = start_idxs[ii]
            end_idx = end_idxs[ii]

            trajs = p.bT_test_rollouts[start_idx:end_idx]
            n_traj = end_idx - start_idx

            colors = []

            offsets = np.zeros((n_traj, 2))
            for jj, traj in enumerate(trajs):
                (T,) = traj.shape
                T_state: HerdOs.State = traj.state_now
                T_herder_pos = T_state.base.herder_state[:, 0, :2]

                t_idx = min(kk, T - 1)
                offsets[jj, :] = T_herder_pos[t_idx, :]

                if kk < T:
                    colors.append(color_alive)
                else:
                    colors.append(color_dead)

            circ_collections[ii].set_offsets(offsets)
            circ_collections[ii].set_facecolor(colors)
        return circ_collections + [kk_text]

    pbar = tqdm.tqdm(total=T_max, unit="frame", desc="Generating eval trajs animation")

    def on_progress(current_frame: int, total_frames: int):
        n_done = current_frame + 1
        pbar.total = total_frames
        pbar.n = n_done
        pbar.refresh()

    anim = FuncAnimation(fig, update, init_func=init, frames=T_max, blit=True)
    plot_dir = plots_dir / "eval_trajs_anim"
    plot_dir.mkdir(parents=True, exist_ok=True)
    anim_path = plot_dir / f"eval_trajs_step{p.train_step}.mp4"

    writer = FFMpegWriter(
        fps=30,
        codec="libx264",
        extra_args=["-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p"],
    )

    anim.save(anim_path, writer=writer, dpi=200, progress_callback=on_progress)
    plt.close(fig)


def viz_collect_data(p: CallbackProps):
    train_step = p.train_step

    # Only plot every 1_000 steps.
    if train_step % 5_000 != 0:
        return

    agent = p.agent
    b_data: PPOData = jax.device_get(p.info_update["debug/b_data"])
    Tb_rollout: RolloutOutput = jax.device_get(p.Tb_rollout)

    env = p.env
    env_base = env.base
    cfg_base = env_base.cfg
    cfg_agent = agent.cfg

    # Find rollouts where the herder starts inside circle c1.
    Tb_state: HerdOs.State = Tb_rollout.state_now
    # b_state0 = jtu.tree_map()
    b_pos0 = Tb_state.base.herder_state[0, :, 0, :2]
    c_pos = np.array(env_base.centers)
    c_radii = np.array(env_base.radiuses)
    b_in_c1 = np.linalg.norm(b_pos0 - c_pos[0], axis=-1) < c_radii[0]

    batch_size = len(b_pos0)
    b_Q = b_data.Q

    # # Get one index where the herder starts inside c1.
    # logger.info("n rollouts start inside c1: {}/{}".format(np.sum(b_in_c1), batch_size))
    # idx = np.argmax(b_in_c1)
    #
    # T_rollout: RolloutOutput = jtu.tree_map(lambda Tb_x: Tb_x[:, idx], Tb_rollout)
    # T_state: HerdOs.State = T_rollout.state_now
    # T_term = T_rollout.term
    # T_predicates = T_rollout.predicates
    # T_r = T_predicates["herder_c1"]
    # T_q = -T_predicates["herder_oob"]
    #
    # n_temporal_nodes = env.n_temporal_nodes
    # t_V_dummy = np.zeros(n_temporal_nodes)
    #
    # temporal_node_idx = T_state.temporal_node_idx[0]
    # dag_node_idx = env.temporal_nodes[temporal_node_idx]
    # dag_node = env.dag_nodes[dag_node_idx]
    # assert isinstance(dag_node, DAGReachAvoid)
    # logger.debug("Evaluating reach")
    # T_r2 = agent.evaluate_dag(dag_node.reach, t_V_dummy, T_predicates)
    # logger.debug("Evaluating avoid")
    # T_q2 = agent.evaluate_dag(dag_node.avoid, t_V_dummy, T_predicates)
    #
    # T = len(T_r)
    # T_V = T_V_next = np.zeros(T)
    # gamma, lam = cfg_agent.gamma, cfg_agent.gae_lambda
    # bellman = BellmanMaxMin(T_q, T_r)
    # T_Q_gae = gae_generalized(T_V, T_V_next, T_term, bellman, gamma, lam)
    # T_pos = Tb_state.base.herder_state[:, idx, 0, :2]
    #
    # logger.info("T_q    : {}".format(T_q))
    # logger.info("T_q2   : {}".format(T_q2))
    # logger.info("T_r    : {}".format(T_r))
    # logger.info("T_r2   : {}".format(T_r2))
    # logger.info("T_term : {}".format(T_term))
    # logger.info("T_Q_gae: {}".format(T_Q_gae))
    #
    # Tb_data: PPOData = jtu.tree_map(lambda b_x: ei.rearrange(b_x, "(T b) ... -> T b ...", T=T, b=batch_size), b_data)
    # Tb_Q = Tb_data.Q
    # Tb_state: HerdOs.State = Tb_data.state
    #
    # # If there are any Tb_Q > 1.0, but are outside the circle... ???
    # Tb_pos = Tb_state.base.herder_state[:, :, 0, :2]
    #
    # Tb_Q_high = Tb_Q >= 0.97
    # Tb_in_circ = np.linalg.norm(Tb_pos - c_pos[0], axis=-1) < c_radii[0]
    #
    # Tb_weird = Tb_Q_high & (~Tb_in_circ)
    # if np.any(Tb_weird):
    #     T_idx, b_idx = np.where(Tb_weird)
    #     T_idx, b_idx = T_idx[0], b_idx[0]
    #     Q_weird = Tb_Q[T_idx, b_idx]
    #     logger.warning("Weird high Q-value ({:.4f}) found at T={}, b={}".format(Q_weird, T_idx, b_idx))
    #
    #     T_rollout: RolloutOutput = jtu.tree_map(lambda Tb_x: Tb_x[:, b_idx], Tb_rollout)
    #     T_predicates = T_rollout.predicates
    #     T_term = T_rollout.term
    #     T_r = T_predicates["herder_c1"]
    #     T_q = -T_predicates["herder_oob"]
    #
    #     T_V = T_V_next = np.zeros(T)
    #
    #     bellman = BellmanMaxMin(T_q, T_r)
    #     T_Q_gae = gae_generalized(T_V, T_V_next, T_term, bellman, gamma, lam)
    #
    #     logger.info("T_q    : {}".format(T_q))
    #     logger.info("T_r    : {}".format(T_r))
    #     logger.info("T_term : {}".format(T_term))
    #     logger.info("T_Q_gae: {}".format(T_Q_gae))
    #     logger.debug("Now trying to set_trace inside the method...")
    #
    #     agent.compute_A_Q(Tb_rollout, debug=True)
    #
    #     # ------------------------------------
    #
    #     ipdb.set_trace()

    # ---------------------------------------------------------------
    b_state: HerdOs.State = b_data.state
    b_pos = b_state.base.herder_state[:, 0, :2]

    fig, ax = plt.subplots()
    env.base.setup_ax(ax)

    # cmap = get_BuRd_trunc(0.1).reversed()
    cmap = get_BuRd_smooth().reversed()

    # Visualize the herder positions colored by Q-values.
    sc = ax.scatter(b_pos[:, 0], b_pos[:, 1], c=b_Q, cmap=cmap, s=5, vmin=-1, vmax=1)
    ax.set_title("Q∈[{:.2f}, {:.2f}] | mean={:.2f}".format(np.min(b_Q), np.max(b_Q), b_Q.mean()))
    fig.colorbar(sc, ax=ax)

    # # Visualize the trajectory of the selected rollout.
    # ax.plot(T_pos[:, 0], T_pos[:, 1], color="C2", lw=1.0)
    # ax.plot(T_pos[0, 0], T_pos[0, 1], marker="s", color="C2", ms=3)

    plot_dir = p.run.plots_dir / "collect_data"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plot_dir / f"collect_data_step{p.train_step}.jpg"
    fig.savefig(fig_path, bbox_inches="tight", dpi=500)
    plt.close(fig)
    # -------------------------------------------------------------------------
    # Visualize the observation distribution on a histogram.
    b_obs = b_data.obs

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
