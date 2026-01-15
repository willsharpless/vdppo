import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import EllipseCollection

from rraa_rl.src.env.general_task.herd_os import HerdOs
from rraa_rl.trainer import CallbackProps


def plot_eval_trajs(p: CallbackProps):
    plots_dir = p.run.plots_dir
    env = p.env

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
        node_name = type(node).__name__
        ax.set_title(f"Node {ii} ({node_name}) | {n_temporal_nodes[ii]} trajs")

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
    fig, axes = plt.subplots(1, ncol, figsize=figsize, layout="constrained")
    if ncol == 1:
        axes = [axes]

    cfg = env.base.cfg

    color_alive = "C1"
    color_dead = "C0"

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

    # Create a patch collection
    anim = FuncAnimation(fig, update, init_func=init, frames=T_max)
    plot_dir = plots_dir / "eval_trajs_anim"
    plot_dir.mkdir(parents=True, exist_ok=True)
    anim_path = plot_dir / f"eval_trajs_step{p.train_step}.mp4"
    anim.save(anim_path, fps=30, dpi=300)
