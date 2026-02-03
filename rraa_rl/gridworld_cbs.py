from pathlib import Path
from typing import Callable
from colour import hsl2hex

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from flax import struct
from matplotlib.colors import to_rgba
from matplotlib.colors import LinearSegmentedColormap

from rraa_rl.distribution import tfd
from rraa_rl.jax_utils import jax_vmap
from rraa_rl.lcrl.lcrl_wrapper import LCRLWrapper
from rraa_rl.agents.lcrl_mappo import LCRLMAPPOAgent
from rraa_rl.ldba.ldba import LDBAState
from rraa_rl.src.env.general_task.env import AugObs, AugObsAutomata, StateWithTemporalNode
from rraa_rl.src.env.general_task.gridworld import GridworldMA, GridworldMABase, GridworldMAState
from rraa_rl.trainer import CallbackProps
from rraa_rl.agents.vd_mappo import VDMAPPOAgent

plt.style.use("seaborn-v0_8-darkgrid")

def animate_eval_trajs(p: CallbackProps):
    if isinstance(p.agent, VDMAPPOAgent):
        animate_eval_trajs_vd(p)
    else:
        animate_eval_trajs_base(p)


def save_animation_blit(
    fig: plt.Figure,
    animated_artists: list,
    anim_path: Path,
    T_max: int,
    update_fn: Callable[[int], None],
    fps: int = 30,
    desc: str = "Generating animation",
):
    """
    Run an animation loop with blitting and write frames to a video file.

    Args:
        fig: The matplotlib figure to animate.
        animated_artists: List of artists that are animated (will have set_animated(True) called).
        anim_path: Path to save the output video.
        T_max: Number of frames to render.
        update_fn: Callback function that takes the frame index and updates the artists.
        fps: Frames per second for the output video.
        desc: Description for the progress bar.
    """
    for artist in animated_artists:
        artist.set_animated(True)

    fig.canvas.draw()
    bg = fig.canvas.copy_from_bbox(fig.bbox)

    with iio.imopen(anim_path, "w", plugin="pyav") as writer:
        writer.init_video_stream("libx264", fps=fps)

        pbar = tqdm.trange(T_max, unit="frame", desc=desc)
        for kk in pbar:
            # Restore background
            fig.canvas.restore_region(bg)

            # Update artists via callback
            update_fn(kk)

            # Draw only animated artists onto the restored background
            for artist in animated_artists:
                artist.axes.draw_artist(artist)

            # Update the buffer
            fig.canvas.blit(fig.bbox)

            # Grab frame (ensure contiguous uint8 RGB)
            frame_rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame_rgb = np.ascontiguousarray(frame_rgba[..., :3])

            # If the frame dimensions are not divisible by 16, pad for libx264 compatibility
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

    def update_fn(kk: int):
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

    plot_dir = plots_dir / "eval_trajs_anim"
    plot_dir.mkdir(parents=True, exist_ok=True)
    anim_path = plot_dir / f"eval_trajs_step{p.train_step}.mp4"

    animated_artists = all_circs + [kk_text]
    save_animation_blit(fig, animated_artists, anim_path, T_max, update_fn)

def animate_gridworld_traj(
    anim_path: Path,
    env: GridworldMA,
    cfg: GridworldMABase.Cfg,
    T_state: GridworldMA.State,
    # T_labels: list[str] | None = None,
    fps: int = 30,
):
    figsize = np.array([4, 3])
    fig, ax = plt.subplots(figsize=figsize)
    # env_base = DeliveryBase(cfg)
    env_base = env.base
    # mult = cfg.pos_multiplier
    mult = 1
    env_base.setup_ax(ax)

    # The grid cells are 1x1
    agent_radius = 0.2

    # node_idx = env.temporal_nodes[jj]
    # node = env.dag_nodes[node_idx]
    # node_name = type(node).__name__[3:]
    # ax.set_title(f"Node {jj} ({node_name})")

    circs = []
    for agent_idx in range(env.n_agents):
        circ = plt.Circle((0, 0), agent_radius, facecolor="C1", edgecolor="none")
        ax.add_patch(circ)
        circs.append(circ)

    all_circs = circs

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

    def update_fn(kk: int):
        kk_text.set_text(f"Step {kk: 3}")
        T_pos = T_state.base.pos[kk]

        for agent_idx, circ in enumerate(circs):
            pos = T_pos[agent_idx, :2]
            circ.center = pos

    T = len(T_state.base.pos)
    artists = all_circs + [kk_text]
    save_animation_blit(fig, artists, anim_path, T, fps=fps, update_fn=update_fn)


def animate_eval_trajs_base(p: CallbackProps):
    plots_dir = p.run.plots_dir
    env: LCRLWrapper = p.env
    cfg = env.base.cfg

    n_traj_anim = 8

    bT_test_rollouts = p.bT_test_rollouts

    bT_states: list[LCRLWrapper.State[GridworldMAState]] = [traj.state_now for traj in bT_test_rollouts]
    b_pos0 = [T_state.base.pos[0] for T_state in bT_states]

    n_batch = len(b_pos0)
    batch_indices = np.arange(n_traj_anim)

    # Try and make every index correspond to a different base state.
    pos_seen = []
    idx_try = 0
    for ii in range(n_traj_anim):
        if len(pos_seen) == 0:
            batch_indices[ii] = idx_try
            idx_try += 1
            pos_seen.append(b_pos0[batch_indices[ii]])
        else:
            while idx_try < n_batch:
                pos_candidate = b_pos0[idx_try]
                is_new = all(not np.allclose(pos_candidate, pos_prev) for pos_prev in pos_seen)
                if is_new:
                    batch_indices[ii] = idx_try
                    idx_try += 1
                    pos_seen.append(b_pos0[batch_indices[ii]])
                    break
                else:
                    idx_try += 1

    T_max = 0
    for ii in range(n_traj_anim):
        traj = bT_test_rollouts[batch_indices[ii]]
        T_max = max(T_max, len(traj.term))

    ncol = n_traj_anim

    colors_automata = plt.get_cmap("tab20", env.ldba.n_states).colors
    color_alive = to_rgba("C0", 0.0)
    color_dead = np.array(to_rgba("C0"))

    figsize = 0.9 * np.array([3 * ncol, 3])
    fig, axes = plt.subplots(1, ncol, figsize=figsize, dpi=150, layout="none")

    # The grid cells are 1x1
    agent_radius = 0.2

    agent_collections: dict[int, list[plt.Circle]] = {}
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

    misc_texts = {}
    for ii in range(n_traj_anim):
        # bottom right.
        misc_texts[ii] = axes[ii].text(
            0.98,
            0.02,
            "",
            transform=axes[ii].transAxes,
            verticalalignment="bottom",
            horizontalalignment="right",
            color="white",
            fontsize=8,
            bbox=dict(facecolor="black", alpha=0.5, pad=2),
        )

    fig.tight_layout()

    def update_fn(kk: int):
        kk_text.set_text(f"Step {kk: 3}")
        for ii in range(n_traj_anim):
            traj = bT_test_rollouts[batch_indices[ii]]
            (T,) = traj.shape
            T_state: LCRLWrapper.State[GridworldMAState] = traj.state_now
            T_herder_pos = T_state.base.pos[:, :, :2]

            T_state_next: LCRLWrapper.State[GridworldMAState] = traj.state_next
            T_herder_pos_next = T_state_next.base.pos[:, :, :2]

            is_dead = kk >= T
            t_idx = min(kk, T - 1)

            circs = agent_collections[ii]
            for agent_idx, circ in enumerate(circs):
                if is_dead:
                    pos = T_herder_pos_next[-1, agent_idx, :]
                else:
                    pos = T_herder_pos[t_idx, agent_idx, :]
                circ.center = pos

                if isinstance(T_state, LCRLWrapper.State):
                    automata_state = T_state.ldba_state.state[t_idx]
                    circ.set_facecolor(colors_automata[automata_state])

                if kk < T:
                    circ.set_edgecolor(color_alive)
                else:
                    circ.set_edgecolor(color_dead)

            if isinstance(T_state, LCRLWrapper.State):
                automata_state = T_state.ldba_state.state[t_idx]
                misc_texts[ii].set_text(f"Automata: {automata_state}")

    plot_dir = plots_dir / "eval_trajs_anim"
    plot_dir.mkdir(parents=True, exist_ok=True)
    anim_path = plot_dir / f"eval_trajs_step{p.train_step}.mp4"

    animated_artists = all_circs + [kk_text] + list(misc_texts.values())
    save_animation_blit(fig, animated_artists, anim_path, T_max + 1, update_fn)


class VizValues(struct.PyTreeNode):
    @staticmethod
    def create():
        return VizValues()

    @jax.jit
    def get_value_vd(self, agent: VDMAPPOAgent):
        env: GridworldMA = agent.env
        env_base: GridworldMABase = env.base

        bb_state_base = env_base.get_all_states()
        len_x, len_y = bb_state_base.pos.shape[:2]

        bb_obs = jax_vmap(env_base.get_obs, rep=2)(bb_state_base)
        bb_obs_aug = AugObs(None, None, bb_obs, None)
        bbt_V = agent.network.select("critic")(bb_obs_aug)
        assert bbt_V.shape == (len_x, len_y, env.n_temporal_nodes)

        def get_actions(temporal_idx: jnp.ndarray):
            bb_temporal_idx = jnp.full((len_x, len_y), temporal_idx)
            bb_state = StateWithTemporalNode(bb_temporal_idx, bb_state_base)
            bb_obs_ = jax_vmap(env.get_obs, rep=2)(bb_state)

            def get_mode_and_prob(obs_):
                act_dist: tfd.JointDistributionSequential = agent.network.select("actor")(obs_)
                n_act = act_dist.mode()
                n_log_probs = act_dist.log_prob_parts(n_act)
                entropies_list = [dist.entropy() for dist in act_dist.model]
                n_entropy = jnp.stack(entropies_list, axis=0)
                n_probs = jnp.array([jnp.exp(lp).squeeze() for lp in n_log_probs])
                assert n_probs.shape == (env.n_agents,)
                return n_act, n_probs, n_entropy

            bbn_act, bbn_probs, bbn_entropy = jax_vmap(get_mode_and_prob, rep=2)(bb_obs_)
            return bbn_act, bbn_probs, bbn_entropy

        t_temporal_idx = jnp.arange(env.n_temporal_nodes)
        bbtn_act, bbtn_probs, bbtn_entropy = jax.vmap(get_actions, out_axes=2)(t_temporal_idx)
        assert bbtn_probs.shape == (len_x, len_y, env.n_temporal_nodes, env.n_agents)

        return bbt_V, bbtn_act, bbtn_probs, bbtn_entropy

    @jax.jit
    def get_value_lcrl(self, agent: LCRLMAPPOAgent):
        env: LCRLWrapper = agent.env
        env_base: GridworldMABase = env.base

        bb_state_base = env_base.get_all_states()
        len_x, len_y = bb_state_base.pos.shape[:2]

        bb_obs = jax_vmap(env_base.get_obs, rep=2)(bb_state_base)
        bb_obs_aug = AugObsAutomata(None, bb_obs, None)
        bbt_V = agent.network.select("critic")(bb_obs_aug)
        assert bbt_V.shape == (len_x, len_y, env.ldba.n_states)

        def get_actions(automata_idx: jnp.ndarray):
            bb_automata_idx = jnp.full((len_x, len_y), automata_idx)
            bb_ldba_state = LDBAState(bb_automata_idx, jnp.zeros((len_x, len_y), dtype=bool))
            bb_state = LCRLWrapper.State(bb_ldba_state, bb_state_base)
            bb_obs_ = jax_vmap(env.get_obs, rep=2)(bb_state)

            def get_mode_and_prob(obs_):
                act_dist: tfd.JointDistributionSequential = agent.network.select("actor")(obs_)
                n_act = act_dist.mode()
                n_log_probs = act_dist.log_prob_parts(n_act)[:-1]
                entropies_list = [dist.entropy() for dist in act_dist.model]
                n_entropy = jnp.stack(entropies_list[:-1], axis=0)
                n_probs = jnp.array([jnp.exp(lp).squeeze() for lp in n_log_probs])
                assert n_probs.shape == (env.n_agents,)
                return n_act, n_probs, n_entropy

            bbn_act, bbn_probs, bbn_entropy = jax_vmap(get_mode_and_prob, rep=2)(bb_obs_)
            return bbn_act, bbn_probs, bbn_entropy

        t_automata_idx = jnp.arange(env.ldba.n_states)
        bbtn_act, bbtn_probs, bbtn_entropy = jax.vmap(get_actions, out_axes=2)(t_automata_idx)
        assert bbtn_probs.shape == (len_x, len_y, env.ldba.n_states, env.n_agents)

        return bbt_V, bbtn_act, bbtn_probs, bbtn_entropy

    def __call__(self, p: CallbackProps):
        if p.env.n_agents > 1:
            return

        bbtn_act: list[jnp.ndarray]  # list for each agent.

        match p.agent:
            case LCRLMAPPOAgent():
                env: LCRLWrapper = p.agent.env
                bbt_V, bbtn_act, bbtn_probs, bbtn_entropy = jax.device_get(self.get_value_lcrl(p.agent))
                n_automata_states = env.ldba.n_states
                discrete_state_name = "Automata"
            case VDMAPPOAgent():
                env: GridworldMA = p.agent.env
                bbt_V, bbtn_act, bbtn_probs, bbtn_entropy = jax.device_get(self.get_value_vd(p.agent))
                n_automata_states = env.n_temporal_nodes
                discrete_state_name = "Temporal"
            case _:
                raise NotImplementedError("")

        env_base: GridworldMABase = env.base

        ncol = n_automata_states
        # figsize = 0.9 * np.array([3 * ncol, 3])
        # fig, axes = plt.subplots(1, ncol, figsize=figsize, layout="constrained")

        blue = hsl2hex([0.57, 0.5, 0.55])
        light_blue = hsl2hex([0.4, 1.0, 0.9])
        red = hsl2hex([0.028, 0.62, 0.59])
        light_red = hsl2hex([0.2, 1.0, 0.95])
        white = hsl2hex([0.0, 0.0, 1.0])
        # sdf_cm = LinearSegmentedColormap.from_list("SDF", [(0, red), (0.4, light_red), (0.5, white), (0.7, light_blue), (1.0, blue)], N=256)
        sdf_cm = LinearSegmentedColormap.from_list("SDF", [(0, red), (0.4, light_red), (0.5, white), (1.0, blue)], N=256)

        cmap = sdf_cm
        bbt_probs = bbtn_probs.squeeze(3)
        bbt_act = bbtn_act[0]
        action_to_str = [".", "↑", "↓", "→", "←"]
        plt.style.use("seaborn-v0_8-darkgrid")

        # vmin, vmax = bbt_V.min(), bbt_V.max()
        vmin, vmax = -1, 1.
        for ii in range(n_automata_states):
            fig, ax = plt.subplots(figsize=np.array([6,4]), dpi=400)

            im = ax.imshow(bbt_V[:, :, ii].T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            env_base.setup_ax(ax)
            # cbar = fig.colorbar(im, ax=ax)

            # For each cell, annotate with the action mode.
            for (x, y), prob in np.ndenumerate(bbt_probs[:, :, ii]):
                # if wall #, skip annotation
                # if env_base.map.d_raw['#'][x,y]:
                #     continue

                if bbt_V[x, y, ii] < 0:
                    continue

                action_mode = bbt_act[x, y, ii][0]
                ax.text(
                    x,
                    y,
                    action_to_str[action_mode],
                    # color="white" if prob < 0.5 else "black",  # viridis is dark blue to yellow
                    # fontfamily="DejaVu Sans Mono",
                    fontsize=20,
                    ha="center",
                    va="center",
                )

            plot_dir = p.run.plots_dir / "V" / f"Node{ii}"
            plot_dir.mkdir(parents=True, exist_ok=True)
            fig_path = plot_dir / f"V_step{p.train_step}_Node{ii}.png"
            fig.savefig(fig_path, bbox_inches="tight", dpi=400, pad_inches=1e-2)
            plt.close(fig)

        # -----------------------------------------
        if env.n_agents > 1:
            return

        bbt_probs = bbtn_probs.squeeze(3)
        bbt_entropy = bbtn_entropy.squeeze(3)
        bbt_act = bbtn_act[0]

        ncol = n_automata_states
        nrow = 2
        figsize = 0.9 * np.array([3 * ncol, 3 * nrow])
        fig, axes = plt.subplots(2, ncol, figsize=figsize, layout="constrained")

        action_to_str = [".", "↑", "↓", "→", "←"]

        # first row: plot probabilities
        for ii, ax in enumerate(axes[0, :]):
            im = ax.imshow(bbt_probs[:, :, ii].T, origin="lower", cmap="viridis", vmin=0, vmax=1)
            env_base.setup_ax(ax)
            ax.set_title(f"{discrete_state_name} state {ii}")
            cbar = fig.colorbar(im, ax=ax)

            # For each cell, annotate with the action mode.
            for (x, y), prob in np.ndenumerate(bbt_probs[:, :, ii]):
                action_mode = bbt_act[x, y, ii][0]
                ax.text(
                    x,
                    y,
                    action_to_str[action_mode],
                    color="white" if prob < 0.5 else "black",  # viridis is dark blue to yellow
                    fontfamily="DejaVu Sans Mono",
                    fontsize=8,
                    ha="center",
                    va="center",
                )

        # second row: plot entropies.
        vmin, vmax = bbt_entropy.min(), bbt_entropy.max()
        for ii, ax in enumerate(axes[1, :]):
            bb_entropy = bbt_entropy[:, :, ii]
            im = ax.imshow(bb_entropy.T, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
            env_base.setup_ax(ax)
            ax.set_title(
                f"{discrete_state_name} {ii} Entropy ∈ [{bb_entropy.min():.1e}, {bb_entropy.max():.1e}]",
                fontsize="small",
                fontfamily="DejaVu Sans",
            )
            cbar = fig.colorbar(im, ax=ax)

        plot_dir = p.run.plots_dir / "pol"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig_path = plot_dir / f"pol_step{p.train_step}.jpg"
        fig.savefig(fig_path, bbox_inches="tight", dpi=500)
        plt.close(fig)


def collect_cb(p: CallbackProps):
    if not isinstance(p.agent, LCRLMAPPOAgent):
        return

    # # Count how many of each automata state is present in the rollout.
    # Tb_rollout: RolloutOutput = jax.device_get(p.Tb_rollout)
    # Tb_state_now: LCRLWrapper.State[GridworldMAState] = Tb_rollout.state_now
    # Tb_automata_idx = Tb_state_now.ldba_state.state
    #
    # idx_min, idx_max = np.min(Tb_automata_idx), np.max(Tb_automata_idx)
    # assert idx_min >= 0
    # assert idx_max < p.env.ldba.n_states
    #
    # counts = np.array([np.sum(Tb_automata_idx == ii) for ii in range(p.env.ldba.n_states)])
    # # logger.info("Automata state counts: {}".format(counts))
    #
    # if counts[-1] == 0:
    #     return
    # else:
    #     logger.success("Reached ".format(counts))
