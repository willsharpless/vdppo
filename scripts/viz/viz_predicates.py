import functools as ft
import pathlib

import einops as ei
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import matplotlib.pyplot as plt
import numpy as np

from rraa_rl.common.geometry import AABB, dist_pt_to_aabb
from rraa_rl.common.jax_utils import jax_vmap
from rraa_rl.env.general_task.env import StateWithTemporalNode
from rraa_rl.env.general_task.get_env import get_env
from rraa_rl.env.general_task.herd_base import HerdBaseState
from rraa_rl.env.general_task.herding import Herding
from rraa_rl.common.plot_utils import get_BuRd_smooth


def main():
    env: Herding = get_env("herding_dbg")
    cfg = env.cfg.base
    halfsize = cfg.halfsize

    def get_predicates():
        n_x = 513
        n_y = 513

        b_x = jnp.linspace(-halfsize[0], halfsize[0], num=n_x)
        b_y = jnp.linspace(-halfsize[1], halfsize[1], num=n_y)
        bb_X, bb_Y = jnp.meshgrid(b_x, b_y)

        bb_pos = jnp.stack([bb_X, bb_Y], axis=-1)

        key = jax.random.PRNGKey(0)
        bb_key = ei.rearrange(jax.random.split(key, num=n_x * n_y), "(x y) ... -> x y ...", x=n_x, y=n_y)
        bb_state: StateWithTemporalNode[HerdBaseState] = jax_vmap(env.reset, rep=2)(bb_key)

        with jdc.copy_and_mutate(bb_state) as bb_state:
            bb_state.base.herder_state = bb_state.base.herder_state.at[:, :, 0, :2].set(bb_pos)
            bb_state.base.herder_state = bb_state.base.herder_state.at[:, :, 0, 2:4].set(0.0)

            bb_state.base.herd_state = bb_state.base.herd_state.at[:, :, 0, :2].set(np.array([2.0, 0.0]))

            bb_state.temporal_node_idx = bb_state.temporal_node_idx.at[:].set(0)

        bb_predicates = jax_vmap(env.get_predicates, rep=2)(bb_state)

        return bb_X, bb_Y, bb_predicates

    bb_X, bb_Y, bb_predicates = jax.jit(get_predicates)()
    n_predicates = len(bb_predicates)

    nrow = 2
    ncol = int(np.ceil(n_predicates / 2))

    cmap = get_BuRd_smooth().reversed()

    figsize = np.array([4 * ncol, 3 * nrow])
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, layout="constrained", squeeze=False)
    axes = axes.flatten()

    for ii, ax in enumerate(axes[:n_predicates]):
        if ii == len(axes) - 1:
            break

        ax: plt.Axes
        env.base.setup_ax(ax)
        pred_name = list(bb_predicates.keys())[ii]

        ax.set_title(f"Predicate: {pred_name}")

        im = ax.contourf(bb_X, bb_Y, bb_predicates[pred_name], levels=50, cmap=cmap, vmin=-1, vmax=1, alpha=0.5)
        fig.colorbar(im, ax=ax)

    b_x = np.linspace(-2, 2, num=64)
    b_y = b_x
    bb_X, bb_Y = np.meshgrid(b_x, b_y)
    bb_pos = jnp.stack([bb_X, bb_Y], axis=-1)

    aabb = AABB(np.array([-1, -1]), np.array([+1, +1]))
    bb_dist = jax_vmap(ft.partial(dist_pt_to_aabb, aabb=aabb), rep=2)(bb_pos)

    ax = axes[-1]
    cm = ax.contourf(bb_X, bb_Y, bb_dist, levels=50)
    fig.colorbar(cm, ax=ax)

    ax.contour(bb_X, bb_Y, bb_dist < 1.0, levels=50)

    plot_dir = pathlib.Path("dbg_plots")
    fig_path = plot_dir / "viz_predicates.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
