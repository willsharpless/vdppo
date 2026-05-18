import ipdb
import jax.numpy as jnp
import jax.random as jr
import jax_dataclasses as jdc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from vdppo.env.general_task.env import StateWithTemporalNode
from vdppo.env.general_task.herding import Herding, HerdingCfg


def main():
    cfg_os = HerdingCfg()
    env = Herding(cfg_os)

    cfg = cfg_os.base

    key = jr.PRNGKey(12345)
    state = env.reset(key)

    # herd_state = jnp.array([[-1.5 * cfg.agent_radius, 0.0], [1.5 * cfg.agent_radius, 0.0]])
    herd_state = jnp.array([[-1.5 * cfg.agent_radius, 0.0]])
    with jdc.copy_and_mutate(state) as state_new:
        state_new.base.herd_state = herd_state
    # state = state._replace(herd_state=herd_state)
    state = state_new

    action = jnp.array([1, 1])

    env_states = [state]
    for kk in range(50):
        step = env.step(state, action)
        state = step.envstate
        env_states.append(step.envstate)

    # Visualize the rollout as an animation.
    figsize = np.array([6, 6])
    fig, ax = plt.subplots(layout="constrained", figsize=figsize)

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

    env.base.setup_ax(ax)

    # Plot the herd agents.
    herd_circs = []
    for _ in range(cfg.n_herd):
        circ = plt.Circle((0, 0), cfg.agent_radius, color="blue")
        herd_circs.append(circ)
        ax.add_patch(circ)

    # Plot the herder agents.
    herder_circs = []
    for _ in range(cfg.n_herders):
        circ = plt.Circle((0, 0), cfg.agent_radius, color="red")
        herder_circs.append(circ)
        ax.add_patch(circ)

    def init():
        return herd_circs + herder_circs + [kk_text]

    def update(kk: int):
        envstate = env_states[kk]
        for ii in range(cfg.n_herd):
            herd_pos = envstate.base.herd_state[ii, :2]
            herd_circs[ii].center = (herd_pos[0], herd_pos[1])
        for jj in range(cfg.n_herders):
            herder_pos = envstate.base.herder_state[jj, :2]
            herder_circs[jj].center = (herder_pos[0], herder_pos[1])
        kk_text.set_text(f"Step: {kk}")
        return herd_circs + herder_circs + [kk_text]

    anim = FuncAnimation(fig, update, frames=len(env_states), init_func=init, blit=True)
    anim.save("herding_rollout.mp4", fps=10, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
