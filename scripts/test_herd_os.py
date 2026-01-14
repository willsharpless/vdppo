import ipdb
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from rraa_rl.src.env.general_task.herd_os import HerdOs, HerdOsCfg, HerdOsState


def main():
    cfg = HerdOsCfg()
    env = HerdOs(cfg)

    key = jr.PRNGKey(12345)
    state = env.reset(key)

    action = jnp.array([1, 1])

    env_states = [state]
    for kk in range(10):
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

    ax.set_xlim(-cfg.halfsize[0], cfg.halfsize[0])
    ax.set_ylim(-cfg.halfsize[1], cfg.halfsize[1])

    # Plot the herd circle.
    herd_circle = plt.Circle((0, 0), cfg.herded_radius, color="lightgray", alpha=0.5)
    ax.add_patch(herd_circle)

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
            herd_pos = envstate.herd_state[ii, :2]
            herd_circs[ii].center = (herd_pos[0], herd_pos[1])
        for jj in range(cfg.n_herders):
            herder_pos = envstate.herder_state[jj, :2]
            herder_circs[jj].center = (herder_pos[0], herder_pos[1])
        kk_text.set_text(f"Step: {kk}")
        return herd_circs + herder_circs + [kk_text]

    anim = FuncAnimation(fig, update, frames=len(env_states), init_func=init, blit=True)
    anim.save("herd_os_rollout.mp4", fps=10, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
