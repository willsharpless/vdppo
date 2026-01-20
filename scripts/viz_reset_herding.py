import ipdb
import jax
import jax.random as jr
import matplotlib.pyplot as plt
from cyclopts import App

from rraa_rl import herd_os_cbs
from rraa_rl.run import Run
from rraa_rl.src.env.general_task.env import Env
from rraa_rl.src.env.general_task.get_env import get_env
from rraa_rl.src.env.general_task.herd_base import HerdBaseCfg, HerdBasePlay, HerdingHerdCfg
from rraa_rl.src.env.general_task.herd_os import HerdOs, HerdOsPlay
from rraa_rl.trainer import Trainer
from rraa_rl.vd_mappo import VDMAPPOAgent


def main():
    env: HerdOs = get_env("HerdOs")

    agent_radius = env.cfg.base.agent_radius

    n_plot = 2
    ncol = n_plot

    figsize = (4 * ncol, 3)

    fig, axes = plt.subplots(1, ncol, figsize=figsize, layout="constrained")

    for ii, ax in enumerate(axes):
        key = jr.PRNGKey(12340 + ii)
        state = jax.device_get(env.base.reset(key))

        # Plot the herd and herders.
        herd_pos = state.herd_state[:, :2]
        herder_pos = state.herder_state[:, :2]

        for hp in herder_pos:
            circle = plt.Circle((hp[0], hp[1]), agent_radius, color="C1", alpha=0.5)
            ax.add_artist(circle)

        for hp in herd_pos:
            circle = plt.Circle((hp[0], hp[1]), agent_radius, color="C3", alpha=0.5)
            ax.add_artist(circle)

        # herd_circle_center = info["herd/circle_center"]
        # herd_circle_radius = info["herd/radius"]
        #
        # circ = plt.Circle(
        #     (herd_circle_center[0], herd_circle_center[1]),
        #     herd_circle_radius,
        #     color="C3",
        #     alpha=0.2,
        #     linestyle="--",
        #     fill=False,
        #     linewidth=2,
        # )
        # ax.add_artist(circ)

        env.base.setup_ax(ax)

    fig.savefig("viz_reset_herding.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
