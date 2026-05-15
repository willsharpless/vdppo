import ipdb
import matplotlib.pyplot as plt
import numpy as np

from rraa_rl.env.general_task.get_env import get_env_and_cbs
from rraa_rl.env.general_task.gridworld import GridworldMA


def main():
    env: GridworldMA
    env, _, _ = get_env_and_cbs("gridworld_map5")

    predicates = env.base.map.predicates

    fig, ax = plt.subplots()
    env.setup_ax(ax)
    im = ax.imshow(predicates["w"].T, alpha=0.2, origin="lower", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax)

    assert predicates["w"][0, 3] > 0
    assert predicates["w"][3, 0] < 0

    fig.savefig("test.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
