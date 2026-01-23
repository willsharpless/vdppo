import time

import einops as ei
import imageio.v2 as imageio
import imageio.v3 as iio
import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
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
from rraa_rl.src.env.general_task.env import AugObs, EnvStep
from rraa_rl.src.env.general_task.get_env import get_env_and_cbs
from rraa_rl.src.env.general_task.gridworld import GridworldMA, GridworldMABase, GridworldMAState
from rraa_rl.src.rl.utils.utils import get_BuRd_smooth
from rraa_rl.trainer import CallbackProps
from rraa_rl.vd_mappo import PPOData, VDMAPPOAgent


def main():
    env, _, _ = get_env_and_cbs("gridworld_map1", "lcrl")

    pos0 = np.array([0, 0])

    # Go to A, then hit wall.
    # actions = "UURD"
    actions = "UURLDDRRRUURRRDDLRUU"

    action_dict = dict(S=0, U=1, D=2, R=3, L=4)

    actions = [action_dict[a] for a in actions]

    state: LCRLWrapper.State[GridworldMAState] = env.reset(jr.PRNGKey(0))
    with jdc.copy_and_mutate(state) as state:
        state.base.pos = jnp.array(pos0)[None, :]

    for kk, action in enumerate(actions):
        logger.info("automata: {} | pos: {}".format(state.ldba_state.state, state.base.pos))
        action = [jnp.array([action]), jnp.array([0])]
        step: EnvStep[LCRLWrapper.State[GridworldMAState]] = env.step(state, action)
        state_new = step.envstate
        predicates = [k for k, v in step.predicates.items() if jnp.all(v > 0)]
        logger.info(
            "-> automata: {} | pos: {} | term={}, trunc={} | predicates: {}".format(
                state_new.ldba_state.state, state_new.base.pos, step.term, step.trunc, predicates
            )
        )
        state = state_new
        logger.info("---")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
