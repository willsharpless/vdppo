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

    state: LCRLWrapper.State[GridworldMAState] = env.reset(jr.PRNGKey(0))

    logger.info("automata: {} | pos: {}".format(state.ldba_state.state, state.base.pos))

    action = [jnp.array([1]), jnp.array([0])]

    step: EnvStep[LCRLWrapper.State[GridworldMAState]] = env.step(state, action)
    state_new = step.envstate

    logger.info("automata: {} | pos: {}".format(state_new.ldba_state.state, state_new.base.pos))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
