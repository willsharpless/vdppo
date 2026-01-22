import time

import einops as ei
import imageio.v2 as imageio
import imageio.v3 as iio
import ipdb
import jax
import jax.numpy as jnp
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
from rraa_rl.src.env.general_task.env import AugObs
from rraa_rl.src.env.general_task.gridworld import GridworldMA, GridworldMABase, GridworldMAState
from rraa_rl.src.rl.utils.utils import get_BuRd_smooth
from rraa_rl.trainer import CallbackProps
from rraa_rl.vd_mappo import PPOData, VDMAPPOAgent


def animate_eval_trajs(p: CallbackProps):
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

    raise NotImplementedError("TODO")
