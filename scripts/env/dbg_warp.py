import ipdb
import jax.numpy as jnp
from collections import defaultdict
import mujoco
import imageio
import matplotlib.pyplot as plt
import pickle
from typing import Protocol

import jax
from PIL import Image, ImageDraw, ImageFont
import jax.random as jr
import numpy as np
import tqdm
import yaml
from attrs import define
from cyclopts import Parameter
from loguru import logger

import wandb
from rraa_rl.cfg_utils import Cfg
from rraa_rl.collector import Collector, RolloutOutput, extract_info_from_rollout
from rraa_rl.envs.scene import ManipScene, SceneBaseMinState
from rraa_rl.lcrl_mappo import LCRLMAPPOAgent
from rraa_rl.rollout_temporal_analysis import evaluate_ltl_finite
from rraa_rl.rollout_utils import extract_rollouts_eval
from rraa_rl.run import Run
from rraa_rl.src.env.general_task.env import Env, StateWithTemporalNode
from rraa_rl.src.get_agent_cfg import get_vd_agent_cfg
from rraa_rl.vd_mappo import VDMAPPOAgent

def main():
    cfg = ManipScene.Cfg()
    env = ManipScene(cfg)

    n_envs_train = 16
    bs = n_envs_train

    key = jr.PRNGKey(0)
    b_state = env.reset_batch(key, batch_size=bs, init=False)
    b_action = [jnp.zeros((bs, 5), dtype=jnp.int32)]

    vmap_step_fn = jax.jit(jax.vmap(env.step))
    b_out = vmap_step_fn(b_state, b_action)

    print(b_out.envstate.temporal_node_idx)

if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()