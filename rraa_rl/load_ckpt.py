import pathlib
import pickle
from typing import NamedTuple

import cyclopts
import flax
import ipdb
import jax
import jax.random as jr
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import numpy as np
import yaml
from loguru import logger
from matplotlib.colors import to_rgba

from rraa_rl.collector import Collector
from rraa_rl.gridworld_cbs import save_animation_blit
from rraa_rl.rollout_temporal_analysis import evaluate_ltl_finite
from rraa_rl.rollout_utils import extract_rollouts_eval
from rraa_rl.run import Run
from rraa_rl.src.env.general_task.env import Env, StateWithTemporalNode
from rraa_rl.src.env.general_task.get_env import get_env_and_cbs
from rraa_rl.src.env.general_task.gridworld import GridworldMA, GridworldMAState
from rraa_rl.vd_mappo import VDMAPPOAgent
from rraa_rl.lcrl_mappo import LCRLMAPPOAgent


class LoadCkptResult(NamedTuple):
    run: Run
    agent: VDMAPPOAgent | LCRLMAPPOAgent
    env: Env
    cfg_dict: dict


def load_ckpt(run_path: pathlib.Path, step: int | None = None, alg: str = "vd"):
    # Load the configs.
    yaml_path = run_path / "config.yaml"
    with open(yaml_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    run = Run.fromdict(cfg_dict["run"])
    env_name = run.env_name
    # agent_name = run.agent_name # sometimes capitalizes, differing from previous signature

    env: GridworldMA
    env, _, _ = get_env_and_cbs(env_name, agent_name=alg)

    if alg == "vd":
        agent_cfg = VDMAPPOAgent.Cfg.fromdict(cfg_dict["agent"])
        agent = VDMAPPOAgent.create(123, agent_cfg, env)
    elif alg == "lcrl":
        agent_cfg = LCRLMAPPOAgent.Cfg.fromdict(cfg_dict["agent"])
        agent = LCRLMAPPOAgent.create(123, agent_cfg, env)

    ckpts_path = run_path / "ckpts"
    if step is None:
        latest_ckpt = sorted(ckpts_path.glob("params_*.pkl"))
        assert latest_ckpt, f"No checkpoints found in {ckpts_path}"

        load_path = latest_ckpt[-1]
    else:
        load_path = ckpts_path / f"params_{step:09}.pkl"
        if not load_path.exists():
            available = sorted(ckpts_path.glob("params_*.pkl"))
            raise FileNotFoundError(f"Checkpoint not found: {load_path}. Available: {available}")
    logger.info(f"Restoring from {load_path}")

    with load_path.open("rb") as f:
        load_dict = pickle.load(f)

    agent: VDMAPPOAgent | LCRLMAPPOAgent = flax.serialization.from_state_dict(agent, load_dict["agent"])

    return LoadCkptResult(run=run, agent=agent, env=env, cfg_dict=cfg_dict)