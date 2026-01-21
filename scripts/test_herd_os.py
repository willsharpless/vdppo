import pathlib
import pickle
from typing import Protocol

import cyclopts
import flax
import ipdb
import jax
import jax.random as jr
import numpy as np
import tqdm
import wandb
from attrs import define
from loguru import logger

from rraa_rl import herd_os_cbs
from rraa_rl.collector import Collector, RolloutOutput, extract_info_from_rollout
from rraa_rl.distribution import tfd
from rraa_rl.rollout_temporal_analysis import evaluate_ltl_finite, evaluate_triggers
from rraa_rl.rollout_utils import extract_rollouts_eval
from rraa_rl.run import Run
from rraa_rl.src.env.general_task.herd_os import HerdOs
from rraa_rl.trainer import Trainer
from rraa_rl.vd_mappo import VDMAPPOAgent

app = cyclopts.App()


@app.default()
def main(run_path: pathlib.Path):
    env = HerdOs()
    seed = 123
    cfg = VDMAPPOAgent.Cfg()
    agent = VDMAPPOAgent.create(seed, cfg, env)

    ckpts_path = run_path / "ckpts"
    latest_ckpt = sorted(ckpts_path.glob("params_*.pkl"))
    assert latest_ckpt, f"No checkpoints found in {ckpts_path}"

    load_path = latest_ckpt[-1]
    logger.info(f"Restoring from {load_path}")

    with load_path.open("rb") as f:
        load_dict = pickle.load(f)

    agent: VDMAPPOAgent = flax.serialization.from_state_dict(agent, load_dict["agent"])

    n_envs_test = 1

    env_eval_transition = env.with_temporal_transitions()
    collector = Collector.create(
        key=jr.PRNGKey(1234),
        env=env_eval_transition,
        cfg=Collector.Cfg(n_envs=n_envs_test),
    )

    b_state0 = env.get_eval_states(collector.cfg.n_envs)
    Tb_rollout, info_collect = agent.collect_eval_with_states(collector, b_state0, env.eval_T)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
