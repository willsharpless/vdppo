from typing import Protocol

import attrs
import ipdb
import jax
import jax.random as jr
import jax_dataclasses as jdc
import numpy as np
import tqdm
from attrs import define
from valtr.valtr import to_dag

import wandb
from rraa_rl.collector import Collector, RolloutOutput, extract_info_from_rollout
from rraa_rl.rollout_temporal_analysis import evaluate_triggers
from rraa_rl.rollout_utils import extract_rollouts_eval
from rraa_rl.run import Run
from rraa_rl.src.env.general_task.herd_os import HerdOs
from rraa_rl.vd_mappo import VDMAPPOAgent


@define
class CallbackProps:
    run: Run

    train_step: int
    agent: VDMAPPOAgent
    bT_test_rollouts: list[RolloutOutput]
    test_trigger_dict: dict[tuple[str, str], np.ndarray]

    collector_train: Collector
    Tb_rollout: RolloutOutput
    info_update: dict

    @property
    def env(self):
        return self.agent.env


class Callback(Protocol):
    def __call__(self, p: CallbackProps) -> None: ...


class Trainer:
    agent: VDMAPPOAgent

    def __init__(self, agent: VDMAPPOAgent):
        self.agent = agent
        self.b_state0 = None

    def train(
        self,
        run: Run,
        env: HerdOs,
        eval_cbs: list[Callback] = None,
        collect_cbs: list[Callback] = None,
        debug: bool = False,
    ):
        eval_cbs = eval_cbs if eval_cbs is not None else []

        key_base = jr.PRNGKey(124521)
        key_base, key_collector, key_eval = jr.split(key_base, 3)

        n_envs_train = 1024
        n_envs_test = 128

        collector = Collector.create(
            key=key_collector,
            env=env,
            cfg=Collector.Cfg(n_envs=n_envs_train),
        )
        collector_eval = Collector.create(
            key=key_collector,
            env=env,
            cfg=Collector.Cfg(n_envs=n_envs_test),
        )

        n_train_steps = 100_000

        eval_every = 10_000
        log_every = 100

        if not debug:
            wandb.init(project="vd_mappo", name=run.wandb_name)

        cb_props = CallbackProps(run, -1, self.agent, None, None, collector, None, None)

        pbar = tqdm.trange(n_train_steps)
        for train_step in pbar:
            if train_step % eval_every == 0:
                pbar.set_description(f"Eval at step {train_step}")
                trajs, trigger_dict, info_eval = self.eval(collector_eval, key_eval)

                cb_props.train_step = train_step
                cb_props.agent = self.agent
                cb_props.bT_test_rollouts = trajs
                cb_props.test_trigger_dict = trigger_dict
                cb_props.collector_train = collector

                for cb in eval_cbs:
                    cb_name = cb.__name__ if hasattr(cb, "__name__") else str(type(cb))
                    pbar.set_description(f"Running eval callback {cb_name}")
                    cb(cb_props)

            # Collect rollout data
            pbar.set_description(f"Collecting rollouts")
            collector, Tb_rollout, info_collect = self.agent.collect_batch(collector, self.agent.cfg.rollout_T)

            info_collect2 = {}
            if train_step % log_every == 0:
                # Extract easy-to-log info from the rollouts, e.g., average reset age
                info_collect2 = extract_info_from_rollout(Tb_rollout)

            # Update agent using the collected rollout.
            key_base, key_update = jr.split(key_base, 2)
            pbar.set_description(f"Updating {type(self.agent).__name__}...")
            self.agent, info_update = self.agent.update(Tb_rollout, key_update)

            if len(collect_cbs) > 0:
                cb_props.train_step = train_step
                cb_props.agent = self.agent
                cb_props.Tb_rollout = Tb_rollout
                cb_props.info_update = info_update

                for cb in collect_cbs:
                    cb_name = cb.__name__ if hasattr(cb, "__name__") else str(type(cb))
                    pbar.set_description(f"Running eval callback {cb_name}")
                    cb(cb_props)

            # Update progress bar.
            pbar.update(1)

            # Log info to wandb
            if train_step % log_every == 0:
                # Remove any keys that start with debug/
                info_update_log = {k: v for k, v in info_update.items() if not k.startswith("debug/")}

                info_collect2 = {f"collect/{k}": v for k, v in info_collect2.items()}
                info_collect = {f"collect/{k}": float(v) for k, v in info_collect.items()}
                log_dict = {"step": train_step}
                log_dict = log_dict | info_collect | info_collect2 | info_update_log

                if not debug:
                    wandb.log(log_dict, step=train_step)

    def eval(self, collector: Collector, key_eval: jr.PRNGKey):
        env = self.agent.env

        if self.b_state0 is None:
            self.b_state0 = env.get_eval_states(collector.cfg.n_envs)

        Tb_rollout, info_collect = self.agent.collect_eval_with_states(collector, self.b_state0, env.eval_T)
        Tb_rollout = jax.device_get(Tb_rollout)
        bT_rollout = Tb_rollout.switch01()

        # Extract each rollout
        trajs = extract_rollouts_eval(bT_rollout)

        # Evaluate the satisfaction of each temporal node.
        trigger_dict = evaluate_triggers(env, trajs)
        # Compute the average satisfaction rate for each trigger
        info_trigger = {f"Eval/Triggers/{k[0]}->{k[1]}": float(np.mean(v)) for k, v in trigger_dict.items()}

        info = info_trigger
        return trajs, trigger_dict, info
