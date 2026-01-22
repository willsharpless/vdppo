import pickle
from typing import Protocol

import jax
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
from rraa_rl.rollout_temporal_analysis import evaluate_ltl_finite
from rraa_rl.rollout_utils import extract_rollouts_eval
from rraa_rl.run import Run
from rraa_rl.src.env.general_task.env import Env
from rraa_rl.vd_mappo import VDMAPPOAgent


@define
class CallbackProps:
    run: Run

    train_step: int
    agent: VDMAPPOAgent
    bT_test_rollouts: list[RolloutOutput]
    bT_test_rollout: RolloutOutput
    test_trigger_dict: dict[tuple[str, str], np.ndarray]
    temporal_values_dict: dict[int, np.ndarray]

    collector_train: Collector
    Tb_rollout: RolloutOutput
    info_update: dict

    @property
    def env(self):
        return self.agent.env


class Callback(Protocol):
    def __call__(self, p: CallbackProps) -> None: ...


@Parameter("*", group="Trainer")
@define
class TrainerCfg(Cfg):
    eval_every: int = 5_000
    log_every: int = 100
    save_every: int = 5_000


class Trainer:
    Cfg = TrainerCfg

    agent: VDMAPPOAgent

    def __init__(self, agent: VDMAPPOAgent, cfg: TrainerCfg):
        self.cfg = cfg
        self.agent = agent
        self.b_state0 = None

    def train(
        self,
        run: Run,
        env: Env,
        eval_cbs: list[Callback] = None,
        collect_cbs: list[Callback] = None,
        debug: bool = False,
        wandb_config: dict | None = None,
    ):
        eval_cbs = eval_cbs if eval_cbs is not None else []

        key_base = jr.PRNGKey(124521)
        key_base, key_collector, key_eval = jr.split(key_base, 3)

        n_envs_train = self.agent.cfg.n_envs_train
        n_envs_test = 128

        logger.debug("Constructing collector...")
        collector = Collector.create(
            key=key_collector,
            env=env,
            cfg=Collector.Cfg(n_envs=n_envs_train),
        )
        logger.debug("Constructing collector_eval...")
        collector_eval = Collector.create(
            key=key_collector,
            env=env,
            cfg=Collector.Cfg(n_envs=n_envs_test, auto_reset=False, ignore_trunc=True),
        )

        # Save configs.
        cfg_to_save = {
            "agent": self.agent.cfg.asdict(),
            "trainer": self.cfg.asdict(),
        }
        # Save as yaml.
        yaml_path = run.run_dir / "config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(cfg_to_save, f)
        logger.success("Saved config to {}".format(yaml_path))

        n_train_steps = 100_000

        if not debug:
            wandb_config = wandb_config if wandb_config is not None else {}
            wandb_config = {
                "env": run.env_name,
                "noun": run.noun,
                "name": run.name,
            } | wandb_config
            wandb.init(project="vd_mappo", name=run.wandb_name, config=wandb_config)

        cb_props = CallbackProps(run, -1, self.agent, None, None, None, None, collector, None, None)

        pbar = tqdm.trange(n_train_steps, mininterval=0.25)
        for train_step in pbar:
            if train_step % self.cfg.eval_every == 0:
                pbar.set_description(f"Eval at step {train_step}")
                trajs, bT_rollout, trigger_dict, info_eval = self.eval(collector_eval, key_eval)

                temporal_values_dict = info_eval.pop("debug/temporal_values_dict")

                cb_props.train_step = train_step
                cb_props.agent = self.agent
                cb_props.bT_test_rollouts = trajs
                cb_props.bT_test_rollout = bT_rollout
                cb_props.test_trigger_dict = trigger_dict
                cb_props.temporal_values_dict = temporal_values_dict
                cb_props.collector_train = collector

                for cb in eval_cbs:
                    cb_name = cb.__name__ if hasattr(cb, "__name__") else str(type(cb))
                    pbar.set_description(f"Running eval callback {cb_name}")
                    cb(cb_props)

                # Log
                log_dict = {"step": train_step}
                log_dict = log_dict | info_eval
                if wandb.run is not None:
                    wandb.log(log_dict, step=train_step)

            if train_step % self.cfg.save_every == 0:
                pbar.set_description(f"Saving at step {train_step}")

                ckpts_dir = run.ckpts_dir
                save_dict = dict(agent=self.agent.to_state_dict())
                pkl_path = ckpts_dir / "params_{:09}.pkl".format(train_step)
                with open(pkl_path, "wb") as f:
                    pickle.dump(save_dict, f)

            # Collect rollout data
            pbar.set_description(f"Collecting rollouts")
            collector, Tb_rollout, info_collect = self.agent.collect_batch(collector, self.agent.cfg.rollout_T)

            info_collect2 = {}
            if train_step % self.cfg.log_every == 0:
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
            if train_step % self.cfg.log_every == 0:
                # Remove any keys that start with debug/
                info_update_log = {k: v for k, v in info_update.items() if not k.startswith("debug/")}

                info_collect2 = {f"collect/{k}": v for k, v in info_collect2.items()}
                info_collect = {f"collect/{k}": float(v) for k, v in info_collect.items()}
                log_dict = {"step": train_step}
                log_dict = log_dict | info_collect | info_collect2 | info_update_log

                if wandb.run is not None:
                    wandb.log(log_dict, step=train_step)

    def eval(self, collector: Collector, key_eval: jr.PRNGKey):
        env = self.agent.env

        if self.b_state0 is None:
            self.b_state0 = env.get_eval_states(collector.cfg.n_envs)

        Tb_rollout, info_collect = self.agent.collect_eval_with_states(
            collector, self.b_state0, env.eval_T, temporal_transitions=True
        )
        Tb_rollout = jax.device_get(Tb_rollout)
        bT_rollout = Tb_rollout.switch01()

        # Extract each rollout
        trajs = extract_rollouts_eval(bT_rollout)

        # Evaluate the LTL satisfaction over each trajectory.
        temporal_node_values_l: dict[int, list[float]] = {}
        for traj in trajs:
            T_temporal_node_idx: np.ndarray = traj.temporal_node_idx
            temporal_node_idx = T_temporal_node_idx[0]
            dag_node_idx = env.temporal_nodes[temporal_node_idx]
            dag_value = evaluate_ltl_finite(env, traj.predicates_next, which=np)[dag_node_idx]

            temporal_node_value = temporal_node_values_l.get(temporal_node_idx, [])
            temporal_node_value.append(dag_value)
            temporal_node_values_l[temporal_node_idx] = temporal_node_value
        temporal_node_values: dict[int, np.ndarray] = {k: np.array(v) for k, v in temporal_node_values_l.items()}

        # Compute the average satisfaction rate for each temporal node
        info_satisfaction = {}
        for temporal_node_idx, temporal_node_value in temporal_node_values.items():
            node_name = env.temporal_node_names[temporal_node_idx]
            # Satisfy if positive.
            info_satisfaction[f"Eval/Satisfy/{node_name}"] = float(np.mean(temporal_node_value > 0.1))

        # # Evaluate the satisfaction of each temporal node.
        # trigger_dict = evaluate_triggers(env, trajs)
        # # Compute the average satisfaction rate for each trigger
        # info_trigger = {f"Eval/Triggers/{k[0]}->{k[1]}": float(np.mean(v)) for k, v in trigger_dict.items()}
        trigger_dict = {}
        info = info_satisfaction

        # info = info_trigger | info_satisfaction

        info["debug/temporal_values_dict"] = temporal_node_values

        return trajs, bT_rollout, trigger_dict, info
