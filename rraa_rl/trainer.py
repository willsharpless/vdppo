import jax.random as jr
import tqdm
import wandb
from valtr.valtr import to_dag

from rraa_rl.collector import Collector, extract_info_from_rollout
from rraa_rl.src.env.general_task.herd_os import HerdOs
from rraa_rl.vd_mappo import VDMAPPOAgent


class Trainer:
    def __init__(self):
        pass

    def train(self, agent: VDMAPPOAgent, env: HerdOs):
        # > Process the temporal logic specifications.
        key_base = jr.PRNGKey(124521)
        key_base, key_collector = jr.split(key_base, 2)

        n_envs_train = 1024

        collector = Collector.create(
            key=key_collector,
            env=env,
            cfg=Collector.Cfg(n_envs=n_envs_train),
        )

        n_train_steps = 1_000

        eval_every = 100
        log_every = 10

        pbar = tqdm.trange(n_train_steps)
        for train_step in pbar:
            if train_step % eval_every == 0:
                pbar.set_description(f"Eval at step {train_step}")

            # Collect rollout data
            collector, Tb_rollout, info_collect = agent.collect_batch(collector, agent.cfg.rollout_T)

            info_collect2 = {}
            if train_step % log_every == 0:
                # Extract easy-to-log info from the rollouts, e.g., average reset age
                info_collect2 = extract_info_from_rollout(Tb_rollout)

            # Update agent using the collected rollout.
            key_base, key_update = jr.split(key_base, 2)
            agent, info_update = agent.update(Tb_rollout, key_update)

            # Update progress bar.
            pbar.update(1)

            # Log info to wandb
            if train_step % log_every == 0:
                info_collect2 = {f"collect/{k}": v for k, v in info_collect2.items()}
                info_collect = {f"collect/{k}": float(v) for k, v in info_collect.items()}
                log_dict = {"step": train_step}
                log_dict = log_dict | info_collect | info_collect2 | info_update
                # wandb.log(log_dict, step=train_step)
