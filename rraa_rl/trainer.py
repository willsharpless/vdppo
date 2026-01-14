import jax.random as jr
import tqdm
from valtr.valtr import to_dag

import wandb
from rraa_rl.src.env.general_task.herd_os import HerdOs


class Trainer:
    def __init__(self):
        pass

    def train(self, agent: VDPPO2, env: HerdOs):
        # > Process the temporal logic specifications.
        spec = env.specification

        dag_builder, dag_root = to_dag(spec, ir_filename="herd_os_ir.pdf", dag_filename="herd_os_dag.pdf")
        dag_nodes = dag_builder.nodes

        key_base = jr.PRNGKey(124521)
        key_base, key_collector = jr.split(key_base, 2)

        collector = Collector.create(
            key=key_collector,
            env=env,
            cfg=Collector.Cfg(n_envs=n_env_train),
        )

        n_train_steps = 1_000

        eval_every = 100

        pbar = tqdm.trange(n_train_steps)
        for train_step in pbar:
            if train_step % eval_every == 0:
                pbar.set_description(f"Eval at step {train_step}")

            # Collect rollout data
            collector, b_rollout, col_info = agent.collect_batch(collector)

            # Update agent using the collected rollout.
            key_base, key_update = jr.split(key_base, 2)
            agent.update(b_rollout, key_update)

            # Update progress bar.
            pbar.update(1)

            # Log info to wandb
            if train_step % log_every == 0:
                col_info = {f"col_train/{k}": float(v) for k, v in col_info.items()}
                log_dict = {"step": train_step}
                log_dict = log_dict | col_info
                wandb.log(log_dict, step=train_step)
