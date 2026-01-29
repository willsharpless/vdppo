import pathlib

import cyclopts
import ipdb
import jax
import jax.random as jr
import numpy as np
from loguru import logger

from rraa_rl.collector import Collector
from rraa_rl.deliveryreal_cbs import animate_deliveryreal_traj
from rraa_rl.herd_os_cbs import animate_herding_traj
from rraa_rl.rollout_temporal_analysis import evaluate_ltl_finite
from rraa_rl.load_ckpt import load_ckpt
from rraa_rl.rollout_utils import extract_rollouts_eval
from rraa_rl.src.env.general_task.deliveryreal import DeliveryReal
from rraa_rl.src.env.general_task.herd_os import HerdOs
from rraa_rl.vd_mappo import VDMAPPOAgent

app = cyclopts.App()

@app.default()
def main(
    algs: list[str] | None = ["vd"],
    env_name: str = "herdos",
    seed: int = 123,
    n_envs_test: int = 128,
    n_agent: int = 1,
    n_spec: int = 1,
    dense: bool = False,
    eval_T: int | None = None
):
    
    # algs = ["vd", "mppi"] if algs is None else algs
    # algs = ["vd", "lcrl", "mppi"] if algs is None else algs
    # [drl2, lcer]

    # run_path: pathlib.Path, 

    for alg in algs:

        runs_dir = '/datadrive/vd' / env_name / alg.capitalize()
        alg_env_paths = []
        # TODO iterate

        for run_path in alg_env_paths:

            run, agent, env, cfg_dict = load_ckpt(run_path, None)
            collector = Collector.create(
                key=jr.PRNGKey(seed),
                env=env,
                cfg=Collector.Cfg(n_envs=n_envs_test, auto_reset=False, ignore_trunc=True),
            )
            b_state0 = env.get_eval_states(collector.cfg.n_envs, root_only=True)

            collect_opts = {}
            if isinstance(agent, VDMAPPOAgent):
                collect_opts["temporal_transitions"] = True

            eval_T = eval_T or env.eval_T

            Tb_rollout, info_collect = agent.collect_eval_with_states(collector, b_state0, eval_T, **collect_opts)
            Tb_rollout = jax.device_get(Tb_rollout)
            bT_rollout = Tb_rollout.switch01()

            # Extract each rollout
            b_trajs = extract_rollouts_eval(bT_rollout)

            # Compute satisfaction
            b_values = []
            for ii, traj in enumerate(b_trajs):
                debug = ii == 64
                dag_value = evaluate_ltl_finite(env, traj.predicates_next, which=np)[env.dag_root]
                b_values.append(dag_value)
            b_values = np.array(b_values)
            b_is_satisfied = b_values > 0.1

            p_satisified_mean = np.mean(b_is_satisfied)



        # trainer = Trainer(agent, trainer_cfg)
        # out = trainer.eval(trainer.make_eval_collector(env, n_envs_test))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
