from doctest import debug
import pathlib
import re

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
    eval_T: int | None = None,
    debug: bool = False,
):
    
    # algs = ["vd", "mppi"] if algs is None else algs
    # algs = ["vd", "lcrl", "mppi"] if algs is None else algs
    # [drl2, lcer]

    # run_path: pathlib.Path, 

    for alg in algs:

        runs_dir = pathlib.Path('/datadrive/vd') / env_name / alg.upper()
        alg_env_paths = find_runs(runs_dir, alg)

        means = []
        for run_path in alg_env_paths:
            logger.info(f"Evaluating {run_path._str.split('/')[-1]}")

            run, agent, env, cfg_dict = load_ckpt(run_path, None)
            collector = Collector.create(
                key=jr.PRNGKey(seed),
                env=env,
                cfg=Collector.Cfg(n_envs=n_envs_test, auto_reset=False, ignore_trunc=True),
            )
            b_state0 = env.get_eval_states(collector.cfg.n_envs, root_only=True)
            # ipdb.set_trace()

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
                dag_value = evaluate_ltl_finite(env, traj.predicates_next, which=np)[env.dag_root]
                b_values.append(dag_value)
            b_values = np.array(b_values)
            b_is_satisfied = b_values > 0.

            p_satisfied_mean = np.mean(b_is_satisfied)
            means.append(p_satisfied_mean)

            # Animate some bad trajectories to check
            if debug:
                bad_traj = b_values < 0
                bad_traj_sample = np.random.choice(np.where(bad_traj)[0], size=3, replace=False)
                for ix in bad_traj_sample:
                    traj = b_trajs[ix]
                    cfg: HerdOs.Cfg = env.cfg

                    T_state: HerdOs.State = traj.state_now
                    T_pos_herd = T_state.base.herd_state[:, :, :2]
                    T_pos_herder = T_state.base.herder_state[:, :, :2]
                    T_temporal_node_idx = T_state.temporal_node_idx
                    T_labels = [
                        f"Temporal {t_node_idx} ({env.temporal_node_names[t_node_idx]})" for t_node_idx in T_state.temporal_node_idx
                    ]
                    anim_path = run_path / f"bad_eval_animation_score_{b_values[ix]}.mp4"
                    animate_herding_traj(anim_path, cfg.base, T_pos_herd, T_pos_herder, T_temporal_node_idx, T_labels)
                ipdb.set_trace()

        logger.info(f"{alg}, satisfaction: {np.mean(means):.3f}, sd: {np.std(means):.3f}")
        # trainer = Trainer(agent, trainer_cfg)
        # out = trainer.eval(trainer.make_eval_collector(env, n_envs_test))

def find_runs(runs_dir, alg, n_seeds=3):
    alg_env_paths = []
    
    for seed in range(n_seeds):
        candidates = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.endswith(f"{alg}_seed{seed}")]
        
        best_ckpt, best_dir = -1, None
        for d in candidates:
            ckpt_dir = d / 'ckpts'
            if not ckpt_dir.exists():
                continue
            ckpts = list(ckpt_dir.glob('*.pkl'))
            if not ckpts:
                continue
            # Extract checkpoint numbers
            def ckpt_num(f):
                m = re.search(r'(\d+)', f.stem)
                return int(m.group(1)) if m else -1
            max_ckpt = max(ckpts, key=ckpt_num)
            max_ckpt_num = ckpt_num(max_ckpt)
            if max_ckpt_num > best_ckpt:
                best_ckpt = max_ckpt_num
                best_dir = d
        assert best_dir is not None, f"No valid run dir found for {alg} seed {seed} in {runs_dir}"
        assert best_ckpt > 50000, f"Highest checkpoint for {alg} seed {seed} is {best_ckpt}, must be > 50000"
        alg_env_paths.append(best_dir)

    return alg_env_paths

if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
