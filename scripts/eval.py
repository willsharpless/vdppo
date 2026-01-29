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
from rraa_rl.ablation_cbs import animate_ablation_traj
from rraa_rl.rollout_temporal_analysis import evaluate_ltl_finite
from rraa_rl.load_ckpt import load_ckpt
from rraa_rl.rollout_utils import extract_rollouts_eval
from rraa_rl.src.env.general_task.deliveryreal import DeliveryReal
from rraa_rl.src.env.general_task.herd_os import HerdOs
from rraa_rl.vd_mappo import VDMAPPOAgent
import json
import tempfile

app = cyclopts.App()

@app.default()
def main(
    alg: str = "vd",
    env_name: str = "herdos",
    ablation_type: str = "spec",
    seed: int = 123,
    n_envs_test: int = 128,
    eval_T: int | None = None,
    debug: bool = False,
):

    runs_dir = pathlib.Path('/datadrive/vd') / env_name / alg.upper()

    if "ablation" not in env_name:
        loop_range = 1
    else:
        loop_range = 5

    out_file = pathlib.Path('/datadrive/vd') / env_name / f"eval_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.exists():
        with open(out_file, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    loop_means, loop_stds = [], []
    for i in range(loop_range): #looping in case of ablation

        alg_env_paths = find_runs(runs_dir, alg, env_name, ablation_type=ablation_type, i=i)
        # ipdb.set_trace()

        means = []
        for seed_path in alg_env_paths:
            # if seed_path._str.endswith("seed0"):
            #     continue
            logger.info(f"Evaluating {seed_path._str.split('/')[-1]}")

            # If ablation
            if "ablation" in env_name:
                if "spec" in ablation_type or "depth" in ablation_type:
                    n_spec, n_agent = i + 1, 1
                elif "ag" in ablation_type:
                    n_spec = n_agent = i + 1
            else:
                n_spec = n_agent = 1

            run, agent, env, cfg_dict = load_ckpt(seed_path, None, alg, n_spec=n_spec, n_agent=n_agent)
            collector = Collector.create(
                key=jr.PRNGKey(seed),
                env=env,
                cfg=Collector.Cfg(n_envs=n_envs_test, auto_reset=False, ignore_trunc=True),
            )
            b_state0 = env.get_real_eval_states(collector.cfg.n_envs, collector.cfg.n_envs * 8)
            
            # b_state0 = env.get_eval_states(collector.cfg.n_envs, root_only=True)
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
            if debug and p_satisfied_mean < 1.0:
                bad_traj = b_values < 0
                bad_traj_sample = np.random.choice(np.where(bad_traj)[0], size=min(3, bad_traj.sum()), replace=False)
                for ix in bad_traj_sample:
                    traj = b_trajs[ix]
                    cfg: HerdOs.Cfg = env.cfg

                    T_state: HerdOs.State = traj.state_now
                    T_pos_herd = T_state.base.herd_state[:, :, :2]
                    T_pos_herder = T_state.base.herder_state[:, :, :2]
                    # T_temporal_node_idx = T_state.temporal_node_idx
                    T_labels = [
                        f"Temporal {t_node_idx} ({env.temporal_node_names[t_node_idx]})" for t_node_idx in T_state.temporal_node_idx
                    ]
                    anim_path = seed_path / f"bad_eval_animation_{ix}.mp4"
                    if env_name == "herdos":
                        animate_herding_traj(anim_path, cfg.base, T_pos_herd, T_pos_herder, None, T_labels)
                    elif "ablation" in env_name:
                        animate_ablation_traj(anim_path, env, cfg.base, T_pos_herder, None, T_labels)
                # ipdb.set_trace()

        # trainer = Trainer(agent, trainer_cfg)
        # out = trainer.eval(trainer.make_eval_collector(env, n_envs_test))

        loop_means.append(np.mean(means))
        loop_stds.append(np.std(means))

    logger.info(f"{alg}, satisfaction: {loop_means}, sd: {loop_stds}")

    all_results[f"{alg}_{env_name}"] = {
        "means": loop_means,
        "stds": loop_stds,
    } # overwrites any data in the json with same label

    with tempfile.NamedTemporaryFile("w", dir=out_file.parent, delete=False) as tmp_file:
        json.dump(all_results, tmp_file, indent=4)
        temp_file_path = pathlib.Path(tmp_file.name)

    # Atomically replace the original file with the temporary file
    temp_file_path.replace(out_file)
    logger.info(f"Wrote results to {out_file}")

def find_runs(runs_dir, alg, env_name, n_seeds=3, ablation_type="spec", i=0):
    alg_env_paths = []
    
    for seed in range(n_seeds):
        if env_name.startswith("ablation"):
            if ablation_type == "spec" or ablation_type == "depth":
                candidates = [d for d in runs_dir.iterdir() if d.is_dir() and 
                            d.name.endswith(f"{alg}_spc{i+1}_ag1_seed{seed}")]
            if ablation_type == "ag":
                candidates = [d for d in runs_dir.iterdir() if d.is_dir() and 
                            d.name.endswith(f"{alg}_spc{i+1}_ag{i+1}_seed{seed}")]
        else:
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
