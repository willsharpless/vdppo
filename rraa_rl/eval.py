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
from rraa_rl.gridworld_cbs import animate_gridworld_traj
from rraa_rl.rollout_temporal_analysis import evaluate_ltl_finite
from rraa_rl.load_ckpt import load_ckpt
from rraa_rl.rollout_utils import extract_rollouts_eval
from rraa_rl.src.env.general_task.deliveryreal import DeliveryReal
from rraa_rl.src.env.general_task.herd_os import HerdOs
from rraa_rl.vd_mappo import VDMAPPOAgent
from rraa_rl.MPPI import init_mppi
import json
import tempfile

app = cyclopts.App()

@app.default()
def eval(
    alg: str = "vd",
    env_name: str = "herdos",
    ablation_type: str = "spec",
    seed: int = 123,
    n_envs_test: int = 128,
    eval_T: int | None = None,
    debug: bool = False,
    missing: bool = False,
    min_ckpt: int = 50000
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
    missing_runs = []
    for i in range(loop_range): #looping in case of ablation
        if debug and i == 3:
            ipdb.set_trace()
        
        if alg != "mppi":
            alg_env_paths, missing_run = find_runs(runs_dir, alg, env_name, ablation_type=ablation_type, i=i, min_ckpt=min_ckpt)
        else:
            alg_env_paths = ["seed_0", "seed_1", "seed_2"]
        # ipdb.set_trace()

        if alg != "mppi" and missing_run:
            missing_runs.extend(missing_run)

        if missing:
            continue

        means = []
        for seed_path in alg_env_paths:
            # if seed_path._str.endswith("seed0"):
            #     continue
            if alg != "mppi":
                logger.info(f"Evaluating {seed_path._str.split('/')[-1]}")
            else:
                logger.info(f"Evaluating MPPI, {seed_path}")

            # If ablation
            if "ablation" in env_name:
                if "spec" in ablation_type or env_name == "ablation_depth":
                    n_spec, n_agent = i + 1, 1
                elif "ag" in ablation_type:
                    n_spec = n_agent = i + 1
            else:
                n_spec = n_agent = 1

            if alg != "mppi":
                run, agent, env, cfg_dict = load_ckpt(seed_path, None, alg, n_spec=n_spec, n_agent=n_agent)
            else:
                run_seed = jr.PRNGKey(int(seed_path.split('_')[-1]))
                run, agent, env, cfg_dict = init_mppi(env_name, run_seed, n_spec=n_spec, n_agent=n_agent)

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
                    
                    # get bad traj data
                    traj = b_trajs[ix]
                    cfg: HerdOs.Cfg = env.cfg

                    T_state: HerdOs.State = traj.state_now

                    # anim output
                    if alg != "mppi":
                        anim_path = seed_path / f"bad_eval_animation_{ix}.mp4"
                    else:
                        mppi_dir = pathlib.Path('/datadrive/vd') / env_name / alg.upper()
                        mppi_dir.mkdir(parents=True, exist_ok=True)
                        anim_path = mppi_dir / f"bad_eval_animation_{ix}_seed{seed_path.split('_')[-1]}.mp4"

                    # make env anim
                    if 'gridworld' not in env_name:
                        T_pos_herd = T_state.base.herd_state[:, :, :2]
                        T_pos_herder = T_state.base.herder_state[:, :, :2]
                        # T_temporal_node_idx = T_state.temporal_node_idx
                        T_labels = [
                            f"Temporal {t_node_idx} ({env.temporal_node_names[t_node_idx]})" for t_node_idx in T_state.temporal_node_idx
                        ]
                        if env_name == "herdos":
                            animate_herding_traj(anim_path, cfg.base, T_pos_herd, T_pos_herder, None, T_labels)
                        elif "ablation" in env_name:
                            animate_ablation_traj(anim_path, env, cfg.base, T_pos_herder, None, T_labels)
                    else:
                        animate_gridworld_traj(anim_path, env, cfg, T_state)
                # ipdb.set_trace()

        # trainer = Trainer(agent, trainer_cfg)
        # out = trainer.eval(trainer.make_eval_collector(env, n_envs_test))

        loop_means.append(np.mean(means))
        loop_stds.append(np.std(means))

    if missing:
        if missing_runs:
            logger.warning(f"Missing runs for {alg}: {missing_runs}")
        else:
            logger.info(f"All runs found for {alg} - {env_name} ({ablation_type}) !")
        return
    
    logger.info(f"{alg}, satisfaction: {loop_means}, sd: {loop_stds}")

    label = f"{alg}_{env_name}" if not 'ablation' in env_name else f"{alg}_{ablation_type}_{env_name}"
    all_results[label] = {
        "means": loop_means,
        "stds": loop_stds,
    } # overwrites any data in the json with same label

    with tempfile.NamedTemporaryFile("w", dir=out_file.parent, delete=False) as tmp_file:
        json.dump(all_results, tmp_file, indent=4)
        temp_file_path = pathlib.Path(tmp_file.name)

    # Atomically replace the original file with the temporary file
    temp_file_path.replace(out_file)
    logger.info(f"Wrote results to {out_file}")

    return loop_means, loop_stds, missing_runs

def find_runs(runs_dir, alg, env_name, n_seeds=3, ablation_type="spec", i=0, min_ckpt=50000):
    alg_env_paths, missing = [], []
    
    for seed in range(n_seeds):
        if "ablation" in env_name:
            if ablation_type == "spec" or env_name == "ablation_depth":
                candidate_tag = f"{alg}_spc{i+1}_ag1_seed{seed}"
                candidates = [d for d in runs_dir.iterdir() if d.is_dir() and 
                            d.name.endswith(candidate_tag)]
            if ablation_type == "ag":
                candidate_tag = f"{alg}_spc{i+1}_ag{i+1}_seed{seed}"
                candidates = [d for d in runs_dir.iterdir() if d.is_dir() and 
                            d.name.endswith(candidate_tag)]
        else:
            candidate_tag = f"{alg}_seed{seed}"
            candidates = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.endswith(candidate_tag)]

        if candidates:
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
            # assert best_dir is not None, f"No valid run dir found for {alg} seed {seed} in {runs_dir}"
            # assert best_ckpt > 10000, f"Highest checkpoint for {alg} seed {seed} is {best_ckpt}, must be > 10000"
            if best_ckpt < min_ckpt:
                logger.warning(f"SKIPPING SEED {seed}: Low best ckpt ({best_ckpt} < {min_ckpt}) for {candidate_tag}...")
                missing.append(candidate_tag)
                continue
            alg_env_paths.append(best_dir)

        else:
            logger.warning(f"SKIPPING SEED {seed}: No candidates for {candidate_tag}...")
            missing.append(candidate_tag)

    if len(alg_env_paths) < n_seeds:
        logger.warning(f"MISSING SEED(S): {len(alg_env_paths)} viable seeds in {runs_dir}")
        assert len(alg_env_paths) > 0, f"No valid run dir found for {alg} seed {seed} in {runs_dir}"

    return alg_env_paths, missing

if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
