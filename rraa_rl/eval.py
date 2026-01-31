import json
import pathlib
import re
import tempfile
from doctest import debug

import cyclopts
import ipdb
import jax
import jax.random as jr
import numpy as np
from flax.errors import ScopeParamShapeError
from loguru import logger

from rraa_rl.ablation_cbs import animate_ablation_traj
from rraa_rl.collector import Collector
from rraa_rl.deliveryreal_cbs import animate_deliveryreal_traj
from rraa_rl.eval_results import get_eval_label, get_eval_results_path, has_eval_results, save_eval_results
from rraa_rl.gridworld_cbs import animate_gridworld_traj
from rraa_rl.herd_os_cbs import animate_herding_traj
from rraa_rl.load_ckpt import load_ckpt
from rraa_rl.MPPI import init_mppi
from rraa_rl.rollout_temporal_analysis import (evaluate_ltl_finite, evaluate_ltl_finite_dag,
                                               get_ltl_finite_values_rollout)
from rraa_rl.rollout_utils import extract_rollouts_eval
from rraa_rl.src.env.general_task.delivery_base import DeliveryBase
from rraa_rl.src.env.general_task.deliveryreal import DeliveryReal
from rraa_rl.src.env.general_task.herd_os import HerdOs
from rraa_rl.vd_mappo import VDMAPPOAgent

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
    min_ckpt: int = 50000,
    n_seeds: int = 3,
    force: bool = False
):

    runs_dir = pathlib.Path("/datadrive/vd") / env_name / alg.upper()

    is_ablation = "ablation" in env_name

    if is_ablation:
        loop_range = 5
    else:
        loop_range = 1

    loop_range_vals = list(range(loop_range))

    if env_name == "ablation_depth":
        eval_T = eval_T or 512
        eval_T = max(eval_T, 512)
        loop_range_vals = [4]

    # out_file = pathlib.Path("/datadrive/vd") / env_name / f"eval_results.json"
    out_file = get_eval_results_path(env_name)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.exists():
        with open(out_file, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    loop_means, loop_stds = [], []
    missing_runs = []
    for i in loop_range_vals:  # looping in case of ablation
        if debug and i == 3:
            ipdb.set_trace()

        if is_ablation:
            if "spec" in ablation_type or env_name == "ablation_depth":
                n_spec, n_agent = i + 1, 1
            elif "ag" in ablation_type:
                n_spec = n_agent = i + 1
        else:
            n_spec = n_agent = 1

        if alg != "mppi":
            alg_env_paths, missing_run = find_runs(
                runs_dir, alg, env_name, ablation_type=ablation_type, i=i, min_ckpt=min_ckpt, n_seeds=n_seeds
            )
        else:
            missing_run = False
            if is_ablation:
                # For ablations, alg_env_path is like `chest_ablation_vd_spc3_ag1_seed1`
                alg_env_paths = [
                    f"{env_name}_{alg}_spc{n_spec}_ag{n_agent}_seed{mppi_seed}" for mppi_seed in range(n_seeds)
                ]
            else:
                # For normal, alg_env_path is like `thank_herdos_vd_seed1`
                alg_env_paths = [f"{env_name}_{alg}_seed{mppi_seed}" for mppi_seed in range(n_seeds)]

        if alg != "mppi" and missing_run:
            missing_runs.extend(missing_run)

        if missing:
            continue

        means = []
        for seed_path in alg_env_paths:

            # See if we already have results for this run.
            label = get_eval_label(alg, env_name, ablation_type)
            if has_eval_results(label, seed_path, env_name):
                if force:
                    logger.info(f"Already have eval for {seed_path}, but running because force=True.")
                else:
                    logger.info(f"Skipping eval for {seed_path}, already have results.")
                    continue

            # if seed_path._str.endswith("seed0"):
            #     continue
            if alg != "mppi":
                logger.info(f"Evaluating {seed_path._str.split('/')[-1]}")
            else:
                logger.info(f"Evaluating MPPI, {seed_path}")

            # If ablation

            if alg != "mppi":
                try:
                    run, agent, env, cfg_dict = load_ckpt(seed_path, None, alg, n_spec=n_spec, n_agent=n_agent)
                    step = cfg_dict["step"]
                    logger.debug(f"Loaded step {step}")
                except TypeError as e:
                    if "ShapedArray.__init__() got an unexpected keyword argument 'named_shape'" in str(e):
                        # ckpt saved with older version of jax.
                        logger.error("Failed to load ckpt for {} due to jax version mismatch: {}".format(seed_path, e))
                        continue
                    else:
                        raise
            else:
                run_seed = int(seed_path.split("_")[-1].replace("seed", ""))
                run_key = jr.PRNGKey(run_seed)
                run, agent, env, cfg_dict = init_mppi(env_name, run_key, n_spec=n_spec, n_agent=n_agent)
                agent.cfg.n_envs = n_envs_test

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

            try:
                Tb_rollout, info_collect = agent.collect_eval_with_states(collector, b_state0, eval_T, **collect_opts)
            except ScopeParamShapeError as e:
                if "25" in str(e) and "31" in str(e):
                    # Mixup in the centers thing. Try changing centers.
                    logger.warning(
                        "Caught ShapeError during rollout: {}\n"
                        "Trying to fix by changing DeliveryBase centers...".format(e)
                    )

                    assert env_name == "delivery"
                    assert isinstance(env.base, DeliveryBase)

                    default_base_cfg = DeliveryBase.Cfg()
                    env.base.cfg.centers = default_base_cfg.centers
                    env.base.cfg.radiuses = default_base_cfg.radiuses
                    collector = Collector.create(
                        key=jr.PRNGKey(seed),
                        env=env,
                        cfg=Collector.Cfg(n_envs=n_envs_test, auto_reset=False, ignore_trunc=True),
                    )
                    b_state0 = env.get_real_eval_states(collector.cfg.n_envs, collector.cfg.n_envs * 8)

                    try:
                        Tb_rollout, info_collect = agent.collect_eval_with_states(
                            collector, b_state0, eval_T, **collect_opts
                        )
                    except Exception as e:
                        logger.error(
                            "FAILED AGAIN after changing centers during rollout for {}: {}".format(seed_path, e)
                        )
                        logger.error("Giving up.")
                        continue
                else:
                    raise
            Tb_rollout = jax.device_get(Tb_rollout)
            bT_rollout = Tb_rollout.switch01()

            # Extract each rollout
            b_trajs = extract_rollouts_eval(bT_rollout)

            # Compute satisfaction
            eval_results_dict = {}
            eval_formulae = env.get_eval_formulae_dags()
            for formula_name, (dag_nodes, dag_root) in eval_formulae.items():
                eval_results_dict[formula_name] = get_ltl_finite_values_rollout(dag_nodes, dag_root, b_trajs, which=np)

            eval_results = {
                k: {"num_valid": int((b_V > 0.0).sum()), "total": len(b_V), "data": b_V.tolist()}
                for k, b_V in eval_results_dict.items()
            }

            label = get_eval_label(alg, env_name, ablation_type)
            save_eval_results(seed_path, out_file, eval_results, label)

            # b_values = []
            # for ii, traj in enumerate(b_trajs):
            #     dag_value = evaluate_ltl_finite_dag(env, traj.predicates_next, which=np)[env.dag_root]
            #     b_values.append(dag_value)
            # b_values = np.array(b_values)
            b_values_root = eval_results_dict["root"]
            b_is_satisfied = b_values_root > 0.0

            p_satisfied_mean = np.mean(b_is_satisfied)
            means.append(p_satisfied_mean)

            logger.debug("p_satisfied_mean: {}".format(p_satisfied_mean))

            # Animate some bad trajectories to check
            if debug and p_satisfied_mean < 1.0:
                bad_traj = b_values_root < 0
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
                        mppi_dir = pathlib.Path("/datadrive/vd") / env_name / alg.upper()
                        mppi_dir.mkdir(parents=True, exist_ok=True)
                        anim_path = mppi_dir / f"bad_eval_animation_{ix}_seed{seed_path.split('_')[-1]}.mp4"

                    # make env anim
                    if "gridworld" not in env_name:
                        T_pos_herd = T_state.base.herd_state[:, :, :2]
                        T_pos_herder = T_state.base.herder_state[:, :, :2]
                        # T_temporal_node_idx = T_state.temporal_node_idx
                        T_labels = [
                            f"Temporal {t_node_idx} ({env.temporal_node_names[t_node_idx]})"
                            for t_node_idx in T_state.temporal_node_idx
                        ]
                        if env_name == "herdos":
                            animate_herding_traj(anim_path, cfg.base, T_pos_herd, T_pos_herder, None, T_labels)
                        elif is_ablation:
                            animate_ablation_traj(anim_path, env, cfg.base, T_pos_herder, None, T_labels)
                    else:
                        animate_gridworld_traj(anim_path, env, cfg, T_state)
                # ipdb.set_trace()

        # trainer = Trainer(agent, trainer_cfg)
        # out = trainer.eval(trainer.make_eval_collector(env, n_envs_test))

        if len(means) == 0:
            continue

        loop_means.append(np.mean(means))
        loop_stds.append(np.std(means))

    if missing:
        if missing_runs:
            logger.warning(f"Missing runs for {alg}: {missing_runs}")
        else:
            logger.info(f"All runs found for {alg} - {env_name} ({ablation_type}) !")
        return

    if len(loop_means) == 0:
        logger.warning(f"No evaluation results found for {alg} - {env_name} ({ablation_type}) !")
        return [], [], missing_runs

    logger.info(f"{alg}, satisfaction: {loop_means}, sd: {loop_stds}")
    return loop_means, loop_stds, missing_runs


def find_runs(runs_dir, alg, env_name, n_seeds=3, ablation_type="spec", i=0, min_ckpt=50000):
    """Finds paths to runs for each seed."""
    alg_env_paths = []
    missing: list[str] = []

    for seed in range(n_seeds):
        if "ablation" in env_name:
            if ablation_type == "spec" or env_name == "ablation_depth":
                candidate_tag = f"{alg}_spc{i+1}_ag1_seed{seed}"
                candidates = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.endswith(candidate_tag)]
            if ablation_type == "ag":
                candidate_tag = f"{alg}_spc{i+1}_ag{i+1}_seed{seed}"
                candidates = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.endswith(candidate_tag)]
        else:
            candidate_tag = f"{alg}_seed{seed}"
            # Has to end with e.g., "vd_seed0" to get picked up
            candidates = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.endswith(candidate_tag)]

        if len(candidates) > 0:
            best_ckpt, best_dir = -1, None
            for d in candidates:
                ckpt_dir = d / "ckpts"
                if not ckpt_dir.exists():
                    continue
                ckpts = list(ckpt_dir.glob("*.pkl"))
                if not ckpts:
                    continue

                # Extract checkpoint numbers
                def ckpt_num(f):
                    m = re.search(r"(\d+)", f.stem)
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
            logger.critical(f"SKIPPING SEED {seed}: No candidates for {candidate_tag}...")
            missing.append(candidate_tag)

    if len(alg_env_paths) < n_seeds:
        logger.critical(f"MISSING SEED(S): {len(alg_env_paths)} viable seeds in {runs_dir}")
        assert len(alg_env_paths) > 0, f"No valid run dir found for {alg} seed {seed} in {runs_dir}"

    return alg_env_paths, missing


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
