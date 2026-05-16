import pathlib

import cyclopts
import ipdb
import jax
import jax.random as jr
import numpy as np
from loguru import logger
from valtr.reachability import DAGGUSingle, dag_to_str

from rraa_rl.training.collector import Collector
from rraa_rl.callbacks.delivery_cbs import animate_delivery_traj
from rraa_rl.callbacks.deliveryreal_cbs import animate_deliveryreal_traj
from rraa_rl.training.load_ckpt import load_ckpt
from rraa_rl.training.rollout_utils import extract_rollouts_eval
from rraa_rl.env.general_task.delivery import Delivery
from rraa_rl.env.general_task.deliveryreal import DeliveryReal
from rraa_rl.agents.vdppo import VDPPOAgent

app = cyclopts.App()


@app.default()
def main(run_path: pathlib.Path, n_env: int = 1, step: int | None = None, eval_T: int | None = None):
    env: DeliveryReal
    run, agent, env, cfg_dict = load_ckpt(run_path, step)

    collector = Collector.create(
        key=jr.PRNGKey(1234),
        env=env,
        cfg=Collector.Cfg(n_envs=n_env, auto_reset=False, ignore_trunc=True),
    )
    b_state0 = env.get_eval_states(collector.cfg.n_envs, root_only=True)

    # Double check that b_state0 is safe.
    b_is_valid = jax.vmap(env.is_valid_real_eval_state)(b_state0)
    assert np.all(b_is_valid)
    logger.debug("is_valid: {}".format(b_is_valid))

    collect_opts = {}
    if isinstance(agent, VDPPOAgent):
        collect_opts["temporal_transitions"] = True

    if eval_T is None:
        eval_T = env.eval_T
        logger.info("Using default eval_T from env: {}", eval_T)

    # eval_T = eval_T or env.eval_T

    Tb_rollout, info_collect = agent.collect_eval_with_states(collector, b_state0, eval_T, **collect_opts)
    Tb_rollout = jax.device_get(Tb_rollout)
    bT_rollout = Tb_rollout.switch01()

    # Extract each rollout
    b_trajs = extract_rollouts_eval(bT_rollout)

    traj_lens = [traj.shape[0] for traj in b_trajs]
    logger.info("Traj lens: {}".format(traj_lens))

    # Animate all trajectories.
    for ii, traj in enumerate(b_trajs):
        traj_len = traj.shape[0]

        anim_path = run_path / f"eval_animation_{ii:02}.mp4"
        T_state = traj.state_now

        T_temporal_node_idx = T_state.temporal_node_idx
        T_labels = []
        for t_node_idx in T_state.temporal_node_idx:
            dag_id = env.temporal_nodes[t_node_idx]
            dag_node = env.dag_nodes[dag_id]
            assert isinstance(dag_node, DAGGUSingle)
            reach_str = dag_to_str(env.dag_nodes, dag_node.reach)

            label = f"T{t_node_idx} %{dag_id}: {reach_str}"
            T_labels.append(label)

        # T_labels = [
        #     f"Temporal {t_node_idx} ({env.temporal_node_names[t_node_idx]})" for
        # ]

        if run.env_name == "delivery":
            T_state: Delivery.State
            cfg: Delivery.Cfg = env.cfg
            T_pos_herder = T_state.base.herder_state[:, :, :2]
            T_pos_target = T_state.base.centers[:, :2, :2]  # only using first two centers
            animate_delivery_traj(anim_path, env, cfg.base, T_pos_herder, T_pos_target, T_temporal_node_idx, T_labels)
        elif run.env_name == "deliveryreal":
            T_state: DeliveryReal.State
            cfg: DeliveryReal.Cfg = env.cfg
            T_pos_herder = T_state.base.herder_state[:, :, :2]
            T_pos_target = T_state.base.centers[:, :2, :2]
            animate_deliveryreal_traj(
                anim_path, env, cfg.base, T_pos_herder, T_pos_target, T_temporal_node_idx, T_labels
            )

        if traj_len < eval_T:
            # Find out why it terminated by printing out all true predicates at the terminal step.
            pred_final_step = {k: v[-1] > 0 for k, v in traj.predicates_next.items()}
            preds_true = [k for k, v in pred_final_step.items() if v]
            logger.info(
                "Traj {} terminated early at step {}. True predicates at final step: {}".format(
                    ii, traj_len, preds_true
                )
            )

        # if traj_len == 1:
        #     # Why is traj_len 1?
        #     T_state_now = traj.state_now
        #     T_state_nxt = traj.state_next
        #     valid_now = jax.vmap(env.is_valid_real_eval_state)(T_state_now)
        #     valid_nxt = jax.vmap(env.is_valid_real_eval_state)(T_state_nxt)
        #
        #     logger.debug("Traj {} has length 1. valid_now: {}, valid_nxt: {}".format(ii, valid_now, valid_nxt))
        #     ipdb.set_trace()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
