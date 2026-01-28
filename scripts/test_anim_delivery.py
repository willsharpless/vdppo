import pathlib

import cyclopts
import ipdb
import jax
import jax.random as jr

from rraa_rl.collector import Collector
from rraa_rl.delivery_cbs import animate_delivery_traj
from rraa_rl.herd_os_cbs import animate_herding_traj
from rraa_rl.load_ckpt import load_ckpt
from rraa_rl.rollout_utils import extract_rollouts_eval
from rraa_rl.src.env.general_task.delivery import Delivery
from rraa_rl.src.env.general_task.herd_os import HerdOs
from rraa_rl.vd_mappo import VDMAPPOAgent

app = cyclopts.App()


@app.default()
def main(run_path: pathlib.Path, n_env: int = 1, step: int | None = None):
    run, agent, env, cfg_dict = load_ckpt(run_path, step)

    collector = Collector.create(
        key=jr.PRNGKey(1234),
        env=env,
        cfg=Collector.Cfg(n_envs=n_env, auto_reset=False, ignore_trunc=True),
    )
    b_state0 = env.get_eval_states(collector.cfg.n_envs, root_only=True)

    collect_opts = {}
    if isinstance(agent, VDMAPPOAgent):
        collect_opts["temporal_transitions"] = True

    Tb_rollout, info_collect = agent.collect_eval_with_states(collector, b_state0, env.eval_T, **collect_opts)
    Tb_rollout = jax.device_get(Tb_rollout)
    bT_rollout = Tb_rollout.switch01()

    # Extract each rollout
    b_trajs = extract_rollouts_eval(bT_rollout)

    # Animate the first trajectory.
    traj = b_trajs[0]

    # ------------------------------------------------------------
    cfg: Delivery.Cfg = env.cfg

    T_state: Delivery.State = traj.state_now
    T_pos_herder = T_state.base.herder_state[:, :, :2]
    T_temporal_node_idx = T_state.temporal_node_idx
    T_labels = [
        f"Temporal {t_node_idx} ({env.temporal_node_names[t_node_idx]})" for t_node_idx in T_state.temporal_node_idx
    ]
    anim_path = run_path / "eval_animation.mp4"
    animate_delivery_traj(anim_path, cfg.base, T_pos_herder, T_temporal_node_idx, T_labels)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
