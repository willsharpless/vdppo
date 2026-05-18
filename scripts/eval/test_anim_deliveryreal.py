import pathlib

import cyclopts
import ipdb
import jax
import jax.random as jr

from vdppo.training.collector import Collector
from vdppo.callbacks.deliveryreal_cbs import animate_deliveryreal_traj
from vdppo.training.load_ckpt import load_ckpt
from vdppo.training.rollout_utils import extract_rollouts_eval
from vdppo.env.general_task.deliveryreal import DeliveryReal
from vdppo.agents.vdppo import VDPPOAgent

app = cyclopts.App()


@app.default()
def main(run_path: pathlib.Path, n_env: int = 1, step: int | None = None, eval_T: int | None = None):
    run, agent, env, cfg_dict = load_ckpt(run_path, step)

    collector = Collector.create(
        key=jr.PRNGKey(1234),
        env=env,
        cfg=Collector.Cfg(n_envs=n_env, auto_reset=False, ignore_trunc=True),
    )
    b_state0 = env.get_eval_states(collector.cfg.n_envs, root_only=True)

    collect_opts = {}
    if isinstance(agent, VDPPOAgent):
        collect_opts["temporal_transitions"] = True

    eval_T = eval_T or env.eval_T

    Tb_rollout, info_collect = agent.collect_eval_with_states(collector, b_state0, eval_T, **collect_opts)
    Tb_rollout = jax.device_get(Tb_rollout)
    bT_rollout = Tb_rollout.switch01()

    # Extract each rollout
    b_trajs = extract_rollouts_eval(bT_rollout)

    # Animate the first trajectory.
    traj = b_trajs[0]

    # ------------------------------------------------------------
    cfg: DeliveryReal.Cfg = env.cfg

    T_state: DeliveryReal.State = traj.state_now
    T_pos_herder = T_state.base.herder_state[:, :, :2]
    T_pos_target = T_state.base.centers[:, :2, :2] # only using first two centers
    T_temporal_node_idx = T_state.temporal_node_idx
    T_labels = [
        f"Temporal {t_node_idx} ({env.temporal_node_names[t_node_idx]})" for t_node_idx in T_state.temporal_node_idx
    ]
    anim_path = run_path / "eval_animation.mp4"
    # ipdb.set_trace()
    animate_deliveryreal_traj(anim_path, env, cfg.base, T_pos_herder, T_pos_target, T_temporal_node_idx, T_labels)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
