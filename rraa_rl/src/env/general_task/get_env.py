from rraa_rl.src.env.general_task.env import Env
from rraa_rl.src.env.general_task.gridworld import GridworldMA, GridworldMACfg, GridworldMap
from rraa_rl import herd_os_cbs, delivery_cbs, gridworld_cbs
from rraa_rl.src.env.general_task.herd_base import HerdingHerdCfg
from rraa_rl.src.env.general_task.herd_os import HerdOs, HerdOsPlay

def get_cfg_herdos():
    # specification = "F G herd_herded && G !herder_oob"
    # fmt: off
    # spec = "G(!herder_oob && !herd_herder_collide) && (F G (herd_herded) && F( herd_gate_1 )"
    # spec = "(!herder_oob && !herd_herder_collide) U (G (herd_herded && !herder_oob && !herd_herder_collide) ) && F( herd_gate_1 )"
    # spec = "(!herder_oob && !herd_herder_collide) U ( herd_gate_1 && (( !herder_oob && !herd_herder_collide ) U G (herd_herded && !herder_oob && !herd_herder_collide) ) )"
    spec = "(!herder_unsafe) U ( herd_gate_0 && (( !herder_unsafe ) U ( herd_gate_1 && (( !herder_unsafe ) U G (herd_herded && !herder_unsafe) ) ) ) )"
    # fmt: on

    base_cfg = HerdingHerdCfg()
    base_cfg.herd_vel = 0.4
    base_cfg.herd_vel_self = 0.05

    # Multiply by 2.25 from original to align with sizes in real life. 10m in sim = 2.5m in real life.
    # base_cfg.agent_radius = 0.45
    base_cfg.agent_radius = 0.41

    # base_cfg.herded_radius = 2.25
    # base_cfg.herded_radius = 2.0
    base_cfg.herded_radius = 1.8

    base_cfg.p_reset_center = 0.25

    base_cfg.herd_zero = False
    base_cfg.n_herd = 3
    base_cfg.n_herders = 2
    base_cfg.acc_maxs = [2.0, 4.0]
    base_cfg.vel_maxs = [1.0, 2.0]

    base_cfg.trunc_steps = 150

    cfg = HerdOs.Cfg(specification=spec, base=base_cfg)
    cfg.eval_T = 512
    return cfg


def get_env_and_cbs(env_name: str) -> tuple[Env, list, list]:
    env_name = env_name.lower()

    herd_eval_cbs = [
        herd_os_cbs.animate_eval_trajs,
        herd_os_cbs.PlotRootTrajPreds.create(),
        herd_os_cbs.plot_eval_trajs,
    ]
    herd_collect_cbs = []

    gridworld_eval_cbs = [
        gridworld_cbs.animate_eval_trajs
    ]
    gridworld_collect_cbs = []

    if env_name == "herdosplay":
        return HerdOsPlay(), herd_eval_cbs, herd_collect_cbs

    if env_name == "herdos":
        cfg = get_cfg_herdos()
        return HerdOs(cfg), herd_eval_cbs, herd_collect_cbs

    if env_name == "herdos_dbg":
        cfg = get_cfg_herdos()

        # 1 herder, 1 herd for easy viz.
        cfg.base.n_herd = 1
        cfg.base.n_herders = 1
        cfg.base.acc_maxs = [2.0]
        cfg.base.vel_maxs = [1.0]

        return HerdOs(cfg), herd_eval_cbs, herd_collect_cbs

    if env_name == "gridworld_map5":
        map5 = GridworldMap.Map5()
        spec = "F A && F B && !D U K && G( !w )"
        cfg = GridworldMACfg(specification=spec, map=map5)

        return GridworldMA(cfg), gridworld_eval_cbs, gridworld_collect_cbs

    raise ValueError(f"Unknown environment name: {env_name}")
