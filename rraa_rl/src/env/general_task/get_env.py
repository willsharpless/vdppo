from rraa_rl.src.env.general_task.env import Env
from rraa_rl.src.env.general_task.herd_base import HerdingHerdCfg
from rraa_rl.src.env.general_task.herd_os import HerdOs, HerdOsPlay


def get_env(env_name: str) -> Env:
    env_name = env_name.lower()

    if env_name == "herdosplay":
        return HerdOsPlay()

    if env_name == "herdos":
        # specification = "F G herd_herded && G !herder_oob"
        # fmt: off
        # spec = "G(!herder_oob && !herd_herder_collide) && (F G (herd_herded) && F( herd_gate_1 )"
        # spec = "(!herder_oob && !herd_herder_collide) U (G (herd_herded && !herder_oob && !herd_herder_collide) ) && F( herd_gate_1 )"
        # spec = "(!herder_oob && !herd_herder_collide) U ( herd_gate_1 && (( !herder_oob && !herd_herder_collide ) U G (herd_herded && !herder_oob && !herd_herder_collide) ) )"
        spec = "(!is_herder_unsafe) U ( herd_gate_0 && (( !is_herder_unsafe ) U ( herd_gate_1 && (( !is_herder_unsafe ) U G (herd_herded && !is_herder_unsafe) ) ) ) )"
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

        return HerdOs(cfg)

    raise ValueError(f"Unknown environment name: {env_name}")
