from rraa_rl.src.env.general_task.env import Env
from rraa_rl.src.env.general_task.herd_base import HerdingHerdCfg
from rraa_rl.src.env.general_task.herd_os import HerdOsPlay, HerdOs


def get_env(env_name: str) -> Env:
    env_name = env_name.lower()

    if env_name == "herdosplay":
        return HerdOsPlay()

    if env_name == "herdos":
        # specification = "F G herd_herded && G !herder_oob"
        specification = "(!herder_oob) U (G (herd_herded && !herder_oob) )"

        base_cfg = HerdingHerdCfg()
        base_cfg.herd_vel = 0.2
        base_cfg.herd_vel_self = 0.05

        base_cfg.p_reset_center = 0.25

        base_cfg.herd_zero = False
        base_cfg.n_herd = 3
        base_cfg.n_herders = 2
        base_cfg.acc_maxs = [2.0, 4.0]
        base_cfg.vel_maxs = [1.0, 2.0]

        cfg = HerdOs.Cfg(specification=specification, base=base_cfg)
        cfg.eval_T = 256

        return HerdOs(cfg)

    raise ValueError(f"Unknown environment name: {env_name}")
