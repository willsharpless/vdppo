from rraa_rl.lcrl_mappo import LCRLMAPPOAgent
from rraa_rl.vd_mappo import VDMAPPOAgent


def get_vd_agent_cfg(env_name: str):
    env_name = env_name.lower()

    cfg = VDMAPPOAgent.Cfg()
    cfg.actor_lr = 8e-4
    cfg.n_epochs = 2
    cfg.n_minibatches = 4
    cfg.entropy_coef = 1.5e-2
    cfg.rollout_T = 30
    cfg.n_envs_train = 4096

    if env_name == "herdos":
        pass

    if env_name == "gridworld_map1":
        cfg.entropy_coef = 1.5e-2
    if env_name == "gridworld_map5":
        cfg.entropy_coef = 3e-2

    if env_name == "manip_scene":
        cfg.n_envs_train = 256

    return cfg


def get_lcrl_agent_cfg(env_name: str):
    env_name = env_name.lower()

    cfg = LCRLMAPPOAgent.Cfg()
    cfg.actor_lr = 3e-4
    cfg.entropy_coef = 1e-2
    return cfg
