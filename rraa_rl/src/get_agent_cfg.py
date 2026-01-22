from rraa_rl.vd_mappo import VDMAPPOAgent


def get_agent_cfg(env_name: str, agent_name: str):
    env_name = env_name.lower()

    assert agent_name == "VDMAPPO"

    if env_name == "herdos":
        cfg = VDMAPPOAgent.Cfg()
        cfg.actor_lr = 8e-4
        cfg.n_epochs = 2
        cfg.n_minibatches = 4
        cfg.entropy_coef = 1.5e-2
        cfg.rollout_T = 30
        cfg.n_envs_train = 4096

        return cfg

    raise ValueError(f"Unknown env_name {env_name}")
