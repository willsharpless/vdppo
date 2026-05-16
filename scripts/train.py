import ipdb
from cyclopts import App

from rraa_rl.agents.cmdp_mappo import CMDPMAPPOAgent
from rraa_rl.agents.lcrl_mappo import LCRLMAPPOAgent
from rraa_rl.agents.vd_mappo import VDMAPPOAgent
from rraa_rl.lcrl.lcrl_wrapper import LCRLWrapper
from rraa_rl.training.run import Run
from rraa_rl.env.general_task.env import Env
from rraa_rl.env.general_task.get_env import get_env_and_cbs
from rraa_rl.get_agent_cfg import get_lcrl_agent_cfg, get_vd_agent_cfg, get_cmdp_agent_cfg
from rraa_rl.training.trainer import Trainer, TrainerCfg

app = App()


@app.command()
def vd(
    name: str | None = None,
    debug: bool = False,
    env_name: str = "Delivery",
    seed: int = 123,
    trainer_cfg: TrainerCfg = TrainerCfg(),
    n_agent: int = 1,
    n_spec: int = 1,
    dense: bool = False,
    actor_shared_trunk: bool = True,
    value_shared_trunk: bool = True,
    n_layers: int = 2,
    run_callbacks: bool = True,
):
    env, eval_cbs, collect_cbs = get_env_and_cbs(env_name, agent_name="vd", n_agent=n_agent, n_spec=n_spec, dense=dense)
    agent_cfg = get_vd_agent_cfg(env_name)
    agent_cfg.actor_shared_trunk = actor_shared_trunk
    agent_cfg.value_shared_trunk = value_shared_trunk
    agent_cfg.actor_hids = (128,) * n_layers
    agent_cfg.critic_hids = (128,) * n_layers

    if hasattr(env.base, "n_envs"):
        env.base.n_envs = agent_cfg.n_envs_train

    agent = VDMAPPOAgent.create(seed, agent_cfg, env)

    return train(name, debug, env_name, seed, trainer_cfg, env, eval_cbs, collect_cbs, agent, run_callbacks)


@app.command()
def cmdp(
    name: str | None = None,
    debug: bool = False,
    env_name: str = "Delivery",
    seed: int = 123,
    trainer_cfg: TrainerCfg = TrainerCfg(),
    n_agent: int = 1,
    n_spec: int = 1,
    dense: bool = False,
    run_callbacks: bool = True,
):
    env, eval_cbs, collect_cbs = get_env_and_cbs(env_name, agent_name="cmdp", n_agent=n_agent, n_spec=n_spec, dense=dense)
    agent_cfg = get_cmdp_agent_cfg(env_name)

    if hasattr(env.base, "n_envs"):
        env.base.n_envs = agent_cfg.n_envs_train

    agent = CMDPMAPPOAgent.create(seed, agent_cfg, env)

    return train(name, debug, env_name, seed, trainer_cfg, env, eval_cbs, collect_cbs, agent, run_callbacks)


@app.command()
def lcrl(
    name: str | None = None,
    debug: bool = False,
    env_name: str = "Herding",
    seed: int = 123,
    trainer_cfg: TrainerCfg = TrainerCfg(),
    n_agent: int = 1,
    n_spec: int = 1,
    dense: bool = False,
    run_callbacks: bool = True,
):
    env: LCRLWrapper
    env, eval_cbs, collect_cbs = get_env_and_cbs(
        env_name, agent_name="lcrl", n_agent=n_agent, n_spec=n_spec, dense=dense
    )
    agent_cfg = get_lcrl_agent_cfg(env_name)

    if hasattr(env.base, "n_envs"):
        env.base.n_envs = agent_cfg.n_envs_train

    agent = LCRLMAPPOAgent.create(seed, agent_cfg, env)
    # agent_cfg.random_automata_init = True

    env.cfg.random_automata_init = agent_cfg.random_automata_init

    return train(name, debug, env_name, seed, trainer_cfg, env, eval_cbs, collect_cbs, agent, run_callbacks)


def train(
    name: str | None,
    debug: bool,
    env_name: str,
    seed: int,
    trainer_cfg: TrainerCfg,
    env: Env,
    eval_cbs: list,
    collect_cbs: list,
    agent: VDMAPPOAgent | LCRLMAPPOAgent,
    run_callbacks: bool = True,
):
    agent_name = agent.get_agent_name()
    wandb_config = {"seed": seed, "cli_env_name": env_name, "agent_name": agent_name}

    # env_name = f"{type(env).__name__}-{env_name}"
    run = Run.create(env_name=env_name.lower(), agent_name=agent_name, name=name, debug=debug)
    trainer = Trainer(agent, trainer_cfg)
    if not run_callbacks:
        eval_cbs = []
        collect_cbs = []
    trainer.train(run, env, eval_cbs=eval_cbs, collect_cbs=collect_cbs, debug=debug, wandb_config=wandb_config)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
