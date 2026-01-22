import ipdb
from cyclopts import App

from rraa_rl.run import Run
from rraa_rl.src.env.general_task.get_env import get_env_and_cbs
from rraa_rl.src.get_agent_cfg import get_agent_cfg
from rraa_rl.trainer import Trainer, TrainerCfg
from rraa_rl.vd_mappo import VDMAPPOAgent

app = App()


@app.default()
def main(
    name: str | None = None,
    debug: bool = False,
    env_name: str = "HerdOs",
    seed: int = 123,
    trainer_cfg: TrainerCfg = TrainerCfg(),
):
    env, eval_cbs, collect_cbs = get_env_and_cbs(env_name)
    agent_cfg = get_agent_cfg(env_name, agent_name="VDMAPPO")
    agent = VDMAPPOAgent.create(seed, agent_cfg, env)

    wandb_config = {"seed": seed, "cli_env_name": env_name}

    env_name = type(env).__name__
    run = Run.create(env_name=env_name, name=name)
    trainer = Trainer(agent, trainer_cfg)
    trainer.train(run, env, eval_cbs=eval_cbs, collect_cbs=collect_cbs, debug=debug, wandb_config=wandb_config)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
