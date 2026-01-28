import ipdb
from cyclopts import App

from rraa_rl.lcrl.lcrl_wrapper import LCRLEnvCfg, LCRLWrapper
from rraa_rl.lcrl_mappo import LCRLMAPPOAgent
from rraa_rl.run import Run
from rraa_rl.src.env.general_task.env import Env, EnvUsingBase
from rraa_rl.src.env.general_task.get_env import get_env_and_cbs
from rraa_rl.src.get_agent_cfg import get_lcrl_agent_cfg, get_vd_agent_cfg
from rraa_rl.trainer import Trainer, TrainerCfg
from rraa_rl.vd_mappo import VDMAPPOAgent

app = App()

@app.default()
def main(
    algs: list[str] | None = None,
    debug: bool = False,
    env_name: str = "Delivery",
    seed: int = 123,
    n_envs_test: int = 128,
    trainer_cfg: TrainerCfg = TrainerCfg(),
    n_agent: int = 1,
    n_spec: int = 1,
    dense: bool = False
):
    
    algs = ["vd", "mppi"] if algs is None else algs
    # algs = ["vd", "lcrl", "mppi"] if algs is None else algs
    # [drl2, lcer]

    for alg in algs:

        env, _, _ = get_env_and_cbs(env_name, agent_name=alg, n_agent=n_agent, n_spec=n_spec, dense=dense)

        if hasattr(env.base, "n_envs"):
            env.base.n_envs = agent_cfg.n_envs_train

        if alg == "vd":
            agent_cfg = get_vd_agent_cfg(env_name)
            agent = VDMAPPOAgent.create(seed, agent_cfg, env)

        elif alg == "lcrl":
            agent_cfg = get_lcrl_agent_cfg(env_name)
            agent = LCRLMAPPOAgent.create(seed, agent_cfg, env)
            env.cfg.random_automata_init = False

        elif alg == "mppi":
            # mppi_cfg = MPPICfg()
            # agent = MPPI.create(seed, mppi_cfg, env)
            pass

        else:
            raise ValueError(f"Unknown alg {alg}")  


        trainer = Trainer(agent, trainer_cfg)
        out = trainer.eval(trainer.make_eval_collector(env, n_envs_test))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
