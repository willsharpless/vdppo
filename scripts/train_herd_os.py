import ipdb
from cyclopts import App

from rraa_rl import herd_os_cbs
from rraa_rl.run import Run
from rraa_rl.src.env.general_task.env import Env
from rraa_rl.src.env.general_task.herd_base import HerdBaseCfg, HerdBasePlay
from rraa_rl.src.env.general_task.herd_os import HerdOs, HerdOsPlay
from rraa_rl.trainer import Trainer
from rraa_rl.vd_mappo import VDMAPPOAgent

app = App()


def get_env(env_name: str) -> Env:
    env_name = env_name.lower()

    if env_name == "herdosplay":
        return HerdOsPlay()

    if env_name == "herdos":
        # specification = "F G herd_herded && G !herder_oob"
        specification = "(!herder_oob) U (G (herd_herded && !herder_oob) )"

        base_cfg = HerdBaseCfg()
        base_cfg.herd_zero = False
        base_cfg.n_herd = 1
        base_cfg.n_herders = 2
        base_cfg.acc_maxs = [1.0, 2.0]
        base_cfg.vel_maxs = [0.5, 1.0]

        cfg = HerdOs.Cfg(specification=specification, base=base_cfg)
        cfg.eval_T = 256

        return HerdOs(cfg)

    raise ValueError(f"Unknown environment name: {env_name}")


@app.default()
def main(name: str | None = None, debug: bool = False, env_name: str = "HerdOsPlay", seed: int = 123):
    env = get_env(env_name)

    cfg = VDMAPPOAgent.Cfg()
    agent = VDMAPPOAgent.create(seed, cfg, env)

    eval_cbs = [
        herd_os_cbs.animate_eval_trajs,
        herd_os_cbs.PlotRootTrajPreds.create(),
        herd_os_cbs.plot_eval_trajs,
        herd_os_cbs.VizValues.create(),
    ]
    # eval_cbs = [herd_os_cbs.plot_eval_trajs, VizValues.create()]
    collect_cbs = [herd_os_cbs.viz_collect_data, herd_os_cbs.viz_obs_histogram]

    env_name = type(env).__name__
    run = Run.create(env_name=env_name, name=name)
    trainer = Trainer(agent)
    trainer.train(run, env, eval_cbs=eval_cbs, collect_cbs=collect_cbs, debug=debug)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
