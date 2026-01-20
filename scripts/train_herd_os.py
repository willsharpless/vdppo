import ipdb
from cyclopts import App

from rraa_rl import herd_os_cbs
from rraa_rl.run import Run
from rraa_rl.src.env.general_task.get_env import get_env
from rraa_rl.trainer import Trainer
from rraa_rl.vd_mappo import VDMAPPOAgent

app = App()


@app.default()
def main(name: str | None = None, debug: bool = False, env_name: str = "HerdOsPlay", seed: int = 123):
    env = get_env(env_name)

    cfg = VDMAPPOAgent.Cfg()
    cfg.actor_lr = 8e-4
    cfg.n_epochs = 2
    cfg.n_minibatches = 4
    cfg.entropy_coef = 1.5e-2
    cfg.rollout_T = 30
    cfg.n_envs_train = 4096

    agent = VDMAPPOAgent.create(seed, cfg, env)

    eval_cbs = [
        herd_os_cbs.animate_eval_trajs,
        herd_os_cbs.PlotRootTrajPreds.create(),
        herd_os_cbs.plot_eval_trajs,
        # herd_os_cbs.VizValues.create(),
    ]
    # eval_cbs = [herd_os_cbs.plot_eval_trajs, VizValues.create()]
    # collect_cbs = [herd_os_cbs.viz_collect_data, herd_os_cbs.viz_obs_histogram]
    collect_cbs = []

    env_name = type(env).__name__
    run = Run.create(env_name=env_name, name=name)
    trainer = Trainer(agent)
    trainer.train(run, env, eval_cbs=eval_cbs, collect_cbs=collect_cbs, debug=debug)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
