import pathlib
from typing import Annotated

import cyclopts
import ipdb
import jax.random as jr
from cyclopts import App, Parameter

from rraa_rl import herd_os_cbs
from rraa_rl.distribution import tfd
from rraa_rl.run import Run
from rraa_rl.src.env.general_task.herd_base import HerdBase, HerdBasePlay
from rraa_rl.src.env.general_task.herd_os import HerdOs
from rraa_rl.trainer import Trainer
from rraa_rl.vd_mappo import VDMAPPOAgent

app = App()


@app.default()
def main(name: str | None = None, debug: bool = False):
    base_cfg = HerdBasePlay.Cfg()
    base_cfg.n_herders = 2
    base_cfg.acc_maxs = [1.0, 2.0]
    base_cfg.vel_maxs = [0.5, 1.0]

    env_cfg = HerdOs.Cfg(base=base_cfg)

    env = HerdOs(cfg=env_cfg)
    seed = 123
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

    run = Run.create(env_name="HerdOs", name=name)
    trainer = Trainer(agent)
    trainer.train(run, env, eval_cbs=eval_cbs, collect_cbs=collect_cbs, debug=debug)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
