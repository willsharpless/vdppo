import cyclopts
import ipdb
import jax.random as jr

from rraa_rl import herd_os_cbs
from rraa_rl.distribution import tfd
from rraa_rl.run import Run
from rraa_rl.src.env.general_task.herd_os import HerdOs
from rraa_rl.trainer import Trainer
from rraa_rl.vd_mappo import VDMAPPOAgent

app = cyclopts.App()


@app.default()
def main(debug: bool = False):
    env = HerdOs()
    seed = 123
    cfg = VDMAPPOAgent.Cfg()
    agent = VDMAPPOAgent.create(seed, cfg, env)

    eval_cbs = [herd_os_cbs.plot_eval_trajs, herd_os_cbs.animate_eval_trajs]

    run = Run.create("HerdOs")
    trainer = Trainer(agent)
    trainer.train(run, env, eval_cbs, debug=debug)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
