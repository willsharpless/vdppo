import cyclopts
import ipdb
import jax.random as jr

from rraa_rl.distribution import tfd
from rraa_rl.src.env.general_task.herd_os import HerdOs
from rraa_rl.trainer import Trainer
from rraa_rl.vd_mappo import VDMAPPOAgent

app = cyclopts.App()


@app.default()
def main():
    env = HerdOs()
    seed = 123
    cfg = VDMAPPOAgent.Cfg()
    agent = VDMAPPOAgent.create(seed, cfg, env)

    trainer = Trainer()
    trainer.train(agent, env)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
