import ipdb
from cyclopts import App

from rraa_rl import herd_os_cbs
from rraa_rl import delivery_cbs
from rraa_rl.run import Run
from rraa_rl.src.env.general_task.env import Env
from rraa_rl.src.env.general_task.delivery_base import DeliveryBaseCfg, DeliveryBasePlay
from rraa_rl.src.env.general_task.delivery import Delivery, DeliveryPlay
from rraa_rl.trainer import Trainer
from rraa_rl.vd_mappo import VDMAPPOAgent
import os 

app = App()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_env(env_name: str) -> Env:
    env_name = env_name.lower()

    if env_name == "deliveryplay":
        return DeliveryPlay()

    if env_name == "delivery":
        specification = "F target0 && F target1 && G(!obstacles) && G(!oob)"
        # specification = "F target0 && F target1 && G(!obstacles) && G(!oob) && G(!collide)"

        # 2 ag by default
        base_cfg = DeliveryBaseCfg()

        # 1 agent test
        base_cfg.n_herders = 1
        base_cfg.n_herd = 1
        base_cfg.acc_maxs = [1.0]
        base_cfg.vel_maxs = [0.5]

        cfg = Delivery.Cfg(specification=specification, base=base_cfg)
        return Delivery(cfg)

    raise ValueError(f"Unknown environment name: {env_name}")


@app.default()
def main(name: str | None = None, debug: bool = False, env_name: str = "delivery", seed: int = 123):
    env = get_env(env_name)

    cfg = VDMAPPOAgent.Cfg()
    agent = VDMAPPOAgent.create(seed, cfg, env)

    eval_cbs = [
        delivery_cbs.animate_eval_trajs,
        delivery_cbs.PlotRootTrajPreds.create(),
        delivery_cbs.plot_eval_trajs,
        delivery_cbs.VizValues.create(),
    ]
    # eval_cbs = [herd_os_cbs.plot_eval_trajs, VizValues.create()]
    collect_cbs = [delivery_cbs.viz_collect_data, delivery_cbs.viz_obs_histogram]

    env_name = type(env).__name__
    run = Run.create(env_name=env_name, name=name)
    trainer = Trainer(agent)
    trainer.train(run, env, eval_cbs=eval_cbs, collect_cbs=collect_cbs, debug=debug)


if __name__ == "__main__":
    # with ipdb.launch_ipdb_on_exception():
    #     app()
    app()
