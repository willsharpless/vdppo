import ipdb
from cyclopts import App

from rraa_rl.callbacks import delivery_cbs
from rraa_rl.training.run import Run
from rraa_rl.env.general_task.delivery import Delivery
from rraa_rl.env.general_task.delivery_base import DeliveryBaseCfg
from rraa_rl.env.general_task.env import Env
from rraa_rl.training.trainer import Trainer, TrainerCfg
from rraa_rl.agents.vd_mappo import VDMAPPOAgent

app = App()

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
# os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')

def get_env(env_name: str) -> Env:
    env_name = env_name.lower()

    if env_name == "delivery":
        specification = "F target0 && G(!oob)"
        # specification = "F target0 && G(!obstacles) && G(!oob)"
        # specification = "F target0 && F target1 && G(!obstacles) && G(!oob)"
        # specification = "F target0 && F target1 && G(!obstacles) && G(!oob) && G(!collide)"
        # specification = "F target0 && F target1 && G(!obstacles) && G(!oob) && G(F(ags_to_base_agent))"
        
        # specification = "G(F target0) && G(F target1) && G(!obstacles) && G(!oob) && G(F(ags_to_base_agent))"

        # to come
        # specification = "F target0 && F target1 && G(!obstacles) && G(!oob) && G(ag_at_target => ag_to_base_agent)"
        # specification = "G(F target0 && F target1) && G(!obstacles && !oob) && G(!ag_at_target || ag_to_base_agent)"
        # specification = "G(F target0 && F target1) && G(!obstacles && !oob) && G(!ag1_at_target || ag1_to_base_agent) && G(!ag2_at_target || ag2_to_base_agent)"

        ## TODO add !collide !!!

        ## 2 ag by default
        base_cfg = DeliveryBaseCfg()

        ## 1 agent test
        # base_cfg.n_herders = 1
        # base_cfg.n_herd = 1
        # base_cfg.acc_maxs = [1.0]
        # base_cfg.vel_maxs = [0.5]

        # base_cfg.n_herders = 2
        # base_cfg.n_herd = 2
        # base_cfg.acc_maxs = [3.0, 3.0]
        # base_cfg.vel_maxs = [1.0, 1.0]

        # 3 agent test with base agent (last agent)
        base_cfg.base_agent = True
        base_cfg.n_herders = 3
        base_cfg.n_herd = 3
        base_cfg.acc_maxs = [2.0, 2.0, 1.0]
        base_cfg.vel_maxs = [1.0, 1.0, 0.1]
        base_cfg.dynamic_targets = True
        base_cfg.update_targets = True

        cfg = Delivery.Cfg(specification=specification, base=base_cfg)
        return Delivery(cfg)

    raise ValueError(f"Unknown environment name: {env_name}")


@app.default()
def main(
    name: str | None = None,
    debug: bool = False,
    env_name: str = "delivery",
    seed: int = 123,
    trainer_cfg: TrainerCfg = TrainerCfg(),
):
    env = get_env(env_name)

    cfg = VDMAPPOAgent.Cfg()
    agent = VDMAPPOAgent.create(seed, cfg, env)

    eval_cbs = [
        delivery_cbs.animate_eval_trajs,
        delivery_cbs.PlotRootTrajPreds.create(),
        delivery_cbs.plot_eval_trajs,
        delivery_cbs.VizValues.create(),
    ]
    # eval_cbs = [herding_cbs.plot_eval_trajs, VizValues.create()]
    collect_cbs = [delivery_cbs.viz_collect_data, delivery_cbs.viz_obs_histogram]

    env_name = type(env).__name__
    run = Run.create(env_name=env_name, name=name)
    trainer = Trainer(agent, cfg=trainer_cfg)
    trainer.train(run, env, eval_cbs=eval_cbs, collect_cbs=collect_cbs, debug=debug)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
    # app()
