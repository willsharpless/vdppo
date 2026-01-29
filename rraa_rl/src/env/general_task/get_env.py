import jax.numpy as jnp

from rraa_rl import delivery_cbs, deliveryreal_cbs, gridworld_cbs, herd_os_cbs, ablation_cbs
from rraa_rl.jax_utils import tree_stack
from rraa_rl.lcrl.lcrl_wrapper import LCRLEnvCfg, LCRLWrapper
from rraa_rl.ldba.ldba import LDBA, Guard, Transition, parse_ltl2ldba
from rraa_rl.src.env.general_task.delivery import Delivery, DeliveryBase, DeliveryBaseCfg, DeliveryCfg
from rraa_rl.src.env.general_task.env import Env
from rraa_rl.src.env.general_task.gridworld import GridworldMA, GridworldMACfg, GridworldMap
from rraa_rl.src.env.general_task.herd_base import HerdingHerdCfg
from rraa_rl.src.env.general_task.herd_os import HerdOs
from rraa_rl.src.env.general_task.delivery import DeliveryBase, DeliveryBaseCfg, Delivery, DeliveryCfg
from rraa_rl.src.env.general_task.deliveryreal import DeliveryRealBase, DeliveryRealBaseCfg, DeliveryReal, DeliveryRealCfg
from rraa_rl.src.env.general_task.get_env_ldba import get_env_ldba
import ipdb
from loguru import logger


def get_cfg_herdos():
    # specification = "F G herd_herded && G !herder_oob"
    # fmt: off
    # spec = "G(!herder_unsafe) && (F G (herd_herded) && F( herd_gate_1 )"
    # spec = "(!herder_oob && !herd_herder_collide) U (G (herd_herded && !herder_oob && !herd_herder_collide) ) && F( herd_gate_1 )"
    # spec = "(!herder_oob && !herd_herder_collide) U ( herd_gate_1 && (( !herder_oob && !herd_herder_collide ) U G (herd_herded && !herder_oob && !herd_herder_collide) ) )"

    # This below is equivalent to the uncommented version, verified by spot.
    # spec = "G(!herder_unsafe) && F( herd_gate_0 && F( herd_gate_1 ) ) && F G (herd_herded)"
    spec = "(!herder_unsafe) U ( herd_gate_0 && (( !herder_unsafe ) U ( herd_gate_1 && (( !herder_unsafe ) U G (herd_herded && !herder_unsafe) ) ) ) )"
    # fmt: on

    base_cfg = HerdingHerdCfg()
    base_cfg.herd_vel = 0.4
    base_cfg.herd_vel_self = 0.05

    # Multiply by 2.25 from original to align with sizes in real life. 10m in sim = 2.5m in real life.
    # base_cfg.agent_radius = 0.45
    base_cfg.agent_radius = 0.41

    # base_cfg.herded_radius = 2.25
    # base_cfg.herded_radius = 2.0
    base_cfg.herded_radius = 1.8

    base_cfg.p_reset_center = 0.25
    base_cfg.p_reset_task = 0.2
    base_cfg.p_reset_herd = 0.1
    base_cfg.p_reset_gate = 0.2
    base_cfg.p_reset_gap = 0.01

    base_cfg.herd_zero = False
    base_cfg.n_herd = 3
    base_cfg.n_herders = 2
    base_cfg.acc_maxs = [2.0, 4.0]
    base_cfg.vel_maxs = [1.0, 2.0]

    base_cfg.trunc_steps = 150

    cfg = HerdOs.Cfg(specification=spec, base=base_cfg)
    cfg.eval_T = 512
    return cfg, spec


def get_env_and_cbs(
    env_name: str, agent_name: str, n_agent: int = 1, n_spec: int = 1, dense: bool = False,
) -> tuple[Env, list, list]:
    env_name = env_name.lower()

    herd_eval_cbs = [
        herd_os_cbs.env_layout_plot,
        herd_os_cbs.animate_eval_trajs,
        herd_os_cbs.PlotRootTrajPreds.create(),
        herd_os_cbs.plot_eval_trajs,
    ]
    herd_collect_cbs = []
    if agent_name == 'lcrl':
        herd_eval_cbs = [herd_os_cbs.animate_eval_trajs_multi_agent_LDBA]

    gridworld_eval_cbs = [gridworld_cbs.animate_eval_trajs, gridworld_cbs.VizValues.create()]
    gridworld_collect_cbs = [gridworld_cbs.collect_cb]

    delivery_eval_cbs = [
        delivery_cbs.env_layout_plot,
        delivery_cbs.animate_eval_trajs,
        delivery_cbs.PlotRootTrajPreds.create(),
        delivery_cbs.plot_eval_trajs,
        delivery_cbs.VizValues.create(),
    ]
    delivery_collect_cbs = [delivery_cbs.viz_collect_data, delivery_cbs.viz_obs_histogram]
    if agent_name == 'lcrl':
        delivery_eval_cbs, delivery_collect_cbs = [delivery_cbs.animate_eval_trajs_multi_agent_LDBA], []

    deliveryreal_eval_cbs = [
        deliveryreal_cbs.env_layout_plot,
        deliveryreal_cbs.animate_eval_trajs,
        deliveryreal_cbs.PlotRootTrajPreds.create(),
        deliveryreal_cbs.plot_eval_trajs,
        deliveryreal_cbs.VizValues.create(),
    ]
    deliveryreal_collect_cbs = [deliveryreal_cbs.viz_collect_data, deliveryreal_cbs.viz_obs_histogram]

    manip_eval_cbs = []
    manip_collect_cbs = []

    ablation_eval_cbs = [
        ablation_cbs.env_layout_plot,
        ablation_cbs.animate_eval_trajs,
        ablation_cbs.plot_eval_trajs,
        ablation_cbs.VizValues.create(),
    ]
    ablation_collect_cbs = [ablation_cbs.viz_collect_data, ablation_cbs.viz_obs_histogram]
    if agent_name == 'lcrl':
        ablation_eval_cbs, ablation_collect_cbs = [ablation_cbs.animate_eval_trajs_multi_agent_LDBA], []

    manip_eval_cbs = []
    manip_collect_cbs = []

    if env_name == "herdos":
        cfg, spec = get_cfg_herdos()
        env = HerdOs(cfg)
        cbs = herd_eval_cbs, herd_collect_cbs

    elif env_name == "herdos_hardware":
        cfg, spec = get_cfg_herdos()
        cfg.base.wall_thick_x = 0.95
        env = HerdOs(cfg)
        cbs = herd_eval_cbs, herd_collect_cbs

    elif env_name == "herdos_dbg":
        cfg, spec = get_cfg_herdos()

        # 1 herder, 1 herd for easy viz.
        cfg.base.n_herd = 1
        cfg.base.n_herders = 1
        cfg.base.acc_maxs = [2.0]
        cfg.base.vel_maxs = [1.0]

        env = HerdOs(cfg)
        cbs = herd_eval_cbs, herd_collect_cbs

    elif env_name == "gridworld_map1":
        map5 = GridworldMap.Map1()
        # spec = "F A"
        spec = "F A && F B && G( !w )"
        cfg = GridworldMACfg(specification=spec, map=map5)

        env = GridworldMA(cfg)
        cbs = gridworld_eval_cbs, gridworld_collect_cbs

    elif env_name == "gridworld_map5":
        map5 = GridworldMap.Map5()
        spec = "F A && F B && !D U K && G( !w )"
        cfg = GridworldMACfg(specification=spec, map=map5)

        env = GridworldMA(cfg)
        cbs = gridworld_eval_cbs, gridworld_collect_cbs

    elif env_name == "gridworld_map6":
        map6 = GridworldMap.Map6()
        spec = "F( C && F G ( (q U (A && q )) && (q U (B && q )) ) )"
        cfg = GridworldMACfg(specification=spec, map=map6)

        env = GridworldMA(cfg)
        cbs = gridworld_eval_cbs, gridworld_collect_cbs
    elif env_name == "manip_scene":
        from rraa_rl.envs.scene import ManipScene

        # spec = "F( drawer_open && F( cube_in_drawer ))"

        # The following two specs are equivalent, verified by spot.
        # spec = "F( drawer_open && F( cube_in_drawer )) && F G( drawer_closed )"
        spec = "F( drawer_open && F( cube_in_drawer && F G( drawer_closed )))"
        cfg = ManipScene.Cfg(specification=spec)
        env = ManipScene(cfg)
        cbs = manip_eval_cbs, manip_collect_cbs

    elif env_name == "gridworld_map7":
        map7 = GridworldMap.Map7()
        spec = "(!q U C) && F( C && F G ( (q U (A && q )) && (q U (B && q )) ) )"
        cfg = GridworldMACfg(specification=spec, map=map7)

        env = GridworldMA(cfg)
        cbs = gridworld_eval_cbs, gridworld_collect_cbs

    elif env_name == "delivery":
        # specification = "F target0 && G(!oob)"
        # specification = "G(F target0) && G(!oob)"
        # specification = "F target0 && G(!obstacles) && G(!oob)"

        # specification = "F target0 && G(!obstacles) && G(!oob)"
        # specification = "F target0 && F target1 && G(!obstacles) && G(!oob)"
        # specification = "F target0 && F target1 && G(!obstacles) && G(!oob) && G(!collide)"
        # specification = "F target0 && F target1 && G(!obstacles) && G(!oob) && G(F(ags_to_base_agent))"

        # specification = "G(F target0) && G(F target1) && G(!oob) && G(F(ags_to_base_agent))"
        # specification = "G(F target0) && G(F target1) && G(!oob) && G(!aerial_collide) && G(F ag0_base) && G(F ag1_base)"
        # specification = "G(F ag0_target0) && G(F ag1_target1) && G(!oob) && G(!aerial_collide) && G(F ag0_base) && G(F ag1_base)"
        spec = "G(F ag0_target0) && G(F ag1_target1) && G(!obstacles) && G(!oob) && G(!aerial_collide) && G(F ag0_base) && G(F ag1_base)"
        # specification = "G(F target0_dense) && G(F target1_dense) && G(!oob) && G(!aerial_collide) && G(F ag0_base) && G(F ag1_base)"
        # specification = "G(F target0) && G(F target1) && G(!obstacles) && G(!oob) && G(!aerial_collide) && G(F ag0_base) && G(F ag1_base)"

        # to come
        # specification = "F target0 && F target1 && G(!obstacles) && G(!oob) && G(ag_at_target => ag_to_base_agent)"
        # specification = "G(F target0 && F target1) && G(!obstacles && !oob) && G(!ag_at_target || ag_to_base_agent)"
        # specification = "G(F target0 && F target1) && G(!obstacles && !oob) && G(!ag1_at_target || ag1_to_base_agent) && G(!ag2_at_target || ag2_to_base_agent)"

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
        base_cfg.centers = [
            [-2.0, 0.0],
            [3.0, 1.0],
        ]
        base_cfg.radiuses = [0.5, 0.5]
        base_cfg.update_cond_fn = (
            "agent_in_respective_target" if "ag0_target0" in spec else "any_agent_in_target"
        )
        base_cfg.update_cond_fn = 'agent_in_respective_target'

        cfg = Delivery.Cfg(specification=spec, base=base_cfg)
        env = Delivery(cfg)

        cbs = delivery_eval_cbs, delivery_collect_cbs

    elif env_name == "deliveryreal":
        
        spec = "G(F ag0_target0) && G(F ag1_target1) && G(!obstacles) && G(!oob) && G(!aerial_collide) && G(F ag0_base) && G(F ag1_base)"

        base_cfg = DeliveryRealBaseCfg()

        cfg = DeliveryReal.Cfg(specification=spec, base=base_cfg)
        env = DeliveryReal(cfg)

        cbs = deliveryreal_eval_cbs, deliveryreal_collect_cbs

    elif env_name == "ablation":
        ## N spec and N agent Ablation Env (Double Integrator)

        spec = "G(!oob) && G(!obstacles)" if n_agent == 1 else "G(!oob) && G(!obstacles) && G(!collide)"
        for i in range(n_spec):
            spec += f" && F target{i}" if not dense else f" && F target{i}_dense"

        base_cfg = DeliveryBaseCfg()

        base_cfg.n_herders = n_agent
        base_cfg.n_herd = n_agent
        base_cfg.acc_maxs = [2.0] * n_agent
        base_cfg.vel_maxs = [1.0] * n_agent
        base_cfg.dynamic_targets = False
        base_cfg.update_targets = False

        cfg = Delivery.Cfg(specification=spec, base=base_cfg)
        env = Delivery(cfg)

        cbs = ablation_eval_cbs, ablation_collect_cbs

    elif env_name == "ablation_depth":
        ## N spec and N agent Ablation Env (Double Integrator)

        spec = "G(!oob) && G(!obstacles)" if n_agent == 1 else "G(!oob) && G(!obstacles) && G(!collide)"

        dense_tag = "_dense" if dense else ""
        if n_spec == 1:
            spec += f" && F(target0{dense_tag})"
        elif n_spec == 2:
            spec += f" && F(target0{dense_tag} && F(target1{dense_tag}))"
        elif n_spec == 3:
            spec += f" && F(target0{dense_tag} && F(target1{dense_tag} && F(target2{dense_tag})))"
        elif n_spec == 4:
            spec += f" && F(target0{dense_tag} && F(target1{dense_tag} && F(target2{dense_tag} && F(target3{dense_tag}))))"
        elif n_spec == 5:
            spec += f" && F(target0{dense_tag} && F(target1{dense_tag} && F(target2{dense_tag} && F(target3{dense_tag} && F(target4{dense_tag})))))"

        base_cfg = DeliveryBaseCfg()

        base_cfg.n_herders = n_agent
        base_cfg.n_herd = n_agent
        base_cfg.acc_maxs = [2.0] * n_agent
        base_cfg.vel_maxs = [1.0] * n_agent
        base_cfg.dynamic_targets = False
        base_cfg.update_targets = False

        cfg = Delivery.Cfg(specification=spec, base=base_cfg)
        env = Delivery(cfg)

        cbs = ablation_eval_cbs, ablation_collect_cbs

    else:
        raise ValueError(f"Unknown environment name: {env_name}")
    
    if agent_name == 'lcrl':
        lcrl_env_cfg = LCRLEnvCfg(specification=spec)
        ldba = get_env_ldba(env_name, n_spec=n_spec)
        env = LCRLWrapper(lcrl_env_cfg, env.base, ldba)

    return env, cbs[0], cbs[1]
