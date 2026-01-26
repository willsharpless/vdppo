import jax.numpy as jnp

from rraa_rl import delivery_cbs, gridworld_cbs, herd_os_cbs
from rraa_rl.envs.scene import ManipScene
from rraa_rl.jax_utils import tree_stack
from rraa_rl.lcrl.lcrl_wrapper import LCRLEnvCfg, LCRLWrapper
from rraa_rl.ldba.ldba import LDBA, Guard, Transition, parse_ltl2ldba
from rraa_rl.src.env.general_task.env import Env
from rraa_rl.src.env.general_task.gridworld import GridworldMA, GridworldMACfg, GridworldMap
from rraa_rl.src.env.general_task.herd_base import HerdingHerdCfg
from rraa_rl.src.env.general_task.herd_os import HerdOs


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

    base_cfg.herd_zero = False
    base_cfg.n_herd = 3
    base_cfg.n_herders = 2
    base_cfg.acc_maxs = [2.0, 4.0]
    base_cfg.vel_maxs = [1.0, 2.0]

    base_cfg.trunc_steps = 150

    cfg = HerdOs.Cfg(specification=spec, base=base_cfg)
    cfg.eval_T = 512
    return cfg


def get_env_and_cbs(env_name: str, agent_name: str) -> tuple[Env, list, list]:
    env_name = env_name.lower()

    herd_eval_cbs = [
        herd_os_cbs.animate_eval_trajs,
        herd_os_cbs.PlotRootTrajPreds.create(),
        herd_os_cbs.plot_eval_trajs,
    ]
    herd_collect_cbs = []

    gridworld_eval_cbs = [gridworld_cbs.animate_eval_trajs, gridworld_cbs.VizValues.create()]
    gridworld_collect_cbs = [gridworld_cbs.collect_cb]

    manip_eval_cbs = []
    manip_collect_cbs = []

    if env_name == "herdos":
        cfg = get_cfg_herdos()
        env = HerdOs(cfg)
        cbs = herd_eval_cbs, herd_collect_cbs

    elif env_name == "herdos_dbg":
        cfg = get_cfg_herdos()

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

        if agent_name == "lcrl":
            base = env.base

            # Acceptance: 1 Inf(0)
            # AP: 1 "goal"
            # --BODY--
            # State: 0
            # [!0] 0
            # [0] 1
            # State: 1
            # [t] 1 {0}

            # hoa_text = """
            # HOA: v1
            # tool: "owl ltl2ldba" "21.0"
            # name: "Automaton for F(goal)"
            # owlArgs: "ltl2ldba" "-f" "F goal"
            # Start: 0
            # acc-name: Buchi
            # Acceptance: 1 Inf(0)
            # properties: trans-acc no-univ-branch
            # properties: deterministic unambiguous
            # properties: complete
            # AP: 1 "goal"
            # --BODY--
            # State: 0
            # [!0] 0
            # [0] 1
            # State: 1
            # [t] 1 {0}
            # --END--
            # """
            # ldba = parse_ltl2ldba(hoa_text)

            # spec = "F A"
            # predicate_order = ["A"]
            # n_states = 2
            # BIT_A = 1 << 0
            # guard = Guard(pos_mask=jnp.array(0b0, dtype=jnp.int32), neg_mask=jnp.array(BIT_A, dtype=jnp.int32))
            # t0 = Transition(src=0, dst=0, guard=guard)
            #
            # guard = Guard(pos_mask=jnp.array(BIT_A, dtype=jnp.int32), neg_mask=jnp.array(0b0, dtype=jnp.int32))
            # t1 = Transition(src=0, dst=1, guard=guard)
            #
            # transitions = tree_stack([t0, t1], axis=0)
            #
            # epsilon_src = jnp.array([], dtype=jnp.int32)
            # epsilon_dst = jnp.array([], dtype=jnp.int32)
            #
            # # (n_accepting_sets=1, n_states=2). Only state 1 is accepting (True).
            # accepting_sets = jnp.array([[0, 1]], dtype=jnp.bool)

            # spec = "F A && F B"
            # predicate_order = ["A", "B"]
            # n_states = 4
            # BIT_A = 1 << 0
            # BIT_B = 1 << 1
            #
            # transition_specs = [
            #     # From state 0
            #     (0, 0, 0b00, BIT_A | BIT_B),  # !a & !b -> 0
            #     (0, 1, BIT_A, BIT_B),  # a & !b -> 1
            #     (0, 2, BIT_B, BIT_A),  # !a & b -> 2
            #     (0, 1, BIT_A | BIT_B, 0b0),  # a & b -> 1
            #     # From state 1
            #     (1, 1, 0b00, BIT_B),  # !b -> 1
            #     (1, 3, BIT_B, 0b0),  # b -> 3
            #     # From state 2
            #     (2, 2, 0b00, BIT_A),  # !a
            #     (2, 3, BIT_A, 0b0),  # a -> 3
            #     # From state 3 (accepting)
            #     (3, 3, 0b0, 0b0),  # t -> 3
            # ]
            #
            # # Build arrays from specs
            # src_list = []
            # dst_list = []
            # pos_mask_list = []
            # neg_mask_list = []
            #
            # for src, dst, pos_mask, neg_mask in transition_specs:
            #     src_list.append(src)
            #     dst_list.append(dst)
            #     pos_mask_list.append(pos_mask)
            #     neg_mask_list.append(neg_mask)
            #
            # transitions = Transition(
            #     src=jnp.array(src_list, dtype=jnp.int32),
            #     dst=jnp.array(dst_list, dtype=jnp.int32),
            #     guard=Guard(
            #         pos_mask=jnp.array(pos_mask_list, dtype=jnp.int32),
            #         neg_mask=jnp.array(neg_mask_list, dtype=jnp.int32),
            #     ),
            # )
            #
            # epsilon_src = jnp.array([], dtype=jnp.int32)
            # epsilon_dst = jnp.array([], dtype=jnp.int32)
            #
            # # (n_accepting_sets=1, n_states=4). Only state 3 is accepting (True).
            # accepting_sets = jnp.array([[0, 0, 0, 1]], dtype=jnp.bool)

            spec = "F A && F B && G( !w )"
            predicate_order = ["A", "B", "w"]
            n_states = 4
            BIT_A = 1 << 0
            BIT_B = 1 << 1
            BIT_w = 1 << 2

            transition_specs = [
                # From state 0
                (0, 0, 0b00, BIT_A | BIT_B | BIT_w),  # !a & !b -> 0
                (0, 1, BIT_A, BIT_B | BIT_w),  # a & !b -> 1
                (0, 2, BIT_B, BIT_A | BIT_w),  # !a & b -> 2
                (0, 1, BIT_A | BIT_B, BIT_w),  # a & b -> 1
                # From state 1
                (1, 1, 0b00, BIT_B | BIT_w),  # !b -> 1
                (1, 3, BIT_B, BIT_w),  # b -> 3
                # From state 2
                (2, 2, 0b00, BIT_A | BIT_w),  # !a
                (2, 3, BIT_A, BIT_w),  # a -> 3
                # From state 3 (accepting)
                (3, 3, 0b0, BIT_w),  # t -> 3
            ]

            # Build arrays from specs
            src_list = []
            dst_list = []
            pos_mask_list = []
            neg_mask_list = []

            for src, dst, pos_mask, neg_mask in transition_specs:
                src_list.append(src)
                dst_list.append(dst)
                pos_mask_list.append(pos_mask)
                neg_mask_list.append(neg_mask)

            transitions = Transition(
                src=jnp.array(src_list, dtype=jnp.int32),
                dst=jnp.array(dst_list, dtype=jnp.int32),
                guard=Guard(
                    pos_mask=jnp.array(pos_mask_list, dtype=jnp.int32),
                    neg_mask=jnp.array(neg_mask_list, dtype=jnp.int32),
                ),
            )

            epsilon_src = jnp.array([], dtype=jnp.int32)
            epsilon_dst = jnp.array([], dtype=jnp.int32)

            # (n_accepting_sets=1, n_states=4). Only state 3 is accepting (True).
            accepting_sets = jnp.array([[0, 0, 0, 1]], dtype=jnp.bool)

            lcrl_env_cfg = LCRLEnvCfg(specification=spec)
            ldba = LDBA(transitions, epsilon_src, epsilon_dst, accepting_sets, n_states, predicate_order)
            env = LCRLWrapper(lcrl_env_cfg, base, ldba)

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
        spec = "F( drawer_open && F( cube_in_drawer ))"
        cfg = ManipScene.Cfg(specification=spec)
        env = ManipScene(cfg)
        cbs = manip_eval_cbs, manip_collect_cbs
    else:
        raise ValueError(f"Unknown environment name: {env_name}")

    return env, cbs[0], cbs[1]
