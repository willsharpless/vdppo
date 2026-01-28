import jax.numpy as jnp

from rraa_rl import ablation_cbs, delivery_cbs, gridworld_cbs, herd_os_cbs
# from rraa_rl.envs.scene import ManipScene
from rraa_rl.jax_utils import tree_stack
from rraa_rl.lcrl.lcrl_wrapper import LCRLEnvCfg, LCRLWrapper
from rraa_rl.ldba.ldba import LDBA, Guard, Transition, parse_ltl2ldba
from rraa_rl.src.env.general_task.delivery import Delivery, DeliveryBase, DeliveryBaseCfg, DeliveryCfg
from rraa_rl.src.env.general_task.env import Env
from rraa_rl.src.env.general_task.gridworld import GridworldMA, GridworldMACfg, GridworldMap
from rraa_rl.src.env.general_task.herd_base import HerdingHerdCfg
from rraa_rl.src.env.general_task.herd_os import HerdOs


def get_env_ldba(env_name: str, n_spec: int = 1, n_agent: int = 1) -> tuple[Env, list, list]:

    if env_name == "herdos":

        return create_herding_ldba()

        # Automaton for: (!herder_unsafe) U ( herd_gate_0 && (( !herder_unsafe ) U ( herd_gate_1 && (( !herder_unsafe ) U G (herd_herded && !herder_unsafe) ) ) ) )
        spec = "(!herder_unsafe) U ( herd_gate_0 && (( !herder_unsafe ) U ( herd_gate_1 && (( !herder_unsafe ) U G (herd_herded && !herder_unsafe) ) ) ) )"
        predicate_order = ["herder_unsafe", "herd_gate_0", "herd_gate_1", "herd_herded"]
        n_states = 4

        BIT_unsafe = 1 << 0
        BIT_gate0 = 1 << 1
        BIT_gate1 = 1 << 2
        BIT_herded = 1 << 3

        # States:
        # 0: initial - waiting for gate_0, !unsafe must hold
        # 1: passed gate_0, waiting for gate_1, !unsafe must hold
        # 2: passed gate_1, waiting for herded (entering G phase), !unsafe must hold
        # 3: accepting - in G(herded && !unsafe) phase

        transition_specs = [
            # From state 0: waiting for gate_0, !unsafe required
            (0, 0, 0b0, BIT_gate0 | BIT_unsafe),  # !gate0 & !unsafe -> 0
            (0, 1, BIT_gate0, BIT_unsafe),  # gate0 & !unsafe -> 1
            # From state 1: passed gate_0, waiting for gate_1, !unsafe required
            (1, 1, 0b0, BIT_gate1 | BIT_unsafe),  # !gate1 & !unsafe -> 1
            (1, 2, BIT_gate1, BIT_unsafe),  # gate1 & !unsafe -> 2
            # From state 2: passed gate_1, waiting for herded, !unsafe required
            (2, 2, 0b0, BIT_herded | BIT_unsafe),  # !herded & !unsafe -> 2
            (2, 3, BIT_herded, BIT_unsafe),  # herded & !unsafe -> 3
            # From state 3: accepting - G(herded && !unsafe)
            (3, 3, BIT_herded, BIT_unsafe),  # herded & !unsafe -> 3
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

        # Only state 3 is accepting
        accepting_sets = jnp.array([[0, 0, 0, 1]], dtype=jnp.bool)

    elif env_name == "herdos_dbg":
        # cfg = get_cfg_herdos()

        # # 1 herder, 1 herd for easy viz.
        # cfg.base.n_herd = 1
        # cfg.base.n_herders = 1
        # cfg.base.acc_maxs = [2.0]
        # cfg.base.vel_maxs = [1.0]

        # env = HerdOs(cfg)
        # cbs = herd_eval_cbs, herd_collect_cbs
        raise NotImplementedError("""HerdOsDbg environment is not implemented in this snippet.""")

    # elif env_name == "gridworld_map0":

    #     return create_gridworld_map0_ldba()

    elif env_name == "gridworld_map1":

        return create_gridworld_map1_ldba()

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

    elif env_name == "gridworld_map5":

        return create_gridworld_map5_ldba()

        # Automaton for: F A && F B && !D U K && G( !w )
        spec = "F A && F B && !D U K && G( !w )"
        predicate_order = ["A", "B", "D", "K", "w"]
        n_states = 8

        BIT_A = 1 << 0
        BIT_B = 1 << 1
        BIT_D = 1 << 2
        BIT_K = 1 << 3
        BIT_w = 1 << 4

        # States:
        # 0: initial - need A, B, K (before K, !D must hold)
        # 1: have A, need B, K
        # 2: have B, need A, K
        # 3: have A & B, need K
        # 4: have K, need A, B
        # 5: have K & A, need B
        # 6: have K & B, need A
        # 7: accepting - have all (A, B, K achieved)

        transition_specs = [
            # From state 0: need A, B, K (!D until K)
            (0, 0, 0b0, BIT_A | BIT_B | BIT_K | BIT_D | BIT_w),  # !a & !b & !k & !d & !w -> 0
            (0, 1, BIT_A, BIT_B | BIT_K | BIT_D | BIT_w),  # a & !b & !k & !d & !w -> 1
            (0, 2, BIT_B, BIT_A | BIT_K | BIT_D | BIT_w),  # !a & b & !k & !d & !w -> 2
            (0, 3, BIT_A | BIT_B, BIT_K | BIT_D | BIT_w),  # a & b & !k & !d & !w -> 3
            (0, 4, BIT_K, BIT_A | BIT_B | BIT_w),  # !a & !b & k & !w -> 4 (D doesn't matter once K)
            (0, 5, BIT_A | BIT_K, BIT_B | BIT_w),  # a & !b & k & !w -> 5
            (0, 6, BIT_B | BIT_K, BIT_A | BIT_w),  # !a & b & k & !w -> 6
            (0, 7, BIT_A | BIT_B | BIT_K, BIT_w),  # a & b & k & !w -> 7
            # From state 1: have A, need B, K (!D until K)
            (1, 1, 0b0, BIT_B | BIT_K | BIT_D | BIT_w),  # !b & !k & !d & !w -> 1
            (1, 3, BIT_B, BIT_K | BIT_D | BIT_w),  # b & !k & !d & !w -> 3
            (1, 5, BIT_K, BIT_B | BIT_w),  # !b & k & !w -> 5
            (1, 7, BIT_B | BIT_K, BIT_w),  # b & k & !w -> 7
            # From state 2: have B, need A, K (!D until K)
            (2, 2, 0b0, BIT_A | BIT_K | BIT_D | BIT_w),  # !a & !k & !d & !w -> 2
            (2, 3, BIT_A, BIT_K | BIT_D | BIT_w),  # a & !k & !d & !w -> 3
            (2, 6, BIT_K, BIT_A | BIT_w),  # !a & k & !w -> 6
            (2, 7, BIT_A | BIT_K, BIT_w),  # a & k & !w -> 7
            # From state 3: have A & B, need K (!D until K)
            (3, 3, 0b0, BIT_K | BIT_D | BIT_w),  # !k & !d & !w -> 3
            (3, 7, BIT_K, BIT_w),  # k & !w -> 7
            # From state 4: have K, need A, B (D no longer matters)
            (4, 4, 0b0, BIT_A | BIT_B | BIT_w),  # !a & !b & !w -> 4
            (4, 5, BIT_A, BIT_B | BIT_w),  # a & !b & !w -> 5
            (4, 6, BIT_B, BIT_A | BIT_w),  # !a & b & !w -> 6
            (4, 7, BIT_A | BIT_B, BIT_w),  # a & b & !w -> 7
            # From state 5: have K & A, need B
            (5, 5, 0b0, BIT_B | BIT_w),  # !b & !w -> 5
            (5, 7, BIT_B, BIT_w),  # b & !w -> 7
            # From state 6: have K & B, need A
            (6, 6, 0b0, BIT_A | BIT_w),  # !a & !w -> 6
            (6, 7, BIT_A, BIT_w),  # a & !w -> 7
            # From state 7: accepting
            (7, 7, 0b0, BIT_w),  # !w -> 7
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

        # Only state 7 is accepting
        accepting_sets = jnp.array([[0, 0, 0, 0, 0, 0, 0, 1]], dtype=jnp.bool)

    elif env_name == "gridworld_map6":

        return create_gridworld_map6_ldba()

        spec = "F( C && F G ( (q U (A && q )) && (q U (B && q )) ) )"
        raise NotImplementedError(f"""{env_name} LDBA is not implemented""")

    elif env_name == "manip_scene":

        raise NotImplementedError(f"""{env_name} LDBA is not implemented""")

    elif env_name == "gridworld_map7":

        return create_gridworld_map7_ldba()
    
        # Automaton for: (!q U C) && F( C && F G ( (q U (A && q )) && (q U (B && q )) ) )
        spec = "(!q U C) && F( C && F G ( (q U (A && q )) && (q U (B && q )) ) )"
        predicate_order = ["A", "B", "C", "q"]
        n_states = 6

        BIT_A = 1 << 0
        BIT_B = 1 << 1
        BIT_C = 1 << 2
        BIT_q = 1 << 3

        # States:
        # 0: initial - waiting for C, !q must hold
        # 1: seen C (first requirement satisfied), now need to reach state where G(...) starts
        # 2: in the G(...) phase, need A (with q), need B (with q), q must hold
        # 3: in G phase, have A, need B, q must hold
        # 4: in G phase, have B, need A, q must hold
        # 5: accepting - in G phase, have both A and B achieved, q must continue to hold

        transition_specs = [
            # From state 0: waiting for C, !q must hold
            (0, 0, 0b0, BIT_C | BIT_q),  # !c & !q -> 0
            (0, 1, BIT_C, BIT_q),  # c & !q -> 1
            # From state 1: C achieved, can transition to G phase
            (1, 1, 0b0, 0b0),  # any -> 1 (waiting)
            # Epsilon transition to state 2 to enter the G phase (non-deterministic choice)
            # From state 2: in G phase, need A&q and B&q, q must hold
            (2, 2, BIT_q, BIT_A | BIT_B),  # q & !a & !b -> 2
            (2, 3, BIT_A | BIT_q, BIT_B),  # a & q & !b -> 3
            (2, 4, BIT_B | BIT_q, BIT_A),  # b & q & !a -> 4
            (2, 5, BIT_A | BIT_B | BIT_q, 0b0),  # a & b & q -> 5
            # From state 3: have A, need B, q must hold
            (3, 3, BIT_q, BIT_B),  # q & !b -> 3
            (3, 5, BIT_B | BIT_q, 0b0),  # b & q -> 5
            # From state 4: have B, need A, q must hold
            (4, 4, BIT_q, BIT_A),  # q & !a -> 4
            (4, 5, BIT_A | BIT_q, 0b0),  # a & q -> 5
            # From state 5: accepting, q must continue to hold forever
            (5, 5, BIT_q, 0b0),  # q -> 5
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

        # Epsilon transition from state 1 to state 2 (entering the G phase)
        epsilon_src = jnp.array([1], dtype=jnp.int32)
        epsilon_dst = jnp.array([2], dtype=jnp.int32)

        # Only state 5 is accepting
        accepting_sets = jnp.array([[0, 0, 0, 0, 0, 1]], dtype=jnp.bool)

    elif env_name == "delivery":

        return create_delivery_ldba()

        # Automaton for: G(F ag0_target0) && G(F ag1_target1) && G(!obstacles) && G(!oob) && G(!aerial_collide) && G(F ag0_base) && G(F ag1_base)
        spec = "G(F ag0_target0) && G(F ag1_target1) && G(!obstacles) && G(!oob) && G(!aerial_collide) && G(F ag0_base) && G(F ag1_base)"
        predicate_order = ["ag0_target0", "ag1_target1", "obstacles", "oob", "aerial_collide", "ag0_base", "ag1_base"]
        n_states = 1  # Single state automaton with multiple accepting conditions

        BIT_ag0_t0 = 1 << 0
        BIT_ag1_t1 = 1 << 1
        BIT_obs = 1 << 2
        BIT_oob = 1 << 3
        BIT_collide = 1 << 4
        BIT_ag0_base = 1 << 5
        BIT_ag1_base = 1 << 6

        # For Generalized Büchi, we stay in state 0 and track acceptance via the accepting sets
        # The safety conditions (G !obstacles, G !oob, G !aerial_collide) are enforced on all transitions

        transition_specs = [
            # Self-loop on state 0, must satisfy safety: !obstacles & !oob & !aerial_collide
            (0, 0, 0b0, BIT_obs | BIT_oob | BIT_collide),
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

        # Generalized Büchi: 4 accepting sets (one for each G F predicate)
        # State 0 is accepting for each set when the corresponding predicate is true
        # This requires special handling - we track which accepting conditions are satisfied per transition
        # For standard LDBA, we'd expand this, but here's the generalized form:

        # Actually for LDBA with Generalized Büchi, we need transition-based acceptance
        # Let me restructure this properly with explicit state tracking:

        # Alternative: Expand to track "seen since last full cycle" for each G F requirement
        n_states = 16  # 2^4 states to track which of the 4 G F conditions have been seen

        # States encode: [seen_ag0_t0, seen_ag1_t1, seen_ag0_base, seen_ag1_base] as binary
        # State 0: 0000 - seen nothing
        # State 15: 1111 - seen all (then reset to track next cycle)

        transition_specs = []

        for state in range(16):
            seen_ag0_t0 = (state >> 0) & 1
            seen_ag1_t1 = (state >> 1) & 1
            seen_ag0_base = (state >> 2) & 1
            seen_ag1_base = (state >> 3) & 1

            # Generate all combinations of incoming predicates
            for ag0_t0 in [0, 1]:
                for ag1_t1 in [0, 1]:
                    for ag0_base in [0, 1]:
                        for ag1_base in [0, 1]:
                            new_seen_ag0_t0 = seen_ag0_t0 | ag0_t0
                            new_seen_ag1_t1 = seen_ag1_t1 | ag1_t1
                            new_seen_ag0_base = seen_ag0_base | ag0_base
                            new_seen_ag1_base = seen_ag1_base | ag1_base

                            new_state = (
                                (new_seen_ag0_t0 << 0)
                                | (new_seen_ag1_t1 << 1)
                                | (new_seen_ag0_base << 2)
                                | (new_seen_ag1_base << 3)
                            )

                            # If all seen (state 15), reset to reflect new observations only
                            if new_state == 15:
                                new_state = (ag0_t0 << 0) | (ag1_t1 << 1) | (ag0_base << 2) | (ag1_base << 3)

                            # Build mask
                            pos_mask = 0
                            neg_mask = BIT_obs | BIT_oob | BIT_collide  # Safety always required

                            if ag0_t0:
                                pos_mask |= BIT_ag0_t0
                            else:
                                neg_mask |= BIT_ag0_t0
                            if ag1_t1:
                                pos_mask |= BIT_ag1_t1
                            else:
                                neg_mask |= BIT_ag1_t1
                            if ag0_base:
                                pos_mask |= BIT_ag0_base
                            else:
                                neg_mask |= BIT_ag0_base
                            if ag1_base:
                                pos_mask |= BIT_ag1_base
                            else:
                                neg_mask |= BIT_ag1_base

                            transition_specs.append((state, new_state, pos_mask, neg_mask))

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

        # State 15 (all seen) is the accepting state - reaching it means one full cycle completed
        accepting_sets = jnp.array([[1 if i == 15 else 0 for i in range(16)]], dtype=jnp.bool)

    elif env_name == "ablation":
        ## N spec and N agent Ablation Env (Double Integrator)

        return create_ablation_ldba(n_agent=n_agent, n_spec=n_spec, dense=False)

    elif env_name == "ablation_depth":
        ## N spec and N agent Ablation Env (Double Integrator)

        return create_ablation_depth_ldba(n_agent=n_agent, n_spec=n_spec)

    else:
        raise ValueError(f"Unknown environment name: {env_name}")

    return LDBA(transitions, epsilon_src, epsilon_dst, accepting_sets, n_states, predicate_order)


def create_herding_ldba() -> LDBA:
    """
    Create LDBA for:
    "(!herder_unsafe) U (herd_gate_0 && ((!herder_unsafe) U (herd_gate_1 &&
     ((!herder_unsafe) U G(herd_herded && !herder_unsafe)))))"

    States:
        0: Initial - waiting for gate_0
        1: gate_0 reached - waiting for gate_1
        2: gate_0 reached + herded - waiting for gate_1
        3: herded but no gate_0 yet
        4: Both gates reached - waiting for permanent herded
        5: Accepting - all conditions met (gates passed, permanently herded & safe)
       -1: Sink (implicit, for safety violation)

    Predicate order: ["herder_unsafe", "herd_gate_0", "herd_gate_1", "herd_herded"]
    Bits:             0 = unsafe,       1 = gate_0,      2 = gate_1,     3 = herded
    """

    # Define predicate order (determines bit positions in labels)
    predicate_order = ["herder_unsafe", "herd_gate_0", "herd_gate_1", "herd_herded"]

    # Bit positions
    UNSAFE_BIT = 1 << 0  # 0b0001
    GATE_0_BIT = 1 << 1  # 0b0010
    GATE_1_BIT = 1 << 2  # 0b0100
    HERDED_BIT = 1 << 3  # 0b1000

    # === Define all transitions ===
    # Format: (src, dst, pos_mask, neg_mask)
    transition_specs = [
        # From state 0 (initial: waiting for gate_0)
        # [!0 & !1] 0                           # !unsafe & !gate_0 → 0
        (0, 0, 0b0000, UNSAFE_BIT | GATE_0_BIT),
        # [!0 & 1 & !2] 1                       # !unsafe & gate_0 & !gate_1 → 1
        (0, 1, GATE_0_BIT, UNSAFE_BIT | GATE_1_BIT),
        # [!0 & 1 & !2 & 3] 2                   # !unsafe & gate_0 & !gate_1 & herded → 2
        (0, 2, GATE_0_BIT | HERDED_BIT, UNSAFE_BIT | GATE_1_BIT),
        # [!0 & !1 & 3] 3                       # !unsafe & !gate_0 & herded → 3
        (0, 3, HERDED_BIT, UNSAFE_BIT | GATE_0_BIT),
        # [!0 & 1 & 2] 4                        # !unsafe & gate_0 & gate_1 → 4
        (0, 4, GATE_0_BIT | GATE_1_BIT, UNSAFE_BIT),
        # [!0 & 1 & 2 & 3] 5                    # !unsafe & gate_0 & gate_1 & herded → 5
        (0, 5, GATE_0_BIT | GATE_1_BIT | HERDED_BIT, UNSAFE_BIT),
        # From state 1 (gate_0 reached, waiting for gate_1)
        # [!0 & !2] 1                           # !unsafe & !gate_1 → 1
        (1, 1, 0b0000, UNSAFE_BIT | GATE_1_BIT),
        # [!0 & !2 & 3] 2                       # !unsafe & !gate_1 & herded → 2
        (1, 2, HERDED_BIT, UNSAFE_BIT | GATE_1_BIT),
        # [!0 & 2] 4                            # !unsafe & gate_1 → 4
        (1, 4, GATE_1_BIT, UNSAFE_BIT),
        # [!0 & 2 & 3] 5                        # !unsafe & gate_1 & herded → 5
        (1, 5, GATE_1_BIT | HERDED_BIT, UNSAFE_BIT),
        # From state 2 (gate_0 reached + herded, waiting for gate_1)
        # [!0 & !2 & 3] 2                       # !unsafe & !gate_1 & herded → 2
        (2, 2, HERDED_BIT, UNSAFE_BIT | GATE_1_BIT),
        # [!0 & 2 & 3] 5                        # !unsafe & gate_1 & herded → 5
        (2, 5, GATE_1_BIT | HERDED_BIT, UNSAFE_BIT),
        # From state 3 (herded but no gate_0 yet)
        # [!0 & 1 & !2 & 3] 2                   # !unsafe & gate_0 & !gate_1 & herded → 2
        (3, 2, GATE_0_BIT | HERDED_BIT, UNSAFE_BIT | GATE_1_BIT),
        # [!0 & !1 & 3] 3                       # !unsafe & !gate_0 & herded → 3
        (3, 3, HERDED_BIT, UNSAFE_BIT | GATE_0_BIT),
        # [!0 & 1 & 2 & 3] 5                    # !unsafe & gate_0 & gate_1 & herded → 5
        (3, 5, GATE_0_BIT | GATE_1_BIT | HERDED_BIT, UNSAFE_BIT),
        # From state 4 (both gates reached, waiting for herded)
        # [!0] 4                                # !unsafe → 4
        (4, 4, 0b0000, UNSAFE_BIT),
        # [!0 & 3] 5                            # !unsafe & herded → 5
        (4, 5, HERDED_BIT, UNSAFE_BIT),
        # From state 5 (accepting)
        # [!0 & 3] 5 {0}                        # !unsafe & herded → 5 (accepting)
        (5, 5, HERDED_BIT, UNSAFE_BIT),
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

    # === Epsilon Transitions ===
    # None needed - automaton is deterministic (no non-determinism to resolve)
    epsilon_src = jnp.array([], dtype=jnp.int32)
    epsilon_dst = jnp.array([], dtype=jnp.int32)

    # === Accepting Sets ===
    # Shape: (n_accepting_sets, n_states) = (1, 6)
    # State 5 is in accepting set 0
    accepting_sets = jnp.array(
        [
            # State: 0  1  2  3  4  5
            [0, 0, 0, 0, 0, 1],  # Accepting set 0: only state 5 is accepting
        ],
        dtype=jnp.bool,
    )

    # === Create LDBA ===
    ldba = LDBA(
        transitions=transitions,
        epsilon_src=epsilon_src,
        epsilon_dst=epsilon_dst,
        accepting_sets=accepting_sets,
        n_states=6,
        predicate_order=predicate_order,
    )

    return ldba


def create_delivery_ldba() -> LDBA:
    """
    Create LDBA for:
    "G(F ag0_target0) && G(F ag1_target1) && G(!obstacles) && G(!oob) &&
     G(!aerial_collide) && G(F ag0_base) && G(F ag1_base)"

    This is a multi-agent patrolling task where:
    - Agent 0 must repeatedly visit target0 and base
    - Agent 1 must repeatedly visit target1 and base
    - All agents must avoid obstacles, out-of-bounds, and aerial collisions

    States track which of the 4 GF conditions have been satisfied in current cycle:
        0: Initial/reset - need all 4
        1: Need ag1_base (have ag0_target0, ag1_target1, ag0_base)
        2: Need ag1_target1, ag0_base, ag1_base (have ag0_target0)
        3: Need ag0_base, ag1_base (have ag0_target0, ag1_target1)
        4: Need ag0_target0 OR cycle complete
       -1: Sink (implicit, safety violation)

    Predicate order: ["ag0_target0", "ag1_target1", "obstacles", "oob",
                      "aerial_collide", "ag0_base", "ag1_base"]
    """

    # Define predicate order (determines bit positions in labels)
    predicate_order = [
        "ag0_target0",  # bit 0
        "ag1_target1",  # bit 1
        "obstacles",  # bit 2
        "oob",  # bit 3
        "aerial_collide",  # bit 4
        "ag0_base",  # bit 5
        "ag1_base",  # bit 6
    ]

    # Bit positions
    AG0_T0 = 1 << 0  # 0b0000001
    AG1_T1 = 1 << 1  # 0b0000010
    OBST = 1 << 2  # 0b0000100
    OOB = 1 << 3  # 0b0001000
    AERIAL = 1 << 4  # 0b0010000
    AG0_BASE = 1 << 5  # 0b0100000
    AG1_BASE = 1 << 6  # 0b1000000

    # Safety bits (always in neg_mask)
    SAFETY = OBST | OOB | AERIAL  # 0b0011100

    # === Define all transitions ===
    # Format: (src, dst, pos_mask, neg_mask, accepting)
    transition_specs = [
        # ============ State 0: Initial/reset - need all 4 ============
        # [!2 & !3 & !4] 0
        (0, 0, 0, SAFETY, False),
        # [0 & 1 & !2 & !3 & !4 & 5 & !6] 1
        (0, 1, AG0_T0 | AG1_T1 | AG0_BASE, SAFETY | AG1_BASE, False),
        # [0 & !1 & !2 & !3 & !4] 2
        (0, 2, AG0_T0, SAFETY | AG1_T1, False),
        # [0 & 1 & !2 & !3 & !4 & !5] 3
        (0, 3, AG0_T0 | AG1_T1, SAFETY | AG0_BASE, False),
        # [0 & 1 & !2 & !3 & !4 & 5 & 6] 4  (first part of OR)
        (0, 4, AG0_T0 | AG1_T1 | AG0_BASE | AG1_BASE, SAFETY, False),
        # [!0 & !2 & !3 & !4] 4  (second part of OR)
        (0, 4, 0, SAFETY | AG0_T0, False),
        # ============ State 1: Need ag1_base ============
        # [!2 & !3 & !4 & !6] 1
        (1, 1, 0, SAFETY | AG1_BASE, False),
        # [0 & !1 & !2 & !3 & !4 & 6] 2 {0}
        (1, 2, AG0_T0 | AG1_BASE, SAFETY | AG1_T1, True),
        # [0 & 1 & !2 & !3 & !4 & !5 & 6] 3 {0}
        (1, 3, AG0_T0 | AG1_T1 | AG1_BASE, SAFETY | AG0_BASE, True),
        # [0 & 1 & !2 & !3 & !4 & 5 & 6] 4 {0}  (first part of OR)
        (1, 4, AG0_T0 | AG1_T1 | AG0_BASE | AG1_BASE, SAFETY, True),
        # [!0 & !2 & !3 & !4 & 6] 4 {0}  (second part of OR)
        (1, 4, AG1_BASE, SAFETY | AG0_T0, True),
        # ============ State 2: Need ag1_target1, ag0_base, ag1_base ============
        # [!1 & !2 & !3 & !4] 2
        (2, 2, 0, SAFETY | AG1_T1, False),
        # [1 & !2 & !3 & !4 & 5 & !6] 1
        (2, 1, AG1_T1 | AG0_BASE, SAFETY | AG1_BASE, False),
        # [1 & !2 & !3 & !4 & !5] 3
        (2, 3, AG1_T1, SAFETY | AG0_BASE, False),
        # [1 & !2 & !3 & !4 & 5 & 6] 4 {0}
        (2, 4, AG1_T1 | AG0_BASE | AG1_BASE, SAFETY, True),
        # ============ State 3: Need ag0_base, ag1_base ============
        # [!2 & !3 & !4 & 5 & !6] 1
        (3, 1, AG0_BASE, SAFETY | AG1_BASE, False),
        # [0 & !1 & !2 & !3 & !4 & 5 & 6] 2 {0}
        (3, 2, AG0_T0 | AG0_BASE | AG1_BASE, SAFETY | AG1_T1, True),
        # [!2 & !3 & !4 & !5] 3
        (3, 3, 0, SAFETY | AG0_BASE, False),
        # [0 & 1 & !2 & !3 & !4 & 5 & 6] 4 {0}  (first part of OR)
        (3, 4, AG0_T0 | AG1_T1 | AG0_BASE | AG1_BASE, SAFETY, True),
        # [!0 & !2 & !3 & !4 & 5 & 6] 4 {0}  (second part of OR)
        (3, 4, AG0_BASE | AG1_BASE, SAFETY | AG0_T0, True),
        # ============ State 4: Need ag0_target0 OR cycle complete ============
        # [0 & 1 & !2 & !3 & !4 & 5 & !6] 1
        (4, 1, AG0_T0 | AG1_T1 | AG0_BASE, SAFETY | AG1_BASE, False),
        # [0 & !1 & !2 & !3 & !4] 2
        (4, 2, AG0_T0, SAFETY | AG1_T1, False),
        # [0 & 1 & !2 & !3 & !4 & !5] 3
        (4, 3, AG0_T0 | AG1_T1, SAFETY | AG0_BASE, False),
        # [!0 & !2 & !3 & !4] 4
        (4, 4, 0, SAFETY | AG0_T0, False),
        # [0 & 1 & !2 & !3 & !4 & 5 & 6] 4 {0}
        (4, 4, AG0_T0 | AG1_T1 | AG0_BASE | AG1_BASE, SAFETY, True),
    ]

    # Build arrays from specs
    src_list = []
    dst_list = []
    pos_mask_list = []
    neg_mask_list = []

    for src, dst, pos_mask, neg_mask, _ in transition_specs:
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

    # === Epsilon Transitions ===
    # None needed - automaton handles GF via accepting transitions
    epsilon_src = jnp.array([], dtype=jnp.int32)
    epsilon_dst = jnp.array([], dtype=jnp.int32)

    # === Accepting Sets ===
    # Shape: (n_accepting_sets, n_states) = (1, 5)
    # States with accepting transitions: 1, 2, 3, 4
    # But accepting is on TRANSITIONS, not states. For the frontier tracking,
    # we mark states that HAVE accepting outgoing transitions.
    # Looking at the HOA: states 1, 2, 3, 4 all have at least one {0} transition
    accepting_sets = jnp.array(
        [
            # State: 0  1  2  3  4
            [0, 1, 1, 1, 1],  # States with accepting transitions
        ],
        dtype=jnp.bool,
    )

    # === Create LDBA ===
    ldba = LDBA(
        transitions=transitions,
        epsilon_src=epsilon_src,
        epsilon_dst=epsilon_dst,
        accepting_sets=accepting_sets,
        n_states=5,
        predicate_order=predicate_order,
    )

    return ldba

import jax.numpy as jnp
from typing import List


def create_ablation_ldba(n_spec: int, n_agent: int, dense: bool = False) -> LDBA:
    """
    Create LDBA for ablation environment:
    "G(!oob) && G(!obstacles) [&& G(!collide)] && F target0 && F target1 && ... && F target{n_spec-1}"
    
    Multi-agent navigation with multiple sequential targets.
    - Must avoid out-of-bounds and obstacles (always)
    - Must avoid collisions if n_agent > 1
    - Must eventually visit each target (in any order)
    
    States track which targets have been visited:
        State encoding: each bit represents whether target_i has been visited
        State 0: No targets visited yet
        State 2^n_spec - 1: All targets visited (accepting)
        -1: Sink (implicit, safety violation)
    
    Args:
        n_spec: Number of targets to visit
        n_agent: Number of agents (determines if collision avoidance is needed)
        dense: If True, uses dense target predicates (target{i}_dense)
    
    Predicate order: ["oob", "obstacles", "collide" (if n_agent > 1), 
                      "target0", "target1", ..., "target{n_spec-1}"]
                      (or "target0_dense", etc. if dense=True)
    """
    
    # Build predicate order
    predicate_order = ["oob", "obstacles"]
    
    # Add collision predicate if multiple agents
    if n_agent > 1:
        predicate_order.append("collide")
    
    # Add target predicates
    suffix = "_dense" if dense else ""
    for i in range(n_spec):
        predicate_order.append(f"target{i}{suffix}")
    
    # Calculate bit positions
    OOB_BIT = 1 << 0
    OBSTACLES_BIT = 1 << 1
    
    if n_agent > 1:
        COLLIDE_BIT = 1 << 2
        TARGET_START_BIT = 3
        SAFETY = OOB_BIT | OBSTACLES_BIT | COLLIDE_BIT
    else:
        COLLIDE_BIT = 0
        TARGET_START_BIT = 2
        SAFETY = OOB_BIT | OBSTACLES_BIT
    
    # Target bits
    TARGET_BITS = [(1 << (TARGET_START_BIT + i)) for i in range(n_spec)]
    
    # Number of states: 2^n_spec (one for each subset of targets visited)
    n_states = 2 ** n_spec
    
    # === Define all transitions ===
    transition_specs = []
    
    for state in range(n_states):
        # For each state, generate transitions based on which targets we can visit next
        # State encoding: bit i set means target i has been visited
        
        # Self-loop: stay in same state if no new targets visited and safe
        # neg_mask = SAFETY | (all unvisited target bits)
        unvisited_targets = 0
        for i in range(n_spec):
            if not (state & (1 << i)):  # target i not yet visited
                unvisited_targets |= TARGET_BITS[i]
        
        transition_specs.append((state, state, 0, SAFETY | unvisited_targets))
        
        # Transitions to states with one additional target visited
        for i in range(n_spec):
            if not (state & (1 << i)):  # target i not yet visited in current state
                next_state = state | (1 << i)
                
                # Must have target_i set, must be safe
                pos_mask = TARGET_BITS[i]
                # neg_mask includes safety violations
                neg_mask = SAFETY
                
                transition_specs.append((state, next_state, pos_mask, neg_mask))
    
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
    
    # === Epsilon Transitions ===
    epsilon_src = jnp.array([], dtype=jnp.int32)
    epsilon_dst = jnp.array([], dtype=jnp.int32)
    
    # === Accepting Sets ===
    # Only the final state (all targets visited) is accepting
    accepting_sets = jnp.zeros((1, n_states), dtype=jnp.bool_)
    accepting_sets = accepting_sets.at[0, n_states - 1].set(True)  # State 2^n_spec - 1
    
    # === Create LDBA ===
    ldba = LDBA(
        transitions=transitions,
        epsilon_src=epsilon_src,
        epsilon_dst=epsilon_dst,
        accepting_sets=accepting_sets,
        n_states=n_states,
        predicate_order=predicate_order,
    )
    
    return ldba


def create_ablation_depth_ldba(n_spec: int, n_agent: int) -> LDBA:
    """
    Create LDBA for ablation_depth environment:
    "G(!oob) && G(!obstacles) [&& G(!collide)] && F(target0 && F(target1 && F(...)))"
    
    Multi-agent navigation with nested temporal goals - targets must be visited in order.
    - Must avoid out-of-bounds and obstacles (always)
    - Must avoid collisions if n_agent > 1
    - Must visit targets sequentially: first target0, then target1, then target2, etc.
    
    States track progress through the sequence:
        0: Initial - need to visit target0 first
        1: target0 visited - need to visit target1 next
        2: target0, target1 visited - need to visit target2 next
        ...
        n_spec: All targets visited in order (accepting)
        -1: Sink (implicit, safety violation)
    
    Args:
        n_spec: Number of targets to visit in sequence
        n_agent: Number of agents (determines if collision avoidance is needed)
    
    Predicate order: ["oob", "obstacles", "collide" (if n_agent > 1),
                      "target0", "target1", ..., "target{n_spec-1}"]
    """
    
    # Build predicate order
    predicate_order = ["oob", "obstacles"]
    
    # Add collision predicate if multiple agents
    if n_agent > 1:
        predicate_order.append("collide")
    
    # Add target predicates
    for i in range(n_spec):
        predicate_order.append(f"target{i}")
    
    # Calculate bit positions
    OOB_BIT = 1 << 0
    OBSTACLES_BIT = 1 << 1
    
    if n_agent > 1:
        COLLIDE_BIT = 1 << 2
        TARGET_START_BIT = 3
        SAFETY = OOB_BIT | OBSTACLES_BIT | COLLIDE_BIT
    else:
        COLLIDE_BIT = 0
        TARGET_START_BIT = 2
        SAFETY = OOB_BIT | OBSTACLES_BIT
    
    # Target bits
    TARGET_BITS = [(1 << (TARGET_START_BIT + i)) for i in range(n_spec)]
    
    # Number of states: n_spec + 1
    # States 0 to n_spec-1: waiting for targets 0 to n_spec-1
    # State n_spec: accepting (all targets visited in order)
    n_states = n_spec + 1
    
    # === Define all transitions ===
    transition_specs = []
    
    for state in range(n_spec):
        # State i: waiting for target_i
        
        # Self-loop: stay in state i if target_i not reached and safe
        # Must NOT have target_i, must be safe
        transition_specs.append((state, state, 0, SAFETY | TARGET_BITS[state]))
        
        # Transition to next state: visit target_i
        # Must have target_i, must be safe
        next_state = state + 1
        transition_specs.append((state, next_state, TARGET_BITS[state], SAFETY))
    
    # Final accepting state: self-loop while safe
    # Once all targets visited, just need to remain safe
    transition_specs.append((n_spec, n_spec, 0, SAFETY))
    
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
    
    # === Epsilon Transitions ===
    epsilon_src = jnp.array([], dtype=jnp.int32)
    epsilon_dst = jnp.array([], dtype=jnp.int32)
    
    # === Accepting Sets ===
    # Only the final state (all targets visited in sequence) is accepting
    accepting_sets = jnp.zeros((1, n_states), dtype=jnp.bool_)
    accepting_sets = accepting_sets.at[0, n_spec].set(True)  # Final state
    
    # === Create LDBA ===
    ldba = LDBA(
        transitions=transitions,
        epsilon_src=epsilon_src,
        epsilon_dst=epsilon_dst,
        accepting_sets=accepting_sets,
        n_states=n_states,
        predicate_order=predicate_order,
    )
    
    return ldba

def create_gridworld_map1_ldba() -> LDBA:
    """
    Create LDBA for:
    "F(a) & F(b) & G(!w)"
    
    (Eventually a, eventually b, and always not w)

    States:
        0: Initial - waiting for both a and b
        1: b seen - waiting for a
        2: a seen - waiting for b
        3: Accepting - both a and b seen, stay safe
       -1: Sink (implicit, for w violation)

    Predicate order: ["a", "b", "w"]
    Bits:             0 = a,  1 = b,  2 = w
    """

    # Define predicate order (determines bit positions in labels)
    predicate_order = ["A", "B", "w"]

    # Bit positions
    A_BIT = 1 << 0  # 0b001
    B_BIT = 1 << 1  # 0b010
    W_BIT = 1 << 2  # 0b100

    # === Define all transitions ===
    # Format: (src, dst, pos_mask, neg_mask)
    transition_specs = [
        # From state 0 (initial: waiting for both a and b)
        # [!0 & 1 & !2] 1                       # !a & b & !w → 1
        (0, 1, B_BIT, A_BIT | W_BIT),
        # [!0 & !1 & !2] 0                      # !a & !b & !w → 0
        (0, 0, 0b000, A_BIT | B_BIT | W_BIT),
        # [0 & !1 & !2] 2                       # a & !b & !w → 2
        (0, 2, A_BIT, B_BIT | W_BIT),
        # [0 & 1 & !2] 3                        # a & b & !w → 3
        (0, 3, A_BIT | B_BIT, W_BIT),

        # From state 1 (b seen, waiting for a)
        # [!0 & !2] 1                           # !a & !w → 1
        (1, 1, 0b000, A_BIT | W_BIT),
        # [0 & !2] 3                            # a & !w → 3
        (1, 3, A_BIT, W_BIT),

        # From state 2 (a seen, waiting for b)
        # [!1 & !2] 2                           # !b & !w → 2
        (2, 2, 0b000, B_BIT | W_BIT),
        # [1 & !2] 3                            # b & !w → 3
        (2, 3, B_BIT, W_BIT),

        # From state 3 (accepting: both seen)
        # [!2] 3 {0}                            # !w → 3 (accepting)
        (3, 3, 0b000, W_BIT),
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

    # === Epsilon Transitions ===
    # None needed - automaton is deterministic
    epsilon_src = jnp.array([], dtype=jnp.int32)
    epsilon_dst = jnp.array([], dtype=jnp.int32)

    # === Accepting Sets ===
    # Shape: (n_accepting_sets, n_states) = (1, 4)
    # State 3 is in accepting set 0
    accepting_sets = jnp.array(
        [
            # State: 0  1  2  3
            [0, 0, 0, 1],  # Accepting set 0: only state 3 is accepting
        ],
        dtype=jnp.bool,
    )

    # === Create LDBA ===
    ldba = LDBA(
        transitions=transitions,
        epsilon_src=epsilon_src,
        epsilon_dst=epsilon_dst,
        accepting_sets=accepting_sets,
        n_states=4,
        predicate_order=predicate_order,
    )

    return ldba

def create_gridworld_map5_ldba() -> LDBA:
    """
    Create LDBA for:
    "F(a) & F(b) & G(!w) & ((!d) U k)"
    
    (Eventually a, eventually b, always not w, and not d until k)

    States:
        0: Initial - waiting for a, b, and k
        1: k seen - waiting for a and b
        2: a seen - waiting for b and k
        3: b seen - waiting for a and k
        4: b and k seen - waiting for a
        5: a and b seen - waiting for k
        6: a and k seen - waiting for b
        7: Accepting - a, b, and k all seen, stay safe
       -1: Sink (implicit, for w or d violation)

    Predicate order: ["a", "b", "d", "k", "w"]
    Bits:             0 = a,  1 = b,  2 = d,  3 = k,  4 = w
    """

    # Define predicate order (determines bit positions in labels)
    predicate_order = ["A", "B", "D", "K", "w"]

    # Bit positions
    A_BIT = 1 << 0  # 0b00001
    B_BIT = 1 << 1  # 0b00010
    D_BIT = 1 << 2  # 0b00100
    K_BIT = 1 << 3  # 0b01000
    W_BIT = 1 << 4  # 0b10000

    # === Define all transitions ===
    # Format: (src, dst, pos_mask, neg_mask)
    transition_specs = [
        # From state 0 (initial: waiting for a, b, and k)
        # [!0 & !1 & 3 & !4] 1                  # !a & !b & k & !w → 1
        (0, 1, K_BIT, A_BIT | B_BIT | W_BIT),
        # [0 & !1 & !2 & !3 & !4] 2             # a & !b & !d & !k & !w → 2
        (0, 2, A_BIT, B_BIT | D_BIT | K_BIT | W_BIT),
        # [!0 & 1 & !2 & !3 & !4] 3             # !a & b & !d & !k & !w → 3
        (0, 3, B_BIT, A_BIT | D_BIT | K_BIT | W_BIT),
        # [!0 & 1 & 3 & !4] 4                   # !a & b & k & !w → 4
        (0, 4, B_BIT | K_BIT, A_BIT | W_BIT),
        # [!0 & !1 & !2 & !3 & !4] 0            # !a & !b & !d & !k & !w → 0
        (0, 0, 0b00000, A_BIT | B_BIT | D_BIT | K_BIT | W_BIT),
        # [0 & 1 & !2 & !3 & !4] 5              # a & b & !d & !k & !w → 5
        (0, 5, A_BIT | B_BIT, D_BIT | K_BIT | W_BIT),
        # [0 & !1 & 3 & !4] 6                   # a & !b & k & !w → 6
        (0, 6, A_BIT | K_BIT, B_BIT | W_BIT),
        # [0 & 1 & 3 & !4] 7                    # a & b & k & !w → 7
        (0, 7, A_BIT | B_BIT | K_BIT, W_BIT),

        # From state 1 (k seen: waiting for a and b)
        # [!0 & !1 & !4] 1                      # !a & !b & !w → 1
        (1, 1, 0b00000, A_BIT | B_BIT | W_BIT),
        # [!0 & 1 & !4] 4                       # !a & b & !w → 4
        (1, 4, B_BIT, A_BIT | W_BIT),
        # [0 & !1 & !4] 6                       # a & !b & !w → 6
        (1, 6, A_BIT, B_BIT | W_BIT),
        # [0 & 1 & !4] 7                        # a & b & !w → 7
        (1, 7, A_BIT | B_BIT, W_BIT),

        # From state 2 (a seen: waiting for b and k)
        # [!1 & !2 & !3 & !4] 2                 # !b & !d & !k & !w → 2
        (2, 2, 0b00000, B_BIT | D_BIT | K_BIT | W_BIT),
        # [1 & !2 & !3 & !4] 5                  # b & !d & !k & !w → 5
        (2, 5, B_BIT, D_BIT | K_BIT | W_BIT),
        # [!1 & 3 & !4] 6                       # !b & k & !w → 6
        (2, 6, K_BIT, B_BIT | W_BIT),
        # [1 & 3 & !4] 7                        # b & k & !w → 7
        (2, 7, B_BIT | K_BIT, W_BIT),

        # From state 3 (b seen: waiting for a and k)
        # [!0 & !2 & !3 & !4] 3                 # !a & !d & !k & !w → 3
        (3, 3, 0b00000, A_BIT | D_BIT | K_BIT | W_BIT),
        # [!0 & 3 & !4] 4                       # !a & k & !w → 4
        (3, 4, K_BIT, A_BIT | W_BIT),
        # [0 & !2 & !3 & !4] 5                  # a & !d & !k & !w → 5
        (3, 5, A_BIT, D_BIT | K_BIT | W_BIT),
        # [0 & 3 & !4] 7                        # a & k & !w → 7
        (3, 7, A_BIT | K_BIT, W_BIT),

        # From state 4 (b and k seen: waiting for a)
        # [!0 & !4] 4                           # !a & !w → 4
        (4, 4, 0b00000, A_BIT | W_BIT),
        # [0 & !4] 7                            # a & !w → 7
        (4, 7, A_BIT, W_BIT),

        # From state 5 (a and b seen: waiting for k)
        # [!2 & !3 & !4] 5                      # !d & !k & !w → 5
        (5, 5, 0b00000, D_BIT | K_BIT | W_BIT),
        # [3 & !4] 7                            # k & !w → 7
        (5, 7, K_BIT, W_BIT),

        # From state 6 (a and k seen: waiting for b)
        # [!1 & !4] 6                           # !b & !w → 6
        (6, 6, 0b00000, B_BIT | W_BIT),
        # [1 & !4] 7                            # b & !w → 7
        (6, 7, B_BIT, W_BIT),

        # From state 7 (accepting: all conditions met)
        # [!4] 7 {0}                            # !w → 7 (accepting)
        (7, 7, 0b00000, W_BIT),
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

    # === Epsilon Transitions ===
    # None needed - automaton is deterministic
    epsilon_src = jnp.array([], dtype=jnp.int32)
    epsilon_dst = jnp.array([], dtype=jnp.int32)

    # === Accepting Sets ===
    # Shape: (n_accepting_sets, n_states) = (1, 8)
    # State 7 is in accepting set 0
    accepting_sets = jnp.array(
        [
            # State: 0  1  2  3  4  5  6  7
            [0, 0, 0, 0, 0, 0, 0, 1],  # Accepting set 0: only state 7 is accepting
        ],
        dtype=jnp.bool,
    )

    # === Create LDBA ===
    ldba = LDBA(
        transitions=transitions,
        epsilon_src=epsilon_src,
        epsilon_dst=epsilon_dst,
        accepting_sets=accepting_sets,
        n_states=8,
        predicate_order=predicate_order,
    )

    return ldba

def create_gridworld_map6_ldba() -> LDBA:
    """
    Create LDBA for:
    "F(c & F(G((q U (a & q)) & (q U (b & q)))))"
    
    (Eventually c, then eventually globally: while q holds, eventually reach a&q and b&q)

    States:
        0: Initial - waiting for c
        1: c seen - waiting to enter the G(...) phase, can also try accepting conditions
        2: In G phase - a seen (with q), waiting for b (with q)
        3: In G phase - checking/accepting state (b seen or waiting for a)
       -1: Sink (implicit, for q violation in accepting phase)

    Predicate order: ["c", "q", "a", "b"]
    Bits:             0 = c,  1 = q,  2 = a,  3 = b
    """

    # Define predicate order (determines bit positions in labels)
    predicate_order = ["C", "q", "A", "B"]

    # Bit positions
    C_BIT = 1 << 0  # 0b0001
    Q_BIT = 1 << 1  # 0b0010
    A_BIT = 1 << 2  # 0b0100
    B_BIT = 1 << 3  # 0b1000

    # === Define all transitions ===
    # Format: (src, dst, pos_mask, neg_mask)
    transition_specs = [
        # From state 0 (initial: waiting for c)
        # [!0] 0                                # !c → 0
        (0, 0, 0b0000, C_BIT),
        # [0] 1                                 # c → 1
        (0, 1, C_BIT, 0b0000),

        # From state 1 (c seen: waiting to enter G phase)
        # [t] 1                                 # true → 1 (can always stay)
        (1, 1, 0b0000, 0b0000),
        # [1 & 2 & !3] 2                        # q & a & !b → 2
        (1, 2, Q_BIT | A_BIT, B_BIT),
        # [1 & (2 & 3 | !2)] 3                  # q & (a & b | !a) → 3
        # This is: q & (!a | b) which splits into two transitions:
        # q & !a → 3
        (1, 3, Q_BIT, A_BIT),
        # q & a & b → 3
        (1, 3, Q_BIT | A_BIT | B_BIT, 0b0000),

        # From state 2 (a seen with q, waiting for b with q)
        # [1 & 3] 3 {0}                         # q & b → 3 (accepting)
        (2, 3, Q_BIT | B_BIT, 0b0000),
        # [1 & !3] 2                            # q & !b → 2
        (2, 2, Q_BIT, B_BIT),

        # From state 3 (checking/accepting state)
        # [1 & 2 & !3] 2                        # q & a & !b → 2
        (3, 2, Q_BIT | A_BIT, B_BIT),
        # [1 & 2 & 3] 3 {0}                     # q & a & b → 3 (accepting)
        (3, 3, Q_BIT | A_BIT | B_BIT, 0b0000),
        # [1 & !2] 3                            # q & !a → 3
        (3, 3, Q_BIT, A_BIT),
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

    # === Epsilon Transitions ===
    # None needed - automaton is deterministic
    epsilon_src = jnp.array([], dtype=jnp.int32)
    epsilon_dst = jnp.array([], dtype=jnp.int32)

    # === Accepting Sets ===
    # Shape: (n_accepting_sets, n_states) = (1, 4)
    # States 2→3 and 3→3 transitions with {0} are accepting
    # But for state-based acceptance, we mark states that have accepting outgoing transitions
    # Looking at the HOA format, accepting transitions go TO state 3
    # In Büchi acceptance, we need to visit accepting transitions infinitely often
    # States 2 and 3 have accepting transitions (marked with {0})
    accepting_sets = jnp.array(
        [
            # State: 0  1  2  3
            [0, 0, 1, 1],  # Accepting set 0: states 2 and 3 have accepting transitions
        ],
        dtype=jnp.bool,
    )

    # === Create LDBA ===
    ldba = LDBA(
        transitions=transitions,
        epsilon_src=epsilon_src,
        epsilon_dst=epsilon_dst,
        accepting_sets=accepting_sets,
        n_states=4,
        predicate_order=predicate_order,
    )

    return ldba

def create_gridworld_map7_ldba() -> LDBA:
    """
    Create LDBA for:
    "((!q U c) & F(c & F(G((q U (a & q)) & (q U (b & q))))))"
    
    (Not q until c, and eventually c, then eventually globally: 
     while q holds, repeatedly reach a&q and b&q)

    States:
        0: Initial - waiting for c (must have !q until then)
        1: c seen - waiting to enter the G(...) phase
        2: In G phase - checking/accepting state (b seen or waiting for a)
        3: In G phase - a seen (with q), waiting for b (with q)
       -1: Sink (implicit, for q violation before c, or q violation in accepting phase)

    Predicate order: ["q", "c", "a", "b"]
    Bits:             0 = q,  1 = c,  2 = a,  3 = b
    """

    # Define predicate order (determines bit positions in labels)
    predicate_order = ["q", "C", "A", "B"]

    # Bit positions
    Q_BIT = 1 << 0  # 0b0001
    C_BIT = 1 << 1  # 0b0010
    A_BIT = 1 << 2  # 0b0100
    B_BIT = 1 << 3  # 0b1000

    # === Define all transitions ===
    # Format: (src, dst, pos_mask, neg_mask)
    transition_specs = [
        # From state 0 (initial: waiting for c, must have !q)
        # [!0 & !1] 0                           # !q & !c → 0
        (0, 0, 0b0000, Q_BIT | C_BIT),
        # [1] 1                                 # c → 1
        (0, 1, C_BIT, 0b0000),

        # From state 1 (c seen: waiting to enter G phase)
        # [0 & (2 & 3 | !2)] 2                  # q & (a & b | !a) → 2
        # This splits into two transitions:
        # q & !a → 2
        (1, 2, Q_BIT, A_BIT),
        # q & a & b → 2
        (1, 2, Q_BIT | A_BIT | B_BIT, 0b0000),
        # [t] 1                                 # true → 1 (can always stay)
        (1, 1, 0b0000, 0b0000),
        # [0 & 2 & !3] 3                        # q & a & !b → 3
        (1, 3, Q_BIT | A_BIT, B_BIT),

        # From state 2 (checking/accepting state)
        # [0 & 2 & 3] 2 {0}                     # q & a & b → 2 (accepting)
        (2, 2, Q_BIT | A_BIT | B_BIT, 0b0000),
        # [0 & !2] 2                            # q & !a → 2
        (2, 2, Q_BIT, A_BIT),
        # [0 & 2 & !3] 3                        # q & a & !b → 3
        (2, 3, Q_BIT | A_BIT, B_BIT),

        # From state 3 (a seen with q, waiting for b with q)
        # [0 & 3] 2 {0}                         # q & b → 2 (accepting)
        (3, 2, Q_BIT | B_BIT, 0b0000),
        # [0 & !3] 3                            # q & !b → 3
        (3, 3, Q_BIT, B_BIT),
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

    # === Epsilon Transitions ===
    # None needed - automaton is deterministic
    epsilon_src = jnp.array([], dtype=jnp.int32)
    epsilon_dst = jnp.array([], dtype=jnp.int32)

    # === Accepting Sets ===
    # Shape: (n_accepting_sets, n_states) = (1, 4)
    # Accepting transitions (marked {0}) occur on:
    #   2→2 with q & a & b
    #   3→2 with q & b
    # States 2 and 3 have accepting outgoing transitions
    accepting_sets = jnp.array(
        [
            # State: 0  1  2  3
            [0, 0, 1, 1],  # Accepting set 0: states 2 and 3 have accepting transitions
        ],
        dtype=jnp.bool,
    )

    # === Create LDBA ===
    ldba = LDBA(
        transitions=transitions,
        epsilon_src=epsilon_src,
        epsilon_dst=epsilon_dst,
        accepting_sets=accepting_sets,
        n_states=4,
        predicate_order=predicate_order,
    )

    return ldba