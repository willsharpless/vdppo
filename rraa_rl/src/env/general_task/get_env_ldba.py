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


def get_env_ldba(env_name: str, n_spec: int = 1) -> tuple[Env, list, list]:

    if env_name == "herdos":

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

    elif env_name == "gridworld_map1":

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

        spec = "F( C && F G ( (q U (A && q )) && (q U (B && q )) ) )"
        raise NotImplementedError(f"""{env_name} LDBA is not implemented""")

    elif env_name == "manip_scene":

        raise NotImplementedError(f"""{env_name} LDBA is not implemented""")

    elif env_name == "gridworld_map7":
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

        # Predicates: oob, obstacles, target0, ..., target{n_spec-1}
        predicate_order = ["oob", "obstacles"] + [f"target{i}_dense" if dense else f"target{i}" for i in range(n_spec)]
        n_states = 2**n_spec

        BIT_oob = 1 << 0
        BIT_obs = 1 << 1
        BIT_targets = [1 << (2 + i) for i in range(n_spec)]

        transition_specs = []
        for state in range(n_states):
            # For each possible combination of targets reached
            for obs in range(2):  # obstacles: 0 or 1
                for oob in range(2):  # oob: 0 or 1
                    for target_bits in range(2**n_spec):
                        # Build predicate mask for this input
                        pos_mask = 0
                        neg_mask = 0
                        if oob:
                            pos_mask |= BIT_oob
                        else:
                            neg_mask |= BIT_oob
                        if obs:
                            pos_mask |= BIT_obs
                        else:
                            neg_mask |= BIT_obs
                        for i in range(n_spec):
                            if (target_bits >> i) & 1:
                                pos_mask |= BIT_targets[i]
                            else:
                                neg_mask |= BIT_targets[i]
                        # Safety: only allow transitions if oob==0 and obs==0
                        if oob or obs:
                            continue
                        # Compute next state: mark any new targets as seen
                        new_state = state
                        for i in range(n_spec):
                            if (target_bits >> i) & 1:
                                new_state |= 1 << i
                        transition_specs.append((state, new_state, pos_mask, neg_mask))

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
        accepting_sets = jnp.zeros((1, n_states), dtype=jnp.bool)
        accepting_sets[0, n_states - 1] = True  # Only all targets seen is accepting

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
        dtype=jnp.int32,
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
        dtype=jnp.int32,
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
