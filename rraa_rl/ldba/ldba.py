import functools as ft
import re
from typing import NamedTuple

import ipdb
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from attrs import define
from loguru import logger


class Guard(NamedTuple):
    """
    The guard is satisfied by label σ iff:
    (σ & pos_mask) == pos_mask  AND  (σ & neg_mask) == 0
    """

    # Both scalar ints.
    pos_mask: jnp.ndarray
    neg_mask: jnp.ndarray

    def evaluate(self, label: jnp.ndarray) -> jnp.ndarray:
        assert label.shape == ()
        pos_satisfied = (label & self.pos_mask) == self.pos_mask
        neg_satisfied = (label & self.neg_mask) == 0
        return pos_satisfied & neg_satisfied


class Transition(NamedTuple):
    src: int
    dst: int
    guard: Guard

    def is_eligible(self, state: jnp.ndarray, label: jnp.ndarray):
        return (state == self.src) & self.guard.evaluate(label)


@define
class LDBA:
    transitions: Transition

    # (n_epsilon_transitions,)
    epsilon_src: jnp.ndarray
    # (n_epsilon_transitions,)
    epsilon_dst: jnp.ndarray

    # (n_accepting_sets, n_states). 1 if state is in accepting set.
    accepting_sets: jnp.ndarray

    n_states: int
    predicate_order: list[str]

    @property
    def sink_state(self) -> int:
        # Convention used in LCRL
        return -1

    @property
    def n_accepting_sets(self) -> int:
        return self.accepting_sets.shape[0]

    @property
    def n_epsilon_transitions(self) -> int:
        return self.epsilon_src.shape[0]

    def step_epsilon(self, state: jnp.ndarray, epsilon_action: jnp.ndarray):
        # epsilon_action is an int. Should be in [1, n_epsilon_transitions], or 0 for no-op.
        epsilon_idx = epsilon_action - 1
        epsilon_idx_safe = jnp.where(epsilon_action == 0, 0, epsilon_idx)

        is_sink_state = state == self.sink_state
        state_safe = jnp.where(is_sink_state, 0, state)
        epsilon_taken = self.epsilon_src[epsilon_idx_safe] == state_safe
        next_state = jnp.where(epsilon_taken, self.epsilon_dst[epsilon_idx_safe], state)
        # If in sink, stay in sink.
        next_state = jnp.where(is_sink_state, self.sink_state, next_state)
        return next_state, epsilon_taken

    def step_noepsilon(self, state: jnp.ndarray, label: jnp.ndarray):
        is_sink_state = state == self.sink_state
        state_safe = jnp.where(is_sink_state, 0, state)

        m_eligible = jax.vmap(ft.partial(Transition.is_eligible, state=state_safe, label=label))(self.transitions)
        has_eligible = jnp.any(m_eligible)
        # There should only be at most one eligible, since we extract the epsilon transitions out.
        idx = jnp.argmax(m_eligible)
        next_state = jnp.where(has_eligible, self.transitions.dst[idx], state)

        # If in sink, stay in sink.
        next_state = jnp.where(is_sink_state, self.sink_state, next_state)
        return next_state

    def step(self, state: jnp.ndarray, label: jnp.ndarray, epsilon_action: jnp.ndarray):
        next_state_noepsilon = self.step_noepsilon(state, label)

        if self.n_epsilon_transitions == 0:
            return next_state_noepsilon, jnp.array(False)

        next_state_epsilon, epsilon_taken = self.step_epsilon(state, epsilon_action)

        # If we tried to take epsilon (epsilon != 0) but it was not eligible, then we fall back to no-epsilon step.
        try_take_epsilon = epsilon_action != 0
        has_taken_epsilon = try_take_epsilon & epsilon_taken
        next_state = jnp.where(has_taken_epsilon, next_state_epsilon, next_state_noepsilon)
        return next_state, has_taken_epsilon

    def predicates_to_label(self, predicates: dict[str, jnp.ndarray]) -> jnp.ndarray:
        label = 0
        for i, pred_name in enumerate(self.predicate_order):
            pred_value = predicates.get(pred_name, jnp.array(False))
            label |= jnp.where(pred_value, 1 << i, 0)
        return label

    def update_frontier(self, frontier_mask: jnp.ndarray, state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # frontier_mask: (n_accepting_sets,). bool
        # 1: Find which accepting sets contain the current state.
        # (n_accepting_sets, ), 1 if the set contains the state.
        accepting_sets = self.accepting_sets[:, state]
        assert accepting_sets.dtype == bool

        # 2: Update the frontier mask. If the set i is in the frontier, then frontier_mask[i] = 1
        satisfied_sets = frontier_mask | accepting_sets

        # Check if there were any 0s that have become 1s.
        has_changed = jnp.any(satisfied_sets & (~frontier_mask))

        # 3: If all sets are satisfied, then reset.
        all_satisfied = jnp.all(satisfied_sets)
        new_frontier_mask = jnp.where(all_satisfied, jnp.zeros_like(frontier_mask, dtype=bool), satisfied_sets)

        assert frontier_mask.dtype == bool
        assert frontier_mask.shape == (self.n_accepting_sets,)
        assert new_frontier_mask.dtype == bool
        assert new_frontier_mask.shape == (self.n_accepting_sets,)

        return new_frontier_mask, has_changed


@jdc.pytree_dataclass
class LDBAState:
    state: jnp.ndarray
    # (n_accepting_sets,). Starts at 0, goes to 1 when the set is satisfied.
    accepting_frontier_mask: jnp.ndarray


def parse_ltl2ldba(hoa_text: str) -> LDBA:
    lines = hoa_text.strip().split("\n")

    # Parse header information
    ap_list = []  # Atomic propositions in order
    start_state = 0
    n_acc_sets = 1

    body_started = False
    body_lines = []

    for line in lines:
        line = line.strip()

        if line == "--BODY--":
            body_started = True
            continue
        elif line == "--END--":
            break

        if not body_started:
            # Parse header
            if line.startswith("AP:"):
                # AP: 2 "goal1" "unsafe"
                ap_match = re.findall(r'"([^"]+)"', line)
                ap_list = ap_match
            elif line.startswith("Start:"):
                start_state = int(line.split(":")[1].strip())
            elif line.startswith("Acceptance:"):
                # Acceptance: 1 Inf(0)
                n_acc_sets = int(line.split(":")[1].strip().split()[0])
        else:
            body_lines.append(line)

    logger.debug("start_state: {}".format(start_state))
    logger.debug("")

    # Parse body - extract states and transitions
    states_data = {}  # state_id -> list of (guard_str, dst, acc_sets)
    current_state = None

    for line in body_lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("State:"):
            current_state = int(line.split(":")[1].strip().split()[0])
            states_data[current_state] = []
        else:
            # Parse transition: [guard] dst {acc_sets}?
            # Examples: [!0 & 1] 2, [t] 1 {0}, [!0] 0
            trans_match = re.match(r"\[([^\]]+)\]\s+(\d+)(?:\s+\{([^}]*)\})?", line)
            if trans_match:
                guard_str = trans_match.group(1)
                dst = int(trans_match.group(2))
                acc_str = trans_match.group(3)
                acc_sets = []
                if acc_str:
                    acc_sets = [int(x.strip()) for x in acc_str.split(",") if x.strip()]
                states_data[current_state].append((guard_str, dst, acc_sets))

    # Determine number of states
    all_states = set(states_data.keys())
    for state_transitions in states_data.values():
        for _, dst, _ in state_transitions:
            all_states.add(dst)
    n_states = max(all_states) + 1

    # Parse guards and identify epsilon transitions (non-deterministic choices)
    # A transition is an epsilon transition if there's another transition from the same state
    # that is strictly more general (i.e., the epsilon transition's guard is a subset)

    def parse_guard(guard_str: str, n_aps: int) -> tuple[int, int]:
        """
        Parse guard string and return (pos_mask, neg_mask).
        pos_mask: bits that must be 1
        neg_mask: bits that must be 0
        """
        guard_str = guard_str.strip()

        # Handle 't' (true) - always satisfied
        if guard_str == "t":
            return (0, 0)

        pos_mask = 0
        neg_mask = 0

        # Split by '&' and parse each literal
        literals = [lit.strip() for lit in guard_str.split("&")]

        for lit in literals:
            lit = lit.strip()
            if not lit:
                continue

            if lit.startswith("!"):
                # Negative literal
                ap_idx = int(lit[1:])
                neg_mask |= 1 << ap_idx
            else:
                # Positive literal
                ap_idx = int(lit)
                pos_mask |= 1 << ap_idx

        return (pos_mask, neg_mask)

    def guard_implies(g1: tuple[int, int], g2: tuple[int, int]) -> bool:
        """
        Check if guard g1 implies guard g2 (g1 is more specific).
        g1 implies g2 iff:
        - All positive requirements of g2 are in g1
        - All negative requirements of g2 are in g1
        """
        pos1, neg1 = g1
        pos2, neg2 = g2
        # g2's positive requirements must be subset of g1's
        # g2's negative requirements must be subset of g1's
        return ((pos2 & pos1) == pos2) and ((neg2 & neg1) == neg2)

    n_aps = len(ap_list)

    # Process transitions and identify epsilon transitions
    # Epsilon transitions arise from non-determinism: when a more specific guard
    # leads to a different state than a more general guard from the same source

    transitions = []  # List of (src, dst, pos_mask, neg_mask)
    epsilon_transitions = []  # List of (src, dst)
    accepting_states_per_set = [set() for _ in range(n_acc_sets)]

    for src, trans_list in states_data.items():
        parsed_trans = []
        for guard_str, dst, acc_sets in trans_list:
            guard = parse_guard(guard_str, n_aps)
            parsed_trans.append((guard, dst, acc_sets))

            # Record accepting states
            for acc_set_idx in acc_sets:
                accepting_states_per_set[acc_set_idx].add(dst)

        # Identify epsilon transitions: if trans A is strictly more specific than trans B
        # and they go to different destinations, then A is an epsilon transition
        for i, (guard_i, dst_i, acc_i) in enumerate(parsed_trans):
            is_epsilon = False
            for j, (guard_j, dst_j, acc_j) in enumerate(parsed_trans):
                if i == j:
                    continue
                # Check if guard_i implies guard_j (guard_i is more specific)
                # and they go to different destinations
                if guard_implies(guard_i, guard_j) and dst_i != dst_j:
                    # guard_i is more specific and leads somewhere else
                    # This is non-determinism, so guard_i transition is an epsilon
                    is_epsilon = True
                    break

            if is_epsilon:
                epsilon_transitions.append((src, dst_i))
            else:
                transitions.append((src, dst_i, guard_i[0], guard_i[1]))

    # Build JAX arrays
    if transitions:
        src_arr = jnp.array([t[0] for t in transitions], dtype=jnp.int32)
        dst_arr = jnp.array([t[1] for t in transitions], dtype=jnp.int32)
        pos_mask_arr = jnp.array([t[2] for t in transitions], dtype=jnp.int32)
        neg_mask_arr = jnp.array([t[3] for t in transitions], dtype=jnp.int32)

        guard = Guard(pos_mask=pos_mask_arr, neg_mask=neg_mask_arr)
        transition_obj = Transition(src=src_arr, dst=dst_arr, guard=guard)
    else:
        # Empty transitions (shouldn't happen in practice)
        guard = Guard(pos_mask=jnp.array([], dtype=jnp.int32), neg_mask=jnp.array([], dtype=jnp.int32))
        transition_obj = Transition(src=jnp.array([], dtype=jnp.int32), dst=jnp.array([], dtype=jnp.int32), guard=guard)

    # Build epsilon transition arrays
    if epsilon_transitions:
        epsilon_src = jnp.array([e[0] for e in epsilon_transitions], dtype=jnp.int32)
        epsilon_dst = jnp.array([e[1] for e in epsilon_transitions], dtype=jnp.int32)
    else:
        epsilon_src = jnp.array([], dtype=jnp.int32)
        epsilon_dst = jnp.array([], dtype=jnp.int32)

    # Build accepting sets matrix: (n_acc_sets, n_states)
    accepting_sets_matrix = jnp.zeros((n_acc_sets, n_states), dtype=jnp.int32)
    for acc_set_idx, states in enumerate(accepting_states_per_set):
        for state in states:
            accepting_sets_matrix = accepting_sets_matrix.at[acc_set_idx, state].set(1)

    return LDBA(
        transitions=transition_obj,
        epsilon_src=epsilon_src,
        epsilon_dst=epsilon_dst,
        accepting_sets=accepting_sets_matrix,
        n_states=n_states,
        predicate_order=ap_list,
    )
