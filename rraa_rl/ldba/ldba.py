import functools as ft
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from attrs import define
from flax import struct


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

    # @staticmethod
    # def create(transitions: Transition, epsilon_src: jnp.ndarray, epsilon_dst: jnp.ndarray):
    #     return LDBA(transitions=transitions, epsilon_src=epsilon_src, epsilon_dst=epsilon_dst)

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
        next_state = jnp.where(has_eligible, self.transitions[idx].dst, state)

        # If in sink, stay in sink.
        next_state = jnp.where(is_sink_state, self.sink_state, next_state)
        return next_state

    def step(self, state: jnp.ndarray, label: jnp.ndarray, epsilon_action: jnp.ndarray):
        next_state_epsilon, epsilon_taken = self.step_epsilon(state, epsilon_action)
        next_state_noepsilon = self.step_noepsilon(state, label)

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

        # 2: Update the frontier mask. If the set i is in the frontier, then frontier_mask[i] = 1
        satisfied_sets = frontier_mask | accepting_sets

        # Check if there were any 0s that have become 1s.
        has_changed = jnp.any(satisfied_sets & (~frontier_mask))

        # 3: If all sets are satisfied, then reset.
        all_satisfied = jnp.all(satisfied_sets)
        new_frontier_mask = jnp.where(all_satisfied, jnp.zeros_like(frontier_mask), satisfied_sets)

        return new_frontier_mask, has_changed


@jdc.pytree_dataclass
class LDBAState:
    state: jnp.ndarray
    # (n_accepting_sets,). Starts at 0, goes to 1 when the set is satisfied.
    accepting_frontier_mask: jnp.ndarray
