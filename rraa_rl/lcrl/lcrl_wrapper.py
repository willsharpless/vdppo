import functools as ft
from typing import Generic, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jr
import jax_dataclasses as jdc
from attrs import define
from jaxtyping import PRNGKeyArray

from rraa_rl.ldba.ldba import LDBA, LDBAState
from rraa_rl.src.env.general_task.env import AugObsAutomata, BaseEnv, EnvCfg, EnvStep, EnvUsingBase
from rraa_rl.train_utils import tree_where

BaseClassState = TypeVar("BaseClassState")


@define(slots=False)
class LCRLEnvCfg(EnvCfg):
    specification: str = "F G herd_herded"
    random_automata_init: bool = False


@jdc.pytree_dataclass
class LCRLState(Generic[BaseClassState]):
    ldba_state: LDBAState
    base: BaseClassState


class LCRLWrapper(EnvUsingBase):
    State = LCRLState

    def __init__(self, cfg: LCRLEnvCfg, base: BaseEnv, ldba: LDBA):
        self.cfg = cfg
        EnvUsingBase.__init__(self, cfg, self.specification, base_env=base)
        self.ldba = ldba

    @property
    def n_actions_per_agent(self) -> list[list[int]]:
        # Make the epsilon transition a new agent at the end with n_epsilon_action + 1 actions.
        n_actions_per_agent = self.base.n_actions_per_agent
        return [*n_actions_per_agent, [self.ldba.n_epsilon_transitions + 1]]

    def step(self, state: LCRLState, action: list[jnp.ndarray]) -> EnvStep:
        action_base = action[:-1]
        action_epsilon = action[-1]

        base_step: EnvStep = self.base.step(state.base, action_base)
        predicates_float = base_step.predicates

        predicates_bool = {k: v > 0.5 for k, v in predicates_float.items()}

        # Make sure these are bool predicates.
        for k, v in predicates_bool.items():
            assert v.dtype == bool, f"Predicate {k} is not bool, but {v.dtype}"

        label = self.ldba.predicates_to_label(predicates_bool)
        automata_state_new, epsilon_taken = self.ldba.step(state.ldba_state.state, label, action_epsilon)

        # If we take the epsilon, then we don't take the base.step
        base_state = tree_where(epsilon_taken, state.base, base_step.envstate)

        # Update the frontier.
        frontier_mask_new, has_changed = self.ldba.update_frontier(
            state.ldba_state.accepting_frontier_mask, automata_state_new
        )
        assert state.ldba_state.accepting_frontier_mask.dtype == bool
        assert frontier_mask_new.dtype == bool

        ldba_state_new = LDBAState(automata_state_new, frontier_mask_new)
        state_new = LCRLState(ldba_state=ldba_state_new, base=base_state)

        # If we reach the sink state, we terminate.
        in_sink_state = automata_state_new == self.ldba.sink_state
        term = base_step.term | in_sink_state

        info_aug = {"epsilon_taken": epsilon_taken, "has_frontier_changed": has_changed, "into_sink": in_sink_state}
        info = base_step.info | info_aug

        obs = self._augment_obs(state_new, base_step.obs)
        step = base_step._replace(envstate=state_new, obs=obs, info=info, term=term)
        return step

    @property
    def specification(self):
        return self.cfg.specification

    @ft.partial(jax.jit, static_argnames=("self",))
    def reset(self, key: PRNGKeyArray) -> LCRLState:
        base_state = self.base.reset(key)
        accepting_frontier_mask = jnp.zeros(self.ldba.n_accepting_sets, dtype=bool)
        if self.cfg.random_automata_init:
            automata_state0 = jr.randint(key, (), 0, self.ldba.n_states, dtype=jnp.int32)
        else:
            automata_state0 = jnp.array(0, dtype=jnp.int32)
        ldba_state = LDBAState(automata_state0, accepting_frontier_mask)
        return LCRLState(ldba_state, base_state)

    def get_eval_states(self, n_envs: int) -> LCRLState:
        key = jr.PRNGKey(seed=12345)
        return self.reset_batch(key, n_envs)

    def get_obs(self, state: LCRLState) -> AugObsAutomata:
        base_obs = self.base.get_obs(state.base)
        return self._augment_obs(state, base_obs)

    def _augment_obs(self, state: LCRLState, obs: jnp.ndarray):
        obs_aug = self._get_augment_obs(state)
        return AugObsAutomata(state.ldba_state.state, obs, obs_aug)

    def _get_augment_obs(self, state: LCRLState):
        obs, _ = self._augment_obs_and_names(state)
        return obs

    def _augment_obs_and_names(self, state: LCRLState):
        n_ldba_states = self.ldba.n_states
        ldba_obs = jax.nn.one_hot(state.ldba_state.state, n_ldba_states, dtype=jnp.float32)
        ldba_obs_names = [f"ldba_state_{i}" for i in range(n_ldba_states)]
        return ldba_obs, ldba_obs_names
