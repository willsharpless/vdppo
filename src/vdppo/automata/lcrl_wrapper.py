import functools as ft
from typing import Generic, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax_dataclasses as jdc
import numpy as np
from attrs import define
from jaxtyping import PRNGKeyArray

from vdppo.common.jax_utils import tree_cat
from vdppo.automata.ldba import LDBA, LDBAState
from vdppo.env.general_task.env import AugObsAutomata, BaseEnv, EnvCfg, EnvStep, EnvUsingBase
from vdppo.common.train_utils import tree_where

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

    def to_minstate(self, state: LCRLState) -> LCRLState:
        # validate=False because we are changing the structure.
        with jdc.copy_and_mutate(state, validate=False) as state_new:
            state_new.base = self.base.to_minstate(state.base)
        return state_new

    def from_minstate(self, minstate: LCRLState) -> LCRLState:
        # validate=False because we are changing the structure.
        with jdc.copy_and_mutate(minstate, validate=False) as state_new:
            state_new.base = self.base.from_minstate(minstate.base)
        return state_new

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

        from vdppo.env.scene import SceneBase

        if isinstance(self.base, SceneBase):
            state: LCRLState[SceneBase.State]

            # We can't do jnp.where on warp state since it doesn't vmap well.
            # Do it only on qpos and qvel.
            qpos = jnp.where(epsilon_taken, state.base.mjx_data.qpos, base_step.envstate.mjx_data.qpos)
            qvel = jnp.where(epsilon_taken, state.base.mjx_data.qvel, base_step.envstate.mjx_data.qvel)

            mjx_data_new = state.base.mjx_data.replace(qpos=qpos, qvel=qvel)

            with jdc.copy_and_mutate(base_step.envstate) as base_state:
                base_state.mjx_data = mjx_data_new
            base_state = self.base.mjx_forward(base_state)
        else:
            base_state = tree_where(epsilon_taken, state.base, base_step.envstate)

        # # Warp is jank. I guess this is to avoid tracers interacting with warp data.
        # def where_should_epsilon(x, y):
        #     if epsilon_taken.shape and epsilon_taken.shape[0] != x.shape[0]:
        #         return y
        #
        #     if epsilon_taken.shape:
        #         epsilon_taken_reshaped = jnp.reshape(epsilon_taken, [x.shape[0]] + [1] * (len(x.shape) - 1))
        #
        #     return jnp.where(epsilon_taken_reshaped, x, y)
        #
        # # If we take the epsilon, then we don't take the base.step
        # # base_state = tree_where(epsilon_taken, state.base, base_step.envstate)
        # base_state = jtu.tree_map(where_should_epsilon, state.base, base_step.envstate)

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

    def get_eval_states(self, n_envs: int, root_only: bool = False) -> LCRLState:
        key = jr.PRNGKey(seed=12345)
        # Override the automata state to be the initial state.
        if root_only:
            states: LCRLState = self.reset_batch(key, n_envs)
            with jdc.copy_and_mutate(states) as states:
                states.ldba_state.state = jnp.zeros((n_envs,), dtype=jnp.int32)
            return states

        # Assign envs evenly to each temporal node.
        n_envs_per_node = np.full((self.ldba.n_states,), n_envs // self.ldba.n_states)
        n_envs_per_node[0] = n_envs - n_envs_per_node[1:].sum()

        # Do this because manipulating warp state is very problematic. Not all fields should be batched.
        b_state_base = self.base.reset_batch_with_pattern(key, tuple(n_envs_per_node))

        ldba_state_list = []
        for ii, n_envs_this in enumerate(n_envs_per_node):
            ldba_state_list.append(jnp.full((n_envs_this,), ii))
        b_ldba_state = jnp.concatenate(ldba_state_list, axis=0)

        b_accepting_frontier_mask = jnp.zeros((n_envs, self.ldba.n_accepting_sets), dtype=bool)
        b_ldba = LDBAState(b_ldba_state, b_accepting_frontier_mask)
        states = LCRLState(b_ldba, b_state_base)

        # # evenly divide n_envs across ldba_states
        # with jdc.copy_and_mutate(states) as states:
        #     n_envs_per_node = jnp.full((self.ldba.n_states,), n_envs // self.ldba.n_states)
        #     n_envs_per_node = n_envs_per_node.at[:(n_envs % self.ldba.n_states)].add(1)
        #     states.ldba_state.state = jnp.repeat(jnp.arange(self.ldba.n_states), n_envs_per_node)

        return states

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

    def get_real_eval_states(
        self,
        n_envs: int,
        n_envs_to_sample: int,
        root_only: bool = True,
    ):
        from vdppo.env.scene import SceneBase

        if isinstance(self.base, SceneBase):
            base_key = jr.PRNGKey(seed=12345)
            b_state = self.reset_batch(base_key, n_envs)
            if root_only:
                with jdc.copy_and_mutate(b_state) as b_state:
                    b_state.ldba_state.state = jnp.zeros((n_envs,), dtype=jnp.int32)
            return b_state

        assert root_only
        n_obtained = 0

        b_valid_fn = jax.jit(jax.vmap(self.is_valid_real_eval_state))

        valid_states_list = []

        ii = 0
        base_key = jr.PRNGKey(seed=12345)
        while n_obtained < n_envs_to_sample:
            key = jr.fold_in(base_key, ii)
            # Rejection sampling using is_valid_real_eval_state.
            m_state_base = self.base.reset_batch_eval(key, n_envs_to_sample)
            accepting_frontier_mask = jnp.zeros((n_envs_to_sample, self.ldba.n_accepting_sets), dtype=bool)
            automata_state0 = jnp.zeros(n_envs_to_sample, dtype=jnp.int32)
            ldba_state = LDBAState(automata_state0, accepting_frontier_mask)
            b_state0 = LCRLState(ldba_state, m_state_base)
            b_valid = b_valid_fn(b_state0)
            n_valid = jnp.sum(b_valid)

            state0_valid = jtu.tree_map(lambda x: x[b_valid], b_state0)
            valid_states_list.append(state0_valid)
            n_obtained += int(n_valid)

        # Concatenate, then trim to n_envs.
        b_state0_all = tree_cat(valid_states_list, axis=0)
        b_state0_all = jtu.tree_map(lambda x: x[:n_envs], b_state0_all)
        return b_state0_all
