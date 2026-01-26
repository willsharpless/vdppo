import functools as ft
# from typing import Any, Callable, Protocol, Self, Tuple
from typing import Any, Callable, Protocol, Tuple

import einops as ei
import ipdb
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax_dataclasses as jdc
import numpy as np
from attrs import define
from flax import struct
from jaxtyping import Bool, Float, PRNGKeyArray
from loguru import logger
from typing_extensions import Self

from rraa_rl.jax_utils import switch01, tree_where_dim0
from rraa_rl.src.env.general_task.env import Env, EnvStep
from rraa_rl.train_utils import tree_where


class SampleActionFn(Protocol):
    def __call__(self, obs: Any, key: PRNGKeyArray) -> Any: ...


class GetActionFn(Protocol):
    def __call__(self, obs: Any) -> Any: ...


class BatchResetFn(Protocol):
    def __call__(self, env: Env, b_key: PRNGKeyArray, b_state: Any) -> Any: ...


class TruncateFn(Protocol):
    """Function that can be passed to the collector to truncate episodes early."""

    def __call__(self, env: Env, b_key: PRNGKeyArray, b_step: EnvStep) -> EnvStep: ...


class SwitchFn(Protocol):
    """Function that can be passed to the collector to modify the temporal_node_idx."""

    def __call__(self, env: Env, key: PRNGKeyArray, step: EnvStep) -> EnvStep: ...


@jdc.pytree_dataclass
class CollectorState:
    b_state: Any  # Current states of each environment.
    b_obs: Any


@struct.dataclass
class RolloutOutput:
    state_now: Any
    state_next: Any
    obs_now: Any
    obs_next: Any
    act: jnp.ndarray

    predicates_next: dict

    term: jnp.ndarray
    """Termination flags after taking action."""

    trunc: jnp.ndarray
    """Truncation flags after taking action."""

    logprob: jnp.ndarray
    """Log probabilities of the actions taken."""

    info: dict
    """Additional info from the environment."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Get n_envs and n_steps."""
        return self.term.shape

    def switch01(self) -> "RolloutOutput":
        return jtu.tree_map(switch01, self)

    @staticmethod
    def from_rollout(
        colstate: CollectorState,
        b_step_result: EnvStep,
        b_act: jnp.ndarray,
        b_logprob: jnp.ndarray,
    ) -> "RolloutOutput":
        return RolloutOutput(
            state_now=colstate.b_state,
            state_next=b_step_result.envstate,
            obs_now=colstate.b_obs,
            obs_next=b_step_result.obs,
            act=b_act,
            predicates_next=b_step_result.predicates,
            term=b_step_result.term,
            trunc=b_step_result.trunc,
            logprob=b_logprob,
            info=b_step_result.info,
        )

    @property
    def temporal_node_idx(self):
        assert hasattr(self.state_now, "temporal_node_idx")
        return self.state_now.temporal_node_idx


@define
class CollectorCfg:
    n_envs: int

    auto_reset: bool = True
    """False for evals to make it easier to track episode ends."""

    ignore_trunc: bool = False
    """If True, then remove all truncations from the collected data."""

    use_minstate: bool = True
    """IF True, then use the minstate representation for storage to save memory."""


class Collector(struct.PyTreeNode):
    Cfg = CollectorCfg

    collect_idx: int
    key: PRNGKeyArray
    collect_state: CollectorState
    env: Env = struct.field(pytree_node=False)
    cfg: CollectorCfg = struct.field(pytree_node=False)

    @classmethod
    def create(cls, key: PRNGKeyArray, env: Env, cfg: CollectorCfg, init: bool = True):
        key, key_init = jr.split(key)
        b_state = env.reset_batch(key_init, cfg.n_envs, init=init)
        b_obs = jax.jit(jax.vmap(env.get_obs))(b_state)

        collector_state = CollectorState(b_state=b_state, b_obs=b_obs)
        return Collector(
            collect_idx=0,
            key=key,
            collect_state=collector_state,
            env=env,
            cfg=cfg,
        )

    def reset_with_state(self, b_state: Any) -> Self:
        with jdc.copy_and_mutate(self) as self_new:
            self_new.collect_state.b_state = b_state
            self_new.collect_state.b_obs = jax.vmap(self.env.get_obs)(b_state)

        return self_new

    def step_single_fn(self, sample_action: SampleActionFn, key: PRNGKeyArray, state: Any, obs: Any):
        action, logprob = sample_action(obs, key)
        step_result = self.env.step(state, action)
        return step_result, action, logprob

    def step_single_fn_det(self, get_action: GetActionFn, state: Any, obs: Any):
        action = get_action(obs)
        step_result = self.env.step(state, action)
        return step_result, action

    def collect_batch(
        self,
        sample_action: SampleActionFn,
        T: int,
        reset_fn: BatchResetFn = None,
        truncate_fn: TruncateFn = None,
        switch_fn: SwitchFn = None,
    ) -> tuple[Self, RolloutOutput, dict]:
        if reset_fn is None:
            reset_fn = _default_reset_fn

        def loop(carry: tuple[CollectorState], args):
            b_key = args
            (colstate,) = carry
            b3_key = jax.vmap(ft.partial(jr.split, num=4))(b_key)
            b_key_pol, b_key_reset = b3_key[:, 0], b3_key[:, 1]
            b_key_truncate, b_key_switch = b3_key[:, 2], b3_key[:, 3]

            step_single_fn = ft.partial(self.step_single_fn, sample_action)
            b_step_result: EnvStep
            b_step_result, b_act, b_logprob = jax.vmap(step_single_fn)(b_key_pol, colstate.b_state, colstate.b_obs)

            if switch_fn is not None:
                # Switch the temporal_node_idx based on some criteria. Possible use the value function.
                b_step_result = switch_fn(self.env, b_key_truncate, b_step_result)

            if truncate_fn is not None:
                # Additional function to truncate episodes early (potentially using extra info)
                b_step_result = truncate_fn(self.env, b_key_truncate, b_step_result)

            # NOTE: Reset DOESN'T change the step, only the colstate.
            out = RolloutOutput.from_rollout(colstate, b_step_result, b_act, b_logprob)

            if self.cfg.use_minstate:
                # Convert to minstate to save memory.
                b_state_now = jax.vmap(self.env.to_minstate)(out.state_now)
                b_state_next = jax.vmap(self.env.to_minstate)(out.state_next)
                out = out.replace(state_now=b_state_now, state_next=b_state_next)

            # Sample new states from the environments for resets
            if self.cfg.auto_reset:
                b_state_reset = reset_fn(self.env, b_key_reset, colstate.b_state)
                b_obs_reset = jax.vmap(self.env.get_obs)(b_state_reset)

                b_should_reset = b_step_result.term | b_step_result.trunc

                # b_state_new = jax.vmap(tree_where)(b_should_reset, b_state_reset, b_step_result.envstate)

                # Warp is jank. I guess this is to avoid tracers interacting with warp data.
                def where_should_reset(x, y):
                    if b_should_reset.shape and b_should_reset.shape[0] != x.shape[0]:
                        ipdb.set_trace()
                        return y

                    if b_should_reset.shape:
                        should_reset = jnp.reshape(b_should_reset, [x.shape[0]] + [1] * (len(x.shape) - 1))

                    return jnp.where(should_reset, x, y)

                b_state_new = where_should_reset(b_state_reset, b_step_result.envstate)

                b_obs_new = jax.vmap(tree_where)(b_should_reset, b_obs_reset, b_step_result.obs)
            else:
                b_state_new = b_step_result.envstate
                b_obs_new = b_step_result.obs

            # Create the updated collector state
            colstate_new_ = jdc.replace(colstate, b_state=b_state_new, b_obs=b_obs_new)
            carry_new = (colstate_new_,)
            return carry_new, out

        carry0 = (self.collect_state,)
        key = jr.fold_in(self.key, self.collect_idx)
        Tb_keys = jr.split(key, T * self.cfg.n_envs).reshape(T, self.cfg.n_envs, 2)
        (colstate_new,), Tb_rollout = lax.scan(loop, carry0, Tb_keys, length=T)

        self_new = self.replace(collect_idx=self.collect_idx + 1, collect_state=colstate_new)
        collect_info = {}
        return self_new, Tb_rollout, collect_info

    def collect_full_traj_det(
        self,
        get_action: GetActionFn,
        T: int,
        reset_fn: BatchResetFn = None,
        switch_fn: SwitchFn = None,
    ) -> tuple[Self, RolloutOutput, dict]:
        if reset_fn is None:
            reset_fn = _default_reset_fn

        def loop(carry: tuple[CollectorState], args):
            b_key = args
            (colstate,) = carry

            b2_key = jax.vmap(ft.partial(jr.split, num=2))(b_key)
            b_key_reset, b_key_switch = b2_key[:, 0], b2_key[:, 1]

            step_single_fn = ft.partial(self.step_single_fn_det, get_action)
            b_step_result: EnvStep
            b_step_result, b_act = jax.vmap(step_single_fn)(colstate.b_state, colstate.b_obs)
            b_logprob = jnp.zeros(self.cfg.n_envs)

            if switch_fn is not None:
                # Switch the temporal_node_idx based on some criteria. Possible use the value function.
                b_step_result = jax.vmap(ft.partial(switch_fn, self.env))(b_key_switch, b_step_result)

            if self.cfg.ignore_trunc:
                logger.debug("Ignoring truncations in collected data.")
                b_step_result = b_step_result._replace(trunc=jnp.zeros_like(b_step_result.trunc))

            # NOTE: Reset DOESN'T change the step, only the colstate.
            out = RolloutOutput.from_rollout(colstate, b_step_result, b_act, b_logprob)

            if self.cfg.use_minstate:
                # Convert to minstate to save memory.
                b_state_now = jax.vmap(self.env.to_minstate)(out.state_now)
                b_state_next = jax.vmap(self.env.to_minstate)(out.state_next)
                out = out.replace(state_now=b_state_now, state_next=b_state_next)

            # Sample new states from the environments for resets
            if self.cfg.auto_reset:
                b_state_reset = reset_fn(self.env, b_key_reset, colstate.b_state)
                b_obs_reset = jax.vmap(self.env.get_obs)(b_state_reset)

                b_should_reset = b_step_result.term | b_step_result.trunc
                b_state_new = jax.vmap(tree_where)(b_should_reset, b_state_reset, b_step_result.envstate)
                b_obs_new = jax.vmap(tree_where)(b_should_reset, b_obs_reset, b_step_result.obs)
            else:
                b_state_new = b_step_result.envstate
                b_obs_new = b_step_result.obs

            # Create the updated collector state
            colstate_new_ = jdc.replace(colstate, b_state=b_state_new, b_obs=b_obs_new)
            carry_new = (colstate_new_,)
            return carry_new, out

        carry0 = (self.collect_state,)
        key = jr.fold_in(self.key, self.collect_idx)
        Tb_keys = jr.split(key, T * self.cfg.n_envs).reshape(T, self.cfg.n_envs, 2)
        (colstate_new,), Tb_rollout = lax.scan(loop, carry0, Tb_keys, length=T)

        self_new = self.replace(collect_idx=self.collect_idx + 1, collect_state=colstate_new)
        collect_info = {}
        return self_new, Tb_rollout, collect_info


def _default_reset_fn(env: Env, b_key: PRNGKeyArray, b_state: Any) -> Any:
    batch_size = len(b_key)
    return env.reset_batch(b_key[0], batch_size)


def extract_info_from_rollout(Tb_rollout: RolloutOutput) -> dict:
    """Extract easy-to-log info from the rollout."""
    T, b = Tb_rollout.shape

    Tb_term = jax.device_get(Tb_rollout.term)
    Tb_trunc = jax.device_get(Tb_rollout.trunc)
    Tb_info = jax.device_get(Tb_rollout.info)
    Tb_age = Tb_info["age"]

    info_out = {}

    # Average episode length at termination or truncation.
    Tb_done = Tb_term | Tb_trunc
    if np.any(Tb_done):
        info_out["avg_episode_length"] = np.mean(Tb_age.flatten()[Tb_done.flatten()])

    # Fraction of episodes terminated vs truncated.
    if Tb_term.sum() > 0 and Tb_trunc.sum() > 0:
        n_done = Tb_term.sum() + Tb_trunc.sum()
        info_out["frac_term"] = Tb_term.sum() / n_done

    return info_out
