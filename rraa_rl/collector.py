import functools as ft
from typing import Any, Callable, Protocol, Self, Tuple

import einops as ei
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

from rraa_rl.jax_utils import tree_where_dim0
from rraa_rl.src.env.general_task.env import EnvStep
from rraa_rl.src.env.general_task.herd_os import HerdOs


class SampleActionFn(Protocol):
    def __call__(self, obs: Any, key: PRNGKeyArray) -> Any: ...


class BatchResetFn(Protocol):
    def __call__(self, env: HerdOs, b_key: PRNGKeyArray, b_state: Any) -> Any: ...


@struct.dataclass
class RolloutOutput:
    state_now: Any
    state_next: Any
    obs_now: Any
    obs_next: Any
    act: jnp.ndarray

    predicates: dict

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


@struct.dataclass
class CollectorState:
    b_state: Any  # Current states of each environment.
    b_obs: Any


@define
class CollectorCfg:
    n_envs: int


class Collector(struct.PyTreeNode):
    Cfg = CollectorCfg

    collect_idx: int
    key: PRNGKeyArray
    collect_state: CollectorState
    env: HerdOs = struct.field(pytree_node=False)
    cfg: CollectorCfg = struct.field(pytree_node=False)

    @classmethod
    def create(cls, key: PRNGKeyArray, env: HerdOs, cfg: CollectorCfg):
        key, key_init = jr.split(key)
        b_state = env.reset_batch(key_init, cfg.n_envs)
        b_obs = jax.vmap(env.get_obs)(b_state)

        collector_state = CollectorState(b_state=b_state, b_obs=b_obs)
        return Collector(
            collect_idx=0,
            key=key,
            collect_state=collector_state,
            env=env,
            cfg=cfg,
        )

    def step_single_fn(self, sample_action: SampleActionFn, key: PRNGKeyArray, state: Any, obs: Any):
        action, logprob = sample_action(obs, key)
        step_result = self.env.step(state, action)
        return step_result, action, logprob

    def collect_batch(
        self, sample_action: SampleActionFn, T: int, reset_fn: BatchResetFn = None
    ) -> tuple[Self, RolloutOutput, dict]:
        if reset_fn is None:
            reset_fn = _default_reset_fn

        def loop(carry: tuple[CollectorState], args):
            b_key = args
            (colstate,) = carry
            b2_key = jax.vmap(jr.split)(b_key)
            b_key_pol, b_key_reset = b2_key[:, 0], b2_key[:, 1]

            step_single_fn = ft.partial(self.step_single_fn, sample_action)
            b_step_result: EnvStep
            b_step_result, b_act, b_logprob = jax.vmap(step_single_fn)(b_key_pol, colstate.b_state, colstate.b_obs)

            # Sample new states from the environments for resets
            b_state_reset = reset_fn(self.env, b_key_reset, colstate.b_state)
            b_obs_reset = jax.vmap(self.env.get_obs)(b_state_reset)

            b_should_reset = b_step_result.term | b_step_result.trunc
            b_state_new = tree_where_dim0(b_should_reset, b_state_reset, b_step_result.envstate, which=jnp)
            b_obs_new = tree_where_dim0(b_should_reset, b_obs_reset, b_step_result.obs, which=jnp)

            # Create the updated collector state
            colstate_new = jdc.replace(colstate, b_state=b_state_new, b_obs=b_obs_new)
            carry_new = (colstate_new,)
            out = RolloutOutput(
                state_now=colstate.b_state,
                state_next=b_obs_new,
                obs_now=colstate.b_obs,
                obs_next=b_obs_new,
                act=b_act,
                predicates=b_step_result.predicates,
                term=b_step_result.term,
                trunc=b_step_result.trunc,
                logprob=b_logprob,
                info=b_step_result.info,
            )
            return carry_new, out

        carry0 = (self.collect_state,)
        key = jr.fold_in(self.key, self.collect_idx)
        Tb_keys = jr.split(key, T * self.cfg.n_envs).reshape(T, self.cfg.n_envs, 2)
        (colstate_new,), Tb_rollout = lax.scan(loop, carry0, Tb_keys, length=T)

        self_new = self.replace(collect_idx=self.collect_idx + 1, collect_state=colstate_new)
        collect_info = {}
        return self_new, Tb_rollout, collect_info


def _default_reset_fn(env: HerdOs, b_key: PRNGKeyArray, b_state: Any) -> Any:
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
