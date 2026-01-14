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
from flax import struct
from jaxtyping import Bool, Float, PRNGKeyArray
from loguru import logger

from rraa_rl.src.env.general_task.herd_os import HerdOs


class SampleActionFn(Protocol):
    def __call__(self, state: Any, key: PRNGKeyArray) -> Any: ...


@struct.dataclass
class RolloutOutput:
    T_state_now: Any
    T_state_next: Any
    T_obs_now: Any
    T_obs_next: Any
    T_action: jnp.ndarray

    T_predicates: dict

    T_term: jnp.ndarray
    """Termination flags after taking action."""

    T_trunc: jnp.ndarray
    """Truncation flags after taking action."""

    T_logprob: jnp.ndarray
    """Log probabilities of the actions taken."""

    T_info: dict
    """Additional info from the environment."""


@struct.dataclass
class CollectorState:
    b_state: Any  # Current states of each environment.
    b_obs: Any


@struct.dataclass
class CollectorCfg:
    n_envs: int


class Collector(struct.PyTreeNode):
    collect_idx: int
    key: PRNGKeyArray
    collect_state: CollectorState
    env: HerdOs = struct.field(pytree_node=False)
    cfg: CollectorCfg = struct.field(pytree_node=False)

    @classmethod
    def create(cls, key: PRNGKeyArray, env: HerdOs, cfg: CollectorCfg):
        key, key_init = jr.split(key)
        b_key_init = jr.split(key_init, cfg.n_envs)
        b_reset_result = jax.vmap(env.reset)(b_key_init)
        b_state = b_reset_result.state

        collector_state = CollectorState(b_state=b_state)
        return Collector(
            collect_idx=0,
            key=key,
            collect_state=collector_state,
            env=env,
            cfg=cfg,
        )

    def collect_batch(self, sample_action: SampleActionFn, batch_size: int) -> tuple[Self, RolloutOutput, dict]:
        def loop(carry, args):
            b_key = args
            b_colstate, = carry
            b2_key = jax.vmap(jr.split)(b_key)
            b_key_pol, b_key_reset = b2_key[:, 0], b2_key[:, 1]

            # Sample new states from the environments for resets

            # Create the updated collector state
            b_colstate_new = jdc.replace(b_colstate, b_state=b_state_new)
            carry_new = (b_colstate_new,)
            out = RolloutOutput(
                T_reset_id=b_colstate.b_reset_id,
                T_state_now=b_state_now,
                T_state_next=b_state_next,
                T_obs_now=step_result.b_obs_now,
                T_obs_next=b_obs_next,
                T_act=b_act,
                T_rew=b_rew,
                TM_cost=b_cost,
                TM_h_now=b_h_now,
                TM_h_next=b_h_next,
                T_term=b_term,
                T_trunc=b_trunc,
                T_logprob=b_logprob,
                T_info=b_info,
            )
            return carry_new, out
