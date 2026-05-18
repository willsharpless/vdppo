from typing import Callable, NamedTuple, Protocol

import jax
import jax.numpy as jnp
from loguru import logger


class BellmanUpdate(Protocol):
    def __call__(self, T_term: jnp.ndarray, T_V_next_k: jnp.ndarray, gamma: float):
        raise NotImplementedError("")


class BellmanMin(NamedTuple):
    T_q: jnp.ndarray

    def __call__(self, T_term: jnp.ndarray, T_V_next_k: jnp.ndarray, gamma: float):
        """(1-gamma) * q + gamma * min(q, V_next)"""
        T_q = self.T_q
        assert len(T_q) == len(T_term)

        #    V_next = inf is identity for min.
        T_V_next_k_masked = jnp.where(T_term, jnp.inf, T_V_next_k)
        T_Q_curr = (1 - gamma) * T_q + gamma * jnp.minimum(T_q, T_V_next_k_masked)
        return T_Q_curr


class BellmanMax(NamedTuple):
    T_r: jnp.ndarray

    def __call__(self, T_term: jnp.ndarray, T_V_next_k: jnp.ndarray, gamma: float):
        """(1-gamma) * r + gamma * max(r, V_next)"""
        T_r = self.T_r
        assert len(T_r) == len(T_term)

        #    V_next = -inf is identity for max.
        T_V_next_k_masked = jnp.where(T_term, -jnp.inf, T_V_next_k)
        T_Q_curr = (1 - gamma) * T_r + gamma * jnp.maximum(T_r, T_V_next_k_masked)
        return T_Q_curr


class BellmanMaxMin(NamedTuple):
    T_q: jnp.ndarray
    T_r: jnp.ndarray

    def __call__(self, T_term: jnp.ndarray, T_V_next_k: jnp.ndarray, gamma: float):
        """(1-gamma) * min(r, q) + gamma * max(r, min(q, V_next) )"""
        T_q, T_r = self.T_q, self.T_r
        assert len(T_q) == len(T_r) == len(T_term)

        T_V_next_k_masked = jnp.where(T_term, -jnp.inf, T_V_next_k)
        T_Q_curr = (1 - gamma) * jnp.minimum(T_q, T_r) + gamma * jnp.maximum(T_r, jnp.minimum(T_q, T_V_next_k_masked))
        return T_Q_curr


class BellmanGUSingle(NamedTuple):
    T_q: jnp.ndarray
    T_r: jnp.ndarray
    T_V_nextGU: jnp.ndarray
    T_V_nextGU_next: jnp.ndarray

    def __call__(self, T_term: jnp.ndarray, T_V_next_k: jnp.ndarray, gamma: float):
        T_q, T_r = self.T_q, self.T_r
        T_V_nextGU, T_V_nextGU_next = self.T_V_nextGU, self.T_V_nextGU_next
        assert len(T_q) == len(T_r) == len(T_term)

        # Reach r AND be able to reach the next GU.
        # T_r_tilde = jnp.minimum(T_r, T_V_nextGU)
        T_r_tilde = jnp.minimum(T_r, T_V_nextGU_next)
        # T_r_tilde = T_r

        T_V_next_k_masked = jnp.where(T_term, -jnp.inf, T_V_next_k)
        T_disc_term = jnp.minimum(T_q, T_r)
        T_Q_curr = (1 - gamma) * T_disc_term + gamma * jnp.maximum(T_r_tilde, jnp.minimum(T_q, T_V_next_k_masked))
        return T_Q_curr


def gae_generalized(
    T_V_next: jnp.ndarray,
    T_term: jnp.ndarray,
    T_next_different: jnp.ndarray,
    bellman_update: BellmanUpdate,
    gamma: float,
    lam: float,
):
    """
    Args:
        T_V_next:
        T_term: True if the episode terminated after taking the action at time t.
        T_next_different: True if the next state is from a different rollout (i.e., env reset). term | trunc
        bellman_update:
        gamma:
        lam:

    Returns:

    """
    T = len(T_V_next)
    assert len(T_V_next) == len(T_term) == T

    def body(carry, _):
        T_Q_avg, T_weight_sum, T_V_next_k, T_isvalid, coef = carry

        # 1: Apply Bellman.
        # T_V_next_k_masked = jnp.where(T_term, jnp.inf, T_V_next_k)
        # T_Q_curr = (1 - gamma) * T_q + gamma * jnp.minimum(T_q, T_V_next_k_masked)
        T_Q_curr = bellman_update(T_term, T_V_next_k, gamma=gamma)

        # 2: Stable weighted update.
        T_coef = coef * T_isvalid
        T_weight_sum_new = T_weight_sum + T_coef

        # Incremental update: avg = avg + (w / total_w) * (new - avg)
        # Use jnp.where to prevent division by zero for masked indices
        T_step_size = jnp.where(T_weight_sum_new > 0, T_coef / T_weight_sum_new, 0.0)
        T_Q_avg_new = T_Q_avg + T_step_size * (T_Q_curr - T_Q_avg)

        # 3. Shift for next depth. Can't use inf, since 0 * inf = nan.
        T_V_next_k_shift = jnp.concatenate([T_Q_curr[1:], jnp.array([1.337e8])], axis=0)
        T_isvalid_shift = jnp.concatenate([T_isvalid[1:], jnp.array([0.0])], axis=0)

        # 4. If next state is from a different rollout, then isvalid=0. Don't propagate from a different episode.
        T_isvalid_shift = jnp.where(T_next_different, 0.0, T_isvalid_shift)

        carry_new = (
            T_Q_avg_new,
            T_weight_sum_new,
            T_V_next_k_shift,
            T_isvalid_shift,
            coef * lam,
        )
        return carry_new, None

    T_Q_avg0 = jnp.zeros(T)
    T_weight_sum0 = jnp.zeros(T)
    T_isvalid = jnp.ones(T)
    carry0 = (T_Q_avg0, T_weight_sum0, T_V_next, T_isvalid, 1.0)
    carry_final, _ = jax.lax.scan(body, carry0, None, length=T)
    T_Q_avg = carry_final[0]
    return T_Q_avg


def sum_gae(
    T_V: jnp.ndarray,
    T_V_next: jnp.ndarray,
    T_term: jnp.ndarray,
    T_next_diff: jnp.ndarray,
    T_rew: jnp.ndarray,
    gamma: float,
    gae_lambda: float,
):
    """Standard GAE implementation. Correctly handles truncations."""
    T = len(T_V_next)
    assert len(T_V_next) == len(T_term) == len(T_next_diff) == len(T_rew) == T

    T_delta = T_rew + gamma * T_V_next * (1.0 - T_term) - T_V

    def body(gae_next, inp):
        delta, next_same_ep = inp
        gae = delta + gamma * gae_lambda * next_same_ep * gae_next
        return gae, gae

    T_next_same_ep = 1.0 - T_next_diff
    T_inp = (T_delta, T_next_same_ep)
    _, T_A_gae = jax.lax.scan(body, 0.0, T_inp, length=T, reverse=True)
    T_Q_gae = T_A_gae + T_V
    return T_A_gae, T_Q_gae
