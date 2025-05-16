import einops as ei
from functools import partial
from typing import Tuple, NamedTuple

import jax
import jax.numpy as jnp

class Transition_sac(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

class Transition_cppo(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    value_cost: jnp.ndarray
    reward: jnp.ndarray
    cost: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    reach: jnp.ndarray
    avoid: jnp.ndarray

class Transition_rr_cppo(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    value_cost: jnp.ndarray
    reward: jnp.ndarray
    cost: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    reach1: jnp.ndarray
    reach2: jnp.ndarray

class Transition_reach(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    value_reach: jnp.ndarray
    reward: jnp.ndarray
    energy: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    reach: jnp.ndarray


class Transition_a(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avoid: jnp.ndarray

class Transition_raa(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    reach: jnp.ndarray
    avoid: jnp.ndarray
    value_avoid: jnp.ndarray 
    has_reached: jnp.ndarray
    policy_taken: jnp.ndarray

class Transition_ra(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    reach: jnp.ndarray
    avoid: jnp.ndarray

class Transition_rr(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    value_reach1: jnp.ndarray
    value_reach2: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    reach1: jnp.ndarray
    reach2: jnp.ndarray
    has_reached_1: jnp.ndarray
    has_reached_2: jnp.ndarray
    policy_taken: jnp.ndarray

class Transition_r(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    reach: jnp.ndarray

class Transition_r1(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    reach1: jnp.ndarray

class Transition_r2(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    reach2: jnp.ndarray

class Transition_rraa(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    value_reach1: jnp.ndarray
    value_reach2: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    reach1: jnp.ndarray
    reach2: jnp.ndarray

@partial(jax.jit)
def calculate_advantage(
    gae_nval_gamma_lambda: Tuple[jnp.ndarray, jnp.ndarray, float, float],
    transition
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, float, float], jnp.array]:
    gae, next_value, Gamma, Lambda = gae_nval_gamma_lambda
    value, reward, done = transition[0], transition[1], transition[2]
    delta = reward + Gamma * next_value * (1 - done) - value
    gae = delta + Gamma * Lambda * (1 - done) * gae
    return (gae, value, Gamma, Lambda), gae


@partial(jax.jit)
def calculate_gae(
    gamma: float,
    gae_lambda: float,
    value: jnp.ndarray,
    reward: jnp.ndarray,
    done: jnp.ndarray,
    last_value: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    _, advantages = jax.lax.scan(
        calculate_advantage,
        (jnp.zeros_like(last_value), last_value, gamma, gae_lambda),
        (value, reward, done),
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + value

@partial(jax.jit)
def calculate_gae_reach(
    gamma: float,
    gae_lambda: float,
    Tp1_hs: jnp.ndarray,
    Tp1_Vhs: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    Tp1, nh = Tp1_hs.shape
    T = Tp1 - 1

    def loop(carry, inp):
        ii, hs, Vhs = inp
        next_Vhs_row, gae_coeffs = carry

        mask = jnp.arange(T + 1) < ii + 1
        mask_h = mask[:, None]

        # DP for Vh.
        Vhs_row = mask_h * jnp.minimum(hs, gamma * next_Vhs_row)

        normed_gae_coeffs = gae_coeffs / gae_coeffs.sum()
        Qhs_GAE = ei.einsum(Vhs_row, normed_gae_coeffs, "Tp1 nh, Tp1 -> nh")

        # Setup Vs_row for next timestep.
        Vhs_row = Vhs_row.at[ii + 1, :].set(Vhs)

        # Update GAE coeffs. [1] -> [λ 1] -> [λ² λ 1]
        gae_coeffs = jnp.roll(gae_coeffs, 1)
        gae_coeffs = gae_coeffs.at[0].set(gae_coeffs[1] * gae_lambda)

        return (Vhs_row, gae_coeffs), Qhs_GAE

    init_gae_coeffs = jnp.zeros(T + 1)
    init_gae_coeffs = init_gae_coeffs.at[0].set(1.0)

    init_Vhs = jnp.zeros((T + 1, nh)).at[0, :].set(Tp1_Vhs[T, :])
    init_carry = (init_Vhs, init_gae_coeffs)

    ts = jnp.arange(T)[::-1]
    inps = (ts, Tp1_hs[1:], Tp1_Vhs[:-1])

    _, Qhs_GAEs = jax.lax.scan(loop, init_carry, inps, reverse=True)
    return Qhs_GAEs - Tp1_Vhs[:-1], Qhs_GAEs

@partial(jax.jit)
def calculate_gae_reach2(
    gamma: float,
    gae_lambda: float,
    T_hs: jnp.ndarray,
    T_Vhs: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    Tp1, nh = T_hs.shape
    T = Tp1 - 1

    def loop(carry, inp):
        ii, hs, Vhs = inp
        next_Vhs_row, gae_coeffs = carry

        mask = jnp.arange(T + 1) < ii + 1
        mask_h = mask[:, None]

        # DP for Vh.
        disc_to_h = (1 - gamma) * hs + gamma * next_Vhs_row
        Vhs_row = mask_h * jnp.minimum(hs, disc_to_h)

        normed_gae_coeffs = gae_coeffs / gae_coeffs.sum()
        Qhs_GAE = ei.einsum(Vhs_row, normed_gae_coeffs, "Tp1 nh, Tp1 -> nh")

        # Setup Vs_row for next timestep.
        Vhs_row = Vhs_row.at[ii + 1, :].set(Vhs)

        # Update GAE coeffs. [1] -> [λ 1] -> [λ² λ 1]
        gae_coeffs = jnp.roll(gae_coeffs, 1)
        gae_coeffs = gae_coeffs.at[0].set(gae_coeffs[1] * gae_lambda)

        return (Vhs_row, gae_coeffs), Qhs_GAE

    init_gae_coeffs = jnp.zeros(T + 1)
    init_gae_coeffs = init_gae_coeffs.at[0].set(1.0)

    init_Vhs = jnp.zeros((T + 1, nh)).at[0, :].set(T_Vhs[T, :])
    init_carry = (init_Vhs, init_gae_coeffs)

    ts = jnp.arange(T)[::-1]
    inps = (ts, T_hs[:-1], T_Vhs[1:])

    _, Qhs_GAEs = jax.lax.scan(loop, init_carry, inps, reverse=True)
    return Qhs_GAEs - T_Vhs[:-1], Qhs_GAEs

@partial(jax.jit)
def calculate_gae_reach3(
    gamma: float,
    gae_lambda: float,
    T_hs: jnp.ndarray,
    T_Vhs: jnp.ndarray,
    done: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    Tp1, nh = T_hs.shape
    T = Tp1 - 1

    def loop(carry, inp):
        ii, hs, Vhs, done_row = inp
        next_Vhs_row, gae_coeffs = carry

        # Update GAE coeffs. [1] -> [1 λ] -> [1 λ λ²]
        gae_coeffs = jnp.roll(gae_coeffs, 1, axis=0) * gae_lambda * (1 - done_row)
        gae_coeffs = gae_coeffs.at[0, :].set(1.0)

        mask = jnp.arange(T + 1) < ii + 1
        mask_h = mask[:, None]

        # DP for Vh.
        done_row_processed = jnp.where(jnp.isnan(done_row * jnp.inf), 0, done_row * jnp.inf)
        disc_to_h = (1 - gamma) * hs + gamma * (next_Vhs_row + done_row_processed)
        # disc_to_h = (1 - gamma) * hs + gamma * (next_Vhs_row)
        Vhs_row = jnp.minimum(hs, disc_to_h)
        Vhs_row = mask_h * Vhs_row

        normed_gae_coeffs = gae_coeffs / jnp.sum(gae_coeffs, axis=0)
        Qhs_GAE = jnp.sum(Vhs_row * normed_gae_coeffs, axis=0)

        # Setup Vs_row for next timestep.
        Vhs_row = jnp.roll(Vhs_row, 1, axis=0)
        Vhs_row = Vhs_row.at[0, :].set(Vhs)

        return (Vhs_row, gae_coeffs), Qhs_GAE

    done = jnp.array(done, dtype=int)
    init_gae_coeffs = jnp.zeros((T + 1, nh))

    init_Vhs = jnp.zeros((T + 1, nh)).at[0, :].set(T_Vhs[T, :])
    init_carry = (init_Vhs, init_gae_coeffs)

    ts = jnp.arange(T)[::-1]
    inps = (ts, T_hs[:-1], T_Vhs[1:], done)

    _, Qhs_GAEs = jax.lax.scan(loop, init_carry, inps, reverse=True)
    return Qhs_GAEs - T_Vhs[:-1], Qhs_GAEs


@partial(jax.jit)
def calculate_gae_reach4(
    gamma: float,
    gae_lambda: float,
    T_hs: jnp.ndarray,
    T_Vhs: jnp.ndarray,
    done: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    Tp1, nh = T_hs.shape
    T = Tp1 - 1

    def loop(carry, inp):
        ii, hs, Vhs, done_row = inp
        next_Vhs_row, gae_coeffs, pre_done_row = carry

        # Update GAE coeffs. [1] -> [1, λ/(1-λ)] -> [1 λ λ²/(1-λ)] -> [1 λ λ² λ³/(1-λ)]
        gae_coeffs = (jnp.roll(gae_coeffs, 1, axis=0) * gae_lambda * (1 - pre_done_row) +
                      jnp.roll(gae_coeffs, 1, axis=0) * (gae_lambda / (1 - gae_lambda)) * pre_done_row) * (1 - done_row)
        gae_coeffs = gae_coeffs.at[0, :].set(1.0)

        mask = jnp.arange(T + 1) < ii + 1
        mask_h = mask[:, None]

        # DP for Vh.
        done_row_processed = jnp.where(jnp.isnan(done_row * jnp.inf), 0, done_row * jnp.inf)
        disc_to_h = (1 - gamma) * hs + gamma * (next_Vhs_row + done_row_processed)

        Vhs_row = jnp.minimum(hs, disc_to_h)
        Vhs_row = mask_h * Vhs_row

        normed_gae_coeffs = gae_coeffs / jnp.sum(gae_coeffs, axis=0)
        Qhs_GAE = jnp.sum(Vhs_row * normed_gae_coeffs, axis=0)

        # Setup Vs_row for next timestep.
        Vhs_row = jnp.roll(Vhs_row, 1, axis=0)
        Vhs_row = Vhs_row.at[0, :].set(Vhs)

        return (Vhs_row, gae_coeffs, done_row), Qhs_GAE

    done = jnp.array(done, dtype=int)
    init_gae_coeffs = jnp.zeros((T + 1, nh))

    init_Vhs = jnp.zeros((T + 1, nh)).at[0, :].set(T_Vhs[T, :])
    init_carry = (init_Vhs, init_gae_coeffs, jnp.zeros(nh, dtype=int))

    ts = jnp.arange(T)[::-1]
    inps = (ts, T_hs[:-1], T_Vhs[1:], done)

    _, Qhs_GAEs = jax.lax.scan(loop, init_carry, inps, reverse=True)
    return Qhs_GAEs - T_Vhs[:-1], Qhs_GAEs

@partial(jax.jit)
def calculate_gae_avoid4(
    gamma: float,
    gae_lambda: float,
    T_hs: jnp.ndarray,
    T_Vhs: jnp.ndarray,
    done: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    Tp1, nh = T_hs.shape
    T = Tp1 - 1

    def loop(carry, inp):
        ii, hs, Vhs, done_row = inp
        next_Vhs_row, gae_coeffs, pre_done_row = carry

        # Update GAE coeffs. [1] -> [1, λ/(1-λ)] -> [1 λ λ²/(1-λ)] -> [1 λ λ² λ³/(1-λ)]
        gae_coeffs = (jnp.roll(gae_coeffs, 1, axis=0) * gae_lambda * (1 - pre_done_row) +
                      jnp.roll(gae_coeffs, 1, axis=0) * (gae_lambda / (1 - gae_lambda)) * pre_done_row) * (1 - done_row)
        gae_coeffs = gae_coeffs.at[0, :].set(1.0)

        mask = jnp.arange(T + 1) < ii + 1
        mask_h = mask[:, None]

        # DP for Vh.
        done_row_processed = jnp.where(jnp.isnan(done_row * jnp.inf), 0, done_row * -jnp.inf) # MAXIMUM NEEDS -INF
        disc_to_h = (1 - gamma) * hs + gamma * (next_Vhs_row + done_row_processed)

        Vhs_row = jnp.maximum(hs, disc_to_h) # AVOID IS MAXIMUM
        Vhs_row = mask_h * Vhs_row

        normed_gae_coeffs = gae_coeffs / jnp.sum(gae_coeffs, axis=0)
        Qhs_GAE = jnp.sum(Vhs_row * normed_gae_coeffs, axis=0)

        # Setup Vs_row for next timestep.
        Vhs_row = jnp.roll(Vhs_row, 1, axis=0)
        Vhs_row = Vhs_row.at[0, :].set(Vhs)

        return (Vhs_row, gae_coeffs, done_row), Qhs_GAE

    done = jnp.array(done, dtype=int)
    init_gae_coeffs = jnp.zeros((T + 1, nh))

    init_Vhs = jnp.zeros((T + 1, nh)).at[0, :].set(T_Vhs[T, :])
    init_carry = (init_Vhs, init_gae_coeffs, jnp.zeros(nh, dtype=int))

    ts = jnp.arange(T)[::-1]
    inps = (ts, T_hs[:-1], T_Vhs[1:], done)

    _, Qhs_GAEs = jax.lax.scan(loop, init_carry, inps, reverse=True)
    return Qhs_GAEs - T_Vhs[:-1], Qhs_GAEs

@partial(jax.jit)
def calculate_gae_reachavoid4(
    gamma: float,
    gae_lambda: float,
    T_ls: jnp.ndarray,
    T_gs: jnp.ndarray,
    T_Vs: jnp.ndarray,
    done: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    Tp1, nh = T_ls.shape
    T = Tp1 - 1

    def loop(carry, inp):
        ii, ls, gs, Vs, done_row = inp
        next_Vs_row, gae_coeffs, pre_done_row = carry

        # Update GAE coeffs. [1] -> [1, λ/(1-λ)] -> [1 λ λ²/(1-λ)] -> [1 λ λ² λ³/(1-λ)]
        gae_coeffs = (jnp.roll(gae_coeffs, 1, axis=0) * gae_lambda * (1 - pre_done_row) +
                      jnp.roll(gae_coeffs, 1, axis=0) * (gae_lambda / (1 - gae_lambda)) * pre_done_row) * (1 - done_row)
        gae_coeffs = gae_coeffs.at[0, :].set(1.0)

        mask = jnp.arange(T + 1) < ii + 1
        mask_lg = mask[:, None]

        # DP for Vh.
        done_row_processed = jnp.where(jnp.isnan(done_row * jnp.inf), 0, done_row * jnp.inf)

        # disc_to_h = (1 - gamma) * hs + gamma * (next_Vhs_row + done_row_processed) # OSWIN BRT backup
        # Vhs_row = jnp.minimum(hs, disc_to_h)
        # Vhs_row = mask_h * Vhs_row

        Vs_row = (1 - gamma) * jnp.maximum(ls, gs) + gamma * jnp.maximum(jnp.minimum(ls, next_Vs_row + done_row_processed), gs)
        Vs_row = mask_lg * Vs_row

        normed_gae_coeffs = gae_coeffs / jnp.sum(gae_coeffs, axis=0)
        Qhs_GAE = jnp.sum(Vs_row * normed_gae_coeffs, axis=0)

        # Setup Vs_row for next timestep.
        Vs_row = jnp.roll(Vs_row, 1, axis=0)
        Vs_row = Vs_row.at[0, :].set(Vs)

        return (Vs_row, gae_coeffs, done_row), Qhs_GAE

    done = jnp.array(done, dtype=int)
    init_gae_coeffs = jnp.zeros((T + 1, nh))

    init_Vs = jnp.zeros((T + 1, nh)).at[0, :].set(T_Vs[T, :])
    init_carry = (init_Vs, init_gae_coeffs, jnp.zeros(nh, dtype=int))

    ts = jnp.arange(T)[::-1]
    inps = (ts, T_ls[:-1], T_gs[:-1], T_Vs[1:], done)

    _, Qhs_GAEs = jax.lax.scan(loop, init_carry, inps, reverse=True)
    return Qhs_GAEs - T_Vs[:-1], Qhs_GAEs


@partial(jax.jit)
def calculate_indexs(
    gamma: float,
    reward: jnp.ndarray,
    energy: jnp.ndarray,
    T_hs: jnp.ndarray,
    T_Vhs: jnp.ndarray,
    T_Vs: jnp.ndarray,
) -> jnp.ndarray:

    Tp1, nh = T_Vs.shape
    T = Tp1 - 1

    def loop(carry, inp):
        ii, reach, reward, Vhs, Vs = inp
        (next_Vs_row, next_Vhs_row, next_mask) = carry

        # DP for Vh and reward.
        disc_to_h = (1 - gamma) * reach + gamma * next_Vhs_row
        Vhs_row = next_mask * jnp.minimum(reach, disc_to_h)
        Vs_row = next_mask * (reward + gamma * next_Vs_row)
        index = jnp.argmin(jnp.maximum(Vs_row - energy[-ii-1, :], Vhs_row), axis=0)

        # Setup Vs_row for next timestep.
        Vhs_row = Vhs_row.at[ii + 1, :].set(Vhs)
        Vs_row = Vs_row.at[ii + 1, :].set(Vs)
        next_mask = jnp.roll(next_mask, 1)
        next_mask = next_mask.at[0, :].set(1.)

        return (Vs_row, Vhs_row, next_mask), index

    ts = jnp.arange(T)[::-1]
    init_Vs = jnp.ones((T + 1, nh)) * jnp.inf
    init_Vhs = jnp.ones((T + 1, nh)) * jnp.inf
    init_mask = jnp.ones((T + 1, 1)) * jnp.inf
    init_Vs = init_Vs.at[0, :].set(T_Vs[T, :])
    init_Vhs = init_Vhs.at[0, :].set(T_Vhs[T, :])
    init_mask = init_mask.at[0, :].set(1.)
    init_carry = (init_Vs, init_Vhs, init_mask)
    inps = (ts, T_hs[:-1], reward, T_Vhs[:-1], T_Vs[1:])
    _, indexs = jax.lax.scan(loop, init_carry, inps, reverse=True)

    return indexs

@partial(jax.jit)
def calculate_indexs2(
    gamma: float,
    reward: jnp.ndarray,
    energy: jnp.ndarray,
    T_hs: jnp.ndarray,
) -> jnp.ndarray:

    Tp1, nh = T_hs.shape
    T = Tp1 - 1

    def loop(carry, inp):
        ii, reach, reward = inp
        (next_Vs_row, next_Vhs_row, next_mask_1, done) = carry

        Vs_row = next_mask_1 * (reward + gamma * next_Vs_row)
        Vs_row = Vs_row.at[ii, :].set(0)

        Vhs_row = next_Vhs_row.at[ii, :].set(reach)

        V_total = jnp.maximum(Vs_row - energy[-ii-1, :], Vhs_row)[::-1]

        index = jnp.argmin(V_total, axis=0)

        done = done.at[index, jnp.arange(nh)].set(1.0)

        next_mask_1 = jnp.roll(next_mask_1, 1)
        next_mask_1 = next_mask_1.at[0, :].set(1.)

        return (Vs_row, Vhs_row, next_mask_1, done), index

    ts = jnp.arange(T)[::-1]
    init_Vs = jnp.ones((T, nh)) * jnp.inf
    init_Vhs = jnp.ones((T, nh)) * jnp.inf
    done = jnp.full((T, nh), 0.)
    init_mask_1 = jnp.ones((T, 1)) * jnp.inf
    init_carry = (init_Vs, init_Vhs, init_mask_1, done)
    inps = (ts, T_hs[:-1], reward)
    end, indexs = jax.lax.scan(loop, init_carry, inps, reverse=True)

    return indexs, end[3]

@partial(jax.jit)
def calculate_indexs3(
    gamma: float,
    reward: jnp.ndarray,
    energy: jnp.ndarray,
    T_hs: jnp.ndarray,
    last_value: jnp.ndarray,
    last_value_reach: jnp.ndarray,
) -> jnp.ndarray:

    Tp1, nh = T_hs.shape
    T = Tp1 - 1

    def loop(carry, inp):
        ii, reach, reward = inp
        (next_Vs_row, next_Vhs_row, next_mask_1, done) = carry

        Vs_row = next_mask_1 * (reward + gamma * next_Vs_row)
        Vs_row = Vs_row.at[ii, :].set(0)

        Vhs_row = next_Vhs_row.at[ii, :].set(reach)

        V_total = jnp.maximum(Vs_row - energy[-ii-1, :], Vhs_row)[::-1]
        V_next = jnp.maximum(jnp.power(gamma, ii) * last_value + V_total[-1, :] - energy[-ii-1, :], last_value_reach)
        V_total_1 = jnp.concatenate((V_total, V_next))

        index_1 = jnp.argmin(V_total_1, axis=0)
        done = done.at[index_1, jnp.arange(nh)].set(1.0)

        next_mask_1 = jnp.roll(next_mask_1, 1)
        next_mask_1 = next_mask_1.at[0, :].set(1.)

        return (Vs_row, Vhs_row, next_mask_1, done), index_1

    ts = jnp.arange(T)[::-1]
    init_Vs = jnp.ones((T, nh)) * jnp.inf
    init_Vhs = jnp.ones((T, nh)) * jnp.inf
    done = jnp.full((T+1, nh), 0.)
    init_mask_1 = jnp.ones((T, 1)) * jnp.inf
    init_carry = (init_Vs, init_Vhs, init_mask_1, done)
    inps = (ts, T_hs[:-1], reward)
    end, indexs = jax.lax.scan(loop, init_carry, inps, reverse=True)

    return indexs, end[3]

@partial(jax.jit)
def calculate_indexs3_rr(
    gamma: float,
    reward: jnp.ndarray,
    T_hs: jnp.ndarray,
    last_value: jnp.ndarray,
) -> jnp.ndarray:

    Tp1, nh = T_hs.shape
    T = Tp1 - 1

    def loop(carry, inp):
        ii, reach, reward = inp
        (next_Vs_row, next_mask_1, done) = carry

        # Vs_row = (1 - gamma) * reach + gamma * jnp.min(reach, next_Vs_row) # DOESNT WORK W JAX
        # Vs_row = (1 - gamma) * reach + gamma * jnp.min(reach, next_Vs_row) # DOESNT WORK W JAX
        Vs_row = next_mask_1 * (reward + gamma * next_Vs_row)
        Vs_row = Vs_row.at[ii, :].set(reach)
        Vs_row = Vs_row.at[ii, :].set(reach)

        V_total = Vs_row[::-1]
        V_next = jnp.maximum(jnp.power(gamma, ii) * last_value + V_total[-1, :], last_value)
        V_total_1 = jnp.concatenate((V_total, V_next))

        # Vs_row = next_mask_1 * (reward + gamma * next_Vs_row) # OSWINS
        # Vs_row = next_mask_1 * (reward + gamma * next_Vs_row)
        # Vhs_row = Vs_row.at[ii, :].set(reach)

        # # ACCUMULATION VERSION LIKE OSWINS
        # V_total = jnp.maximum(Vs_row, Vhs_row)[::-1]
        # V_next = jnp.maximum(jnp.power(gamma, ii) * last_value + V_total[-1, :], last_value)
        # V_total_1 = jnp.concatenate((V_total, V_next))

        # MIN ACCUMULATION
        # V_total = jnp.minimum(Vs_row, Vhs_row)[::-1]
        # V_next = jnp.minimum(jnp.power(gamma, ii) * last_value + V_total[-1, :], last_value)
        # V_total_1 = jnp.concatenate((V_total, V_next))

        # JUST REACH VERSION
        # V_total = Vhs_row[::-1]
        # Vhs_row = next_Vhs_row.at[ii, :].set(reach)
        # V_total_1 = jnp.concatenate((V_total, last_value))

        index_1 = jnp.argmin(V_total_1, axis=0)
        # done = done.at[index_1, jnp.arange(nh)].set(1.0) # NO DONE

        next_mask_1 = jnp.roll(next_mask_1, 1)
        next_mask_1 = next_mask_1.at[0, :].set(1.)

        return (Vs_row, next_mask_1, done), index_1

    ts = jnp.arange(T)[::-1]
    init_Vs = jnp.ones((T, nh)) * jnp.inf
    done = jnp.full((T+1, nh), 0.)
    init_mask_1 = jnp.ones((T, 1)) * jnp.inf
    init_carry = (init_Vs, init_mask_1, done)
    inps = (ts, T_hs[:-1], reward)
    end, indexs = jax.lax.scan(loop, init_carry, inps, reverse=True)

    return indexs, end[-1]

@partial(jax.jit)
def calculate_indexs_rr(
    gamma: float,
    reward: jnp.ndarray,
    T_hs: jnp.ndarray,
    T_Vs: jnp.ndarray,
) -> jnp.ndarray:

    Tp1, nh = T_Vs.shape
    T = Tp1 - 1

    def loop(carry, inp):
        ii, reach, Vs = inp
        (next_Vs_row, next_mask, done) = carry

        # DP for Vh and reward.
        disc_to_h = (1 - gamma) * reach + gamma * next_Vs_row
        Vs_row = next_mask * jnp.minimum(reach, disc_to_h)
        # Vhs_row = next_mask * jnp.minimum(reach, disc_to_h)
        # Vs_row = next_mask * (reward + gamma * next_Vs_row)
        index = jnp.argmin(Vs_row, axis=0)
        done = done.at[index, jnp.arange(nh)].set(1.0)

        # Setup Vs_row for next timestep.
        # Vhs_row = Vhs_row.at[ii + 1, :].set(Vhs)
        Vs_row = Vs_row.at[ii + 1, :].set(Vs)
        next_mask = jnp.roll(next_mask, 1)
        next_mask = next_mask.at[0, :].set(1.)

        return (Vs_row, next_mask, done), index

    ts = jnp.arange(T)[::-1]
    init_Vs = jnp.ones((T + 1, nh)) * jnp.inf
    # init_Vhs = jnp.ones((T + 1, nh)) * jnp.inf
    init_mask = jnp.ones((T + 1, 1)) * jnp.inf
    init_Vs = init_Vs.at[0, :].set(T_Vs[T, :])
    # init_Vhs = init_Vhs.at[0, :].set(T_Vhs[T, :])
    init_mask = init_mask.at[0, :].set(1.)
    done = jnp.full((T+1, nh), 0.)
    init_carry = (init_Vs, init_mask, done)
    inps = (ts, T_hs[:-1], T_Vs[1:])
    end, indexs = jax.lax.scan(loop, init_carry, inps, reverse=True)

    return indexs, end[-1]

@partial(jax.jit)
def calculate_advantage2(
    gae_nval_gamma_lambda: Tuple[jnp.ndarray, jnp.ndarray, float, float],
    inp
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, float, float], jnp.array]:

    gae, next_value, Gamma, Lambda = gae_nval_gamma_lambda
    transition, done, next_done = inp
    reward = transition.reward
    value = transition.value
    delta = (reward + Gamma * next_value * (1 - next_done)) * (1 - done) - value
    gae = delta + Gamma * Lambda * (1 - next_done) * (1 - done) * gae
    return (gae, value, Gamma, Lambda), gae


@partial(jax.jit)
def calculate_gae2(
    gamma: float,
    gae_lambda: float,
    trajectory_batch,
    done: jnp.ndarray,
    last_value: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    next_done = jnp.roll(done, -1, axis=0)
    next_done = next_done.at[-1, :].set(next_done[-2, :])
    _, advantages = jax.lax.scan(
        calculate_advantage2,
        (jnp.zeros_like(last_value), last_value, gamma, gae_lambda),
        (trajectory_batch, done, next_done),
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + trajectory_batch.value


@partial(jax.jit)
def calculate_gae3(
    gamma: float,
    gae_lambda: float,
    value_append: jnp.ndarray,
    rewards: jnp.ndarray,
    index: jnp.ndarray,
    last_value: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    T, nh = rewards.shape

    def loop(carry, inp):
        ii, value, reward = inp
        (adv, l_value) = carry

        is_one_step = jnp.where((index_new - 1 == ii), jnp.zeros_like(rewards), jnp.ones_like(rewards))
        delta = reward + gamma * l_value * is_one_step - value
        new_adv = delta + gamma * gae_lambda * is_one_step * adv
        new_adv = jnp.where((index_new - 1 >= ii) * (index_prev_2 <= ii), new_adv, adv)

        return (new_adv, value), None

    index_new = jnp.where(index == T, T+1, index)
    index_prev_1 = jnp.repeat(jnp.arange(nh).reshape(1, nh), T, axis=0)
    index_prev_2 = jnp.repeat(jnp.arange(T).reshape(T, 1), nh, axis=1)
    advantages = -value_append[index, index_prev_1]
    advantages = jnp.where(index == T, 0, advantages)
    inps = (jnp.arange(T), value_append[:-1, :], rewards)
    init_carry = (advantages, last_value)
    end, _ = jax.lax.scan(loop, init_carry, inps, reverse=True)

    return end[0], end[0] + value_append[:-1, :]
