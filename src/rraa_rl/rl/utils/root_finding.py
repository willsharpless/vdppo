from typing import NamedTuple

import jax.lax as lax
import jax.numpy as jnp

class BisectionState(NamedTuple):
    lo: jnp.ndarray
    hi: jnp.ndarray
    # -1: decreasing, +1: increasing, 0: root not in [lower, upper]
    sign: jnp.ndarray

class Bisection:

    def __init__(self, train_state_energy, train_state_h, variance, mean, obsv, threshold, n_iters):
        self.train_state_energy = train_state_energy
        self.train_state_h = train_state_h
        self.variance = variance
        self.mean = mean
        self.obsv = obsv
        self.n_iters = n_iters
        self.threshold = threshold

    def init_state(self, lb, ub) -> BisectionState:
        xlb, xub = jnp.ones(self.obsv.shape[0]) * lb, jnp.ones(self.obsv.shape[0]) * ub
        obsv_lb = self.obsv.at[:, -1].set(xlb)
        obsv_ub = self.obsv.at[:, -1].set(xub)

        ylb = jnp.maximum(self.train_state_energy.apply_fn(self.train_state_energy.params, obsv_lb) - (xlb * self.variance + self.mean),
                          self.train_state_h.apply_fn(self.train_state_h.params, obsv_lb) + self.threshold)

        yub = jnp.maximum(self.train_state_energy.apply_fn(self.train_state_energy.params, obsv_ub) - (xub * self.variance + self.mean),
                          self.train_state_h.apply_fn(self.train_state_h.params, obsv_ub) + self.threshold)

        is_incr = (ylb < 0) & (yub >= 0)
        is_decr = (ylb > 0) & (yub <= 0)
        sign = jnp.where(is_incr, 1, jnp.where(is_decr, -1, 0))
        return BisectionState(xlb, xub, sign)

    def update_step(self, state: BisectionState, _):
        x = 0.5 * (state.lo + state.hi)
        obsv = self.obsv.at[:, -1].set(x)
        y = jnp.maximum(self.train_state_energy.apply_fn(self.train_state_energy.params, obsv) - (x * self.variance + self.mean),
                          self.train_state_h.apply_fn(self.train_state_h.params, obsv) + self.threshold)
        # If x is too large, then bound becomes [lo, x].
        #                    else bound becomes [x, hi]
        too_large = state.sign * y > 0

        hi = jnp.where(too_large, x, state.hi)
        lo = jnp.where(too_large, state.lo, x)

        return BisectionState(lo, hi, state.sign), (x, y, None)

    def _update(self, state: BisectionState, _):
        new_state, (x, y, _) = self.update_step(state, _)
        return new_state, _

    def refine_output(self, state: BisectionState):
        return 0.5 * (state.lo + state.hi)

    def run(self):
        init_state = self.init_state(-1., 1.)
        final_state, _ = lax.scan(self._update, init_state, None, length=self.n_iters)
        best_x = self.refine_output(final_state)
        return best_x, None

    def run_detailed(self):
        init_state = self.init_state(-1., 1.)
        final_state, outputs = lax.scan(self.update_step, init_state, None, length=self.n_iters)
        best_x = self.refine_output(final_state)
        return best_x, outputs