import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct
import numpy as np
from jax_f16.f16 import F16

@struct.dataclass
class EnvState:
    state: jax.Array = struct.field(default_factory=jax.Array)
    time: int = 0
    energy: float = 0.
    reach: float = 0.
    avoid: int = 0.

@struct.dataclass
class EnvParams:
    min_energy: float = -400.0
    max_energy: float = 800.0
    max_steps_in_episode: int = 5000

def rotz(psi):
    c, s = jnp.cos(psi), jnp.sin(psi)
    return jnp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def roty(theta):
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotx(phi):
    c, s = jnp.cos(phi), jnp.sin(phi)
    return jnp.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def compute_f16_vel_angles(state):
    """Compute cos / sin of [gamma, sigma], the pitch & yaw of the velocity vector."""
    assert state.shape == (F16.NX,)
    # 1: Compute {}^{W}R^{F16}.
    R_W_F16 = rotz(state[F16.PSI]) @ roty(state[F16.THETA]) @ rotx(state[F16.PHI])
    assert R_W_F16.shape == (3, 3)

    # 2: Compute v_{F16}
    ca, sa = jnp.cos(state[F16.ALPHA]), jnp.sin(state[F16.ALPHA])
    cb, sb = jnp.cos(state[F16.BETA]), jnp.sin(state[F16.BETA])
    v_F16 = jnp.array([ca * cb, sb, sa * cb])

    # 3: Compute v_{W}
    v_W = R_W_F16 @ v_F16
    assert v_W.shape == (3,)

    # 4: Back out cos and sin of gamma and sigma.
    cos_sigma = v_W[0]
    sin_sigma = v_W[1]
    sin_gamma = v_W[2]

    out = jnp.array([cos_sigma, sin_sigma, sin_gamma])
    assert out.shape == (3,)
    return out


class F16Avoid(environment.Environment):

    def __init__(self):
        super().__init__()
        self.obs_shape = (26,)
        self.NX = F16.NX
        self.NU = F16.NU

        (self.VT, self.ALPHA, self.BETA,
         self.PHI, self.THETA, self.PSI,
         self.P, self.Q, self.R,
         self.PN, self.PE, self.H,
         self.POW, self.NZINT, self.PSINT,
         self.NYRINT, self.FREEZE) = range(self.NX+1)

        self.NZ, self.PS, self.NYR, self.THRTL = range(self.NU)
        self.MORELLI_BOUNDS = np.array(
            [
                [-np.inf, np.inf],  # vt
                # [_d2r(-10), _d2r(45)],  # alpha (rad)
                [-0.17453292519943295, 0.7853981633974483],  # alpha (rad)
                # [_d2r(-30), _d2r(30)],  # beta (rad)
                [-0.5235987755982988, 0.5235987755982988],  # beta (rad)
                [-np.inf, np.inf],  # roll (rad)
                [-np.inf, np.inf],  # pitch (rad)
                [-np.inf, np.inf],  # yaw (rad)
                [-np.inf, np.inf],  # P
                [-np.inf, np.inf],  # Q
                [-np.inf, np.inf],  # R
                [-np.inf, np.inf],  # north pos
                [-np.inf, np.inf],  # east pos
                [-np.inf, np.inf],  # altitude
                [-np.inf, np.inf],  # engine thrust dynamics lag state
                [-np.inf, np.inf],  # Nz integrator
                [-np.inf, np.inf],  # Ps integrator
                [-np.inf, np.inf],  # Ny+R integrator
            ]
        )

        self._dt = 0.05
        self._env = F16()

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for Environment."""
        default = EnvParams()
        return default

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: chex.Array,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:

        assert state.state.shape == (self.NX,)
        assert action.shape == (self.NU,)

        control_raw = jnp.tanh(action)
        control = control_raw * jnp.array([10., 10., 10., 0.5]) + jnp.array([0., 0., 0., 0.5])

        h_1 = self._env.xdot(state.state, control)
        h_2 = self._env.xdot(state.state + h_1 * self._dt / 2, control)
        h_3 = self._env.xdot(state.state + h_2 * self._dt / 2, control)
        h_4 = self._env.xdot(state.state + h_3 * self._dt, control)
        a_state_new = state.state + (h_1 + 2 * h_2 + 2 * h_3 + h_4) * self._dt / 6

        state_high = jnp.full(16, jnp.inf)
        state_low = jnp.full(16, -jnp.inf)
        state_high = state_high.at[self.ALPHA].set(self.MORELLI_BOUNDS[self.ALPHA, 1])
        state_low = state_low.at[self.ALPHA].set(self.MORELLI_BOUNDS[self.ALPHA, 0])
        state_high = state_high.at[self.BETA].set(self.MORELLI_BOUNDS[self.BETA, 1])
        state_low = state_low.at[self.BETA].set(self.MORELLI_BOUNDS[self.BETA, 0])
        state_high = state_high.at[self.THETA].set(jnp.pi / 3)
        state_low = state_low.at[self.THETA].set(-jnp.pi / 3)


        state_high = state_high.at[self.P].set(10.0)
        state_low = state_low.at[self.P].set(-10.0)

        a_state_new = jnp.clip(a_state_new, state_low, state_high)

        is_avoid = self.is_avoid(a_state_new, params)
        avoid_value = jnp.where(is_avoid, -1, state.avoid)
        reach_value = self.is_reach(a_state_new, avoid_value, params)
        a_state_new = jnp.where(avoid_value == 1, a_state_new, state.state)

        next_energy = jnp.clip(state.energy - 4 * jnp.sum(control_raw ** 2), params.min_energy, params.max_energy)
        next_state_new = EnvState(state=a_state_new, time=state.time + 1, energy=next_energy,
                                  reach=reach_value, avoid=avoid_value)
        done = self.is_terminal(next_state_new, params)

        return (
            lax.stop_gradient(self.get_obs(next_state_new)),
            lax.stop_gradient(next_state_new),
            4 * jnp.sum(control_raw ** 2),
            done,
            {"pos_y": state.state[self.PE],
             "pos_x": state.state[self.PN],
             "height": state.state[self.H],
             "state": state.state
             },
        )


    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:

        _MAX_ALT_SAMPLE = 800.0
        bounds = np.array(
            [
                (150.0, 550.0),  # vt
                (-0.17453292519943295 / 10, 0.7853981633974483 / 10),  # alpha (rad)
                (-0.5235987755982988 / 10, 0.5235987755982988 / 10),  # beta (rad)
                (-np.pi / 4, np.pi / 4),  # phi roll
                (-1.0, 1.0),  # theta pitch
                (-1e-4, 1e-4),  # psi yaw
                (-0.5, 0.5),  # P
                (-0.5, 0.5),  # Q
                (-0.5, 0.5),  # R
                (0.0, 1900.0),  # pos_n
                (-200.0, 200.0),  # pos_e
                (300.0, _MAX_ALT_SAMPLE),  # alt.
                (0.0, 10.0),  # power. Consider sampling wider range.
                (-2.0, 2.0),  # nz_int
                (-2.0, 2.0),  # ps_int
                (-2.0, 2.0),  # nyr_int
            ]
        )

        state = jax.random.uniform(key, shape=(16,), minval=bounds[:, 0], maxval=bounds[:, 1])
        is_avoid = self.is_avoid(state, params)
        avoid_value = jnp.where(is_avoid, -1, 1)
        reach_value = self.is_reach(state, avoid_value, params)
        init_energy = jax.random.uniform(
            key, minval=0., maxval=params.max_energy
        )
        state = EnvState(state=state, time=0, energy=init_energy, reach=reach_value, avoid=avoid_value)
        return self.get_obs(state), state

    def is_reach(self, state, avoid_value, params: EnvParams) -> float:
        """Check the reach value of the current state"""
        has_reached_goal = jnp.fabs(state[self.PN] - 2000.) < 25.
        reach = (jnp.fabs(state[self.PN] - 2000.) - 25.) / 5.
        value = jnp.where(has_reached_goal, -300., reach)
        is_avoid = (avoid_value == -1)
        value = jnp.where(is_avoid, 800.0, value)
        return value

    def is_avoid(self, state, params: EnvParams):

        alt_valid = (0. <= state[self.H]) & (state[self.H] <= 1100.0)
        pe_valid = (-200.0 <= state[self.PE]) & (state[self.PE] <= 200.0)
        # avoid_1 = (-200.0 <= state[self.PE]) & (state[self.PE] <= -50.0) & (jnp.fabs(state[self.PN] - 500.) <= 25.)
        # avoid_2 = (0.0 <= state[self.PE]) & (state[self.PE] <= 200.0) & (jnp.fabs(state[self.PN] - 1000.) <= 25.)
        # avoid_3 = (-200.0 <= state[self.PE]) & (state[self.PE] <= -50.0) & (jnp.fabs(state[self.PN] - 1500.) <= 25.)

        return jnp.logical_not(alt_valid & pe_valid)

    def get_obs(self, state: EnvState) -> chex.Array:
        """Return angle in polar coordinates and change."""

        state_obs = state.state

        # Learn position-invariant policy.
        # state_obs = state_obs.at[self.PN].set(0.0)

        # sin-cos encode angles.
        with jax.ensure_compile_time_eval():
            angle_idxs = jnp.array([F16.ALPHA, F16.BETA, F16.PHI, F16.THETA, F16.PSI])
            other_idxs = jnp.setdiff1d(jnp.arange(self.NX), angle_idxs)

        angles = state_obs[angle_idxs]
        other = state_obs[other_idxs]

        angles_enc = jnp.concatenate([jnp.cos(angles), jnp.sin(angles)], axis=0)
        state_enc = jnp.concatenate([other, angles_enc], axis=0)
        assert state_enc.shape == (self.NX + len(angle_idxs),)

        # Add extra features.
        vel_feats = compute_f16_vel_angles(state.state[:F16.NX])
        assert vel_feats.shape == (3,)
        state_enc = jnp.concatenate([state_enc, vel_feats], axis=0)

        # fmt: off
        obs_mean = np.array([3.4e+02, -1.7e-01, 2.9e-01, 1.0e-01, 1.0e+03, 1.0e-01, 3.1e+02, 12.0e+00,
                             3.0e-02, 2.1e-02, 2.4e00, 8.7e-01, 8.9e-01, 7.7e-01, 8.0e-01,
                             7.6e-01, 3.6e-01, -1.3e-02, -7.6e-03, -3.5e-01, -1.4e-03, 5.9e-01, -3.6e-03,
                             5.4e-01])
        obs_std = np.array([1.1e+02, 1.7e+00, 6.3e-01, 3.2e+00, 1.0e+03, 1.3e+02, 2.2e+02, 5.0e+00,
                            1.9e+00, 1.5e+00, 4.5e+00, 1.4e-01, 9.6e-02, 2.6e-01, 2.2e-01,
                            3.7e-01, 3.1e-01, 4.4e-01, 5.8e-01, 4.4e-01, 5.4e-01, 3.5e-01, 3.1e-01,
                            3.8e-01])
        # fmt: on

        state_enc = (state_enc - obs_mean) / obs_std

        # For better stability, clip the state_enc to not be too large.
        state_enc = jnp.clip(state_enc, -10.0, 10.0)

        obs = jnp.concatenate([state_enc, jnp.array([state.avoid, state.energy])]).squeeze()
        return obs

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = state.time >= params.max_steps_in_episode
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "F16Avoid"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 4

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        high = jnp.ones(26) * jnp.finfo(jnp.float32).max
        return spaces.Box(-high, high, shape=(26,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "state": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (16, ),
                    jnp.float32,
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
                "energy": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "reach": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
            }
        )