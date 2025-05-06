import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct
import control as ct


data_1 = jnp.load("./rraa_rl/EFPPO/src/env/wind_field_1.npz")
data_2 = jnp.load("./rraa_rl/EFPPO/src/env/wind_field_2.npz")

@struct.dataclass
class EnvState:
    state: jax.Array = struct.field(default_factory=jax.Array)
    time: int = 0
    reach: float = 0.
    avoid: int = 0
    cost: float = 0.

@struct.dataclass
class EnvParams:
    gamma: float = 0.99
    max_thrust: float = 1.0
    thrust_limit: float = 0.1
    max_steps_in_episode: int = 5000
    index: int = 0
    u_air: jax.Array = struct.field(default_factory=jax.Array)
    v_air: jax.Array = struct.field(default_factory=jax.Array)
    obstacle: jax.Array = struct.field(default_factory=jax.Array)

class WindFieldBaseline(environment.Environment):

    def __init__(self):
        super().__init__()
        self.obs_shape = (13,)

        # [ px py pz | ψ θ ϕ | u v w | r q p ]
        self.AGENT_NX = 12
        # [ f1 f2 f3 f4 ]
        self.AGENT_NU = 4
        self.POS_DIM = 3

        (self.PX, self.PY, self.PZ,
         self.PSI, self.THETA, self.PHI,
         self.U, self.V, self.W,
         self.R, self.Q, self.P) = range(self.AGENT_NX)

        self._dt = 0.02
        self.radius = 0.05

        self.m = 0.0299
        self.Ixx = 1.395 * 10 ** (-5)
        self.Iyy = 1.395 * 10 ** (-5)
        self.Izz = 2.173 * 10 ** (-5)
        self.CT = 3.1582 * 10 ** (-10)
        self.CD = 7.9379 * 10 ** (-12)
        self.d = 0.03973
        # self.u_scale = 1e-2
        self.scale = 0.2

        self.max_linvel = 3.0
        self.max_angvel = 3.0

        self.max_linvel_sample = 0.1
        self.max_angvel_sample = 0.1
        self.default_area_size = 2
        self.pol_goal_max_dist = 1.0
        self.velnorm_thresh = 0.1

        self.K_nom = jnp.array(
            [
                [-0.3895, 0.8958, 1.5283, -0.1888, -1.3328, -3.1956, -0.3749, 0.8761, 1.682, -0.5776, -0.2311, -0.1978],
                [0.3895, -0.8958, 1.5283, -0.1888, 1.3328, 3.1956, 0.3749, -0.8761, 1.682, 0.5776, 0.2311, -0.1978],
                [0.3895, 0.8958, 1.5283, 0.1888, 1.3328, -3.1956, 0.3749, 0.8761, 1.682, -0.5776, 0.2311, 0.1978],
                [-0.3895, -0.8958, 1.5283, 0.1888, -1.3328, 3.1956, -0.3749, -0.8761, 1.682, 0.5776, -0.2311, 0.1978],
            ]
        )
        self.K_ll = jnp.array(
            [
                [-0.1236, -0.1236, -0.0022, -0.0040, -0.0040, -0.0022, -0.0020, 0.0020, 0.1494],
                [-0.1236, 0.1236, 0.0022, -0.0040, 0.0040, 0.0022, 0.0020, 0.0020, 0.1494],
                [0.1236, 0.1236, -0.0022, 0.0040, 0.0040, -0.0022, 0.0020, -0.0020, 0.1494],
                [0.1236, -0.1235, 0.0022, 0.0040, -0.0040, 0.0022, -0.0020, -0.0020, 0.1494],
            ]
        )
        self.K = jnp.array(self._compute_K_nom())

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for Environment."""
        default = EnvParams(u_air=data_2['u_air'], v_air=data_2['v_air'], obstacle=data_2['obs'])
        return default

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: chex.Array,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:

        assert (action.shape[0] == 3)

        control = jnp.tanh(action)

        goal = control * 1.5 + state.state[0:3]

        control_real_1 = self.u_ref(state.state, goal)
        h_1 = self.f_single_change(state.state, control_real_1)
        control_real_2 = self.u_ref(state.state + h_1 * self._dt / 2, goal)
        h_2 = self.f_single_change(state.state + h_1 * self._dt / 2, control_real_2)
        control_real_3 = self.u_ref(state.state + h_2 * self._dt / 2, goal)
        h_3 = self.f_single_change(state.state + h_2 * self._dt / 2, control_real_3)
        control_real_4 = self.u_ref(state.state + h_3 * self._dt, goal)
        h_4 = self.f_single_change(state.state + h_3 * self._dt, control_real_4)
        a_state_new = state.state + (h_1 + 2 * h_2 + 2 * h_3 + h_4) * self._dt / 6

        x_index = params.index % 2
        y_index = params.index // 2
        pos_x = state.state[0]
        pos_y = state.state[1]
        ind_x = ((pos_x * 5. + 15.) * 255. / 60.).astype(int)
        ind_y = ((pos_y * 5. + 15.) * 255. / 60.).astype(int)

        a_state_new = a_state_new.at[6].add(jnp.array(params.u_air)[ind_y + y_index * 128, ind_x + x_index * 128]
                                            * self._dt * self.scale)
        a_state_new = a_state_new.at[7].add(jnp.array(params.v_air)[ind_y + y_index * 128, ind_x + x_index * 128]
                                            * self._dt * self.scale)

        # Clips theta, phi to [-pi/6, pi/6]. Clip uvw to [-0.5, 0.5] and pqr to [-1.0, 1.0]
        pos_lo = jnp.array([-jnp.inf, -jnp.inf, -2.])
        pos_hi = jnp.array([jnp.inf, jnp.inf, 2.])

        # [ psi theta phi ]
        euler_lo = jnp.array([-jnp.pi / 3, -jnp.pi / 3, -jnp.pi / 3])
        euler_hi = jnp.array([jnp.pi / 3, jnp.pi / 3, jnp.pi / 3])

        linvel_lo = jnp.full(3, -self.max_linvel)
        linvel_hi = jnp.full(3, self.max_linvel)

        angvel_lo = jnp.full(3, -self.max_angvel)
        angvel_hi = jnp.full(3, self.max_angvel)

        # Only clip velocities.
        x_lo = jnp.concatenate([pos_lo, euler_lo, linvel_lo, angvel_lo])
        x_hi = jnp.concatenate([pos_hi, euler_hi, linvel_hi, angvel_hi])
        a_state_new = jnp.clip(a_state_new, x_lo, x_hi)

        is_avoid = self.is_avoid(a_state_new, params)
        avoid_value = jnp.where(is_avoid, -1, 1)
        reach_value = self.is_reach(a_state_new, params)

        reward = (
            params.gamma * reach_value - state.reach
        )
        reward = reward.squeeze()

        energy_consumption = (
            jnp.sum(control ** 2) / 2.5
        )

        next_state_new = EnvState(state=a_state_new, time=state.time + 1, reach=reach_value,
                                  avoid=avoid_value, cost=energy_consumption)
        done = self.is_terminal(next_state_new, params)

        return (
            lax.stop_gradient(self.get_obs(next_state_new)),
            lax.stop_gradient(next_state_new),
            reward,
            done,
            {"pos": state.state[0: 3] * 5.,
             "control": control_real_1,
             "action": action,
             "state": state.state,
             "goal": goal,
             },
        )


    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:

        pos_hi = jnp.array([14. / 5., 14. / 5., 2 / 5.])
        pos_low = jnp.array([-14. / 5., -14. / 5., -2 / 5.])
        euler_hi = jnp.array([jnp.pi / 20., jnp.pi / 20., jnp.pi / 20.])
        linvel_hi = jnp.full(3, self.max_linvel_sample)
        angvel_hi = jnp.full(3, self.max_angvel_sample)

        high = jnp.concatenate([pos_hi, euler_hi, linvel_hi, angvel_hi])
        low = jnp.concatenate([pos_low, -euler_hi, -linvel_hi, -angvel_hi])

        state = jax.random.uniform(key, shape=(12,), minval=low, maxval=high)
        is_avoid = self.is_avoid(state, params)
        avoid_value = jnp.where(is_avoid, -1, 1)
        reach_value = self.is_reach(state, params)
        state = EnvState(state=state, time=0, reach=reach_value, avoid=avoid_value, cost=0.)
        return self.get_obs(state), state

    def is_reach(self, obs, params: EnvParams) -> float:
        """Check the reach value of the current state"""
        x_index = params.index % 2
        y_index = params.index // 2
        reach = jnp.sqrt((obs[0] * 5. + 30. * x_index - 15.) ** 2 + (obs[1] * 5. + 30. * y_index - 15.) ** 2)
        has_reached_goal = jnp.maximum(jnp.fabs(obs[0] * 5. + 30. * x_index - 15.), jnp.fabs(obs[1] * 5. + 30. * y_index - 15.)) < 4.
        value = jnp.where(has_reached_goal, -30., reach)
        return value * 10.0

    def is_avoid(self, obs, params: EnvParams):
        x_index = params.index % 2
        y_index = params.index // 2
        pos_x = obs[0]
        pos_y = obs[1]
        ind_x = ((pos_x * 5. + 15.) * 255. / 60.).astype(int)
        ind_y = ((pos_y * 5. + 15.) * 255. / 60.).astype(int)
        avoid_obstacle = ((5. * pos_x > -15.) & (5. * pos_y > -15.) & (5. * pos_x < 15.)
                          & (5. * pos_y < 15.) & jnp.array(params.obstacle)[ind_y + y_index * 128, ind_x + x_index * 128])
        return ((5. * pos_x < -15.) | (5. * pos_y < -15.) | (5. * pos_x > 15.) |
                (5. * pos_y > 15.) | avoid_obstacle)

    def u_hover(self):
        # Want u_scale * sum(control) / m == 9.81
        # total_thrust = 9.81 * self.m / self.u_scale
        total_thrust = 9.81 * self.m
        control = jnp.full(4, total_thrust / 4)
        return control

    def u_ref(self, state, target):
        goal_state = jnp.concatenate([target, jnp.zeros(9)], axis=-1)

        # (12, )
        a_err = state - goal_state
        # a_err_max = jnp.abs(a_err / jnp.linalg.norm(a_err, axis=-1, keepdims=True))
        # a_err = jnp.clip(a_err, -a_err_max, a_err_max)

        # (12, ) -> (4, )
        a_lqr = -(self.K @ a_err)

        # a_lqr = a_lqr.clip(-1, 1)

        return a_lqr

    def f_single_change(self, state, acc):

        # control = self.acc_to_thrust(acc) + self.u_hover()

        assert state.shape == (self.AGENT_NX,)
        assert acc.shape == (self.AGENT_NU,)

        I = jnp.array([self.Ixx, self.Iyy, self.Izz])

        psi, theta, phi = state[self.PSI], state[self.THETA], state[self.PHI]
        u, v, w = state[self.U], state[self.V], state[self.W]
        r, q, p = state[self.R], state[self.Q], state[self.P]

        uvw = jnp.array([u, v, w])
        pqr = jnp.array([p, q, r])

        c_phi, s_phi = jnp.cos(phi), jnp.sin(phi)
        c_th, s_th = jnp.cos(theta), jnp.sin(theta)
        t_th = jnp.tan(theta)
        sec_th_safe = jnp.where(c_th > 1e-3, 1 / c_th, 1e3)

        rotz = jnp.array([[jnp.cos(psi), -jnp.sin(psi), 0],
                          [jnp.sin(psi), jnp.cos(psi), 0],
                          [0, 0, 1]])
        roty = jnp.array([[jnp.cos(theta), 0, jnp.sin(theta)],
                          [0, 1, 0],
                          [-jnp.sin(theta), 0, jnp.cos(theta)]])
        rotx = jnp.array([[1, 0, 0],
                          [0, jnp.cos(phi), -jnp.sin(phi)],
                          [0, jnp.sin(phi), jnp.cos(phi)]])
        R_W_cf = rotz @ roty @ rotx

        # Linear velocity.
        v_Wcf_cf = jnp.array([u, v, w])
        v_Wcf_W = R_W_cf @ v_Wcf_cf
        assert v_Wcf_W.shape == (3,)

        # Euler Angle Dynamics.
        mat = jnp.array(
            [
                [1, s_phi * t_th, c_phi * t_th],
                [0, c_phi, -s_phi],
                [0, s_phi * sec_th_safe, c_phi * sec_th_safe],
            ]
        )
        deuler_rpy = mat @ pqr
        deuler_ypr = deuler_rpy[::-1]

        # deuler_ypr = mat @ pqr

        # Body frame linear acceleration. d/dt [ u v w ]
        acc_cf_g = -R_W_cf[2, :] * 9.81
        # thrust = jnp.sum(control) / self.m

        acc_cf = -jnp.cross(pqr, uvw) + acc_cf_g
        acc_cf = acc_cf.at[2].add(acc[0] + 9.81)



        # Body frame angular acceleration.
        pqr_dot = -jnp.cross(pqr, I * pqr) / I

        # dp_du = jnp.sqrt(2) * self.d * jnp.array([-1.0, -1.0, 1.0, 1.0]) / self.Ixx
        # dq_du = jnp.sqrt(2) * self.d * jnp.array([-1.0, 1.0, 1.0, -1.0]) / self.Iyy
        # dr_du = (self.CD / self.CT) * jnp.array([-1.0, 1.0, -1.0, 1.0]) / self.Izz
        # pqr_dot_thrust = jnp.stack([dp_du, dq_du, dr_du], axis=0) @ control
        pqr_dot_thrust = acc[1:4]

        assert pqr_dot_thrust.shape == (3,)
        # pqr_dot = pqr_dot + pqr_dot_thrust * self.u_scale
        pqr_dot = pqr_dot + pqr_dot_thrust

        rpq_dot = pqr_dot[::-1]
        assert pqr_dot.shape == (3,)

        x_dot = jnp.concatenate([v_Wcf_W, deuler_ypr, acc_cf, rpq_dot], axis=0)
        assert x_dot.shape == (self.AGENT_NX,)

        return x_dot

    def get_obs(self, state: EnvState) -> chex.Array:
        """Return angle in polar coordinates and change."""
        obs = state.state.squeeze()
        return obs

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = state.time >= params.max_steps_in_episode
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "WindField"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 3

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Box(
            low=-params.max_thrust,
            high=params.max_thrust,
            shape=(3,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        high = jnp.ones(13) * jnp.finfo(jnp.float32).max
        return spaces.Box(-high, high, shape=(13,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "state": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (12, ),
                    jnp.float32,
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
                "reach": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
            }
        )

    def thrust_to_acc(self, thrust):
        CT, CD = self.CT, self.CD
        CT, CD = 1, CD / CT

        # Try and avoid catastrophic cancellation by doing the sums early
        assert thrust.shape == (4,)
        w_term = jnp.sum(thrust)
        p_term = jnp.sum(thrust * jnp.array([-1.0, -1.0, 1.0, 1.0]))
        q_term = jnp.sum(thrust * jnp.array([-1.0, 1.0, 1.0, -1.0]))
        r_term = jnp.sum(thrust * jnp.array([-1.0, 1.0, -1.0, 1.0]))

        w_dot = CT * w_term / self.m
        p_dot = CT * np.sqrt(2) * self.d * p_term / self.Ixx
        q_dot = CT * np.sqrt(2) * self.d * q_term / self.Iyy
        r_dot = CD * r_term / self.Izz

        return jnp.array([w_dot, p_dot, q_dot, r_dot])

    def acc_to_thrust(self, acc):
        CT, CD = self.CT, self.CD
        CT, CD = 1, CD / CT

        dw, dp, dq, dr = acc
        # Convert to unnormalized acc.
        wterm = dw * self.m / CT
        pterm = dp * self.Ixx / (CT * np.sqrt(2) * self.d)
        qterm = dq * self.Iyy / (CT * np.sqrt(2) * self.d)
        rterm = dr * self.Izz / CD

        # Solve for thrusts.
        u1 = (wterm - pterm - qterm - rterm) / 4
        u2 = (wterm - pterm + qterm + rterm) / 4
        u3 = (wterm + pterm + qterm - rterm) / 4
        u4 = (wterm + pterm - qterm + rterm) / 4

        return jnp.array([u1, u2, u3, u4])

    def _compute_K_nom(self):
        x_zero, u_zero = jnp.zeros(12), jnp.zeros(4)
        A_hl, B_hl = jax.jacobian(self.f_single_change, argnums=(0, 1))(x_zero, u_zero)

        #                 [   x,   y,   z |  ψ,   θ,   ϕ |  u,   v,   w, | r,   q,   p ]
        Q = jnp.array([30.0, 30.0, 30.0, 100.0, 100.0, 100.0, 1.0, 1.0, 1.0, 100.0, 100.0, 100.0])
        #                [\dot w, \dot p, \dot q, \dot r ]
        R = 1.0 * jnp.array([1.0, 1.0, 1.0, 1.0])

        K, S, E = ct.lqr(A_hl, B_hl, jnp.diag(Q), jnp.diag(R))
        return K