import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct


@struct.dataclass
class EnvState:
    theta: float
    theta_dot: float
    time: int
    energy: float
    reach: float


@struct.dataclass
class EnvParams:
    max_speed: float = 8.0
    max_torque: float = 1.0
    torque_limit: float = 0.1
    min_energy: float = -400.0
    max_energy: float = 800.0
    dt: float = 0.05
    g: float = 10.0  # gravity
    m: float = 1.0  # mass
    l: float = 1.0  # length
    max_steps_in_episode: int = 2000


class PendulumConstraint(environment.Environment):

    def __init__(self):
        super().__init__()
        self.obs_shape = (4,)

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for Environment."""
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: float,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Integrate pendulum ODE and return transition."""
        u = jnp.tanh(action)
        reach_limit = jnp.fabs(u) > params.torque_limit
        energy_consumption = jnp.where(reach_limit, (jnp.fabs(u) ** 2) * 8.0, 0.)
        u = jnp.where(reach_limit, u, 0.)
        reward = (
            energy_consumption
        )
        reward = reward.squeeze()

        newenergy = jnp.clip((state.energy - reward), params.min_energy, params.max_energy)

        newthdot = state.theta_dot + (
            (
                3 * params.g / (2 * params.l) * jnp.sin(state.theta)
                + 3.0 / (params.m * params.l ** 2) * u
            )
            * params.dt
        )

        newthdot = jnp.clip(newthdot, -params.max_speed, params.max_speed)
        newth = state.theta + newthdot * params.dt
        newreach = self.is_reach(state.theta, newthdot, params.dt)

        # Update state dict and evaluate termination conditions
        state = EnvState(
            newth.squeeze(), newthdot.squeeze(), state.time + 1, newenergy, newreach
        )
        done = self.is_terminal(state, params)
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            {"theta": state.theta,
             "theta_dot": state.theta_dot},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling theta, theta_dot."""
        high = jnp.array([jnp.pi, 1, params.max_energy])
        low = jnp.array([-jnp.pi, -1, params.min_energy])
        state = jax.random.uniform(key, shape=(3,), minval=low, maxval=high)
        reach = self.is_reach(state[0], state[1], params.dt)
        state = EnvState(theta=state[0], theta_dot=state[1], time=0, energy=state[2], reach=reach)
        return self.get_obs(state), state

    def is_reach(self, theta, thetadot, dt) -> float:
        """Check the reach value of the current state"""
        theta_normalized = angle_normalize(theta)
        new_theta = (theta_normalized + thetadot * dt).squeeze()
        has_reached_goal = (theta_normalized * new_theta) < 0
        value = jnp.where(has_reached_goal, -2.5, jnp.fabs(theta_normalized))
        return value * 100.0

    def get_obs(self, state: EnvState) -> chex.Array:
        """Return angle in polar coordinates and change."""
        return jnp.array(
            [
                jnp.cos(state.theta),
                jnp.sin(state.theta),
                state.theta_dot,
                state.energy,
            ]
        ).squeeze()

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = state.time >= params.max_steps_in_episode
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "Pendulum"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 1

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Box(
            low=-params.max_torque,
            high=params.max_torque,
            shape=(1,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        high = jnp.array([1.0, 1.0, params.max_speed, jnp.finfo(jnp.float32).max], dtype=jnp.float32)
        return spaces.Box(-high, high, shape=(4,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "theta": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "theta_dot": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
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


def angle_normalize(x: float) -> float:
    """Normalize the angle - radians."""
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi