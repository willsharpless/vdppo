import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct


@struct.dataclass
class EnvState:
    x: int
    energy: float
    reach: float
    avoid: int
    time: int


@struct.dataclass
class EnvParams:
    min_energy: float = -60.
    max_energy: float = 400.
    max_steps_in_episode: int = 200


class GridAvoid(environment.Environment):

    def __init__(self):
        super().__init__()
        self.obs_shape = (4,)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""

        reward = (-2 * action + 3)

        x = (state.x + 2 * action - 1) % 101

        energy = jnp.clip((state.energy - reward), params.min_energy, params.max_energy)

        is_avoid = (x == 20)
        avoid_state = jnp.where(is_avoid, -1, state.avoid)

        reach = self.is_reach(x, avoid_state)

        # Update state dict and evaluate termination conditions
        state = EnvState(x, energy, reach, avoid_state, state.time + 1)
        done = self.is_terminal(state, params)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            {"x": state.x},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        init_energy = jax.random.uniform(
            key, minval=0, maxval=params.max_energy
        )
        init_state = jnp.array(jax.random.randint(key, (1, ), 0, 101), int)[0]

        is_avoid = (init_state == 20)
        avoid = jnp.where(is_avoid, -1, 1)

        reach = self.is_reach(init_state, avoid)
        state = EnvState(
            x=init_state,
            energy=init_energy,
            reach=reach,
            avoid=avoid,
            time=0,
        )
        return self.get_obs(state), state

    def is_reach(self, x, avoid) -> float:
        """Check the reach value of the current state"""
        state = (x + 25) % 101
        reach = jnp.fabs(state - 50.) - 2.5
        has_reached_goal = jnp.fabs(state - 50.) < 3
        reach = jnp.where(has_reached_goal, -5., reach)
        reach = jnp.where((avoid == -1), 50., reach)
        return reach

    def get_obs(self, state: EnvState) -> chex.Array:
        """Applies observation function to state."""
        return jnp.array([jnp.sin((state.x - 50.) * jnp.pi / 50.), jnp.cos((state.x - 50.) * jnp.pi / 50.), state.avoid, state.energy])

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        done = state.time >= params.max_steps_in_episode
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "GridWorld"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(2)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        high = jnp.array(
            [
                jnp.finfo(jnp.float32).max,
                jnp.finfo(jnp.float32).max,
                jnp.finfo(jnp.float32).max,
                jnp.finfo(jnp.float32).max,
            ]
        )
        return spaces.Box(-high, high, (4,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "x": spaces.Box(-jnp.finfo(jnp.float32).max, jnp.finfo(jnp.float32).max, (), jnp.float32),
                "energy": spaces.Box(params.min_energy, params.max_energy, (), jnp.float32),
                "reach": spaces.Box(-jnp.finfo(jnp.float32).max, jnp.finfo(jnp.float32).max, (), jnp.float32),
                "avoid": spaces.Box(-jnp.finfo(jnp.float32).max, jnp.finfo(jnp.float32).max, (), jnp.float32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )