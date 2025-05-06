import jax
import jax.numpy as jnp
from functools import partial
from gymnax.environments import spaces
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper
from flax import struct
from brax.envs.base import State

from .point_random import PointRandom

@struct.dataclass
class EnvState:
    state: State
    energy: float
    reach: float
    avoid: int

@struct.dataclass
class EnvParams:
    min_energy: float = -400.0
    max_energy: float = 800.0


class PointAvoid:
    def __init__(self, backend="mjx"):
        env = PointRandom(backend=backend)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)
        self.default_params = EnvParams()
        self.hazard_pos_array = jnp.array([[1.403247, 0.6281236], [0.42943087, 1.17059302],
                                           [-1.16036429, 0.89811093], [-0.88776483, 1.46420776],
                                           [-0.07556364, -1.10567521], [0.72648704, 0.17957757],
                                           [-0.33115742, 0.83026827], [-1.33470321, -1.3259373]])

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)
        init_energy = jax.random.uniform(
            key, minval=params.min_energy, maxval=params.max_energy
        )
        is_avoid = self.is_avoid(state.obs[0:2])
        avoid_value = jnp.where(is_avoid, -1, 1)
        reach_value = self.is_reach(state.obs[0:2], avoid_value)
        observation = jnp.concatenate([state.obs, jnp.array([avoid_value, init_energy])])
        env_state = EnvState(state, init_energy, reach_value, avoid_value)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.clip(action, -1., 1.)
        energy_consumption = jnp.sum(u ** 2).squeeze() / 2.
        next_state = self._env.step(state.state, u)
        is_avoid = self.is_avoid(next_state.obs[0:2])
        avoid_value = jnp.where(is_avoid, -1, state.avoid)
        reach_value = self.is_reach(next_state.obs[0:2], avoid_value)
        next_energy = jnp.clip(state.energy - energy_consumption, params.min_energy, params.max_energy)
        observation = jnp.concatenate([next_state.obs, jnp.array([avoid_value, next_energy])])
        next_state_new = EnvState(next_state, next_energy, reach_value, avoid_value)

        return observation, next_state_new, energy_consumption, next_state.done > 0.5, {"x": state.state.obs[0],
                                                                                        "y": state.state.obs[1]}

    @partial(jax.jit, static_argnums=(0,))
    def is_reach(self, obs, avoid_value):
        reach = jnp.sqrt(obs[0] ** 2 + obs[1] ** 2) - 0.3
        has_reached_goal = jnp.sqrt(obs[0] ** 2 + obs[1] ** 2) < 0.3
        value = jnp.where(has_reached_goal, -3.0, reach)
        is_avoid = (avoid_value == -1)
        value = jnp.where(is_avoid, 3.0, value)
        return value * 100.0

    @partial(jax.jit, static_argnums=(0,))
    def is_avoid(self, obs):
        avoid = (obs[0] >= 3.) | (obs[0] <= -3.) | (obs[1] >= 3.) | (obs[0] <= -3.)
        avoid_0 = jnp.sum((obs - self.hazard_pos_array[0]) ** 2) < 0.2 * 0.2
        avoid_1 = jnp.sum((obs - self.hazard_pos_array[1]) ** 2) < 0.2 * 0.2
        avoid_2 = jnp.sum((obs - self.hazard_pos_array[2]) ** 2) < 0.2 * 0.2
        avoid_3 = jnp.sum((obs - self.hazard_pos_array[3]) ** 2) < 0.2 * 0.2
        avoid_4 = jnp.sum((obs - self.hazard_pos_array[4]) ** 2) < 0.2 * 0.2
        avoid_5 = jnp.sum((obs - self.hazard_pos_array[5]) ** 2) < 0.2 * 0.2
        avoid_6 = jnp.sum((obs - self.hazard_pos_array[6]) ** 2) < 0.2 * 0.2
        avoid_7 = jnp.sum((obs - self.hazard_pos_array[7]) ** 2) < 0.2 * 0.2
        # return avoid | avoid_0 | avoid_1 | avoid_2 | avoid_3

        return avoid | avoid_0 | avoid_1 | avoid_2 | avoid_3 | avoid_4 | avoid_5 | avoid_6 | avoid_7

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size + 2,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )