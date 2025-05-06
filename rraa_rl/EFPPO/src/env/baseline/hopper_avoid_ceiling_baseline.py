import jax
import jax.numpy as jnp
from functools import partial
from gymnax.environments import spaces
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper
from flax import struct
from brax.envs.base import State

from .hopper_random import HopperRandom

@struct.dataclass
class EnvState:
    state: State
    reach: float
    avoid: int
    cost: float

@struct.dataclass
class EnvParams:
    gamma: float = 0.99
    torque_limit: float = 0.2
    max_torque: float = 1.0


class HopperAvoidCeilingBaseline:
    def __init__(self, backend="positional"):
        env = HopperRandom(backend=backend,
                           exclude_current_positions_from_observation=False,
                           terminate_when_unhealthy=False)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=2)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)
        self.default_params = EnvParams()

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)
        head_pos, _, _, _, _, _ = self.calculate_position(state.obs)
        is_avoid = self.is_avoid(head_pos)
        avoid_value = jnp.where(is_avoid, -1, 1)
        reach_value = self.is_reach(head_pos)
        observation = jnp.concatenate([state.obs, jnp.array([avoid_value])])
        env_state = EnvState(state, reach_value, avoid_value, 0.)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.tanh(action)
        reach_limit_0 = jnp.fabs(u[0] * state.state.obs[-3] / 2.) > params.torque_limit
        energy_consumption_0 = jnp.where(reach_limit_0, (jnp.fabs(u[0] * state.state.obs[-3] / 2.) ** 2) * 0.6, 0.)
        reach_limit_1 = jnp.fabs(u[1] * state.state.obs[-2] / 2.) > params.torque_limit
        energy_consumption_1 = jnp.where(reach_limit_1, (jnp.fabs(u[1] * state.state.obs[-2] / 2.) ** 2) * 0.6, 0.)
        reach_limit_2 = jnp.fabs(u[2] * state.state.obs[-1] / 2.) > params.torque_limit
        energy_consumption_2 = jnp.where(reach_limit_2, (jnp.fabs(u[2] * state.state.obs[-1] / 2.) ** 2) * 0.6, 0.)
        energy_consumption = energy_consumption_0 + energy_consumption_1 + energy_consumption_2
        next_state = self._env.step(state.state, u)
        head_pos, _, _, _, _, _ = self.calculate_position(next_state.obs)
        is_avoid = self.is_avoid(head_pos)
        avoid_value = jnp.where(is_avoid, -1, state.avoid)
        reach_value = self.is_reach(head_pos)
        reward = params.gamma * reach_value - state.reach
        head_pos, jaw_pos, thg_pos, leg_pos, foot_front_pos, foot_back_pos = self.calculate_position(state.state.obs)
        pos_dict = {"head_pos": head_pos, "jaw_pos": jaw_pos, "thg_pos": thg_pos, "leg_pos": leg_pos,
                    "foot_front_pos": foot_front_pos, "foot_back_pos": foot_back_pos}
        observation = jnp.concatenate([next_state.obs, jnp.array([avoid_value])])
        next_state_new = EnvState(next_state, reach_value, avoid_value, energy_consumption)

        return observation, next_state_new, reward, (state.avoid == -1) | (state.reach < 0), pos_dict

    @partial(jax.jit, static_argnums=(0,))
    def calculate_position(self, obs):
        head_pos = jnp.array([obs[0] + 0.2 * jnp.sin(obs[2]),
                              obs[1] + 0.2 * jnp.cos(obs[2])])
        jaw_pos = jnp.array([obs[0] - 0.2 * jnp.sin(obs[2]),
                             obs[1] - 0.2 * jnp.cos(obs[2])])
        thg_pos = jnp.array([jaw_pos[0] - 0.45 * jnp.sin(obs[2] - obs[3]),
                             jaw_pos[1] - 0.45 * jnp.cos(obs[2] - obs[3])])
        leg_pos = jnp.array([thg_pos[0] - 0.5 * jnp.sin(obs[2] - obs[3] - obs[4]),
                             thg_pos[1] - 0.5 * jnp.cos(obs[2] - obs[3] - obs[4])])
        foot_back_pos = jnp.array([leg_pos[0] - 0.13 * jnp.cos(obs[2] - obs[3] - obs[4] - obs[5]),
                                    leg_pos[1] + 0.13 * jnp.sin(obs[2] - obs[3] - obs[4] - obs[5])])
        foot_front_pos = jnp.array([leg_pos[0] + 0.26 * jnp.cos(obs[2] - obs[3] - obs[4] - obs[5]),
                                   leg_pos[1] - 0.26 * jnp.sin(obs[2] - obs[3] - obs[4] - obs[5])])
        return head_pos, jaw_pos, thg_pos, leg_pos, foot_front_pos, foot_back_pos

    @partial(jax.jit, static_argnums=(0,))
    def is_reach(self, head_pos):
        reach = jnp.sqrt((head_pos[0] - 2.0) ** 2 + (head_pos[1] - 1.4) ** 2) - 0.1
        has_reached_goal = jnp.sqrt((head_pos[0] - 2.0) ** 2 + (head_pos[1] - 1.4) ** 2) < 0.1
        value = jnp.where(has_reached_goal, -2.5, reach)
        return value * 100.0

    @partial(jax.jit, static_argnums=(0,))
    def is_avoid(self, head_pos):
        avoid_1 = (head_pos[1] >= 1.3) & (head_pos[0] >= 0.95) & (head_pos[0] <= 1.05)
        return avoid_1

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size + 1),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )
