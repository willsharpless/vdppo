import jax
import jax.numpy as jnp
from functools import partial
from gymnax.environments import spaces
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper
from flax import struct
from brax.envs.base import State
from .half_cheetah_random import HalfCheetahRandom
from .half_cheetah_deterministic import HalfCheetahDeterministic

# @struct.dataclass
# class EnvState:
#     state: State
#     energy: float
#     reach: float
#     avoid: int

# @struct.dataclass
# class EnvParams:
#     min_energy: float = -400.0
#     max_energy: float = 800.0

@struct.dataclass
class EnvStateRA:
    state: jax.Array = struct.field(default_factory=jax.Array)
    time: int = 0
    avoid: float = 0.
    reach: float = 0.
    has_reached: float = 0.

@struct.dataclass
class EnvStateAvoidOnly:
    state: jax.Array = struct.field(default_factory=jax.Array)
    time: int = 0
    avoid: float = 0.

@struct.dataclass
class EnvParamsEmpty:
    max_steps_in_episode: int = 5000
    pass


class HalfCheetahReachAvoid:
    def __init__(self, backend="positional"):
        env = HalfCheetahRandom(backend=backend,
                           exclude_current_positions_from_observation=False)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)
        self.default_params = EnvParamsEmpty()

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)
        head_pos, _, _, _, _, front_foot_pos, _, _, back_foot_pos = self.calculate_position(state.obs)
        avoid_value = self.is_avoid(front_foot_pos, back_foot_pos)
        reach_value = self.is_reach(head_pos)
        has_reached = reach_value < 0
        observation = jnp.concatenate([state.obs, jnp.array([avoid_value, reach_value])])
        env_state = EnvStateRA(state, avoid_value, reach_value, has_reached)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.tanh(action)
        next_state = self._env.step(state.state, u)
        head_pos, _, _, _, _, front_foot_pos, _, _, back_foot_pos = self.calculate_position(next_state.obs)
        
        avoid_value = self.is_avoid(front_foot_pos, back_foot_pos)
        reach_value = self.is_reach(head_pos)
        has_reached = jnp.logical_or(reach_value < 0, state.has_reached)

        (head_pos, neck_pos, back_pos, front_thigh_pos, front_shin_pos,
         front_foot_pos, back_thigh_pos, back_shin_pos, back_foot_pos) = self.calculate_position(state.state.obs)
        pos_dict = {"head_pos": head_pos, "neck_pos": neck_pos, "back_pos": back_pos,
                    "front_thigh_pos": front_thigh_pos, "front_shin_pos": front_shin_pos, "front_foot_pos": front_foot_pos,
                    "back_thigh_pos": back_thigh_pos, "back_shin_pos": back_shin_pos, "back_foot_pos": back_foot_pos}
        observation = jnp.concatenate([next_state.obs, jnp.array([avoid_value, reach_value])])
        next_state_new = EnvStateRA(next_state, avoid_value, reach_value, has_reached)
        reward = 0.

        return observation, next_state_new, reward, next_state.done > 0.5, pos_dict

    @partial(jax.jit, static_argnums=(0,))
    def calculate_position(self, obs):

        back_pos = jnp.array([obs[0] - 0.5 * jnp.cos(obs[2]),
                              obs[1] + 0.5 * jnp.sin(obs[2])])
        neck_pos = jnp.array([obs[0] + 0.5 * jnp.cos(obs[2]),
                              obs[1] - 0.5 * jnp.sin(obs[2])])
        head_pos = jnp.array([neck_pos[0] + 0.1 * jnp.cos(jnp.pi / 4 - obs[2]) +
                              0.15 * jnp.cos(jnp.pi / 2 - 0.87 - obs[2]),
                              neck_pos[1] + 0.1 * jnp.sin(jnp.pi / 4 - obs[2]) +
                              0.15 * jnp.sin(jnp.pi / 2 - 0.87 - obs[2])])
        front_thigh_pos = jnp.array([neck_pos[0] + 0.266 * jnp.cos(0.53 + jnp.pi / 2 + obs[2] + obs[6]),
                                     neck_pos[1] - 0.266 * jnp.sin(0.53 + jnp.pi / 2 + obs[2] + obs[6])])
        front_shin_pos = jnp.array([front_thigh_pos[0] + 0.212 * jnp.cos(-0.6 + jnp.pi / 2 + obs[2] + obs[6] + obs[7]),
                                    front_thigh_pos[1] - 0.212 * jnp.sin(-0.6 + jnp.pi / 2 + obs[2] + obs[6] + obs[7])])
        front_foot_pos = jnp.array([front_shin_pos[0] + 0.14 * jnp.cos(-0.6 + jnp.pi / 2 + obs[2] + obs[6] + obs[7] + obs[8]),
                                    front_shin_pos[1] - 0.14 * jnp.sin(-0.6 + jnp.pi / 2 + obs[2] + obs[6] + obs[7] + obs[8])])
        back_thigh_pos = jnp.array([back_pos[0] + 0.29 * jnp.cos(jnp.pi * 3 / 2 - 3.8 + obs[2] + obs[3]),
                                     back_pos[1] - 0.29 * jnp.sin(jnp.pi * 3 / 2 - 3.8 + obs[2] + obs[3])])
        back_shin_pos = jnp.array([back_thigh_pos[0] + 0.3 * jnp.cos(jnp.pi * 3 / 2 - 2.03 + obs[2] + obs[3] + obs[4]),
                                    back_thigh_pos[1] - 0.3 * jnp.sin(jnp.pi * 3 / 2 - 2.03 + obs[2] + obs[3] + obs[4])])
        back_foot_pos = jnp.array([back_shin_pos[0] + 0.188 * jnp.cos(jnp.pi / 2 - 0.27 + obs[2] + obs[3] + obs[4] + obs[5]),
                                    back_shin_pos[1] - 0.188 * jnp.sin(jnp.pi / 2 - 0.27 + obs[2] + obs[3] + obs[4] + obs[5])])

        return (head_pos, neck_pos, back_pos, front_thigh_pos, front_shin_pos, front_foot_pos,
                back_thigh_pos, back_shin_pos, back_foot_pos)

    @partial(jax.jit, static_argnums=(0,))
    def is_reach(self, head_pos):
        radius, target_pos = 0.25, jnp.array([3.5, 0.0])
        # reach = jnp.sqrt((head_pos[..., 0] - target_pos[0]) ** 2 + (head_pos[1] - target_pos[1]) ** 2) - 0.2
        # has_reached_goal = jnp.sqrt((head_pos[..., 0] - target_pos[0]) ** 2 + (head_pos[1] - target_pos[1]) ** 2) < 0.2
        reach_value = jnp.sqrt((head_pos[..., 0] - target_pos[0]) ** 2) - radius
        # has_reached_goal = reach_value < 0
        # reach_value = jnp.where(has_reached_goal, -3., reach_value)
        # is_avoid = (avoid_value == -1)
        # value = jnp.where(is_avoid, 6., value)
        return reach_value

    @partial(jax.jit, static_argnums=(0,))
    def is_avoid(self, front_foot_pos, back_foot_pos):
        radius, box_halfwidth = 0.05, 0.1

        avoid_box_1_front = -(jnp.maximum((radius/box_halfwidth) * jnp.fabs(front_foot_pos[..., 0] - 2.5), front_foot_pos[..., 1] + 0.5) - radius)
        avoid_box_1_back = -(jnp.maximum((radius/box_halfwidth) * jnp.fabs(back_foot_pos[..., 0] - 2.5), back_foot_pos[..., 1] + 0.5) - radius)
        avoid_box_2_front = -(jnp.maximum((radius/box_halfwidth) * jnp.fabs(front_foot_pos[..., 0] - 4.5), front_foot_pos[..., 1] + 0.5) - radius)
        avoid_box_2_back = -(jnp.maximum((radius/box_halfwidth) * jnp.fabs(back_foot_pos[..., 0] - 4.5), back_foot_pos[..., 1] + 0.5) - radius)
        
        avoid_value = jnp.maximum(
            jnp.maximum(avoid_box_1_front, avoid_box_1_back),
            jnp.maximum(avoid_box_2_front, avoid_box_2_back)
        )
        # avoid_value = jnp.where(avoid_value > 0, 10., avoid_value)

        return 10. * avoid_value

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

class HalfCheetahAvoidOnly:
    def __init__(self, backend="positional"):
        env = HalfCheetahRandom(backend=backend,
                           exclude_current_positions_from_observation=False)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)
        self.default_params = EnvParamsEmpty()

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)
        head_pos, _, _, _, _, front_foot_pos, _, _, back_foot_pos = self.calculate_position(state.obs)
        avoid_value = self.is_avoid(front_foot_pos, back_foot_pos)
        reach_value = self.is_reach(head_pos)
        observation = jnp.concatenate([state.obs, jnp.array([avoid_value, reach_value])])
        env_state = EnvStateAvoidOnly(state, avoid_value)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.tanh(action)
        next_state = self._env.step(state.state, u)
        head_pos, _, _, _, _, front_foot_pos, _, _, back_foot_pos = self.calculate_position(next_state.obs)
        
        avoid_value = self.is_avoid(front_foot_pos, back_foot_pos)
        reach_value = self.is_reach(head_pos)

        (head_pos, neck_pos, back_pos, front_thigh_pos, front_shin_pos,
         front_foot_pos, back_thigh_pos, back_shin_pos, back_foot_pos) = self.calculate_position(state.state.obs)
        pos_dict = {"head_pos": head_pos, "neck_pos": neck_pos, "back_pos": back_pos,
                    "front_thigh_pos": front_thigh_pos, "front_shin_pos": front_shin_pos, "front_foot_pos": front_foot_pos,
                    "back_thigh_pos": back_thigh_pos, "back_shin_pos": back_shin_pos, "back_foot_pos": back_foot_pos}
        observation = jnp.concatenate([next_state.obs, jnp.array([avoid_value, reach_value])])
        next_state_new = EnvStateAvoidOnly(next_state, avoid_value)
        reward = 0.

        return observation, next_state_new, reward, next_state.done > 0.5, pos_dict

    @partial(jax.jit, static_argnums=(0,))
    def calculate_position(self, obs):

        back_pos = jnp.array([obs[0] - 0.5 * jnp.cos(obs[2]),
                              obs[1] + 0.5 * jnp.sin(obs[2])])
        neck_pos = jnp.array([obs[0] + 0.5 * jnp.cos(obs[2]),
                              obs[1] - 0.5 * jnp.sin(obs[2])])
        head_pos = jnp.array([neck_pos[0] + 0.1 * jnp.cos(jnp.pi / 4 - obs[2]) +
                              0.15 * jnp.cos(jnp.pi / 2 - 0.87 - obs[2]),
                              neck_pos[1] + 0.1 * jnp.sin(jnp.pi / 4 - obs[2]) +
                              0.15 * jnp.sin(jnp.pi / 2 - 0.87 - obs[2])])
        front_thigh_pos = jnp.array([neck_pos[0] + 0.266 * jnp.cos(0.53 + jnp.pi / 2 + obs[2] + obs[6]),
                                     neck_pos[1] - 0.266 * jnp.sin(0.53 + jnp.pi / 2 + obs[2] + obs[6])])
        front_shin_pos = jnp.array([front_thigh_pos[0] + 0.212 * jnp.cos(-0.6 + jnp.pi / 2 + obs[2] + obs[6] + obs[7]),
                                    front_thigh_pos[1] - 0.212 * jnp.sin(-0.6 + jnp.pi / 2 + obs[2] + obs[6] + obs[7])])
        front_foot_pos = jnp.array([front_shin_pos[0] + 0.14 * jnp.cos(-0.6 + jnp.pi / 2 + obs[2] + obs[6] + obs[7] + obs[8]),
                                    front_shin_pos[1] - 0.14 * jnp.sin(-0.6 + jnp.pi / 2 + obs[2] + obs[6] + obs[7] + obs[8])])
        back_thigh_pos = jnp.array([back_pos[0] + 0.29 * jnp.cos(jnp.pi * 3 / 2 - 3.8 + obs[2] + obs[3]),
                                     back_pos[1] - 0.29 * jnp.sin(jnp.pi * 3 / 2 - 3.8 + obs[2] + obs[3])])
        back_shin_pos = jnp.array([back_thigh_pos[0] + 0.3 * jnp.cos(jnp.pi * 3 / 2 - 2.03 + obs[2] + obs[3] + obs[4]),
                                    back_thigh_pos[1] - 0.3 * jnp.sin(jnp.pi * 3 / 2 - 2.03 + obs[2] + obs[3] + obs[4])])
        back_foot_pos = jnp.array([back_shin_pos[0] + 0.188 * jnp.cos(jnp.pi / 2 - 0.27 + obs[2] + obs[3] + obs[4] + obs[5]),
                                    back_shin_pos[1] - 0.188 * jnp.sin(jnp.pi / 2 - 0.27 + obs[2] + obs[3] + obs[4] + obs[5])])

        return (head_pos, neck_pos, back_pos, front_thigh_pos, front_shin_pos, front_foot_pos,
                back_thigh_pos, back_shin_pos, back_foot_pos)

    @partial(jax.jit, static_argnums=(0,))
    def is_reach(self, head_pos):
        radius, target_pos = 0.25, jnp.array([3.5, 0.0])
        # reach = jnp.sqrt((head_pos[..., 0] - target_pos[0]) ** 2 + (head_pos[1] - target_pos[1]) ** 2) - 0.2
        # has_reached_goal = jnp.sqrt((head_pos[..., 0] - target_pos[0]) ** 2 + (head_pos[1] - target_pos[1]) ** 2) < 0.2
        reach_value = jnp.sqrt((head_pos[..., 0] - target_pos[0]) ** 2) - radius
        # has_reached_goal = reach_value < 0
        # reach_value = jnp.where(has_reached_goal, -3., reach_value)
        # is_avoid = (avoid_value == -1)
        # value = jnp.where(is_avoid, 6., value)
        return reach_value

    @partial(jax.jit, static_argnums=(0,))
    def is_avoid(self, front_foot_pos, back_foot_pos):
        radius, box_halfwidth = 0.05, 0.1

        avoid_box_1_front = -(jnp.maximum((radius/box_halfwidth) * jnp.fabs(front_foot_pos[..., 0] - 2.5), front_foot_pos[..., 1] + 0.5) - radius)
        avoid_box_1_back = -(jnp.maximum((radius/box_halfwidth) * jnp.fabs(back_foot_pos[..., 0] - 2.5), back_foot_pos[..., 1] + 0.5) - radius)
        avoid_box_2_front = -(jnp.maximum((radius/box_halfwidth) * jnp.fabs(front_foot_pos[..., 0] - 4.5), front_foot_pos[..., 1] + 0.5) - radius)
        avoid_box_2_back = -(jnp.maximum((radius/box_halfwidth) * jnp.fabs(back_foot_pos[..., 0] - 4.5), back_foot_pos[..., 1] + 0.5) - radius)
        
        avoid_value = jnp.maximum(
            jnp.maximum(avoid_box_1_front, avoid_box_1_back),
            jnp.maximum(avoid_box_2_front, avoid_box_2_back)
        )
        # avoid_value = jnp.where(avoid_value > 0, 10., avoid_value)

        return 10. * avoid_value

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

