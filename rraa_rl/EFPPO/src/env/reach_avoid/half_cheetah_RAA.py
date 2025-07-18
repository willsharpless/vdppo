import jax
import jax.numpy as jnp
from functools import partial
from gymnax.environments import spaces
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper
from flax import struct
from brax.envs.base import State
from .half_cheetah_random import HalfCheetahRandom
from .half_cheetah_deterministic import HalfCheetahDeterministic
from copy import deepcopy 

@struct.dataclass
class EnvStateRA:
    state: jax.Array = struct.field(default_factory=jax.Array)
    avoid: float = 0.
    reach: float = 0.
    has_reached: float = 0.

@struct.dataclass
class EnvStateAvoidOnly:
    state: jax.Array = struct.field(default_factory=jax.Array)
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
        (head_pos, neck_pos, back_pos, front_thigh_pos, front_shin_pos,
         front_foot_pos, back_thigh_pos, back_shin_pos, back_foot_pos) = self.calculate_position(state.obs)
        avoid_value = self.is_avoid(head_pos, front_thigh_pos, front_shin_pos, front_foot_pos, back_thigh_pos, back_shin_pos, back_foot_pos)
        reach_value = self.is_reach(head_pos)
        has_reached = reach_value < 0
        observation = jnp.concatenate([state.obs, jnp.array([avoid_value, reach_value])])
        env_state = EnvStateRA(state, avoid_value, reach_value, has_reached)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.tanh(action)
        next_state = self._env.step(state.state, u)
        (head_pos, neck_pos, back_pos, front_thigh_pos, front_shin_pos,
         front_foot_pos, back_thigh_pos, back_shin_pos, back_foot_pos) = self.calculate_position(next_state.obs)
        avoid_value = self.is_avoid(head_pos, front_thigh_pos, front_shin_pos, front_foot_pos, back_thigh_pos, back_shin_pos, back_foot_pos)
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
        radius, target_xpos = 0.25, 5.5
        # reach = jnp.sqrt((head_pos[..., 0] - target_xpos) ** 2 + (head_pos[1] - target_pos[1]) ** 2) - 0.2
        # has_reached_goal = jnp.sqrt((head_pos[..., 0] - target_xpos) ** 2 + (head_pos[1] - target_pos[1]) ** 2) < 0.2
        reach_value = jnp.sqrt((head_pos[..., 0] - target_xpos) ** 2) - radius
        # has_reached_goal = reach_value < 0
        # reach_value = jnp.where(has_reached_goal, -3., reach_value)
        # is_avoid = (avoid_value == -1)
        # value = jnp.where(is_avoid, 6., value)
        return reach_value

    @partial(jax.jit, static_argnums=(0,))
    def is_avoid(self, head_pos, front_thigh_pos, front_shin_pos, front_foot_pos, back_thigh_pos, back_shin_pos, back_foot_pos):
        box_height, box_halfwidth, box_ycenter, box_front_xcenter, box_back_xcenter = 0.05, 0.25, -0.7, 4.5, 6.5

        avoid_box_1_front_foot = -(jnp.maximum((box_height/box_halfwidth) * jnp.fabs(front_foot_pos[..., 0] - box_front_xcenter), front_foot_pos[..., 1] - box_ycenter) - box_height)
        avoid_box_1_back_foot = -(jnp.maximum((box_height/box_halfwidth) * jnp.fabs(back_foot_pos[..., 0] - box_front_xcenter), back_foot_pos[..., 1] - box_ycenter) - box_height)
        avoid_box_2_front_foot = -(jnp.maximum((box_height/box_halfwidth) * jnp.fabs(front_foot_pos[..., 0] - box_back_xcenter), front_foot_pos[..., 1] - box_ycenter) - box_height)
        avoid_box_2_back_foot = -(jnp.maximum((box_height/box_halfwidth) * jnp.fabs(back_foot_pos[..., 0] - box_back_xcenter), back_foot_pos[..., 1] - box_ycenter) - box_height)
        
        avoid_box_1_front_thigh = -(jnp.maximum((box_height/box_halfwidth) * jnp.fabs(front_thigh_pos[..., 0] - box_front_xcenter), front_thigh_pos[..., 1] - box_ycenter) - box_height)
        avoid_box_1_back_thigh = -(jnp.maximum((box_height/box_halfwidth) * jnp.fabs(back_thigh_pos[..., 0] - box_front_xcenter), back_thigh_pos[..., 1] - box_ycenter) - box_height)
        avoid_box_2_front_thigh = -(jnp.maximum((box_height/box_halfwidth) * jnp.fabs(front_thigh_pos[..., 0] - box_back_xcenter), front_thigh_pos[..., 1] - box_ycenter) - box_height)
        avoid_box_2_back_thigh = -(jnp.maximum((box_height/box_halfwidth) * jnp.fabs(back_thigh_pos[..., 0] - box_back_xcenter), back_thigh_pos[..., 1] - box_ycenter) - box_height)

        avoid_box_1_front_shin = -(jnp.maximum((box_height/box_halfwidth) * jnp.fabs(front_shin_pos[..., 0] - box_front_xcenter), front_shin_pos[..., 1] - box_ycenter) - box_height)
        avoid_box_1_back_shin = -(jnp.maximum((box_height/box_halfwidth) * jnp.fabs(back_shin_pos[..., 0] - box_front_xcenter), back_shin_pos[..., 1] - box_ycenter) - box_height)
        avoid_box_2_front_shin = -(jnp.maximum((box_height/box_halfwidth) * jnp.fabs(front_shin_pos[..., 0] - box_back_xcenter), front_shin_pos[..., 1] - box_ycenter) - box_height)
        avoid_box_2_back_shin = -(jnp.maximum((box_height/box_halfwidth) * jnp.fabs(back_shin_pos[..., 0] - box_back_xcenter), back_shin_pos[..., 1] - box_ycenter) - box_height)

        avoid_box_1_head = -(jnp.maximum((box_height/box_halfwidth) * jnp.fabs(head_pos[..., 0] - box_front_xcenter), head_pos[..., 1] - box_ycenter) - box_height)
        avoid_box_2_head = -(jnp.maximum((box_height/box_halfwidth) * jnp.fabs(head_pos[..., 0] - box_back_xcenter), head_pos[..., 1] - box_ycenter) - box_height)

        box_avoid_value_foot = jnp.maximum(
            jnp.maximum(avoid_box_1_front_foot, avoid_box_1_back_foot),
            jnp.maximum(avoid_box_2_front_foot, avoid_box_2_back_foot)
        )
        box_avoid_value_shin = jnp.maximum(
            jnp.maximum(avoid_box_1_front_shin, avoid_box_1_back_shin),
            jnp.maximum(avoid_box_2_front_shin, avoid_box_2_back_shin)
        )
        box_avoid_value_thigh = jnp.maximum(
            jnp.maximum(avoid_box_1_front_thigh, avoid_box_1_back_thigh),
            jnp.maximum(avoid_box_2_front_thigh, avoid_box_2_back_thigh)
        )
        box_avoid_value_head = jnp.maximum(avoid_box_1_head, avoid_box_2_head)
        box_avoid_value = jnp.maximum(
            jnp.maximum(box_avoid_value_foot, box_avoid_value_shin),
            jnp.maximum(box_avoid_value_thigh, box_avoid_value_head)
        )

        ## WALL AVOIDANCE
        wall_height, wall_halfwidth = 1., 0.1
        left_wall_x, right_wall_x = -0.5, 5.5

        avoid_wall_1_front_foot = -(jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(front_foot_pos[..., 0] - left_wall_x), front_foot_pos[..., 1] - box_ycenter) - wall_height)
        avoid_wall_1_back_foot = -(jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(back_foot_pos[..., 0] - left_wall_x), back_foot_pos[..., 1] - box_ycenter) - wall_height)
        avoid_wall_2_front_foot = -(jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(front_foot_pos[..., 0] - right_wall_x), front_foot_pos[..., 1] - box_ycenter) - wall_height)
        avoid_wall_2_back_foot = -(jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(back_foot_pos[..., 0] - right_wall_x), back_foot_pos[..., 1] - box_ycenter) - wall_height)

        avoid_wall_1_front_thigh = -(jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(front_thigh_pos[..., 0] - left_wall_x), front_thigh_pos[..., 1] - box_ycenter) - wall_height)
        avoid_wall_1_back_thigh = -(jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(back_thigh_pos[..., 0] - left_wall_x), back_thigh_pos[..., 1] - box_ycenter) - wall_height)
        avoid_wall_2_front_thigh = -(jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(front_thigh_pos[..., 0] - right_wall_x), front_thigh_pos[..., 1] - box_ycenter) - wall_height)
        avoid_wall_2_back_thigh = -(jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(back_thigh_pos[..., 0] - right_wall_x), back_thigh_pos[..., 1] - box_ycenter) - wall_height)

        avoid_wall_1_front_shin = -(jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(front_shin_pos[..., 0] - left_wall_x), front_shin_pos[..., 1] - box_ycenter) - wall_height)
        avoid_wall_1_back_shin = -(jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(back_shin_pos[..., 0] - left_wall_x), back_shin_pos[..., 1] - box_ycenter) - wall_height)
        avoid_wall_2_front_shin = -(jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(front_shin_pos[..., 0] - right_wall_x), front_shin_pos[..., 1] - box_ycenter) - wall_height)
        avoid_wall_2_back_shin = -(jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(back_shin_pos[..., 0] - right_wall_x), back_shin_pos[..., 1] - box_ycenter) - wall_height)
        
        avoid_wall_1_head = -(jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(head_pos[..., 0] - left_wall_x), head_pos[..., 1] - box_ycenter) - wall_height)
        avoid_wall_2_head = -(jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(head_pos[..., 0] - right_wall_x), head_pos[..., 1] - box_ycenter) - wall_height)

        # wall_avoid_value_foot = jnp.maximum(
        #     jnp.maximum(avoid_wall_1_front_foot, avoid_wall_1_back_foot),
        #     jnp.maximum(avoid_wall_2_front_foot, avoid_wall_2_back_foot)
        # )
        # wall_avoid_value_shin = jnp.maximum(
        #     jnp.maximum(avoid_wall_1_front_shin, avoid_wall_1_back_shin),
        #     jnp.maximum(avoid_wall_2_front_shin, avoid_wall_2_back_shin)
        # )
        # wall_avoid_value_thigh = jnp.maximum(
        #     jnp.maximum(avoid_wall_1_front_thigh, avoid_wall_1_back_thigh),
        #     jnp.maximum(avoid_wall_2_front_thigh, avoid_wall_2_back_thigh)
        # )
        # wall_avoid_value_head = jnp.maximum(avoid_wall_1_head, avoid_wall_2_head)
        
        wall_avoid_value_foot = jnp.maximum(avoid_wall_1_front_foot, avoid_wall_1_back_foot)
        wall_avoid_value_shin = jnp.maximum(avoid_wall_1_front_shin, avoid_wall_1_back_shin)
        wall_avoid_value_thigh = jnp.maximum(avoid_wall_1_front_thigh, avoid_wall_1_back_thigh)
        wall_avoid_value_head = avoid_wall_1_head

        wall_avoid_value = jnp.maximum(
            jnp.maximum(wall_avoid_value_foot, wall_avoid_value_shin),
            jnp.maximum(wall_avoid_value_thigh, wall_avoid_value_head)
        )

        avoid_value = jnp.maximum(box_avoid_value, wall_avoid_value)

        return avoid_value

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
        (head_pos, neck_pos, back_pos, front_thigh_pos, front_shin_pos,
         front_foot_pos, back_thigh_pos, back_shin_pos, back_foot_pos) = self.calculate_position(state.obs)
        avoid_value = self.is_avoid(head_pos, front_thigh_pos, front_shin_pos, front_foot_pos, back_thigh_pos, back_shin_pos, back_foot_pos)
        reach_value = self.is_reach(head_pos)
        observation = jnp.concatenate([state.obs, jnp.array([avoid_value, reach_value])])
        env_state = EnvStateAvoidOnly(state, avoid_value)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.tanh(action)
        next_state = self._env.step(state.state, u)
        (head_pos, neck_pos, back_pos, front_thigh_pos, front_shin_pos,
         front_foot_pos, back_thigh_pos, back_shin_pos, back_foot_pos) = self.calculate_position(next_state.obs)
        avoid_value = self.is_avoid(head_pos, front_thigh_pos, front_shin_pos, front_foot_pos, back_thigh_pos, back_shin_pos, back_foot_pos)
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
        radius, target_xpos = 0.25, 5.5
        # reach = jnp.sqrt((head_pos[..., 0] - target_xpos) ** 2 + (head_pos[1] - target_pos[1]) ** 2) - 0.2
        # has_reached_goal = jnp.sqrt((head_pos[..., 0] - target_xpos) ** 2 + (head_pos[1] - target_pos[1]) ** 2) < 0.2
        reach_value = jnp.sqrt((head_pos[..., 0] - target_xpos) ** 2) - radius
        # has_reached_goal = reach_value < 0
        # reach_value = jnp.where(has_reached_goal, -3., reach_value)
        # is_avoid = (avoid_value == -1)
        # value = jnp.where(is_avoid, 6., value)
        return reach_value

    @partial(jax.jit, static_argnums=(0,))
    def is_avoid(self, head_pos, front_thigh_pos, front_shin_pos, front_foot_pos, back_thigh_pos, back_shin_pos, back_foot_pos):
        box_height, box_halfwidth, box_ycenter, box_front_xcenter, box_back_xcenter = 0.05, 0.25, -0.7, 4.5, 6.5

        avoid_box_1_front_foot = -(jnp.maximum((box_height/box_halfwidth) * jnp.fabs(front_foot_pos[..., 0] - box_front_xcenter), front_foot_pos[..., 1] - box_ycenter) - box_height)
        avoid_box_1_back_foot = -(jnp.maximum((box_height/box_halfwidth) * jnp.fabs(back_foot_pos[..., 0] - box_front_xcenter), back_foot_pos[..., 1] - box_ycenter) - box_height)
        avoid_box_2_front_foot = -(jnp.maximum((box_height/box_halfwidth) * jnp.fabs(front_foot_pos[..., 0] - box_back_xcenter), front_foot_pos[..., 1] - box_ycenter) - box_height)
        avoid_box_2_back_foot = -(jnp.maximum((box_height/box_halfwidth) * jnp.fabs(back_foot_pos[..., 0] - box_back_xcenter), back_foot_pos[..., 1] - box_ycenter) - box_height)
        
        avoid_box_1_front_thigh = -(jnp.maximum((box_height/box_halfwidth) * jnp.fabs(front_thigh_pos[..., 0] - box_front_xcenter), front_thigh_pos[..., 1] - box_ycenter) - box_height)
        avoid_box_1_back_thigh = -(jnp.maximum((box_height/box_halfwidth) * jnp.fabs(back_thigh_pos[..., 0] - box_front_xcenter), back_thigh_pos[..., 1] - box_ycenter) - box_height)
        avoid_box_2_front_thigh = -(jnp.maximum((box_height/box_halfwidth) * jnp.fabs(front_thigh_pos[..., 0] - box_back_xcenter), front_thigh_pos[..., 1] - box_ycenter) - box_height)
        avoid_box_2_back_thigh = -(jnp.maximum((box_height/box_halfwidth) * jnp.fabs(back_thigh_pos[..., 0] - box_back_xcenter), back_thigh_pos[..., 1] - box_ycenter) - box_height)

        avoid_box_1_front_shin = -(jnp.maximum((box_height/box_halfwidth) * jnp.fabs(front_shin_pos[..., 0] - box_front_xcenter), front_shin_pos[..., 1] - box_ycenter) - box_height)
        avoid_box_1_back_shin = -(jnp.maximum((box_height/box_halfwidth) * jnp.fabs(back_shin_pos[..., 0] - box_front_xcenter), back_shin_pos[..., 1] - box_ycenter) - box_height)
        avoid_box_2_front_shin = -(jnp.maximum((box_height/box_halfwidth) * jnp.fabs(front_shin_pos[..., 0] - box_back_xcenter), front_shin_pos[..., 1] - box_ycenter) - box_height)
        avoid_box_2_back_shin = -(jnp.maximum((box_height/box_halfwidth) * jnp.fabs(back_shin_pos[..., 0] - box_back_xcenter), back_shin_pos[..., 1] - box_ycenter) - box_height)

        avoid_box_1_head = -(jnp.maximum((box_height/box_halfwidth) * jnp.fabs(head_pos[..., 0] - box_front_xcenter), head_pos[..., 1] - box_ycenter) - box_height)
        avoid_box_2_head = -(jnp.maximum((box_height/box_halfwidth) * jnp.fabs(head_pos[..., 0] - box_back_xcenter), head_pos[..., 1] - box_ycenter) - box_height)

        box_avoid_value_foot = jnp.maximum(
            jnp.maximum(avoid_box_1_front_foot, avoid_box_1_back_foot),
            jnp.maximum(avoid_box_2_front_foot, avoid_box_2_back_foot)
        )
        box_avoid_value_shin = jnp.maximum(
            jnp.maximum(avoid_box_1_front_shin, avoid_box_1_back_shin),
            jnp.maximum(avoid_box_2_front_shin, avoid_box_2_back_shin)
        )
        box_avoid_value_thigh = jnp.maximum(
            jnp.maximum(avoid_box_1_front_thigh, avoid_box_1_back_thigh),
            jnp.maximum(avoid_box_2_front_thigh, avoid_box_2_back_thigh)
        )
        box_avoid_value_head = jnp.maximum(avoid_box_1_head, avoid_box_2_head)
        box_avoid_value = jnp.maximum(
            jnp.maximum(box_avoid_value_foot, box_avoid_value_shin),
            jnp.maximum(box_avoid_value_thigh, box_avoid_value_head)
        )

        ## WALL AVOIDANCE
        wall_height, wall_halfwidth = 1., 0.1
        left_wall_x, right_wall_x = -0.5, 5.5

        avoid_wall_1_front_foot = -(jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(front_foot_pos[..., 0] - left_wall_x), front_foot_pos[..., 1] - box_ycenter) - wall_height)
        avoid_wall_1_back_foot = -(jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(back_foot_pos[..., 0] - left_wall_x), back_foot_pos[..., 1] - box_ycenter) - wall_height)
        avoid_wall_2_front_foot = -(jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(front_foot_pos[..., 0] - right_wall_x), front_foot_pos[..., 1] - box_ycenter) - wall_height)
        avoid_wall_2_back_foot = -(jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(back_foot_pos[..., 0] - right_wall_x), back_foot_pos[..., 1] - box_ycenter) - wall_height)

        avoid_wall_1_front_thigh = -(jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(front_thigh_pos[..., 0] - left_wall_x), front_thigh_pos[..., 1] - box_ycenter) - wall_height)
        avoid_wall_1_back_thigh = -(jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(back_thigh_pos[..., 0] - left_wall_x), back_thigh_pos[..., 1] - box_ycenter) - wall_height)
        avoid_wall_2_front_thigh = -(jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(front_thigh_pos[..., 0] - right_wall_x), front_thigh_pos[..., 1] - box_ycenter) - wall_height)
        avoid_wall_2_back_thigh = -(jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(back_thigh_pos[..., 0] - right_wall_x), back_thigh_pos[..., 1] - box_ycenter) - wall_height)

        avoid_wall_1_front_shin = -(jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(front_shin_pos[..., 0] - left_wall_x), front_shin_pos[..., 1] - box_ycenter) - wall_height)
        avoid_wall_1_back_shin = -(jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(back_shin_pos[..., 0] - left_wall_x), back_shin_pos[..., 1] - box_ycenter) - wall_height)
        avoid_wall_2_front_shin = -(jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(front_shin_pos[..., 0] - right_wall_x), front_shin_pos[..., 1] - box_ycenter) - wall_height)
        avoid_wall_2_back_shin = -(jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(back_shin_pos[..., 0] - right_wall_x), back_shin_pos[..., 1] - box_ycenter) - wall_height)
        
        avoid_wall_1_head = -(jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(head_pos[..., 0] - left_wall_x), head_pos[..., 1] - box_ycenter) - wall_height)
        avoid_wall_2_head = -(jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(head_pos[..., 0] - right_wall_x), head_pos[..., 1] - box_ycenter) - wall_height)

        # wall_avoid_value_foot = jnp.maximum(
        #     jnp.maximum(avoid_wall_1_front_foot, avoid_wall_1_back_foot),
        #     jnp.maximum(avoid_wall_2_front_foot, avoid_wall_2_back_foot)
        # )
        # wall_avoid_value_shin = jnp.maximum(
        #     jnp.maximum(avoid_wall_1_front_shin, avoid_wall_1_back_shin),
        #     jnp.maximum(avoid_wall_2_front_shin, avoid_wall_2_back_shin)
        # )
        # wall_avoid_value_thigh = jnp.maximum(
        #     jnp.maximum(avoid_wall_1_front_thigh, avoid_wall_1_back_thigh),
        #     jnp.maximum(avoid_wall_2_front_thigh, avoid_wall_2_back_thigh)
        # )
        # wall_avoid_value_head = jnp.maximum(avoid_wall_1_head, avoid_wall_2_head)
        
        wall_avoid_value_foot = jnp.maximum(avoid_wall_1_front_foot, avoid_wall_1_back_foot)
        wall_avoid_value_shin = jnp.maximum(avoid_wall_1_front_shin, avoid_wall_1_back_shin)
        wall_avoid_value_thigh = jnp.maximum(avoid_wall_1_front_thigh, avoid_wall_1_back_thigh)
        wall_avoid_value_head = avoid_wall_1_head

        wall_avoid_value = jnp.maximum(
            jnp.maximum(wall_avoid_value_foot, wall_avoid_value_shin),
            jnp.maximum(wall_avoid_value_thigh, wall_avoid_value_head)
        )

        avoid_value = jnp.maximum(box_avoid_value, wall_avoid_value)

        return avoid_value
    
    ## TANH Verison
    # @partial(jax.jit, static_argnums=(0,))
    # def is_avoid(self, head_pos, front_thigh_pos, front_shin_pos, front_foot_pos, back_thigh_pos, back_shin_pos, back_foot_pos):
    #     box_height, box_halfwidth, box_ycenter, box_front_xcenter, box_back_xcenter = 0.05, 0.25, -0.7, 4.5, 6.5
    #     alpha = 2.

    #     avoid_box_1_front_foot = -jnp.tanh(alpha * (jnp.maximum((box_height/box_halfwidth) * jnp.fabs(front_foot_pos[..., 0] - box_front_xcenter), front_foot_pos[..., 1] - box_ycenter) - box_height))
    #     avoid_box_1_back_foot = -jnp.tanh(alpha * (jnp.maximum((box_height/box_halfwidth) * jnp.fabs(back_foot_pos[..., 0] - box_front_xcenter), back_foot_pos[..., 1] - box_ycenter) - box_height))
    #     avoid_box_2_front_foot = -jnp.tanh(alpha * (jnp.maximum((box_height/box_halfwidth) * jnp.fabs(front_foot_pos[..., 0] - box_back_xcenter), front_foot_pos[..., 1] - box_ycenter) - box_height))
    #     avoid_box_2_back_foot = -jnp.tanh(alpha * (jnp.maximum((box_height/box_halfwidth) * jnp.fabs(back_foot_pos[..., 0] - box_back_xcenter), back_foot_pos[..., 1] - box_ycenter) - box_height))
        
    #     avoid_box_1_front_thigh = -jnp.tanh(alpha * (jnp.maximum((box_height/box_halfwidth) * jnp.fabs(front_thigh_pos[..., 0] - box_front_xcenter), front_thigh_pos[..., 1] - box_ycenter) - box_height))
    #     avoid_box_1_back_thigh = -jnp.tanh(alpha * (jnp.maximum((box_height/box_halfwidth) * jnp.fabs(back_thigh_pos[..., 0] - box_front_xcenter), back_thigh_pos[..., 1] - box_ycenter) - box_height))
    #     avoid_box_2_front_thigh = -jnp.tanh(alpha * (jnp.maximum((box_height/box_halfwidth) * jnp.fabs(front_thigh_pos[..., 0] - box_back_xcenter), front_thigh_pos[..., 1] - box_ycenter) - box_height))
    #     avoid_box_2_back_thigh = -jnp.tanh(alpha * (jnp.maximum((box_height/box_halfwidth) * jnp.fabs(back_thigh_pos[..., 0] - box_back_xcenter), back_thigh_pos[..., 1] - box_ycenter) - box_height))

    #     avoid_box_1_front_shin = -jnp.tanh(alpha * (jnp.maximum((box_height/box_halfwidth) * jnp.fabs(front_shin_pos[..., 0] - box_front_xcenter), front_shin_pos[..., 1] - box_ycenter) - box_height))
    #     avoid_box_1_back_shin = -jnp.tanh(alpha * (jnp.maximum((box_height/box_halfwidth) * jnp.fabs(back_shin_pos[..., 0] - box_front_xcenter), back_shin_pos[..., 1] - box_ycenter) - box_height))
    #     avoid_box_2_front_shin = -jnp.tanh(alpha * (jnp.maximum((box_height/box_halfwidth) * jnp.fabs(front_shin_pos[..., 0] - box_back_xcenter), front_shin_pos[..., 1] - box_ycenter) - box_height))
    #     avoid_box_2_back_shin = -jnp.tanh(alpha * (jnp.maximum((box_height/box_halfwidth) * jnp.fabs(back_shin_pos[..., 0] - box_back_xcenter), back_shin_pos[..., 1] - box_ycenter) - box_height))

    #     avoid_box_1_head = -jnp.tanh(alpha * (jnp.maximum((box_height/box_halfwidth) * jnp.fabs(head_pos[..., 0] - box_front_xcenter), head_pos[..., 1] - box_ycenter) - box_height))
    #     avoid_box_2_head = -jnp.tanh(alpha * (jnp.maximum((box_height/box_halfwidth) * jnp.fabs(head_pos[..., 0] - box_back_xcenter), head_pos[..., 1] - box_ycenter) - box_height))

    #     box_avoid_value_foot = jnp.maximum(
    #         jnp.maximum(avoid_box_1_front_foot, avoid_box_1_back_foot),
    #         jnp.maximum(avoid_box_2_front_foot, avoid_box_2_back_foot)
    #     )
    #     box_avoid_value_shin = jnp.maximum(
    #         jnp.maximum(avoid_box_1_front_shin, avoid_box_1_back_shin),
    #         jnp.maximum(avoid_box_2_front_shin, avoid_box_2_back_shin)
    #     )
    #     box_avoid_value_thigh = jnp.maximum(
    #         jnp.maximum(avoid_box_1_front_thigh, avoid_box_1_back_thigh),
    #         jnp.maximum(avoid_box_2_front_thigh, avoid_box_2_back_thigh)
    #     )
    #     box_avoid_value_head = jnp.maximum(avoid_box_1_head, avoid_box_2_head)
    #     box_avoid_value = jnp.maximum(
    #         jnp.maximum(box_avoid_value_foot, box_avoid_value_shin),
    #         jnp.maximum(box_avoid_value_thigh, box_avoid_value_head)
    #     )

    #     ## WALL AVOIDANCE
    #     wall_height, wall_halfwidth = 1., 0.1
    #     left_wall_x, right_wall_x = -0.5, 5.5

    #     avoid_wall_1_front_foot = -jnp.tanh(alpha * (jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(front_foot_pos[..., 0] - left_wall_x), front_foot_pos[..., 1] - box_ycenter) - wall_height))
    #     avoid_wall_1_back_foot = -jnp.tanh(alpha * (jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(back_foot_pos[..., 0] - left_wall_x), back_foot_pos[..., 1] - box_ycenter) - wall_height))
    #     avoid_wall_2_front_foot = -jnp.tanh(alpha * (jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(front_foot_pos[..., 0] - right_wall_x), front_foot_pos[..., 1] - box_ycenter) - wall_height))
    #     avoid_wall_2_back_foot = -jnp.tanh(alpha * (jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(back_foot_pos[..., 0] - right_wall_x), back_foot_pos[..., 1] - box_ycenter) - wall_height))

    #     avoid_wall_1_front_thigh = -jnp.tanh(alpha * (jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(front_thigh_pos[..., 0] - left_wall_x), front_thigh_pos[..., 1] - box_ycenter) - wall_height))
    #     avoid_wall_1_back_thigh = -jnp.tanh(alpha * (jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(back_thigh_pos[..., 0] - left_wall_x), back_thigh_pos[..., 1] - box_ycenter) - wall_height))
    #     avoid_wall_2_front_thigh = -jnp.tanh(alpha * (jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(front_thigh_pos[..., 0] - right_wall_x), front_thigh_pos[..., 1] - box_ycenter) - wall_height))
    #     avoid_wall_2_back_thigh = -jnp.tanh(alpha * (jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(back_thigh_pos[..., 0] - right_wall_x), back_thigh_pos[..., 1] - box_ycenter) - wall_height))

    #     avoid_wall_1_front_shin = -jnp.tanh(alpha * (jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(front_shin_pos[..., 0] - left_wall_x), front_shin_pos[..., 1] - box_ycenter) - wall_height))
    #     avoid_wall_1_back_shin = -jnp.tanh(alpha * (jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(back_shin_pos[..., 0] - left_wall_x), back_shin_pos[..., 1] - box_ycenter) - wall_height))
    #     avoid_wall_2_front_shin = -jnp.tanh(alpha * (jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(front_shin_pos[..., 0] - right_wall_x), front_shin_pos[..., 1] - box_ycenter) - wall_height))
    #     avoid_wall_2_back_shin = -jnp.tanh(alpha * (jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(back_shin_pos[..., 0] - right_wall_x), back_shin_pos[..., 1] - box_ycenter) - wall_height))
        
    #     avoid_wall_1_head = -jnp.tanh(alpha * (jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(head_pos[..., 0] - left_wall_x), head_pos[..., 1] - box_ycenter) - wall_height))
    #     avoid_wall_2_head = -jnp.tanh(alpha * (jnp.maximum((wall_height/wall_halfwidth) * jnp.fabs(head_pos[..., 0] - right_wall_x), head_pos[..., 1] - box_ycenter) - wall_height))

    #     wall_avoid_value_foot = jnp.maximum(
    #         jnp.maximum(avoid_wall_1_front_foot, avoid_wall_1_back_foot),
    #         jnp.maximum(avoid_wall_2_front_foot, avoid_wall_2_back_foot)
    #     )
    #     wall_avoid_value_shin = jnp.maximum(
    #         jnp.maximum(avoid_wall_1_front_shin, avoid_wall_1_back_shin),
    #         jnp.maximum(avoid_wall_2_front_shin, avoid_wall_2_back_shin)
    #     )
    #     wall_avoid_value_thigh = jnp.maximum(
    #         jnp.maximum(avoid_wall_1_front_thigh, avoid_wall_1_back_thigh),
    #         jnp.maximum(avoid_wall_2_front_thigh, avoid_wall_2_back_thigh)
    #     )
    #     wall_avoid_value_head = jnp.maximum(avoid_wall_1_head, avoid_wall_2_head)
    #     wall_avoid_value = jnp.maximum(
    #         jnp.maximum(wall_avoid_value_foot, wall_avoid_value_shin),
    #         jnp.maximum(wall_avoid_value_thigh, wall_avoid_value_head)
    #     )

    #     avoid_value = jnp.maximum(box_avoid_value, wall_avoid_value)

    #     return avoid_value

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
    
    @partial(jax.jit, static_argnums=(0,))
    def reset_toinput(self, key, reset_obs, params=None):
        # Reset the environment to a specific observation
        
        # # NOTE: OLD ATTEMPT - RUNS BUT MAY NOT WORK ? 
        # old_state = self._env.reset(key)
        # old_state = replace(old_state, obs=reset_obs[:12]) # set the obs

        # Derived from Reset function in: 
        # 1. brax.envs.hopper 
        # 2. brax.envs.wrappers.training (EpisodeWrapper)
        # 3. brax.envs.wrappers.auto_reset (AutoResetWrapper)
        reset_obs = deepcopy(reset_obs[:18])

        ## HOPPER RESET FIX -- DO WE NEED FOR CHEETAH?
        # og_reset_obs = deepcopy(reset_obs)
        # reset_obs = reset_obs.at[1].set(og_reset_obs[1] - 1.25) # FIXME: don't know where this comes from exactly figure it out

        qpos = reset_obs[:9]
        qvel = reset_obs[9:18]
        pipeline_state = self._env.pipeline_init(qpos, qvel)
        obs = self._env._get_obs(pipeline_state)
        reward, done, zero = jnp.zeros(3)
        # metrics = {
        #     'reward_forward': zero,
        #     'reward_ctrl': zero,
        #     'reward_healthy': zero,
        #     'x_position': zero,
        #     'x_velocity': zero,
        # }
        metrics = {
            'x_position': zero,
            'x_velocity': zero,
            'reward_ctrl': zero,
            'reward_run': zero,
        }
        state = State(pipeline_state, obs, reward, done, metrics)
        # Episode Metrics 
        rng = key 
        state.info['steps'] = jnp.zeros(rng.shape[:-1])
        state.info['truncation'] = jnp.zeros(rng.shape[:-1])
        # Keep separate record of episode done as state.info['done'] can be erased
        # by AutoResetWrapper
        state.info['episode_done'] = jnp.zeros(rng.shape[:-1])
        episode_metrics = dict()
        episode_metrics['sum_reward'] = jnp.zeros(rng.shape[:-1])
        episode_metrics['length'] = jnp.zeros(rng.shape[:-1])
        for metric_name in state.metrics.keys():
            episode_metrics[metric_name] = jnp.zeros(rng.shape[:-1])
        state.info['episode_metrics'] = episode_metrics
        state.info['first_pipeline_state'] = state.pipeline_state
        state.info['first_obs'] = state.obs

        (head_pos, neck_pos, back_pos, front_thigh_pos, front_shin_pos,
         front_foot_pos, back_thigh_pos, back_shin_pos, back_foot_pos) = self.calculate_position(state.obs)
        avoid_value = self.is_avoid(head_pos, front_thigh_pos, front_shin_pos, front_foot_pos, back_thigh_pos, back_shin_pos, back_foot_pos)
        reach_value = self.is_reach(head_pos)
        observation = jnp.concatenate([state.obs, jnp.array([avoid_value, reach_value])])
        env_state = EnvStateAvoidOnly(state, avoid_value)

        # FIXME: does the observation need to be transformed?
        # observation = self._env.transform_obs(observation)?

        return observation, env_state