import jax
import jax.numpy as jnp
from functools import partial
from gymnax.environments import spaces
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper
from flax import struct
from brax.envs.base import State
from .half_cheetah_random import HalfCheetahRandom
from copy import deepcopy 

# @struct.dataclass
# class EnvStateRA:
#     state: jax.Array = struct.field(default_factory=jax.Array)
#     avoid: float = 0.
#     reach: float = 0.
#     has_reached: float = 0.

# @struct.dataclass
# class EnvStateAvoidOnly:
#     state: jax.Array = struct.field(default_factory=jax.Array)
#     avoid: float = 0.

@struct.dataclass
class EnvStateRAA:
    state: State
    reach: float
    avoid: int
    min_reach: float # min reach value over trajectory - for state augmentation
    cost: float

@struct.dataclass
class EnvParams:
    gamma: float = 0.99

class HalfCheetahReachAlwaysAvoidBaseline_augmented:
    def __init__(self, backend="positional"):
        env = HalfCheetahRandom(backend=backend,
                           exclude_current_positions_from_observation=False)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)
        self.default_params = EnvParams()

    @partial(jax.jit, static_argnums=(0,))
    def compute_observation(self, state, last_state=None): 
        # Compute observation for constrained MDP
        return jnp.concatenate([state.state.obs, jnp.array([state.min_reach])])

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)
        poses = self.calculate_position(state.obs)
        avoid_value = self.is_avoid(poses)
        reach_value = self.is_reach(poses[0])
        cost = avoid_value
        min_reach = reach_value
        env_state = EnvStateRAA(state, reach_value, avoid_value, min_reach, cost)
        observation = self.compute_observation(state=env_state, last_state=None)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.tanh(action)
        next_state = self._env.step(state.state, u)
        poses = self.calculate_position(next_state.obs)
        avoid_value = self.is_avoid(poses)
        reach_value = self.is_reach(poses[0])
        min_reach = jnp.minimum(state.min_reach, reach_value)

        reward = params.gamma * reach_value - state.reach
        cost = avoid_value

        (head_pos, neck_pos, back_pos, front_thigh_pos, front_shin_pos,
         front_foot_pos, back_thigh_pos, back_shin_pos, back_foot_pos) = self.calculate_position(state.state.obs)
        pos_dict = {"head_pos": head_pos, "neck_pos": neck_pos, "back_pos": back_pos,
                    "front_thigh_pos": front_thigh_pos, "front_shin_pos": front_shin_pos, "front_foot_pos": front_foot_pos,
                    "back_thigh_pos": back_thigh_pos, "back_shin_pos": back_shin_pos, "back_foot_pos": back_foot_pos}

        next_state_new = EnvStateRAA(next_state, reach_value, avoid_value, min_reach, cost)
        observation = self.compute_observation(state=next_state_new, last_state=state)
        done = False # NOTE: Force dones to false for always avoid - make last done true outside #(state.avoid > 0) | (state.reach < 0)

        # return observation, next_state_new, reward, next_state.done > 0.5, pos_dict # NOTE done is always false
        return observation, next_state_new, reward, done, pos_dict

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
    def is_avoid(self, poses):
        
        ## Indices for different body parts
        part_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8] # all
        # part_indices = [0, 3, 4, 5, 6, 7, 8]  # excludes neck_pos and back_pos
        # part_weights = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        ## BOX AVOIDANCE
        box_height, box_halfwidth, box_ycenter, box_front_xcenter, box_back_xcenter = 0.05, 0.25, -0.7, 4.5, 6.5
        box_centers = [box_front_xcenter, box_back_xcenter]
        box_avoid_value = -jnp.inf
        for i in part_indices:
            pos = poses[i]
            for xcenter in box_centers:
                avoid = -(jnp.maximum(
                    (box_height / box_halfwidth) * jnp.fabs(pos[..., 0] - xcenter),
                    pos[..., 1] - box_ycenter
                ) - box_height)
                box_avoid_value = jnp.maximum(box_avoid_value, avoid)

        ## WALL AVOIDANCE
        wall_height, wall_halfwidth = 1., 0.5
        left_wall_x, right_wall_x = -0.9, 5.5
        wall_centers = [left_wall_x]

        wall_avoid_value = -jnp.inf
        for i in part_indices:
            pos = poses[i]
            for xcenter in wall_centers:
                avoid = -(jnp.maximum(
                    (wall_height / wall_halfwidth) * jnp.fabs(pos[..., 0] - xcenter),
                    pos[..., 1] - box_ycenter
                ) - wall_height)
                wall_avoid_value = jnp.maximum(wall_avoid_value, avoid)

        ## Combine Ball Avoidance and Wall Avoidance
        avoid_value = jnp.maximum(10. * box_avoid_value, 0.01 * wall_avoid_value)

        return avoid_value

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size + 1,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )