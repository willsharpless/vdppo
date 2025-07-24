import jax
import jax.numpy as jnp
from functools import partial
from gymnax.environments import spaces
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper
from flax import struct
from brax.envs.base import State
from ..reach_avoid.half_cheetah_random import HalfCheetahRandom
from ..reach_avoid.half_cheetah_deterministic import HalfCheetahDeterministic
from copy import deepcopy 

@struct.dataclass
class EnvStateR1:
    state: jax.Array = struct.field(default_factory=jax.Array)
    reach1: float = 0.

@struct.dataclass
class EnvStateR2:
    state: jax.Array = struct.field(default_factory=jax.Array)
    reach2: float = 0.

@struct.dataclass
class EnvStateRR:
    state: jax.Array = struct.field(default_factory=jax.Array)
    time: int = 0
    reach1: float = 0.
    reach2: float = 0.
    has_reached_1: float = 0.
    has_reached_2: float = 0.
    min_reach1: float = 0.
    min_reach2: float = 0.
    cost : float = 0.

@struct.dataclass
class EnvParamsEmpty:
    gamma: float = 0.99
    pass

class HalfCheetahReachReachBaseline_augmented:

    def compute_cost_accumulated(self, curr_min_reach1_value, curr_min_reach2_value, prev_min_reach1_value=None, prev_min_reach2_value=None, 
                     reach1_value=None, reach2_value=None): 
        if prev_min_reach1_value is None or prev_min_reach2_value is None:
            cost = jnp.maximum(curr_min_reach1_value, curr_min_reach2_value)
        else: 
            cost = jnp.minimum(curr_min_reach1_value, prev_min_reach1_value) + jnp.minimum(curr_min_reach2_value, prev_min_reach2_value)
            # corresponds to accumulated sum cost
        return cost 
    
    def compute_reward(self, state, last_state, params): 
        return params.gamma * jnp.maximum(state.min_reach1, state.min_reach2) - jnp.maximum(last_state.min_reach1, last_state.min_reach2) 
        # corresponds to accumulated max reward
        
        # should we be using this?
        # return params.gamma * (state.min_reach1 + state.min_reach2) - (last_state.min_reach1 + last_state.min_reach2)
        # corresponds to accumulated sum reward

    def __init__(self, backend="positional", use_stl=False):
        env = HalfCheetahRandom(backend=backend,
                           exclude_current_positions_from_observation=False)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)
        self.default_params = EnvParamsEmpty()
        self.use_stl = use_stl

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)
        poses, vels = self.calculate_position(state.obs)
        
        reach1_value = self.is_reach1(poses, vels)
        reach2_value = self.is_reach2(poses, vels)
        has_reached_1 = reach1_value < 0
        has_reached_2 = reach2_value < 0
        
        min_reach1 = reach1_value
        min_reach2 = reach2_value
        cost = self.compute_cost_accumulated(curr_min_reach1_value=min_reach1,
                                             curr_min_reach2_value=min_reach2,
                                             prev_min_reach1_value=None,
                                             prev_min_reach2_value=None,
                                             reach1_value=reach1_value,
                                             reach2_value=reach2_value) 
        
        observation = jnp.concatenate([state.obs, jnp.array([min_reach1, min_reach2])])
        env_state = EnvStateRR(state, reach1_value, reach2_value, has_reached_1, has_reached_2, min_reach1, min_reach2, cost)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.tanh(action)
        next_state = self._env.step(state.state, u)
        poses, vels = self.calculate_position(next_state.obs)
        
        reach1_value = self.is_reach1(poses, vels)
        reach2_value = self.is_reach2(poses, vels)
        
        has_reached_1 = jnp.logical_or(state.has_reached_1, reach1_value < 0)
        has_reached_2 = jnp.logical_or(state.has_reached_2, reach2_value < 0)
        
        min_reach1 = jnp.minimum(state.min_reach1, reach1_value)
        min_reach2 = jnp.minimum(state.min_reach2, reach2_value)
        
        min_reach1_cost_input = deepcopy(min_reach1)
        min_reach2_cost_input = deepcopy(min_reach2)
        reach1_cost_input = deepcopy(reach1_value)
        reach2_cost_input = deepcopy(reach2_value)

        if self.use_stl: 
            reach1_cost_input = jnp.where(has_reached_1, 0, reach2_cost_input)
            reach2_cost_input = jnp.where(has_reached_2, 0, reach1_cost_input)
            min_reach1_cost_input = jnp.where(has_reached_1, 0, min_reach1_cost_input)
            min_reach2_cost_input = jnp.where(has_reached_2, 0, min_reach2_cost_input)
        
        cost = self.compute_cost_accumulated(min_reach1_cost_input, min_reach2_cost_input, state.min_reach1, state.min_reach2, 
                                reach1_value=reach1_cost_input, reach2_value=reach2_cost_input)
            
        (head_pos, neck_pos, back_pos, front_thigh_pos, front_shin_pos,
         front_foot_pos, back_thigh_pos, back_shin_pos, back_foot_pos), vels = self.calculate_position(state.state.obs)
        pos_dict = {"head_pos": head_pos, "neck_pos": neck_pos, "back_pos": back_pos,
                    "front_thigh_pos": front_thigh_pos, "front_shin_pos": front_shin_pos, "front_foot_pos": front_foot_pos,
                    "back_thigh_pos": back_thigh_pos, "back_shin_pos": back_shin_pos, "back_foot_pos": back_foot_pos,
                    "x_vel": vels[0], "z_vel": vels[1], "y_ang_vel": vels[2]}
        
        observation = jnp.concatenate([next_state.obs, jnp.array([reach1_value, reach2_value])])
        next_state_new = EnvStateRR(next_state, reach1_value, reach2_value, has_reached_1, has_reached_2, min_reach1, min_reach2, cost)
        reward = self.compute_reward(state=next_state_new, last_state=state, params=params)

        done = has_reached_1 & has_reached_2

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

        vels = jnp.array([obs[9], obs[10], obs[11]])  # x_vel, z_vel, y_ang_vel
        
        return (head_pos, neck_pos, back_pos, front_thigh_pos, front_shin_pos, front_foot_pos,
                back_thigh_pos, back_shin_pos, back_foot_pos), vels

    # @partial(jax.jit, static_argnums=(0,))
    # def is_reach1(self, poses, vels):
    #     (head_pos, neck_pos, back_pos, front_thigh_pos, front_shin_pos,
    #       front_foot_pos, back_thigh_pos, back_shin_pos, back_foot_pos) = poses
    #     (x_vel, z_vel, y_ang_vel) = vels
        
    #     target_x_vel = 10.
    #     reach1_value = -(x_vel - target_x_vel) # negative when speed achieved
    #     return reach1_value * 10
    
    # @partial(jax.jit, static_argnums=(0,))
    # def is_reach2(self, poses, vels):
    #     (head_pos, neck_pos, back_pos, front_thigh_pos, front_shin_pos,
    #       front_foot_pos, back_thigh_pos, back_shin_pos, back_foot_pos) = poses
    #     (x_vel, z_vel, y_ang_vel) = vels
        
    #     target_ang_vel = 15.
    #     reach2_value = -(jnp.fabs(y_ang_vel) - target_ang_vel) # negative when speed achieved
    #     return reach2_value * 10

    @partial(jax.jit, static_argnums=(0,))
    def is_reach1(self, poses, vels):
        (head_pos, neck_pos, back_pos, front_thigh_pos, front_shin_pos,
          front_foot_pos, back_thigh_pos, back_shin_pos, back_foot_pos) = poses
        (x_vel, z_vel, y_ang_vel) = vels
        
        target_center, radius = [5., 1.], 0.1
        target_pos = front_foot_pos
        reach = jnp.sqrt((target_pos[..., 0] - target_center[0]) ** 2 + (target_pos[..., 1] - target_center[1]) ** 2) - radius
        has_reached_goal = jnp.sqrt((target_pos[..., 0] - target_center[0]) ** 2 + (target_pos[..., 1] - target_center[1]) ** 2) < radius
        value = jnp.where(has_reached_goal, -2.5, reach)
        return value * 100.0
    
    @partial(jax.jit, static_argnums=(0,))
    def is_reach2(self, poses, vels):
        (head_pos, neck_pos, back_pos, front_thigh_pos, front_shin_pos,
          front_foot_pos, back_thigh_pos, back_shin_pos, back_foot_pos) = poses
        (x_vel, z_vel, y_ang_vel) = vels
        
        target_center, radius = [-5., 1.], 0.1
        target_pos = back_foot_pos
        reach = jnp.sqrt((target_pos[..., 0] - target_center[0]) ** 2 + (target_pos[..., 1] - target_center[1]) ** 2) - radius
        has_reached_goal = jnp.sqrt((target_pos[..., 0] - target_center[0]) ** 2 + (target_pos[..., 1] - target_center[1]) ** 2) < radius
        value = jnp.where(has_reached_goal, -2.5, reach)
        return value * 100.0

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