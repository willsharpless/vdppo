import jax
import jax.numpy as jnp
from functools import partial
from gymnax.environments import spaces
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper
from flax import struct
from brax.envs.base import State
from copy import deepcopy
from ..reach_avoid.point_random import PointRandom

SAFETYGYM_TARGET_RIGHT, SAFETYGYM_TARGET_LEFT = [2.5, 2.5], [-2.5, -2.5] # v0
SAFETYGYM_TARGET_RADIUS = 0.3 # v0

# SAFETYGYM_TARGET_RIGHT, SAFETYGYM_TARGET_LEFT = [2., 2.], [-2., -2.] # v1
# SAFETYGYM_TARGET_RADIUS = 0.4 # v1

@struct.dataclass
class EnvStateR1:
    state: jax.Array = struct.field(default_factory=jax.Array)
    reach1: float = 0.

@struct.dataclass
class EnvStateR2:
    state: jax.Array = struct.field(default_factory=jax.Array)
    reach2: float = 0.

@struct.dataclass
class EnvStateRRDecomposed:
    state: jax.Array = struct.field(default_factory=jax.Array)
    reach1: float = 0.
    reach2: float = 0.
    has_reached_1: float = 0.
    has_reached_2: float = 0.

@struct.dataclass
class EnvStateR:
    state: jax.Array = struct.field(default_factory=jax.Array)
    reach: float = 0.

@struct.dataclass
class EnvStateRR:
    state: jax.Array = struct.field(default_factory=jax.Array)
    reach1: float = 0.
    reach2: float = 0.
    has_reached_1: float = 0.
    has_reached_2: float = 0.
    min_reach1: float = 0.
    min_reach2: float = 0.
    cost : float = 0.


class PointRRTemplate:
    def __init__(self, backend="mjx"):
        env = PointRandom(backend=backend)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)
        self.default_params = EnvParamsEmpty()
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        raise NotImplementedError("reset() not implemented in base class")

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        raise NotImplementedError("step() not implemented in base class")

    @partial(jax.jit, static_argnums=(0,))
    def is_reach1(self, state):
        target_center, radius = SAFETYGYM_TARGET_RIGHT, SAFETYGYM_TARGET_RADIUS
        target_pos = state
        reach = jnp.sqrt((target_pos[..., 0] - target_center[0]) ** 2 + \
                         (target_pos[..., 1] - target_center[1]) ** 2) - radius
        value = jnp.where(reach < 0., -3., reach)
        return value * 100.0
    
    @partial(jax.jit, static_argnums=(0,))
    def is_reach2(self, state):
        target_center, radius = SAFETYGYM_TARGET_LEFT, SAFETYGYM_TARGET_RADIUS
        target_pos = state
        reach = jnp.sqrt((target_pos[..., 0] - target_center[0]) ** 2 + \
                         (target_pos[..., 1] - target_center[1]) ** 2) - radius
        value = jnp.where(reach < 0., -3., reach)
        return value * 100.0

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )


class PointRR(PointRRTemplate):
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)
        reach1_value = self.is_reach1(state.obs)
        reach2_value = self.is_reach2(state.obs)
        has_reached_1 = reach1_value < 0
        has_reached_2 = reach2_value < 0
        observation = state.obs
        env_state = EnvStateRRDecomposed(state, reach1_value, reach2_value, has_reached_1, has_reached_2)
        return observation, env_state
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.clip(action, -1., 1.)
        next_state = self._env.step(state.state, u)
        reach1_value = self.is_reach1(next_state.obs)
        reach2_value = self.is_reach2(next_state.obs)
        has_reached_1 = jnp.logical_or(reach1_value < 0, state.has_reached_1)
        has_reached_2 = jnp.logical_or(reach2_value < 0, state.has_reached_2)
        pos_dict = {"x": state.state.obs[0], "y": state.state.obs[1], "theta": jnp.arcsin(state.state.obs[2])}
        observation = next_state.obs
        next_state_new = EnvStateRRDecomposed(next_state, reach1_value, reach2_value, has_reached_1, has_reached_2)
        reward = 0.
        done = jnp.logical_or(has_reached_1, has_reached_2)
        return observation, next_state_new, reward, done, pos_dict

class PointR1(PointRRTemplate):
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)
        reach1_value = self.is_reach1(state.obs)
        observation = state.obs
        env_state = EnvStateR(state, reach1_value)
        return observation, env_state
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.clip(action, -1., 1.)
        next_state = self._env.step(state.state, u)
        reach1_value = self.is_reach1(next_state.obs)
        pos_dict = {"x": state.state.obs[0], "y": state.state.obs[1], "theta": jnp.arcsin(state.state.obs[2])}
        observation = next_state.obs
        next_state_new = EnvStateR(next_state, reach1_value)
        reward = 0.
        done = next_state.done > 0.5
        return observation, next_state_new, reward, done, pos_dict

class PointR2(PointRRTemplate):
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)
        reach2_value = self.is_reach2(state.obs)
        observation = state.obs
        env_state = EnvStateR(state, reach2_value)
        return observation, env_state
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.clip(action, -1., 1.)
        next_state = self._env.step(state.state, u)
        reach2_value = self.is_reach2(next_state.obs)
        pos_dict = {"x": state.state.obs[0], "y": state.state.obs[1], "theta": jnp.arcsin(state.state.obs[2])}
        observation = next_state.obs
        next_state_new = EnvStateR(next_state, reach2_value)
        reward = 0.
        done = next_state.done > 0.5
        return observation, next_state_new, reward, done, pos_dict


@struct.dataclass
class EnvParamsEmpty:
    gamma: float = 0.99
    pass

class PointReachReachBaseline_augmented:
    def __init__(self, backend="mjx", use_stl=False):
        env = PointRandom(backend=backend)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)
        self.default_params = EnvParamsEmpty()
        self.use_stl = use_stl

    def compute_cost_accumulated(self, curr_min_reach1_value, curr_min_reach2_value, prev_min_reach1_value=None, prev_min_reach2_value=None, 
                     reach1_value=None, reach2_value=None): 
        if prev_min_reach1_value is None or prev_min_reach2_value is None:
            cost = jnp.maximum(curr_min_reach1_value, curr_min_reach2_value)
        else: 
            cost = jnp.minimum(curr_min_reach1_value, prev_min_reach1_value) + jnp.minimum(curr_min_reach2_value, prev_min_reach2_value)
            # corresponds to accumulated sum cost
        return cost 
    
    def compute_reward(self, state, last_state, params): 
        # return params.gamma * jnp.maximum(state.min_reach1, state.min_reach2) - jnp.maximum(last_state.min_reach1, last_state.min_reach2) 
        # corresponds to accumulated max reward
        
        return params.gamma * (state.min_reach1 + state.min_reach2) - (last_state.min_reach1 + last_state.min_reach2)
        # corresponds to accumulated sum reward

    @partial(jax.jit, static_argnums=(0,))
    def is_reach1(self, state):
        target_center, radius = SAFETYGYM_TARGET_RIGHT, SAFETYGYM_TARGET_RADIUS
        target_pos = state
        reach = jnp.sqrt((target_pos[..., 0] - target_center[0]) ** 2 + \
                         (target_pos[..., 1] - target_center[1]) ** 2) - radius
        value = jnp.where(reach < 0., -3., reach)
        return value * 100.0
    
    @partial(jax.jit, static_argnums=(0,))
    def is_reach2(self, state):
        target_center, radius = SAFETYGYM_TARGET_LEFT, SAFETYGYM_TARGET_RADIUS
        target_pos = state
        reach = jnp.sqrt((target_pos[..., 0] - target_center[0]) ** 2 + \
                         (target_pos[..., 1] - target_center[1]) ** 2) - radius
        value = jnp.where(reach < 0., -3., reach)
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

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)

        reach1_value = self.is_reach1(state.obs[0:2])
        reach2_value = self.is_reach2(state.obs[0:2])
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
        u = jnp.clip(action, -1., 1.)
        next_state = self._env.step(state.state, u)

        reach1_value = self.is_reach1(next_state.obs[0:2])
        reach2_value = self.is_reach2(next_state.obs[0:2])
        has_reached_1 = jnp.logical_or(reach1_value < 0, state.has_reached_1)
        has_reached_2 = jnp.logical_or(reach2_value < 0, state.has_reached_2)

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

        observation = jnp.concatenate([next_state.obs, jnp.array([reach1_value, reach2_value])])
        next_state_new = EnvStateRR(next_state, reach1_value, reach2_value, has_reached_1, has_reached_2, min_reach1, min_reach2, cost)
        reward = self.compute_reward(state=next_state_new, last_state=state, params=params)

        done = has_reached_1 & has_reached_2

        return observation, next_state_new, reward, done, {"x": state.state.obs[0], "y": state.state.obs[1], "theta": jnp.arcsin(state.state.obs[2])}
