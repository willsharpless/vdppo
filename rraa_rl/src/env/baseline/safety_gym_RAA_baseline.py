import jax
import jax.numpy as jnp
from functools import partial
from gymnax.environments import spaces
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper
from flax import struct
from brax.envs.base import State

from ..reach_avoid.point_random import PointRandom, PointRandomOuter

SAFETYGYM_RAA_TARGET = [0., 0.]
SAFETYGYM_RAA_TARGET_RADIUS = 0.3
SAFETYGYM_RAA_OBSTACLE_RADIUS = 0.2
SAFETYGYM_RAA_BOX_RADIUS = 3.0

@struct.dataclass
class EnvStateRAA:
    state: State
    reach: float
    avoid: int
    min_reach: float # min reach value over trajectory - for state augmentation
    cost: float

@struct.dataclass
class EnvParamsEmpty:
    gamma: float = 0.99
    pass

class PointReachAlwaysAvoidBaseline_augmented:
    def __init__(self, backend="mjx"):
        env = PointRandomOuter(backend=backend)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)
        self.default_params = EnvParamsEmpty()
        self.hazard_pos_array = jnp.array([[1.403247, 0.6281236], [0.42943087, 1.17059302],
                                           [-1.16036429, 0.89811093], [-0.88776483, 1.46420776],
                                           [-0.07556364, -1.10567521], [0.72648704, 0.17957757],
                                           [-0.33115742, 0.83026827], [-1.33470321, -1.3259373]])
        
    @partial(jax.jit, static_argnums=(0,))
    def compute_reward(self, state, action, avoid_value, reach_value, params=None):
        # Compute reward for constrained MDP
        return params.gamma * reach_value - state.reach

    @partial(jax.jit, static_argnums=(0,))
    def compute_cost(self, state, action, avoid_value, reach_value, params=None):
        # Compute cost for constrained MDP
        return state.avoid
    
    @partial(jax.jit, static_argnums=(0,))
    def compute_observation(self, state, last_state=None): 
        # Compute observation for constrained MDP
        return jnp.concatenate([state.state.obs, jnp.array([state.min_reach])])

    @partial(jax.jit, static_argnums=(0,))
    def is_reach(self, state):
        target_center, radius = SAFETYGYM_RAA_TARGET, SAFETYGYM_RAA_TARGET_RADIUS
        target_pos = state
        reach = jnp.sqrt((target_pos[..., 0] - target_center[0]) ** 2 + \
                         (target_pos[..., 1] - target_center[1]) ** 2) - radius
        value = jnp.where(reach < 0., -3., reach)
        return value * 100.0
    
    @partial(jax.jit, static_argnums=(0,))
    def is_avoid(self, state):
        radius = SAFETYGYM_RAA_OBSTACLE_RADIUS
        target_pos = state

        ## AVOID OBSTACLES
        avoid_obstacles = -jnp.inf
        for hazard_pos in self.hazard_pos_array:
            avoid = -(jnp.sqrt((target_pos[..., 0] - hazard_pos[0]) ** 2 + \
                             (target_pos[..., 1] - hazard_pos[1]) ** 2) - radius)
            avoid_obstacles = jnp.maximum(avoid_obstacles, avoid)
        
        ## AVOID WALLS
        avoid_wall = jnp.maximum(jnp.fabs(target_pos[..., 0]), jnp.fabs(target_pos[..., 1])) - SAFETYGYM_RAA_BOX_RADIUS

        avoid = jnp.maximum(10. * avoid_obstacles, 0.1 * avoid_wall)
        value = jnp.where(avoid > 0., 3., avoid)
        return value * 100.0

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

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)

        reach_value = self.is_reach(state.obs[0:2])
        avoid_value = self.is_avoid(state.obs[0:2])

        cost = avoid_value
        min_reach = reach_value
        env_state = EnvStateRAA(state, reach_value, avoid_value, min_reach, cost)
        observation = self.compute_observation(state=env_state, last_state=None)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.clip(action, -1., 1.)
        next_state = self._env.step(state.state, u)

        reach_value = self.is_reach(next_state.obs[0:2])
        avoid_value = self.is_avoid(next_state.obs[0:2])
        min_reach = jnp.minimum(state.min_reach, reach_value)

        reward = self.compute_reward(state=state, action=action, avoid_value=avoid_value, reach_value=reach_value, params=params) #params.gamma * reach_value - state.reach
        cost = self.compute_cost(state=state, action=action, avoid_value=avoid_value, reach_value=reach_value, params=params) #avoid_value

        next_state_new = EnvStateRAA(next_state, reach_value, avoid_value, min_reach, cost)
        observation = self.compute_observation(state=next_state_new, last_state=state)
        done = False

        return observation, next_state_new, reward, done, {"x": state.state.obs[0], "y": state.state.obs[1], "theta": jnp.arcsin(state.state.obs[2])}