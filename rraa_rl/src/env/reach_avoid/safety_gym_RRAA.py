import jax
import jax.numpy as jnp
from functools import partial
from gymnax.environments import spaces
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper
from flax import struct
from brax.envs.base import State

from .point_random import PointRandom

SAFETYGYM_TARGET_RIGHT, SAFETYGYM_TARGET_LEFT = [2.5, 2.5], [-2.5, -2.5] # v0
SAFETYGYM_TARGET_RADIUS = 0.3 # v0

SAFETYGYM_RAA_OBSTACLE_RADIUS = 0.2
SAFETYGYM_RAA_BOX_RADIUS = 3.0

SAFETYGYM_OBSTACLE_SET = jnp.array([[1.403247, 0.6281236], [0.42943087, 1.17059302],
                                    [-1.16036429, 0.89811093], [-0.88776483, 1.46420776],
                                    [-0.07556364, -1.10567521], [0.72648704, 0.17957757],
                                    [-0.33115742, 0.83026827], [-1.33470321, -1.3259373]])

# SAFETYGYM_TARGET_RIGHT, SAFETYGYM_TARGET_LEFT = [2., 2.], [-2., -2.] # v1
# SAFETYGYM_TARGET_RADIUS = 0.4 # v1

@struct.dataclass
class EnvStateA:
    state: jax.Array = struct.field(default_factory=jax.Array)
    avoid: float = 0.

@struct.dataclass
class EnvStateRAA:
    state: jax.Array = struct.field(default_factory=jax.Array)
    reach: float = 0.
    avoid: float = 0.
    has_reached: float = 0.

@struct.dataclass
class EnvStateRRAA:
    state: jax.Array = struct.field(default_factory=jax.Array)
    reach1: float = 0.
    reach2: float = 0.
    avoid: float = 0.
    has_reached_1: float = 0.
    has_reached_2: float = 0.
    has_reached_both: float = 0.

@struct.dataclass
class EnvParamsEmpty:
    pass

class PointRRAATemplate:
    def __init__(self, backend="mjx"):
        env = PointRandom(backend=backend)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)
        self.default_params = EnvParamsEmpty()
        self.hazard_pos_array = SAFETYGYM_OBSTACLE_SET

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

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        raise NotImplementedError("reset() not implemented in base class")

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        raise NotImplementedError("step() not implemented in base class")

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

class PointRRAA(PointRRAATemplate):

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)

        reach1_value = self.is_reach1(state.obs[0:2])
        reach2_value = self.is_reach2(state.obs[0:2])
        has_reached_1 = reach1_value < 0
        has_reached_2 = reach2_value < 0
        has_reached_both = jnp.logical_and(has_reached_1, has_reached_2)

        avoid_value = self.is_avoid(state.obs[0:2])
        
        observation = jnp.concatenate([state.obs, jnp.array([reach1_value, reach2_value, avoid_value])])
        env_state = EnvStateRRAA(state, reach1_value, reach2_value, avoid_value, has_reached_1, has_reached_2, has_reached_both)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.clip(action, -1., 1.)
        next_state = self._env.step(state.state, u)

        reach1_value = self.is_reach1(next_state.obs[0:2])
        reach2_value = self.is_reach2(next_state.obs[0:2])
        has_reached_1 = jnp.logical_or(reach1_value < 0, state.has_reached_1)
        has_reached_2 = jnp.logical_or(reach2_value < 0, state.has_reached_2)
        has_reached_both = jnp.logical_and(has_reached_1, has_reached_2)
        avoid_value = self.is_avoid(next_state.obs[0:2])

        observation = jnp.concatenate([next_state.obs, jnp.array([reach1_value, reach2_value, avoid_value])])
        next_state_new = EnvStateRRAA(next_state, reach1_value, reach2_value, avoid_value, has_reached_1, has_reached_2, has_reached_both)
        reward = 0.
        # done = jnp.logical_or(next_state.done > 0.5, avoid_value > 0) # NOTE not done on reaching to ensure no crash
        done = next_state.done > 0.5

        return observation, next_state_new, reward, done, {"x": state.state.obs[0], "y": state.state.obs[1], "theta": jnp.arcsin(state.state.obs[2])}
    
class PointRAA1(PointRRAATemplate):

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)

        reach1_value = self.is_reach1(state.obs[0:2])
        reach2_value = self.is_reach2(state.obs[0:2])
        has_reached_1 = reach1_value < 0
        avoid_value = self.is_avoid(state.obs[0:2])
        
        observation = jnp.concatenate([state.obs, jnp.array([reach1_value, reach2_value, avoid_value])])
        env_state = EnvStateRAA(state, reach1_value, avoid_value, has_reached_1)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.clip(action, -1., 1.)
        next_state = self._env.step(state.state, u)

        reach1_value = self.is_reach1(next_state.obs[0:2])
        reach2_value = self.is_reach2(next_state.obs[0:2])
        has_reached_1 = jnp.logical_or(reach1_value < 0, state.has_reached_1)
        avoid_value = self.is_avoid(next_state.obs[0:2])

        observation = jnp.concatenate([next_state.obs, jnp.array([reach1_value, reach2_value, avoid_value])])
        next_state_new = EnvStateRAA(next_state, reach1_value, avoid_value, has_reached_1)
        reward = 0.
        # done = jnp.logical_or(next_state.done > 0.5, avoid_value > 0) # NOTE not done on reaching to ensure no crash
        done = next_state.done > 0.5

        return observation, next_state_new, reward, done, {"x": state.state.obs[0], "y": state.state.obs[1], "theta": jnp.arcsin(state.state.obs[2])}
    
    @partial(jax.jit, static_argnums=(0,))
    def reset_toinput(self, key, reset_obs, params=None):
        # reset_obs = deepcopy(reset_obs[:53])
        
        ## Remake Pipeline State
        qpos = reset_obs[0:3]
        qpos.at[2].set(jnp.arcsin(qpos[2])) # FIXME? _get_obs looks like it takes sin/cos of angle
        qvel = reset_obs[4:7]
        pipeline_state = self._env.pipeline_init(qpos, qvel)
        obs = self._env._get_obs(pipeline_state)
        reward, done, zero = jnp.zeros(3)

        ## Define Metrics
        metrics = {
            'forward_reward': zero,
            'reward_linvel': zero,
            'reward_quadctrl': zero,
            'reward_alive': zero,
            'x_position': zero,
            'y_position': zero,
            'distance_from_origin': zero,
            'x_velocity': zero,
            'y_velocity': zero,
        }
        state = State(pipeline_state, obs, reward, done, metrics)
        
        ## Set Auxiliaries
        rng = key 
        for key_name in ['steps', 'truncation', 'episode_done']:
            state.info[key_name] = jnp.zeros(rng.shape[:-1])
        episode_metrics = dict()
        episode_metrics['sum_reward'] = jnp.zeros(rng.shape[:-1])
        episode_metrics['length'] = jnp.zeros(rng.shape[:-1])
        for metric_name in state.metrics.keys():
            episode_metrics[metric_name] = jnp.zeros(rng.shape[:-1])
        state.info['episode_metrics'] = episode_metrics
        state.info['first_pipeline_state'] = state.pipeline_state
        state.info['first_obs'] = state.obs

        ## Set Observation and EnvState
        reach1_value = self.is_reach1(state.obs[0:2])
        reach2_value = self.is_reach2(state.obs[0:2])
        has_reached_1 = jnp.logical_or(reach1_value < 0, state.has_reached_1)
        avoid_value = self.is_avoid(state.obs[0:2])
        observation = jnp.concatenate([state.obs, jnp.array([reach1_value, reach2_value, avoid_value])])
        env_state = EnvStateRAA(state, reach1_value, avoid_value, has_reached_1)

        return observation, env_state
    
class PointRAA2(PointRRAATemplate):

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)

        reach1_value = self.is_reach1(state.obs[0:2])
        reach2_value = self.is_reach2(state.obs[0:2])
        has_reached_2 = reach2_value < 0
        avoid_value = self.is_avoid(state.obs[0:2])
        
        observation = jnp.concatenate([state.obs, jnp.array([reach1_value, reach2_value, avoid_value])])
        env_state = EnvStateRAA(state, reach2_value, avoid_value, has_reached_2)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.clip(action, -1., 1.)
        next_state = self._env.step(state.state, u)

        reach1_value = self.is_reach1(next_state.obs[0:2])
        reach2_value = self.is_reach2(next_state.obs[0:2])
        has_reached_2 = jnp.logical_or(reach2_value < 0, state.has_reached_2)
        avoid_value = self.is_avoid(next_state.obs[0:2])

        observation = jnp.concatenate([next_state.obs, jnp.array([reach1_value, reach2_value, avoid_value])])
        next_state_new = EnvStateRAA(next_state, reach2_value, avoid_value, has_reached_2)
        reward = 0.
        # done = jnp.logical_or(next_state.done > 0.5, avoid_value > 0) # NOTE not done on reaching to ensure no crash
        done = next_state.done > 0.5

        return observation, next_state_new, reward, done, {"x": state.state.obs[0], "y": state.state.obs[1], "theta": jnp.arcsin(state.state.obs[2])}
    
    @partial(jax.jit, static_argnums=(0,))
    def reset_toinput(self, key, reset_obs, params=None):
        # reset_obs = deepcopy(reset_obs[:53])
        
        ## Remake Pipeline State
        qpos = reset_obs[0:3]
        qpos.at[2].set(jnp.arcsin(qpos[2])) # FIXME? _get_obs looks like it takes sin/cos of angle
        qvel = reset_obs[4:7]
        pipeline_state = self._env.pipeline_init(qpos, qvel)
        obs = self._env._get_obs(pipeline_state)
        reward, done, zero = jnp.zeros(3)

        ## Define Metrics
        metrics = {
            'forward_reward': zero,
            'reward_linvel': zero,
            'reward_quadctrl': zero,
            'reward_alive': zero,
            'x_position': zero,
            'y_position': zero,
            'distance_from_origin': zero,
            'x_velocity': zero,
            'y_velocity': zero,
        }
        state = State(pipeline_state, obs, reward, done, metrics)
        
        ## Set Auxiliaries
        rng = key 
        for key_name in ['steps', 'truncation', 'episode_done']:
            state.info[key_name] = jnp.zeros(rng.shape[:-1])
        episode_metrics = dict()
        episode_metrics['sum_reward'] = jnp.zeros(rng.shape[:-1])
        episode_metrics['length'] = jnp.zeros(rng.shape[:-1])
        for metric_name in state.metrics.keys():
            episode_metrics[metric_name] = jnp.zeros(rng.shape[:-1])
        state.info['episode_metrics'] = episode_metrics
        state.info['first_pipeline_state'] = state.pipeline_state
        state.info['first_obs'] = state.obs

        ## Set Observation and EnvState
        reach1_value = self.is_reach1(state.obs[0:2])
        reach2_value = self.is_reach2(state.obs[0:2])
        has_reached_2 = jnp.logical_or(reach2_value < 0, state.has_reached_2)
        avoid_value = self.is_avoid(state.obs[0:2])
        observation = jnp.concatenate([state.obs, jnp.array([reach1_value, reach2_value, avoid_value])])
        env_state = EnvStateRAA(state, reach2_value, avoid_value, has_reached_2)

        return observation, env_state
    
class PointA(PointRRAATemplate):

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)

        reach1_value = self.is_reach1(state.obs[0:2])
        reach2_value = self.is_reach2(state.obs[0:2])
        avoid_value = self.is_avoid(state.obs[0:2])
        
        observation = jnp.concatenate([state.obs, jnp.array([reach1_value, reach2_value, avoid_value])])
        env_state = EnvStateA(state, avoid_value)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.clip(action, -1., 1.)
        next_state = self._env.step(state.state, u)

        reach1_value = self.is_reach1(next_state.obs[0:2])
        reach2_value = self.is_reach2(next_state.obs[0:2])
        avoid_value = self.is_avoid(next_state.obs[0:2])

        observation = jnp.concatenate([next_state.obs, jnp.array([reach1_value, reach2_value, avoid_value])])
        next_state_new = EnvStateA(next_state, avoid_value)
        reward = 0.
        done = next_state.done > 0.5

        return observation, next_state_new, reward, done, {"x": state.state.obs[0], "y": state.state.obs[1], "theta": jnp.arcsin(state.state.obs[2])}
    
    @partial(jax.jit, static_argnums=(0,))
    def reset_toinput(self, key, reset_obs, params=None):
        # reset_obs = deepcopy(reset_obs[:53])
        
        ## Remake Pipeline State
        qpos = reset_obs[0:3]
        qpos.at[2].set(jnp.arcsin(qpos[2])) # FIXME? _get_obs looks like it takes sin/cos of angle
        qvel = reset_obs[4:7]
        pipeline_state = self._env.pipeline_init(qpos, qvel)
        obs = self._env._get_obs(pipeline_state)
        reward, done, zero = jnp.zeros(3)

        ## Define Metrics
        metrics = {
            'forward_reward': zero,
            'reward_linvel': zero,
            'reward_quadctrl': zero,
            'reward_alive': zero,
            'x_position': zero,
            'y_position': zero,
            'distance_from_origin': zero,
            'x_velocity': zero,
            'y_velocity': zero,
        }
        state = State(pipeline_state, obs, reward, done, metrics)
        
        ## Set Auxiliaries
        rng = key 
        for key_name in ['steps', 'truncation', 'episode_done']:
            state.info[key_name] = jnp.zeros(rng.shape[:-1])
        episode_metrics = dict()
        episode_metrics['sum_reward'] = jnp.zeros(rng.shape[:-1])
        episode_metrics['length'] = jnp.zeros(rng.shape[:-1])
        for metric_name in state.metrics.keys():
            episode_metrics[metric_name] = jnp.zeros(rng.shape[:-1])
        state.info['episode_metrics'] = episode_metrics
        state.info['first_pipeline_state'] = state.pipeline_state
        state.info['first_obs'] = state.obs

        ## Set Observation and EnvState
        reach1_value = self.is_reach1(state.obs[0:2])
        reach2_value = self.is_reach2(state.obs[0:2])
        avoid_value = self.is_avoid(state.obs[0:2])
        observation = jnp.concatenate([state.obs, jnp.array([reach1_value, reach2_value, avoid_value])])
        env_state = EnvStateA(state, avoid_value)

        # FIXME: does the observation not need to be transformed?
        # observation = self._env.transform_obs(observation)?

        return observation, env_state
