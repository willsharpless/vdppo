import jax
import jax.numpy as jnp
from functools import partial
from gymnax.environments import spaces
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper
from flax import struct
from brax.envs.base import State

from .point_random import PointRandom

SAFETYGYM_RAA_TARGET = [0., 0.]
SAFETYGYM_RAA_TARGET_RADIUS = 0.3
SAFETYGYM_RAA_OBSTACLE_RADIUS = 0.2
SAFETYGYM_RAA_BOX_RADIUS = 2.5

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

class PointReachAvoidTemplate:
    def __init__(self, backend="mjx"):
        env = PointRandom(backend=backend)
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
        avoid_wall = jnp.maximum(target_pos[..., 0], target_pos[..., 1]) - SAFETYGYM_RAA_BOX_RADIUS

        avoid = jnp.maximum(0.1 * avoid_obstacles, 10 * avoid_wall)
        value = jnp.where(avoid > 0., 3., value)
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

class PointReachAvoid(PointReachAvoidTemplate):

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)

        reach_value = self.is_reach(state.obs[0:2])
        avoid_value = self.is_avoid(state.obs[0:2])
        has_reached = reach_value < 0
        
        observation = jnp.concatenate([state.obs, jnp.array([avoid_value, reach_value])])
        env_state = EnvStateRA(state, avoid_value, reach_value, has_reached)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.clip(action, -1., 1.)
        next_state = self._env.step(state.state, u)

        reach_value = self.is_reach(next_state.obs[0:2])
        avoid_value = self.is_avoid(next_state.obs[0:2])
        has_reached = jnp.logical_or(reach_value < 0, state.has_reached)

        observation = jnp.concatenate([next_state.obs, jnp.array([avoid_value, reach_value])])
        next_state_new = EnvStateRA(next_state, avoid_value, reach_value, has_reached)
        reward = 0.
        done = False

        return observation, next_state_new, reward, done, {"x": state.state.obs[0], "y": state.state.obs[1], "theta": jnp.arcsin(state.state.obs[2])}
    
class PointAvoidOnly(PointReachAvoidTemplate):

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)

        reach_value = self.is_reach(state.obs[0:2])
        avoid_value = self.is_avoid(state.obs[0:2])
        
        observation = jnp.concatenate([state.obs, jnp.array([avoid_value, reach_value])])
        env_state = EnvStateAvoidOnly(state, avoid_value)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.clip(action, -1., 1.)
        next_state = self._env.step(state.state, u)

        reach_value = self.is_reach(next_state.obs[0:2])
        avoid_value = self.is_avoid(next_state.obs[0:2])

        observation = jnp.concatenate([next_state.obs, jnp.array([avoid_value, reach_value])])
        next_state_new = EnvStateAvoidOnly(next_state, avoid_value)
        reward = 0.
        done = False

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
        reach_value = self.is_reach(state.obs[0:2])
        avoid_value = self.is_avoid(state.obs[0:2])
        observation = jnp.concatenate([state.obs, jnp.array([avoid_value, reach_value])])
        env_state = EnvStateAvoidOnly(state, avoid_value)

        # FIXME: does the observation not need to be transformed?
        # observation = self._env.transform_obs(observation)?

        return observation, env_state
