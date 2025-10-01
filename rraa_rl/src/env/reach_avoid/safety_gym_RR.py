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
class EnvStateRR:
    state: jax.Array = struct.field(default_factory=jax.Array)
    reach1: float = 0.
    reach2: float = 0.
    has_reached_1: float = 0.
    has_reached_2: float = 0.

@struct.dataclass
class EnvParamsEmpty:
    pass

class PointReachReachTemplate:
    def __init__(self, backend="mjx"):
        env = PointRandom(backend=backend)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)
        self.default_params = EnvParamsEmpty()
        # self.hazard_pos_array = jnp.array([[1.403247, 0.6281236], [0.42943087, 1.17059302],
        #                                    [-1.16036429, 0.89811093], [-0.88776483, 1.46420776],
        #                                    [-0.07556364, -1.10567521], [0.72648704, 0.17957757],
        #                                    [-0.33115742, 0.83026827], [-1.33470321, -1.3259373]])

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

class PointReachReach(PointReachReachTemplate):

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)

        reach1_value = self.is_reach1(state.obs[0:2])
        reach2_value = self.is_reach2(state.obs[0:2])
        has_reached_1 = reach1_value < 0
        has_reached_2 = reach2_value < 0
        
        observation = jnp.concatenate([state.obs, jnp.array([reach1_value, reach2_value])])
        env_state = EnvStateRR(state, reach1_value, reach2_value, has_reached_1, has_reached_2)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.clip(action, -1., 1.)
        next_state = self._env.step(state.state, u)

        reach1_value = self.is_reach1(next_state.obs[0:2])
        reach2_value = self.is_reach2(next_state.obs[0:2])
        has_reached_1 = jnp.logical_or(reach1_value < 0, state.has_reached_1)
        has_reached_2 = jnp.logical_or(reach2_value < 0, state.has_reached_2)

        observation = jnp.concatenate([next_state.obs, jnp.array([reach1_value, reach2_value])])
        next_state_new = EnvStateRR(next_state, reach1_value, reach2_value, has_reached_1, has_reached_2)
        reward = 0.
        done = jnp.logical_or(next_state.done > 0.5, jnp.logical_and(has_reached_1, has_reached_2))

        return observation, next_state_new, reward, done, {"x": state.state.obs[0], "y": state.state.obs[1], "theta": jnp.arcsin(state.state.obs[2])}
    
class PointReach1(PointReachReachTemplate):

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)

        reach1_value = self.is_reach1(state.obs[0:2])
        reach2_value = self.is_reach2(state.obs[0:2])
        
        observation = jnp.concatenate([state.obs, jnp.array([reach1_value, reach2_value])])
        env_state = EnvStateR1(state, reach1_value)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.clip(action, -1., 1.)
        next_state = self._env.step(state.state, u)

        reach1_value = self.is_reach1(next_state.obs[0:2])
        reach2_value = self.is_reach2(next_state.obs[0:2])

        observation = jnp.concatenate([next_state.obs, jnp.array([reach1_value, reach2_value])])
        next_state_new = EnvStateR1(next_state, reach1_value)
        reward = 0.
        # done = jnp.logical_or(next_state.done > 0.5, jnp.logical_and(has_reached_1, has_reached_2))
        done = next_state.done > 0.5 # FIXME or reach1_value < 0?

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
        observation = jnp.concatenate([state.obs, jnp.array([reach1_value, reach2_value])])
        env_state = EnvStateR1(state, reach1_value)

        # FIXME: does the observation not need to be transformed?
        # observation = self._env.transform_obs(observation)?

        return observation, env_state
    
class PointReach2(PointReachReachTemplate):

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)

        reach1_value = self.is_reach1(state.obs[0:2])
        reach2_value = self.is_reach2(state.obs[0:2])
        
        observation = jnp.concatenate([state.obs, jnp.array([reach1_value, reach2_value])])
        env_state = EnvStateR2(state, reach2_value)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.clip(action, -1., 1.)
        next_state = self._env.step(state.state, u)

        reach1_value = self.is_reach1(next_state.obs[0:2])
        reach2_value = self.is_reach2(next_state.obs[0:2])

        observation = jnp.concatenate([next_state.obs, jnp.array([reach1_value, reach2_value])])
        next_state_new = EnvStateR2(next_state, reach2_value)
        reward = 0.
        # done = jnp.logical_or(next_state.done > 0.5, jnp.logical_and(has_reached_1, has_reached_2))
        done = next_state.done > 0.5 # FIXME or reach1_value < 0?

        return observation, next_state_new, reward, done, {"x": state.state.obs[0], "y": state.state.obs[1], "theta": jnp.arcsin(state.state.obs[2])}
    
    @partial(jax.jit, static_argnums=(0,))
    def reset_toinput(self, key, reset_obs, params=None):
        # reset_obs = deepcopy(reset_obs[:53])
        
        ## Remake Pipeline State
        # qpos = self.sys.init_qpos
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
        observation = jnp.concatenate([state.obs, jnp.array([reach1_value, reach2_value])])
        env_state = EnvStateR2(state, reach2_value)

        # FIXME: does the observation not need to be transformed?
        # observation = self._env.transform_obs(observation)?

        return observation, env_state

    # @partial(jax.jit, static_argnums=(0,))
    # def is_reach(self, obs, avoid_value):
    #     reach = jnp.sqrt(obs[0] ** 2 + obs[1] ** 2) - 0.3
    #     has_reached_goal = jnp.sqrt(obs[0] ** 2 + obs[1] ** 2) < 0.3
    #     value = jnp.where(has_reached_goal, -3.0, reach)
    #     is_avoid = (avoid_value == -1)
    #     value = jnp.where(is_avoid, 3.0, value)
    #     return value * 100.0

    # @partial(jax.jit, static_argnums=(0,))
    # def is_avoid(self, obs):
    #     avoid = (obs[0] >= 3.) | (obs[0] <= -3.) | (obs[1] >= 3.) | (obs[0] <= -3.)
    #     avoid_0 = jnp.sum((obs - self.hazard_pos_array[0]) ** 2) < 0.2 * 0.2
    #     avoid_1 = jnp.sum((obs - self.hazard_pos_array[1]) ** 2) < 0.2 * 0.2
    #     avoid_2 = jnp.sum((obs - self.hazard_pos_array[2]) ** 2) < 0.2 * 0.2
    #     avoid_3 = jnp.sum((obs - self.hazard_pos_array[3]) ** 2) < 0.2 * 0.2
    #     avoid_4 = jnp.sum((obs - self.hazard_pos_array[4]) ** 2) < 0.2 * 0.2
    #     avoid_5 = jnp.sum((obs - self.hazard_pos_array[5]) ** 2) < 0.2 * 0.2
    #     avoid_6 = jnp.sum((obs - self.hazard_pos_array[6]) ** 2) < 0.2 * 0.2
    #     avoid_7 = jnp.sum((obs - self.hazard_pos_array[7]) ** 2) < 0.2 * 0.2
    #     # return avoid | avoid_0 | avoid_1 | avoid_2 | avoid_3

    #     return avoid | avoid_0 | avoid_1 | avoid_2 | avoid_3 | avoid_4 | avoid_5 | avoid_6 | avoid_7
