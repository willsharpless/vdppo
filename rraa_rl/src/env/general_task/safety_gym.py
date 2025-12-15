"""
General Task Safety Gym Environment

Made from Safety Gym RRAA Environment style
Based on Brax PointRandom environment

NOTE: Designed to work for any predicates and so that one env can be used for all decompositions
"""

import jax
import jax.numpy as jnp
from functools import partial
from gymnax.environments import spaces
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper
from flax import struct
from brax.envs.base import State

from .point_random import PointRandom

# SAFETYGYM_TARGET_RIGHT, SAFETYGYM_TARGET_LEFT = [2.5, 2.5], [-2.5, -2.5] # v0
# SAFETYGYM_TARGET_RIGHT, SAFETYGYM_TARGET_LEFT = [2., 2.], [-2., -2.] # v1
SAFETYGYM_TARGET_1, SAFETYGYM_TARGET_2, SAFETYGYM_TARGET_3 = [2., 2.], [-2., -2.], [0., 0.] # v1

SAFETYGYM_TARGET_RADIUS = 0.3 # v0

SAFETYGYM_RAA_OBSTACLE_RADIUS = 0.2
SAFETYGYM_RAA_BOX_RADIUS = 3.0

SAFETYGYM_OBSTACLE_SET = jnp.array([[1.403247, 0.6281236], [0.42943087, 1.17059302],
                                    [-1.16036429, 0.89811093], [-0.88776483, 1.46420776],
                                    [-0.07556364, -1.10567521], [0.72648704, 0.17957757],
                                    [-0.33115742, 0.83026827], [-1.33470321, -1.3259373]])

@struct.dataclass
class EnvStateGeneralTask:
    state: jax.Array = struct.field(default_factory=jax.Array)
    predicate_values: jax.Array = struct.field(default_factory=jax.Array)
    predicate_history_extrema: jax.Array = struct.field(default_factory=jax.Array)

@struct.dataclass
class EnvParamsEmpty:
    pass

class PointGeneralTask:
    def __init__(self, 
                 active_predicates=["reach1", "reach2", "obstacles"], 
                 negated_predicate_mask=jnp.array([0, 0, 1]),
                 backend="mjx"):
        env = PointRandom(backend=backend)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size  
        self.observation_size = (env.observation_size,)
        self.default_params = EnvParamsEmpty()
        self.hazard_pos_array = SAFETYGYM_OBSTACLE_SET
        self.active_predicates = active_predicates
        self.n_active_predicates = len(active_predicates)
        self.negated_predicate_mask = negated_predicate_mask
        
        assert self.n_active_predicates == self.negated_predicate_mask.shape[0], \
            "Number of active predicates must match negated predicate mask length"

    @partial(jax.jit, static_argnums=(0,))
    def predicate_values(self, state):
        values = []
        for predicate in self.active_predicates:
            func = getattr(self, f"is_{predicate}", None)
            if func is not None:
                value = func(state)
                values.append(value)
            else:
                raise NotImplementedError(f"Predicate {predicate} not implemented")
        return jnp.stack(values, axis=-1)
    
    @partial(jax.jit, static_argnums=(0,))
    def predicate_value_extrema(self, state, predicate_values):
        current_values = predicate_values * (1 - 2 * self.negated_predicate_mask) # flip negated for min tracking
        last_maxes = state.predicate_extrema * (1 - 2 * self.negated_predicate_mask)
        maxes = jnp.maximum(current_values, last_maxes)
        maxes = maxes * (1 - 2 * self.negated_predicate_mask)
        return maxes

    @partial(jax.jit, static_argnums=(0,))
    def is_reach1(self, state):
        state = state.obs[0:2]
        target_center, radius = SAFETYGYM_TARGET_1, SAFETYGYM_TARGET_RADIUS
        target_pos = state
        reach = jnp.sqrt((target_pos[..., 0] - target_center[0]) ** 2 + \
                         (target_pos[..., 1] - target_center[1]) ** 2) - radius
        value = jnp.where(reach < 0., -3., reach)
        return value * 100.0
    
    @partial(jax.jit, static_argnums=(0,))
    def is_reach2(self, state):
        state = state.obs[0:2]
        target_center, radius = SAFETYGYM_TARGET_2, SAFETYGYM_TARGET_RADIUS
        target_pos = state
        reach = jnp.sqrt((target_pos[..., 0] - target_center[0]) ** 2 + \
                         (target_pos[..., 1] - target_center[1]) ** 2) - radius
        value = jnp.where(reach < 0., -3., reach)
        return value * 100.0
    
    @partial(jax.jit, static_argnums=(0,))
    def is_reach3(self, state):
        state = state.obs[0:2]
        target_center, radius = SAFETYGYM_TARGET_3, SAFETYGYM_TARGET_RADIUS
        target_pos = state
        reach = jnp.sqrt((target_pos[..., 0] - target_center[0]) ** 2 + \
                         (target_pos[..., 1] - target_center[1]) ** 2) - radius
        value = jnp.where(reach < 0., -3., reach)
        return value * 100.0
    
    @partial(jax.jit, static_argnums=(0,))
    def is_obstacles(self, state):
        state = state.obs[0:2]
        radius = SAFETYGYM_RAA_OBSTACLE_RADIUS
        target_pos = state
        obstacle_type='box'

        ## AVOID OBSTACLES
        avoid_obstacles = -jnp.inf
        for hazard_pos in self.hazard_pos_array:

            if obstacle_type == 'ball':
                avoid = -(jnp.sqrt((target_pos[..., 0] - hazard_pos[0]) ** 2 + \
                                (target_pos[..., 1] - hazard_pos[1]) ** 2) - radius)
            elif obstacle_type == 'box':
                avoid = -(jnp.maximum(jnp.fabs(target_pos[..., 0] - hazard_pos[0]), 
                                    jnp.fabs(target_pos[..., 1] - hazard_pos[1])) - radius)
            else:
                raise NotImplementedError("Obstacle type not implemented")

            avoid_obstacles = jnp.maximum(avoid_obstacles, avoid)
        
        ## AVOID WALLS
        avoid_wall_obstacles = jnp.maximum(jnp.fabs(target_pos[..., 0]), jnp.fabs(target_pos[..., 1])) - SAFETYGYM_RAA_BOX_RADIUS

        avoid = jnp.maximum(10. * avoid_obstacles, 0.1 * avoid_wall_obstacles)
        value = jnp.where(avoid > 0., 3., avoid)
        return value * 100.0

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)

        predicate_values = self.predicate_values(state)
        predicate_extrema = predicate_values

        observation = jnp.concatenate([state.obs, predicate_values])
        env_state = EnvStateGeneralTask(state, predicate_values, predicate_extrema)

        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.clip(action, -1., 1.)
        next_state = self._env.step(state.state, u)

        predicate_values = self.predicate_values(next_state)
        predicate_extrema = self.predicate_value_extrema(next_state, predicate_values)

        observation = jnp.concatenate([next_state.obs, predicate_values])
        next_state_new = EnvStateGeneralTask(next_state, predicate_values, predicate_extrema)

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
        predicate_values = self.predicate_values(state)
        predicate_extrema = self.predicate_value_extrema(state, predicate_values)
        observation = jnp.concatenate([state.obs, predicate_values])
        env_state = EnvStateGeneralTask(state, predicate_values, predicate_extrema)

        return observation, env_state

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size + params.n_active_predicates,), #depends on number of active predicates (reach1, reach2, reach3, avoid -> 4 added)
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )