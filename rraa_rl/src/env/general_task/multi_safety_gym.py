"""
Multi-Agent General Task Safety Gym Environment

Multi-agent version where:
- Obstacle avoidance uses min over all agents (all must avoid)
- Reach targets use max over all agents (any agent can reach)
- Centralized planning with concatenated observations
"""

import jax
import jax.numpy as jnp
from functools import partial
from gymnax.environments import spaces
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper
from flax import struct
from brax.envs.base import State

from ..reach_avoid.multi_point_random import MultiPointRandom

# Target locations
SAFETYGYM_TARGET_1, SAFETYGYM_TARGET_2, SAFETYGYM_TARGET_3 = [2., 2.], [-2., -2.], [0., 0.]
SAFETYGYM_TARGET_RADIUS = 0.3

SAFETYGYM_RAA_OBSTACLE_RADIUS = 0.2
SAFETYGYM_RAA_BOX_RADIUS = 3.0

SAFETYGYM_OBSTACLE_SET = jnp.array([[1.403247, 0.6281236], [0.42943087, 1.17059302],
                                    [-1.16036429, 0.89811093], [-0.88776483, 1.46420776],
                                    [-0.07556364, -1.10567521], [0.72648704, 0.17957757],
                                    [-0.33115742, 0.83026827], [-1.33470321, -1.3259373]])

@struct.dataclass
class EnvStateMultiGeneralTask:
    state: jax.Array = struct.field(default_factory=jax.Array)
    predicate_values: jax.Array = struct.field(default_factory=jax.Array)
    predicate_history_extrema: jax.Array = struct.field(default_factory=jax.Array)

@struct.dataclass
class EnvParamsEmpty:
    pass

class MultiPointGeneralTask:
    def __init__(self, 
                 n_agents=2,
                 active_predicates=["reach1", "reach2", "obstacles"], 
                 negated_predicate_mask=jnp.array([1, 1, 0]),
                 backend="mjx"):
        """Multi-agent general task environment.
        
        Args:
            n_agents: Number of agents
            active_predicates: List of predicate names to compute
            negated_predicate_mask: Mask indicating which predicates are negated
            backend: Physics backend
        """
        env = MultiPointRandom(n_agents=n_agents, backend=backend)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        env = AutoResetWrapper(env)
        self._env = env
        self.n_agents = n_agents
        self.obs_size_per_agent = 7
        self.action_size = env.action_size  
        self.observation_size = (env.observation_size,)
        self.default_params = EnvParamsEmpty()
        self.hazard_pos_array = SAFETYGYM_OBSTACLE_SET
        self.active_predicates = active_predicates
        self.n_active_predicates = len(active_predicates)
        self.negated_predicate_mask = negated_predicate_mask
        
        assert self.n_active_predicates == self.negated_predicate_mask.shape[0], \
            "Number of active predicates must match negated predicate mask length"

    def _get_agent_positions(self, state):
        """Extract positions of all agents from state."""
        # state.obs has shape (n_agents * 7,)
        # First 2 elements of each agent's obs are x, y
        positions = []
        for i in range(self.n_agents):
            start_idx = i * self.obs_size_per_agent
            positions.append(state.obs[start_idx:start_idx+2])
        return jnp.stack(positions)  # Shape: (n_agents, 2)

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
        current_values = predicate_values * (1 - 2 * self.negated_predicate_mask)
        last_maxes = state.predicate_history_extrema * (1 - 2 * self.negated_predicate_mask)
        maxes = jnp.maximum(current_values, last_maxes)
        maxes = maxes * (1 - 2 * self.negated_predicate_mask)
        return maxes

    @partial(jax.jit, static_argnums=(0,))
    def is_reach1_any(self, state):
        """Any agent reaching target 1 (best over agents)."""
        positions = self._get_agent_positions(state)
        target_center, radius = SAFETYGYM_TARGET_1, SAFETYGYM_TARGET_RADIUS
        
        # Compute distance for each agent
        distances = jnp.sqrt(jnp.sum((positions - jnp.array(target_center)) ** 2, axis=1))
        reaches = distances - radius
        
        # Take min (best) over agents
        reach = jnp.min(reaches)
        value = jnp.where(reach < 0., -3., reach)
        return value * 100.0
    
    @partial(jax.jit, static_argnums=(0,))
    def is_reach2_any(self, state):
        """Any agent reaching target 2 (best over agents)."""
        positions = self._get_agent_positions(state)
        target_center, radius = SAFETYGYM_TARGET_2, SAFETYGYM_TARGET_RADIUS
        
        distances = jnp.sqrt(jnp.sum((positions - jnp.array(target_center)) ** 2, axis=1))
        reaches = distances - radius
        
        # Take min (best) over agents
        reach = jnp.min(reaches)
        value = jnp.where(reach < 0., -3., reach)
        return value * 100.0
    
    @partial(jax.jit, static_argnums=(0,))
    def is_reach3_all(self, state):
        """All agents reaching target 3 (worst over agents)."""
        positions = self._get_agent_positions(state)
        target_center, radius = SAFETYGYM_TARGET_3, SAFETYGYM_TARGET_RADIUS
        
        distances = jnp.sqrt(jnp.sum((positions - jnp.array(target_center)) ** 2, axis=1))
        reaches = distances - radius
        
        # Take max (worst) over agents
        reach = jnp.max(reaches)
        value = jnp.where(reach < 0., -3., reach)
        return value * 100.0
    
    @partial(jax.jit, static_argnums=(0,))
    def is_obstacles(self, state):
        """All agents must avoid obstacles (min over agents)."""
        positions = self._get_agent_positions(state)
        radius = SAFETYGYM_RAA_OBSTACLE_RADIUS
        obstacle_type = 'box'

        # Compute worst avoidance value across all agents
        worst_avoid = -jnp.inf
        
        for i in range(self.n_agents):
            agent_pos = positions[i]
            
            # Check against obstacles
            avoid_obstacles = -jnp.inf
            for hazard_pos in self.hazard_pos_array:
                if obstacle_type == 'ball':
                    avoid = -(jnp.sqrt((agent_pos[0] - hazard_pos[0]) ** 2 + \
                                    (agent_pos[1] - hazard_pos[1]) ** 2) - radius)
                elif obstacle_type == 'box':
                    avoid = -(jnp.maximum(jnp.fabs(agent_pos[0] - hazard_pos[0]), 
                                        jnp.fabs(agent_pos[1] - hazard_pos[1])) - radius)
                else:
                    raise NotImplementedError("Obstacle type not implemented")
                
                avoid_obstacles = jnp.maximum(avoid_obstacles, avoid)
            
            # Check against walls
            avoid_wall_obstacles = jnp.maximum(jnp.fabs(agent_pos[0]), jnp.fabs(agent_pos[1])) - SAFETYGYM_RAA_BOX_RADIUS
            
            # Combine obstacles and walls for this agent
            agent_avoid = jnp.maximum(10. * avoid_obstacles, 0.1 * avoid_wall_obstacles)
            
            # Track worst violation across agents (highest value = worst)
            worst_avoid = jnp.maximum(worst_avoid, agent_avoid)
        
        value = jnp.where(worst_avoid > 0., 3., worst_avoid)
        return value * 100.0

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)

        predicate_values = self.predicate_values(state)
        
        observation = jnp.concatenate([state.obs, predicate_values])
        env_state = EnvStateMultiGeneralTask(state, predicate_values, predicate_values)

        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.clip(action, -1., 1.)
        next_state = self._env.step(state.state, u)

        predicate_values = self.predicate_values(next_state)
        predicate_extrema = self.predicate_value_extrema(state, predicate_values)

        observation = jnp.concatenate([next_state.obs, predicate_values])
        next_state_new = EnvStateMultiGeneralTask(next_state, predicate_values, predicate_extrema)

        reward = 0.
        done = next_state.done > 0.5

        # Extract positions and angles for all agents for info
        info = {}
        for i in range(self.n_agents):
            start_idx = i * self.obs_size_per_agent
            info[f"x_{i}"] = state.state.obs[start_idx]
            info[f"y_{i}"] = state.state.obs[start_idx + 1]
            info[f"theta_{i}"] = jnp.arcsin(state.state.obs[start_idx + 2])

        return observation, next_state_new, reward, done, info
    
    @partial(jax.jit, static_argnums=(0,))
    def reset_toinput(self, key, reset_obs, params=None):
        """Reset to a specific observation state."""
        # reset_obs contains concatenated observations for all agents
        
        # Reconstruct qpos and qvel for all agents
        qpos_list = []
        qvel_list = []
        
        for i in range(self.n_agents):
            start_idx = i * self.obs_size_per_agent
            agent_obs = reset_obs[start_idx:start_idx + self.obs_size_per_agent]
            
            # qpos: [x, y, theta]
            qpos_agent = jnp.array([
                agent_obs[0],  # x
                agent_obs[1],  # y
                jnp.arcsin(agent_obs[2])  # theta from sin(theta)
            ])
            qpos_list.append(qpos_agent)
            
            # qvel: [vx, vy, vtheta]
            qvel_agent = agent_obs[4:7]
            qvel_list.append(qvel_agent)
        
        qpos = jnp.concatenate(qpos_list)
        qvel = jnp.concatenate(qvel_list)
        
        pipeline_state = self._env.pipeline_init(qpos, qvel)
        obs = self._env._get_obs(pipeline_state)
        reward, done, zero = jnp.zeros(3)

        # Define Metrics
        metrics = {
            'forward_reward': zero,
            'reward_linvel': zero,
            'reward_quadctrl': zero,
            'reward_alive': zero,
        }
        for i in range(self.n_agents):
            metrics[f'x_position_{i}'] = zero
            metrics[f'y_position_{i}'] = zero
            metrics[f'distance_from_origin_{i}'] = zero
            metrics[f'x_velocity_{i}'] = zero
            metrics[f'y_velocity_{i}'] = zero
            
        state = State(pipeline_state, obs, reward, done, metrics)
        
        # Set Auxiliaries
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

        # Set Observation and EnvState
        predicate_values = self.predicate_values(state)
        observation = jnp.concatenate([state.obs, predicate_values])
        env_state = EnvStateMultiGeneralTask(state, predicate_values, predicate_values)

        return observation, env_state

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size + self.n_active_predicates,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )
