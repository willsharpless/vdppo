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

from ..reach_avoid.multi_point_random import MultiPointRandom, SAFETYGYM_RAA_OBSTACLE_CUSHION_RADIUS, SAFETYGYM_RAA_BOX_CUSHION_RADIUS, SAFETYGYM_OBSTACLE_SET

# Target locations
SAFETYGYM_TARGET_1, SAFETYGYM_TARGET_2, SAFETYGYM_TARGET_3, SAFETYGYM_TARGET_4 = [2., 2.], [-2., -2.], [0.25, 0.7], [0.0, 0.0]
SAFETYGYM_TARGET_RADIUS = 0.3

@struct.dataclass
class EnvStateMultiDynamicGeneralTask:
    state: jax.Array = struct.field(default_factory=jax.Array)
    predicate_values: jax.Array = struct.field(default_factory=jax.Array)
    predicate_history_extrema: jax.Array = struct.field(default_factory=jax.Array)
    tracked_locs: jax.Array = struct.field(default_factory=jax.Array)  # Shape: (n_tracked_locs, 2)

@struct.dataclass
class EnvParamsEmpty:
    pass

class MultiPointDynamicGeneralTask:
    def __init__(self, 
                 n_agents=2,
                 active_predicates=["reach3_any", "obstacles"], 
                 negated_predicate_mask=jnp.array([1, 1, 0]),
                 add_ag_vals_to_obs=False,
                 backend="mjx",
                 fixed_velocity=None,
                 dynamic_predicate_names=["reach3_any"],  # List of predicate names to track locations for
                 dynamic_predicate_resets={"reach3_any": lambda k: jnp.array(SAFETYGYM_TARGET_3)},  # Dict mapping dynamic predicate name to (low, high) bounds for random reset
                 dynamic_predicate_updates={"reach3_any": lambda k, s, l: l},  # Dict mapping dynamic predicate name to dynamics function (key, state, loc) -> new_loc
                #  dynamic_predicate_init_locs={"reach3_any": jnp.array(SAFETYGYM_TARGET_3)}  # Dict mapping tracked predicate name to default location
        ):
        """Multi-agent general task environment with augmented location tracking.
        
        Args:
            n_agents: Number of agents
            active_predicates: List of predicate names to compute
            negated_predicate_mask: Mask indicating which predicates are negated
            add_ag_vals_to_obs: Whether to add per-agent predicate values to observation
            backend: Physics backend
            fixed_velocity: Optional fixed velocity for agents
            dynamic_predicate_names: List of predicate names whose locations will be tracked and augmented to obs
            dynamic_predicate_resets: Dict mapping dynamic predicate name to predicate loc reset function (key, loc) -> loc
            dynamic_predicate_updates: Dict mapping dynamic predicate name to dynamics function (key, state, loc) -> new_loc
            # dynamic_predicate_init_locs: Dict mapping dynamic predicate name to default location [shape (2,)]
        """
        env = MultiPointRandom(n_agents=n_agents, backend=backend, fixed_velocity=fixed_velocity)
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
        self.add_ag_vals_to_obs = add_ag_vals_to_obs
        
        # Set up tracked locations
        self.dynamic_predicate_names = dynamic_predicate_names or []
        self.n_tracked_locs = len(self.dynamic_predicate_names)
        
        # Default location bounds for reach predicates
        default_bounds = (-SAFETYGYM_RAA_BOX_CUSHION_RADIUS, SAFETYGYM_RAA_BOX_CUSHION_RADIUS)
        
        # Default locations for each predicate type
        self.default_pred_locs = {
            "reach1_any": jnp.array(SAFETYGYM_TARGET_1),
            "reach2_any": jnp.array(SAFETYGYM_TARGET_2),
            "reach3_static": jnp.array(SAFETYGYM_TARGET_3),
            "reach3_any": jnp.array(SAFETYGYM_TARGET_3),
            "reach4_all": jnp.array(SAFETYGYM_TARGET_4),
            "obstacles": jnp.array([0.0, 0.0]) # DEBUG FIXME
        }

        # Set up mappings
        self.dynamic_predicate_resets = dynamic_predicate_resets or {}
        self.dynamic_predicate_updates = dynamic_predicate_updates or {}
        
        # Build index mapping from dynamic predicate name to tracked location index
        self.predicate_to_tracked_idx = {name: i for i, name in enumerate(self.dynamic_predicate_names)}
        
        # # Build default locations array for dynamic predicates
        # self.init_tracked_pred_locs = []
        # for name in self.dynamic_predicate_names:
        #     assert name in self.dynamic_predicate_init_locs, f"Initial location for dynamic predicate {name} not specified"
        #     self.init_tracked_pred_locs.append(self.dynamic_predicate_init_locs[name])
        # self.init_tracked_pred_locs = jnp.stack(self.init_tracked_pred_locs) if self.init_tracked_pred_locs else jnp.zeros((0, 2))
        
        # # Set default bounds for dynamic predicates if not specified
        # for name in self.dynamic_predicate_names:
        #     if name not in self.dynamic_predicate_resets:
        #         self.dynamic_predicate_reset_bounds[name] = default_bounds
        
        assert self.n_active_predicates == self.negated_predicate_mask.shape[0], \
            "Number of active predicates must match negated predicate mask length"

    def _get_agent_positions(self, state):
        """Extract positions of all agents from state."""
        positions = []
        for i in range(self.n_agents):
            start_idx = i * self.obs_size_per_agent
            positions.append(state.obs[start_idx:start_idx+2])
        return jnp.stack(positions)

    def _initialize_tracked_locs(self, key):
        """Initialize tracked locations for tracked predicates."""
        if self.n_tracked_locs == 0:
            return jnp.zeros((0, 2))
        
        locs = []
        for name in self.dynamic_predicate_names:
            key, subkey = jax.random.split(key)
            loc = self.dynamic_predicate_resets[name](subkey)
            # if name in self.dynamic_predicate_reset_bounds:
            #     # Random initialization within bounds
            #     key, subkey = jax.random.split(key)
            #     loc = self.dynamic_predicate_resets[name](subkey, loc)
            #     # low, high = self.dynamic_predicate_reset_bounds[name]
            #     # loc = jax.random.uniform(subkey, shape=(2,), minval=low, maxval=high)
            # else:
            #     # Use default location
            #     idx = self.predicate_to_tracked_idx[name]
            #     loc = self.init_tracked_pred_locs[idx]
            locs.append(loc)
        return jnp.stack(locs)  # Shape: (n_tracked_locs, 2)

    def _update_tracked_locs(self, key, state, tracked_locs):
        """Update tracked locations based on dynamics."""
        if self.n_tracked_locs == 0:
            return tracked_locs
        
        new_locs = []
        for i, name in enumerate(self.dynamic_predicate_names):
            loc = tracked_locs[i]
            if name in self.dynamic_predicate_updates:
                # Apply dynamics function
                key, subkey = jax.random.split(key)
                loc = self.dynamic_predicate_updates[name](subkey, state, loc)
            new_locs.append(loc)
        return jnp.stack(new_locs)
    
    def _get_predicate_loc(self, predicate_name, tracked_locs):
        """Get location for a predicate, either from tracked locs or default."""
        if predicate_name in self.predicate_to_tracked_idx:
            idx = self.predicate_to_tracked_idx[predicate_name]
            return tracked_locs[idx]
        else:
            return self.default_pred_locs.get(predicate_name, jnp.array([0.0, 0.0]))

    @partial(jax.jit, static_argnums=(0,))
    def predicate_values(self, state, tracked_locs):
        """Compute predicate values using tracked locations."""
        values = []
        auxs = []
        for predicate in self.active_predicates:
            func = getattr(self, f"is_{predicate}", None)
            if func is not None:
                loc = self._get_predicate_loc(predicate, tracked_locs)
                value, aux = func(state, loc)
                values.append(value)
                auxs.append(aux)
            else:
                raise NotImplementedError(f"Predicate {predicate} not implemented")
        return jnp.stack(values, axis=-1), jnp.stack(auxs, axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def predicate_value_extrema(self, state, predicate_values):
        current_values = predicate_values * (1 - 2 * self.negated_predicate_mask)
        last_maxes = state.predicate_history_extrema * (1 - 2 * self.negated_predicate_mask)
        maxes = jnp.maximum(current_values, last_maxes)
        maxes = maxes * (1 - 2 * self.negated_predicate_mask)
        return maxes

    @partial(jax.jit, static_argnums=(0,))
    def is_reach1_any(self, state, target_center):
        """Any agent reaching target (best over agents)."""
        positions = self._get_agent_positions(state)
        radius = SAFETYGYM_TARGET_RADIUS
        
        distances = jnp.sqrt(jnp.sum((positions - target_center) ** 2, axis=1))
        reaches = distances - radius
        
        reach = jnp.min(reaches)
        value = jnp.where(reach < 0., -3., reach)
        agent_values = jnp.where(reaches < 0., -3., reaches)
        return value * 100.0, agent_values * 100.0

    @partial(jax.jit, static_argnums=(0,))
    def is_reach2_any(self, state, target_center):
        """Any agent reaching target (best over agents)."""
        positions = self._get_agent_positions(state)
        radius = SAFETYGYM_TARGET_RADIUS
        
        distances = jnp.sqrt(jnp.sum((positions - target_center) ** 2, axis=1))
        reaches = distances - radius
        
        reach = jnp.min(reaches)
        value = jnp.where(reach < 0., -3., reach)
        agent_values = jnp.where(reaches < 0., -3., reaches)
        return value * 100.0, agent_values * 100.0
    
    @partial(jax.jit, static_argnums=(0,))
    def is_reach3_any(self, state, target_center):
        """Any agents reaching target (best over agents)."""
        positions = self._get_agent_positions(state)
        radius = SAFETYGYM_TARGET_RADIUS
        
        distances = jnp.sqrt(jnp.sum((positions - target_center) ** 2, axis=1))
        reaches = distances - radius
        
        reach = jnp.min(reaches)
        value = jnp.where(reach < 0., -3., reach)
        agent_values = jnp.where(reaches < 0., -3., reaches)
        return value * 100.0, agent_values * 100.0

    @partial(jax.jit, static_argnums=(0,))
    def is_reach4_all(self, state, target_center):
        """All agents reaching target (worst over agents)."""
        positions = self._get_agent_positions(state)
        radius = SAFETYGYM_TARGET_RADIUS

        distances = jnp.sqrt(jnp.sum((positions - target_center) ** 2, axis=1))
        reaches = distances - radius
        
        reach = jnp.max(reaches)
        value = jnp.where(reach < 0., -3., reach)
        agent_values = jnp.where(reaches < 0., -3., reaches)
        return value * 100.0, agent_values * 100.0

    @partial(jax.jit, static_argnums=(0,))
    def is_obstacles(self, state, target_center):
        """All agents must avoid obstacles (worst over agents)."""
        positions = self._get_agent_positions(state)
        radius = SAFETYGYM_RAA_OBSTACLE_CUSHION_RADIUS
        obstacle_type = 'box'

        worst_avoid = -jnp.inf
        agent_avoids = []

        for i in range(self.n_agents):
            agent_pos = positions[i]
            
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
            
            avoid_wall_obstacles = jnp.maximum(jnp.fabs(agent_pos[0]), jnp.fabs(agent_pos[1])) - SAFETYGYM_RAA_BOX_CUSHION_RADIUS
            
            agent_avoid = jnp.maximum(5. * avoid_obstacles, 0.5 * avoid_wall_obstacles)
            agent_avoids.append(agent_avoid)
            worst_avoid = jnp.maximum(worst_avoid, agent_avoid)

        agent_avoids = jnp.array(agent_avoids)
        value = jnp.where(worst_avoid > 0., 3., worst_avoid)
        agent_values = jnp.where(agent_avoids > 0., 3., agent_avoids)
        return value * 100.0, agent_values * 100.0

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        key, subkey = jax.random.split(key)
        state = self._env.reset(subkey)

        # Initialize tracked locations (randomly if bounds provided)
        key, subkey = jax.random.split(key)
        tracked_locs = self._initialize_tracked_locs(subkey)

        predicate_values, aux = self.predicate_values(state, tracked_locs)
        
        # Augment observation with tracked locations (flattened) and optionally per-agent values
        tracked_locs_flat = tracked_locs.flatten()  # Shape: (n_tracked_locs * 2,)
        if self.add_ag_vals_to_obs:
            observation = jnp.concatenate([state.obs, predicate_values, tracked_locs_flat, aux])
        else:
            observation = jnp.concatenate([state.obs, predicate_values, tracked_locs_flat])
            
        env_state = EnvStateMultiDynamicGeneralTask(state, predicate_values, predicate_values, tracked_locs)

        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.clip(action, -1., 1.)
        next_state = self._env.step(state.state, u)

        # Update tracked locations based on dynamics
        key, subkey = jax.random.split(key)
        tracked_locs = self._update_tracked_locs(subkey, next_state, state.tracked_locs)

        predicate_values, aux = self.predicate_values(next_state, tracked_locs)
        predicate_extrema = self.predicate_value_extrema(state, predicate_values)

        # Augment observation with tracked locations (flattened) and optionally per-agent values
        tracked_locs_flat = tracked_locs.flatten()
        if self.add_ag_vals_to_obs:
            observation = jnp.concatenate([next_state.obs, predicate_values, tracked_locs_flat, aux])
        else:
            observation = jnp.concatenate([next_state.obs, predicate_values, tracked_locs_flat])

        next_state_new = EnvStateMultiDynamicGeneralTask(next_state, predicate_values, predicate_extrema, tracked_locs)

        reward = 0.
        done = next_state.done > 0.5

        # Extract positions and angles for all agents for info
        info = {}
        for i in range(self.n_agents):
            start_idx = i * self.obs_size_per_agent
            info[f"x_{i}"] = state.state.obs[start_idx]
            info[f"y_{i}"] = state.state.obs[start_idx + 1]
            info[f"theta_{i}"] = jnp.arctan2(state.state.obs[start_idx + 2], state.state.obs[start_idx + 3])
        
        # Add tracked location info
        for i, predicate_name in enumerate(self.dynamic_predicate_names):
            info[f"tracked_loc_{predicate_name}_x"] = tracked_locs[i, 0]
            info[f"tracked_loc_{predicate_name}_y"] = tracked_locs[i, 1]

        return observation, next_state_new, reward, done, info
    
    @partial(jax.jit, static_argnums=(0,))
    def reset_toinput(self, key, reset_obs, params=None):
        """Reset to a specific observation state."""
        # Reconstruct qpos and qvel for all agents
        qpos_list = []
        qvel_list = []
        
        for i in range(self.n_agents):
            start_idx = i * self.obs_size_per_agent
            agent_obs = reset_obs[start_idx:start_idx + self.obs_size_per_agent]
            
            qpos_agent = jnp.array([
                agent_obs[0],  # x
                agent_obs[1],  # y
                jnp.arcsin(agent_obs[2])  # theta from sin(theta)
            ])
            qpos_list.append(qpos_agent)
            
            qvel_agent = agent_obs[4:7]
            qvel_list.append(qvel_agent)
        
        qpos = jnp.concatenate(qpos_list)
        qvel = jnp.concatenate(qvel_list)
        
        pipeline_state = self._env.pipeline_init(qpos, qvel)
        obs = self._env._get_obs(pipeline_state)
        reward, done, zero = jnp.zeros(3)

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

        # Initialize tracked locations
        key, subkey = jax.random.split(key)
        tracked_locs = self._initialize_tracked_locs(subkey)

        # Set Observation and EnvState
        predicate_values, aux = self.predicate_values(state, tracked_locs)
        tracked_locs_flat = tracked_locs.flatten()
        if self.add_ag_vals_to_obs:
            observation = jnp.concatenate([state.obs, predicate_values, tracked_locs_flat, aux])
        else:
            observation = jnp.concatenate([state.obs, predicate_values, tracked_locs_flat])
        env_state = EnvStateMultiDynamicGeneralTask(state, predicate_values, predicate_values, tracked_locs)

        return observation, env_state

    def observation_space(self, params):
        aux_size = self.add_ag_vals_to_obs * self.n_active_predicates * self.n_agents
        tracked_locs_size = self.n_tracked_locs * 2  # 2D tracked locations
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size + self.n_active_predicates + tracked_locs_size + aux_size,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )


###############################################################################################################
###############################################################################################################
# DYNAMIC PREDICATE FUNCTIONS

def static_dummy_dynamics(center=jnp.array(SAFETYGYM_TARGET_3)):
    def dynamics_fn(key, state, loc):
        return center
    def reset_fn(key):
        return center
    pred_stats = {"mean_x": center[0], "mean_y": center[1], "std_x": 1., "std_y": 1.}
    return dynamics_fn, reset_fn, pred_stats

def constant_dynamics_with_random_reset(center=jnp.array([0., 0.]), bd=2., reset_distribution="uniform"):
    def dynamics_fn(key, state, loc):
        return center
    def in_box_reset(key):
        if reset_distribution == "uniform":
            return jax.random.uniform(key, shape=(2,), minval=-bd, maxval=bd)
        elif reset_distribution == "truncated_normal":
            return jax.random.truncated_normal(key, mean=center, stddev=bd, minval=-bd, maxval=bd)
        else:
            raise ValueError("Unknown reset distribution")

    if reset_distribution == "uniform":
        pred_stats = {"mean_x": center[0], "mean_y": center[1], "std_x": 10000., "std_y": 10000.}
    elif reset_distribution == "truncated_normal":
        pred_stats = {"mean_x": center[0], "mean_y": center[1], "std_x": bd, "std_y": bd}
    else:
        raise ValueError("Unknown reset distribution")
    
    return dynamics_fn, in_box_reset, pred_stats

def circular_motion_dynamics(center=jnp.array([-0.5, -0.5]), radius=0.5, angular_velocity=0.01):
    """Create a dynamics function that moves location in a circle.
    
    Args:
        center: Center of circular motion
        radius: Radius of circle
        angular_velocity: Angular velocity in radians per step
        
    Returns:
        Dynamics function (key, state, loc) -> new_loc
    """
    def dynamics_fn(key, state, loc):
        # Compute current angle from center
        rel_pos = loc - center
        current_angle = jnp.arctan2(rel_pos[1], rel_pos[0])
        
        # Update angle
        new_angle = current_angle + angular_velocity
        
        # Compute new position
        new_loc = center + radius * jnp.array([jnp.cos(new_angle), jnp.sin(new_angle)])
        return new_loc

    # sample random spot on circle
    def reset_on_circle(key):
        subkey = jax.random.split(key)
        angle = jax.random.uniform(subkey, minval=0, maxval=2 * jnp.pi)
        return center + radius * jnp.array([jnp.cos(angle), jnp.sin(angle)])

    pred_stats = {"mean_x": center[0], "mean_y": center[1], "std_x": radius, "std_y": radius}
    return dynamics_fn, reset_on_circle, pred_stats

def obstacle_edge_dynamics(speed=0.05):
    """Create a dynamics function that moves location around obstacle edges.
    
    Moves around the 6 obstacles that lie above y=0 in a connected path.
    
    Args:
        speed: Speed of movement along obstacle edges
        
    Returns:
        Dynamics function (key, state, loc) -> new_loc
    """
    
    def dynamics_fn(key, state, loc):
        # Find closest obstacle
        distances = jnp.linalg.norm(SAFETYGYM_OBSTACLE_SET - loc, axis=1)
        closest_idx = jnp.argmin(distances)
        
        # Get current and next obstacle
        current_obs = SAFETYGYM_OBSTACLE_SET[closest_idx]
        next_obs = SAFETYGYM_OBSTACLE_SET[(closest_idx + 1) % len(SAFETYGYM_OBSTACLE_SET)]
        
        # Direction to next obstacle
        direction = next_obs - current_obs
        direction_norm = jnp.linalg.norm(direction)
        direction_unit = direction / (direction_norm + 1e-8)
        
        # Move towards next obstacle
        # If close to current obstacle, move toward next
        dist_to_current = jnp.linalg.norm(loc - current_obs)
        
        # Linear interpolation: move along line from current to next obstacle
        new_loc = jnp.where(
            dist_to_current / (direction_norm + 1e-8) < 1.0,
            loc + speed * direction_unit,  # Move toward next
            next_obs + speed * direction_unit  # Wrap to next segment
        )
        
        return new_loc

    # Reset inside the box near obs
    def reset_near_obs(key):
        return jax.random.uniform(key, shape=(2,), minval=-2, maxval=2)

    # pred stats will be mean of obstacles
    mean_x = jnp.mean(SAFETYGYM_OBSTACLE_SET[:, 0])
    mean_y = jnp.mean(SAFETYGYM_OBSTACLE_SET[:, 1])
    std_x = jnp.std(SAFETYGYM_OBSTACLE_SET[:, 0])
    std_y = jnp.std(SAFETYGYM_OBSTACLE_SET[:, 1])
    pred_stats = {"mean_x": mean_x, "mean_y": mean_y, "std_x": std_x, "std_y": std_y}

    return dynamics_fn, reset_near_obs, pred_stats