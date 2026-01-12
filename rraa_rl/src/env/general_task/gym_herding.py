"""
Multi-Agent General Task Safety Gym Environment

Multi-agent version where:
- Obstacle avoidance uses min over all agents (all must avoid)
- Reach targets use max over all agents (any agent can reach)
- Centralized planning with concatenated observations
- Predicates can be associated with auxiliary dynamic states
"""

import jax
import jax.numpy as jnp
from functools import partial
from gymnax.environments import spaces
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper
from flax import struct
from brax.envs.base import State

from ..reach_avoid.multi_point_random import MultiPointRandom, SAFETYGYM_RAA_OBSTACLE_CUSHION_RADIUS, SAFETYGYM_RAA_BOX_CUSHION_RADIUS, SAFETYGYM_OBSTACLE_SET
from ..reach_avoid.multi_point_double_integrator import MultiPointDoubleIntegrator, SAFETYGYM_RAA_OBSTACLE_CUSHION_RADIUS, SAFETYGYM_RAA_BOX_CUSHION_RADIUS, SAFETYGYM_OBSTACLE_SET

# Target locations
SAFETYGYM_TARGET_1, SAFETYGYM_TARGET_2, SAFETYGYM_TARGET_3, SAFETYGYM_TARGET_4 = [2., 2.], [-2., -2.], [0.25, 0.7], [0.0, 0.0]
SAFETYGYM_TARGET_RADIUS = 0.3

HERD_TARGET_RADIUS = 1.
HERDING_COLLISION_RADIUS = 0.2
HERD_CENTER_RADIUS = 1.5

@struct.dataclass
class EnvStateHerd:
    state: jax.Array = struct.field(default_factory=jax.Array)
    predicate_values: jax.Array = struct.field(default_factory=jax.Array)
    predicate_history_extrema: jax.Array = struct.field(default_factory=jax.Array)
    tracked_locs: jax.Array = struct.field(default_factory=jax.Array)  # Shape: (n_tracked_locs, 2)

@struct.dataclass
class EnvParamsEmpty:
    pass

class HerdEnv:
    def __init__(self, 
                 n_agents=2,
                 active_predicates=["reach3_any", "obstacles"], 
                 negated_predicate_mask=jnp.array([1, 0]),
                 add_ag_vals_to_obs=False,
                 episode_length=3000,
                 backend="mjx",
                 fixed_velocity=None,
                 dynamic_predicate_names=["reach3_any"],  # List of predicate names to track locations for
                 dynamic_predicate_resets={"reach3_any": lambda k: jnp.array(SAFETYGYM_TARGET_3)},  # Dict mapping dynamic predicate name to (low, high) bounds for random reset
                 dynamic_predicate_updates={"reach3_any": lambda k, s, l: l},  # Dict mapping dynamic predicate name to dynamics function (key, state, loc) -> new_loc
                #  dynamic_predicate_init_locs={"reach3_any": jnp.array(SAFETYGYM_TARGET_3)}  # Dict mapping tracked predicate name to default location
                dynamics_type="double_integrator",
                rel_acel=[0.5, 1.5, 1., 1., 1.],
                max_acceleration = 1.,
                evaders=[2, 3, 4],
                fixed_policy_fn=None, # will default to herding policy in MultiPointDoubleIntegrator
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
        if dynamics_type == "double_integrator":
            env = MultiPointDoubleIntegrator(n_agents=n_agents, 
                                             backend=backend, 
                                             rel_acel=rel_acel if not fixed_velocity else fixed_velocity,
                                             fixed_policy_agents=evaders,
                                             fixed_policy_fn=fixed_policy_fn,
                                             max_acceleration=max_acceleration,)
            self.obs_size_per_agent = 6
        # elif dynamics_type == "dubins":
        #     env = MultiPointRandom(n_agents=n_agents, backend=backend, fixed_velocity=fixed_velocity)
        #     self.obs_size_per_agent = 7
        else:
            raise NotImplementedError(f"Dynamics type '{dynamics_type}' is not implemented")
        env = EpisodeWrapper(env, episode_length=episode_length, action_repeat=1)
        env = AutoResetWrapper(env)
        self._env = env
        self.n_agents = n_agents
        self.n_controllable_agents = n_agents - len(evaders)
        self.action_size = env.action_size  
        self.observation_size = (env.observation_size,)
        self.default_params = EnvParamsEmpty()
        self.hazard_pos_array = SAFETYGYM_OBSTACLE_SET
        self.active_predicates = active_predicates
        self.n_active_predicates = len(active_predicates)
        self.negated_predicate_mask = negated_predicate_mask
        self.add_ag_vals_to_obs = add_ag_vals_to_obs
        self.dynamics_type = dynamics_type
        self.evaders = evaders
        
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
            "obstacles": jnp.array([0.0, 0.0]),  # DEBUG FIXME
            "collisions": jnp.array([0.0, 0.0])  # DEBUG FIXME
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
    def is_together(self, state, target_center):
        """Reach target where the maximum distance between all evaders is minimized."""
        positions = self._get_agent_positions(state)
        radius = HERD_TARGET_RADIUS

        distances = []
        for ei in self.evaders:
            for ej in self.evaders:
                if ei != ej:
                    distances.append(jnp.linalg.norm(positions[ei] - positions[ej]))
        distances = jnp.array(distances)
        reach = jnp.max(distances) - radius
        value = jnp.where(reach < 0., -3., reach)
        agent_values = jnp.zeros(self.n_agents)
        return value * 100., agent_values * 100.

    @partial(jax.jit, static_argnums=(0,))
    def is_together_center(self, state, target_center):
        """Reach target where the maximum distance between all evaders is minimized."""
        positions = self._get_agent_positions(state)
        radius = HERD_TARGET_RADIUS
        radius_center = HERD_CENTER_RADIUS

        evaders_array = jnp.array(self.evaders)
        centroid = jnp.mean(positions[evaders_array], axis=0)
        reach_together = jnp.max(jnp.linalg.norm(positions[evaders_array] - centroid, axis=1)) - radius

        # Dist centroid to center
        reach_center = jnp.linalg.norm(centroid) - radius_center

        reach = jnp.maximum(reach_together, reach_center)

        value = jnp.where(reach < 0., -3., reach)
        agent_values = jnp.zeros(self.n_agents)
        return value * 100., agent_values * 100.

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
    def is_collisions(self, state, target_center):
        """All agents must avoid one another (worst over agents)."""
        positions = self._get_agent_positions(state)
        radius = HERDING_COLLISION_RADIUS

        worst_avoid = -jnp.inf
        agent_avoids = []

        # Check all pursuers for collision with one another, evaders and walls
        for i in range(self.n_agents):            
            agent_avoid_collision = [-jnp.inf]
            for j in range(self.n_agents):
                if i == j or i in self.evaders:
                    continue # We do not care about evaders colliding with each other DEBUG FIXME?
                avoid_collision = -(jnp.linalg.norm(positions[i] - positions[j]) - radius)
                agent_avoid_collision.append(avoid_collision)
            avoid_collision = jnp.max(jnp.array(agent_avoid_collision))

            avoid_wall_obstacles = jnp.maximum(jnp.fabs(positions[i][0]), jnp.fabs(positions[i][1])) - SAFETYGYM_RAA_BOX_CUSHION_RADIUS

            agent_avoid = jnp.maximum(5. * avoid_collision, 0.5 * avoid_wall_obstacles) # DEBUG FIXME what if we used soft-max?
            agent_avoids.append(agent_avoid)

            # We do not care about evaders colliding at all DEBUG FIXME?
            if i in self.evaders:
                continue  

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
        if self.add_ag_vals_to_obs and self.dynamic_predicate_names:
            observation = jnp.concatenate([state.obs, predicate_values, tracked_locs_flat, aux])
        elif self.dynamic_predicate_names:
            observation = jnp.concatenate([state.obs, predicate_values, tracked_locs_flat])
        elif self.add_ag_vals_to_obs:
            observation = jnp.concatenate([state.obs, predicate_values, aux])
        else:
            observation = jnp.concatenate([state.obs, predicate_values])

        env_state = EnvStateHerd(state, predicate_values, predicate_values, tracked_locs)

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
        if self.add_ag_vals_to_obs and self.dynamic_predicate_names:
            observation = jnp.concatenate([next_state.obs, predicate_values, tracked_locs_flat, aux])
        elif self.dynamic_predicate_names:
            observation = jnp.concatenate([next_state.obs, predicate_values, tracked_locs_flat])
        elif self.add_ag_vals_to_obs:
            observation = jnp.concatenate([next_state.obs, predicate_values, aux])
        else:
            observation = jnp.concatenate([next_state.obs, predicate_values])

        next_state_new = EnvStateHerd(next_state, predicate_values, predicate_extrema, tracked_locs)

        reward = 0.
        done = next_state.done > 0.5

        # Extract positions and angles for all agents for info
        info = {}
        for i in range(self.n_agents):
            start_idx = i * self.obs_size_per_agent
            info[f"x_{i}"] = state.state.obs[start_idx]
            info[f"y_{i}"] = state.state.obs[start_idx + 1]
            if self.dynamics_type == "dubins":
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
        
        if self.dynamics_type == "double_integrator":
            # Double integrator: obs = [x, y, vx, vy, ax, ay]
            # Extract acceleration to compute prev_qd
            accel_list = []
            for i in range(self.n_agents):
                start_idx = i * self.obs_size_per_agent
                agent_obs = reset_obs[start_idx:start_idx + self.obs_size_per_agent]
                
                qpos_agent = jnp.array([
                    agent_obs[0],  # x
                    agent_obs[1],  # y
                ])
                qpos_list.append(qpos_agent)
                
                qvel_agent = jnp.array([
                    agent_obs[2],  # vx
                    agent_obs[3],  # vy
                ])
                qvel_list.append(qvel_agent)
                
                accel_agent = jnp.array([
                    agent_obs[4],  # ax
                    agent_obs[5],  # ay
                ])
                accel_list.append(accel_agent)
        elif self.dynamics_type == "dubins":
            # Dubins: obs = [x, y, sin(theta), cos(theta), vx, vy, vtheta]
            for i in range(self.n_agents):
                start_idx = i * self.obs_size_per_agent
                agent_obs = reset_obs[start_idx:start_idx + self.obs_size_per_agent]
                
                qpos_agent = jnp.array([
                    agent_obs[0],  # x
                    agent_obs[1],  # y
                    jnp.arctan2(agent_obs[2], agent_obs[3])  # theta from sin(theta), cos(theta)
                ])
                qpos_list.append(qpos_agent)
                
                qvel_agent = agent_obs[4:7]
                qvel_list.append(qvel_agent)
        else:
            raise NotImplementedError(f"Dynamics type '{self.dynamics_type}' is not implemented")
        
        qpos = jnp.concatenate(qpos_list)
        qvel = jnp.concatenate(qvel_list)
        
        base_env = self._env
        while hasattr(base_env, '_env'):
            base_env = base_env._env
        
        pipeline_state = base_env.pipeline_init(qpos, qvel)
        
        # Compute acceleration for double integrator (needed for obs)
        if self.dynamics_type == "double_integrator":
            # Use the acceleration from reset_obs
            acceleration = jnp.concatenate(accel_list)
            # Compute prev_qd such that (qvel - prev_qd) / dt = acceleration
            dt = self._env.env.env._dt
            prev_qd = qvel - acceleration * dt
            obs = self._env.env.env._get_obs(pipeline_state, acceleration)
        else:
            prev_qd = None
            obs = self._env._env._env._get_obs(pipeline_state)
        # TODO FIXME, why are these different?
        
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
            if self.dynamics_type == "double_integrator":
                metrics[f'x_acceleration_{i}'] = zero
                metrics[f'y_acceleration_{i}'] = zero
        
        # For double integrator, store previous qd for acceleration computation
        if self.dynamics_type == "double_integrator":
            state_info = {'rng': key, 'prev_qd': prev_qd}
        else:
            state_info = {'rng': key}
            
        state = State(pipeline_state, obs, reward, done, metrics, info=state_info)
        
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
        if self.add_ag_vals_to_obs and self.dynamic_predicate_names:
            observation = jnp.concatenate([state.obs, predicate_values, tracked_locs_flat, aux])
        elif self.dynamic_predicate_names:
            observation = jnp.concatenate([state.obs, predicate_values, tracked_locs_flat])
        elif self.add_ag_vals_to_obs:
            observation = jnp.concatenate([state.obs, predicate_values, aux])
        else:
            observation = jnp.concatenate([state.obs, predicate_values])
        env_state = EnvStateHerd(state, predicate_values, predicate_values, tracked_locs)

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
        return loc
    def reset_fn(key):
        return center
    pred_stats = {"mean_x": center[0], "mean_y": center[1], "std_x": 1., "std_y": 1.}
    return dynamics_fn, reset_fn, pred_stats

def constant_dynamics_with_random_reset(center=jnp.array([0., 0.]), bd=2., reset_distribution="uniform"):
    def dynamics_fn(key, state, loc):
        return loc
    def in_box_reset(key):
        key, subkey = jax.random.split(key)
        if reset_distribution == "uniform":
            return jax.random.uniform(subkey, shape=(2,), minval=-bd, maxval=bd)
        elif reset_distribution == "truncated_normal":
            return jax.random.truncated_normal(subkey, lower=-bd, upper=bd) * bd + center
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
        key, subkey = jax.random.split(key)
        angle = jax.random.uniform(subkey, minval=0, maxval=2 * jnp.pi)
        return center + radius * jnp.array([jnp.cos(angle), jnp.sin(angle)])

    pred_stats = {"mean_x": center[0], "mean_y": center[1], "std_x": radius, "std_y": radius}
    return dynamics_fn, reset_on_circle, pred_stats

def obstacle_weave_dynamics(speed=0.005):
    """Create a dynamics function that moves the location between 10 preset points
    
    Args:
        speed: Speed of movement along obstacle edges
        
    Returns:
        Dynamics function (key, state, loc) -> new_loc
    """

    PATH_POINTS = jnp.array([
        [-0.7, -1.5], 
        [-0.7, 1.2],
        [0., 1.2],
        [0., 1.5],
        [0.9, 1.5],
        [0.9, 0.7],
        [0.2, 0.7],
        [0.2, -0.5],
        [0.6, -0.5],
        [0.6, -1.5],
    ])
    
    def dynamics_fn(key, state, loc):
        # Find closest segment and move along it
        n_points = len(PATH_POINTS)
        
        # For each segment, compute distance from loc to that segment and parametric position
        distances = []
        t_values = []
        for i in range(n_points):
            next_i = (i + 1) % n_points
            segment_start = PATH_POINTS[i]
            segment_end = PATH_POINTS[next_i]
            
            # Compute parametric position along segment
            segment_vec = segment_end - segment_start
            segment_len_sq = jnp.sum(segment_vec ** 2)
            t = jnp.clip(jnp.sum((loc - segment_start) * segment_vec) / (segment_len_sq + 1e-8), 0., 1.)
            t_values.append(t)
            
            # Closest point on segment
            closest_point = segment_start + t * segment_vec
            dist = jnp.sum((loc - closest_point) ** 2)
            distances.append(dist)
        
        distances = jnp.array(distances)
        t_values = jnp.array(t_values)
        
        # When at a waypoint, prefer the forward segment (t=0) over backward segment (t=1)
        # Add a small bias to prefer segments where t < 0.9
        bias = jnp.where(t_values > 0.95, 0.001, 0.0)  # Small penalty for being at end of segment
        closest_segment_idx = jnp.argmin(distances + bias)
        
        # Get the t value for closest segment
        t_on_segment = t_values[closest_segment_idx]
        
        # Get current segment endpoints
        segment_start = PATH_POINTS[closest_segment_idx]
        next_idx = (closest_segment_idx + 1) % n_points
        segment_end = PATH_POINTS[next_idx]
        segment_vec = segment_end - segment_start
        segment_len = jnp.linalg.norm(segment_vec) + 1e-8
        
        # Move along the segment: advance by speed distance along the segment
        # We advance from the current t value on the segment
        new_t = t_on_segment + speed / segment_len
        
        # Clamp to [0, 1] to stay on segment
        new_t = jnp.clip(new_t, 0.0, 1.0)
        
        # Compute new location on the segment
        new_loc = segment_start + new_t * segment_vec
        
        return new_loc

    def reset_on_path(key):
        key, subkey = jax.random.split(key)
        point_idx = jax.random.randint(subkey, (), 0, len(PATH_POINTS))  # Scalar instead of (1,)
        key, subkey = jax.random.split(key)
        a = jax.random.uniform(subkey, (), minval=0., maxval=1.)  # Scalar instead of (1,)
        next_idx = (point_idx + 1) % len(PATH_POINTS)
        return (1 - a) * PATH_POINTS[point_idx] + a * PATH_POINTS[next_idx]

    # pred stats will be mean of obstacles
    mean_x = jnp.mean(PATH_POINTS[:, 0])
    mean_y = jnp.mean(PATH_POINTS[:, 1])
    std_x = jnp.std(PATH_POINTS[:, 0])
    std_y = jnp.std(PATH_POINTS[:, 1])
    pred_stats = {"mean_x": mean_x, "mean_y": mean_y, "std_x": std_x, "std_y": std_y}

    return dynamics_fn, reset_on_path, pred_stats