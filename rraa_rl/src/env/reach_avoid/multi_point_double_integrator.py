import jax
import jax.numpy as jnp
import mujoco
from io import StringIO

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

SAFETYGYM_RAA_OBSTACLE_RADIUS = 0.05
SAFETYGYM_RAA_BOX_RADIUS = 3.15
CUSHION_RADIUS = 0.15
SAFETYGYM_RAA_OBSTACLE_CUSHION_RADIUS = SAFETYGYM_RAA_OBSTACLE_RADIUS + CUSHION_RADIUS
SAFETYGYM_RAA_BOX_CUSHION_RADIUS = SAFETYGYM_RAA_BOX_RADIUS - 2 * CUSHION_RADIUS
# NOTE cushion must be greater than robot radius (0.1)

SAFETYGYM_OBSTACLE_SET = jnp.array([[1.403247, 0.6281236], [0.42943087, 1.17059302],
                                    [-1.16036429, 0.89811093], [-0.88776483, 1.46420776],
                                    [-0.07556364, -1.10567521], [0.72648704, 0.17957757],
                                    [-0.33115742, 0.83026827], [-1.33470321, -1.3259373]])


def default_evader_policy(obs_all_agents, agent_idx, n_agents, obs_size_per_agent=6, 
                          alpha=10.0, n_samples=16, max_accel=1.0, controllable_agents=None,
                          wall_weight=0.6, evader_weight=0.3, pursuer_weight=1.3, 
                          atten_radius_wall=0.8, atten_radius_evader=0.5, atten_radius_pursuer=1.5):
    """Default safety policy that maximizes soft minimum distance to other agents and walls.
    
    Args:
        obs_all_agents: Full observation array for all agents
        agent_idx: Index of the agent to compute action for
        n_agents: Total number of agents
        obs_size_per_agent: Size of observation per agent (default 6)
        alpha: Temperature parameter for soft minimum (higher = closer to hard minimum)
        n_samples: Number of action samples to evaluate
        max_accel: Maximum acceleration magnitude
    
    Returns:
        action: [ax, ay] that maximizes soft minimum distance
    """
    # Extract this agent's position using dynamic_slice (JAX-friendly)
    start_idx = agent_idx * obs_size_per_agent
    agent_obs = jax.lax.dynamic_slice(obs_all_agents, (start_idx,), (obs_size_per_agent,))
    agent_pos = agent_obs[0:2]  # [x, y]
    agent_vel = agent_obs[2:4]  # [vx, vy]
    
    # Sample candidate actions uniformly in a circle
    angles = jnp.linspace(0, 2 * jnp.pi, n_samples, endpoint=False)
    actions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1) * max_accel
    
    def evaluate_action(action):
        """Evaluate soft minimum distance for a given action."""
        # Predict next position (simple Euler integration)
        dt = 0.01  # Small time step for prediction
        next_vel = agent_vel + action * dt
        next_pos = agent_pos + next_vel * dt
        
        weighted_distances, dist_type_int = [], []

        # Apply attenuation: distance -> effective_distance
        def attenuate_distance(dist, weight, mode="none"):
            
            # Hard cutoff: within use weighted, else large value
            if mode == "hard":
                effective_dist = jnp.where(
                    dist < attenuation_radius,
                    dist / weight,  # Smaller weight = more emphasis (smaller effective distance)
                    1e6  # Large value = no effect
                )

            # Smooth: apply Gaussian kernel
            elif mode == "smooth":
                effective_dist = (1 - jnp.exp(-dist**2 / (2 * (attenuation_radius / weight)**2))) * dist
                # effective_dist = jnp.exp(-dist**2 / (2 * (attenuation_radius / weight)**2))

            # No attenuation
            else:
                effective_dist = dist / weight

            return effective_dist
        
        # Distance to walls
        dist_to_north_wall = SAFETYGYM_RAA_BOX_RADIUS - next_pos[1]
        dist_to_south_wall = SAFETYGYM_RAA_BOX_RADIUS + next_pos[1]
        dist_to_east_wall = SAFETYGYM_RAA_BOX_RADIUS - next_pos[0]
        dist_to_west_wall = SAFETYGYM_RAA_BOX_RADIUS + next_pos[0]
        
        weighted_distances.extend([
            attenuate_distance(dist_to_north_wall, wall_weight),
            attenuate_distance(dist_to_south_wall, wall_weight),
            attenuate_distance(dist_to_east_wall, wall_weight),
            attenuate_distance(dist_to_west_wall, wall_weight)
        ])

        dist_type_int.extend([0, 0, 0, 0])

        # Distance to other agents
        for i in range(n_agents):
            # Use dynamic slice for JAX compatibility
            other_start = i * obs_size_per_agent
            other_obs = jax.lax.dynamic_slice(obs_all_agents, (other_start,), (obs_size_per_agent,))
            other_pos = other_obs[0:2]
            
            # Only add distance if not the same agent
            is_same = (i == agent_idx)
            dist = jnp.linalg.norm(next_pos - other_pos)
            
            # Apply attenuation and weighting
            is_controllable = jnp.any(controllable_agents == i)
            agent_weight = jnp.where(is_controllable, pursuer_weight, evader_weight)
            effective_dist = jnp.where(
                is_same,
                1e6,  # Same agent = no effect
                attenuate_distance(dist, agent_weight)
            )
            weighted_distances.append(effective_dist)
            dist_type_int.append(jnp.where(is_controllable, 1, 2))

        weighted_distances = jnp.array(weighted_distances)
        dist_type_int = jnp.array(dist_type_int)

        # Compute soft minimum using log-sum-exp trick
        # soft_min(x) ≈ -log(sum(exp(-alpha * x))) / alpha
        soft_min = -jnp.log(jnp.sum(jnp.exp(-alpha * weighted_distances))) / alpha

        min_effective, argmin = jnp.min(weighted_distances), jnp.argmin(weighted_distances)

        # use dynamic slice to get min_type
        min_type = jax.lax.dynamic_slice(dist_type_int, (argmin,), (1,))

        return soft_min, min_effective, min_type

    # Evaluate all actions
    scores, min_dists, min_types = jax.vmap(evaluate_action)(actions)
    
    # Return action with highest score (maximum soft minimum distance)
    best_idx = jnp.argmax(scores)
    best_action = actions[best_idx]
    min_dist = min_dists[best_idx]
    min_type = min_types[best_idx]

    attenuation_radius = jnp.where(
        min_type == 0, 
        atten_radius_wall, 
        jnp.where(
            min_type == 1, 
            atten_radius_evader, 
            atten_radius_pursuer)
    )
    return best_action * jnp.exp(-min_dist**2 / (2 * (attenuation_radius)**2))


# for testing
def default_evader_policy2(obs_all_agents, agent_idx, n_agents, obs_size_per_agent=6, 
                          alpha=10.0, n_samples=16, max_accel=1.0):
    
    # Sample candidate actions uniformly in a circle
    angles = jnp.linspace(0, 2 * jnp.pi, n_samples, endpoint=False)
    actions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1) * max_accel
    return 0. * actions[0]

class MultiPointDoubleIntegrator(PipelineEnv):

    def __init__(
            self,
            n_agents=2,
            backend='mjx',
            reset_noise_scale=1e-2,
            qd_noise_std=0.0,
            max_acceleration=1.0,  # Maximum acceleration in m/s^2
            rel_acel=None,  # Relative Accelerations
            fixed_policy_agents=None,  # List of agent indices (0-based) with fixed policies
            fixed_policy_fn=None,  # Function that takes (obs, agent_idx) and returns action
            add_obstacles=False,
            **kwargs,
    ):
        """Multi-agent point robot environment with double integrator dynamics.
        
        Args:
            n_agents: Number of agents
            backend: Physics backend ('mjx' or 'generalized')
            reset_noise_scale: Scale of noise for reset
            qd_noise_std: Standard deviation of velocity noise
            max_acceleration: Maximum acceleration magnitude (controls are scaled to this)
            rel_acel: Relative accelerations for each agent. Can be a scalar (same for all) or array of length n_agents.
                       If None, no acceleration limit is applied.
            fixed_policy_agents: List of agent indices with fixed policies (e.g., [0, 2] for agents 0 and 2)
            fixed_policy_fn: Callable that takes (obs, agent_idx) and returns action [ax, ay] for that agent
            add_obstacles: Whether to add obstacles to the environment
        """
        assert n_agents >= 1, "n_agents must be at least 1"
        
        # Setup fixed policy agents
        if fixed_policy_agents is None:
            fixed_policy_agents = []
        self.fixed_policy_agents = jnp.array(fixed_policy_agents, dtype=jnp.int32)
        
        # Determine which agents are controllable
        self.controllable_agents = jnp.array([i for i in range(n_agents) if i not in fixed_policy_agents], dtype=jnp.int32)
        self.n_controllable = len(self.controllable_agents)

        # Store whether to use obstacle-aware reset
        self.add_obstacles = add_obstacles
        
        # Set policy function once at initialization
        # Use default policy if none specified and there are fixed policy agents
        if len(self.fixed_policy_agents) > 0:
            self.fixed_policy_fn = fixed_policy_fn if fixed_policy_fn is not None else self._default_policy_wrapper
        else:
            self.fixed_policy_fn = None
        
        # Generate XML dynamically based on n_agents
        xml_string = self._generate_xml(n_agents)
        sys = mjcf.loads(xml_string)

        n_frames = 5

        if backend == 'mjx':
            sys = sys.tree_replace({
                'opt.solver': mujoco.mjtSolver.mjSOL_NEWTON,
                'opt.disableflags': mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                'opt.iterations': 1,
                'opt.ls_iterations': 4,
            })

        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._reset_noise_scale = reset_noise_scale
        self.qd_noise_std = qd_noise_std
        self.n_agents = n_agents
        self.obs_size_per_agent = 6  # x, y, vx, vy, ax, ay
        self.max_acceleration = max_acceleration

        # Setup relative acceleration per agent
        if rel_acel is None:
            self.rel_acel = None
        else: 
            if jnp.isscalar(rel_acel) or (isinstance(rel_acel, (int, float))):
                # Same relative acceleration for all agents
                rel_acel_full = jnp.full(n_agents, float(rel_acel))
            else:
                # Different relative acceleration per agent
                rel_acel_full = jnp.array(rel_acel)
                assert len(rel_acel_full) == n_agents, f"rel_acel array must have length {n_agents}"
            rel_acel_full_norm = max_acceleration * (rel_acel_full / rel_acel_full.max())
            self.rel_acel = jnp.repeat(rel_acel_full_norm, 2)

        # Store timestep for acceleration computation (can't use self.dt as it's a property)
        self._dt = sys.opt.timestep * kwargs.get('n_frames', n_frames)

    def _generate_xml(self, n_agents):
        """Generate MuJoCo XML for n agents with double integrator dynamics."""
        colors = [
            "1 0 0 1",  # Red
            "0 1 0 1",  # Green
            "0 0 1 1",  # Blue
            "1 1 0 1",  # Yellow
            "1 0 1 1",  # Magenta
            "0 1 1 1",  # Cyan
            "1 0.5 0 1",  # Orange
            "0.5 0 1 1",  # Purple
        ]
        
        # Header - using very low damping for double integrator behavior
        xml = """<mujoco>
    <size njmax="3000" nconmax="1000"/>
    <option timestep="0.002"/>
    <default>
        <geom condim="3" density="1" rgba="1 0 0 1" />
        <joint damping=".001"/>
        <motor ctrlrange="-1 1" ctrllimited="true" forcerange="-.5 .5" forcelimited="true"/>
        <site size="0.032" type="sphere"/>
    </default>
    <worldbody>
        <geom name="floor" size="5 5 0.1" type="plane" condim="3"/>
        
        <!-- Boundary walls -->
"""
        xml += f"""        <geom name="wall_north" pos="0 {SAFETYGYM_RAA_BOX_RADIUS} 0.5" size="{SAFETYGYM_RAA_BOX_RADIUS} 0.05 0.5" type="box" rgba="0.3 0.3 0.3 1" friction="1 0.1 0.1"/>\n"""
        xml += f"""        <geom name="wall_south" pos="0 -{SAFETYGYM_RAA_BOX_RADIUS} 0.5" size="{SAFETYGYM_RAA_BOX_RADIUS} 0.05 0.5" type="box" rgba="0.3 0.3 0.3 1" friction="1 0.1 0.1"/>\n"""
        xml += f"""        <geom name="wall_east" pos="{SAFETYGYM_RAA_BOX_RADIUS} 0 0.5" size="0.05 {SAFETYGYM_RAA_BOX_RADIUS} 0.5" type="box" rgba="0.3 0.3 0.3 1" friction="1 0.1 0.1"/>\n"""
        xml += f"""        <geom name="wall_west" pos="-{SAFETYGYM_RAA_BOX_RADIUS} 0 0.5" size="0.05 {SAFETYGYM_RAA_BOX_RADIUS} 0.5" type="box" rgba="0.3 0.3 0.3 1" friction="1 0.1 0.1"/>\n"""

        # Add obstacles from SAFETYGYM_OBSTACLE_SET
        if self.add_obstacles:
            for idx, obstacle_pos in enumerate(SAFETYGYM_OBSTACLE_SET):
                x, y = float(obstacle_pos[0]), float(obstacle_pos[1])
                xml += f"""        <geom name="obstacle_{idx}" pos="{x} {y} 0.1" size="{SAFETYGYM_RAA_OBSTACLE_RADIUS} {SAFETYGYM_RAA_OBSTACLE_RADIUS} 0.1" type="box" rgba="0.8 0.2 0.2 0.8" friction="1 0.1 0.1"/>\n"""
            
        xml += """        
"""
        
        # Add agents - point masses with only x,y translation (no rotation)
        for i in range(n_agents):
            agent_id = i + 1
            color = colors[i % len(colors)]
            xml += f"""        <!-- Agent {agent_id} -->
        <body name="robot{agent_id}" pos="0 0 .1">
            <joint type="slide" axis="1 0 0" name="x{agent_id}" damping="0.01"/>
            <joint type="slide" axis="0 1 0" name="y{agent_id}" damping="0.01"/>
            <geom name="robot{agent_id}" type="sphere" size=".1" friction="1 0.01 0.01" rgba="{color}"/>
            <site name="robot{agent_id}" rgba="{color[:-2]} .1"/>
        </body>
        
"""
        
        xml += """    </worldbody>
    <actuator>
"""
        
        # Add actuators - direct force control for double integrator
        for i in range(n_agents):
            agent_id = i + 1
            xml += f"""        <!-- Agent {agent_id} actuators - acceleration control -->
        <motor gear="0.3 0 0 0 0 0" joint="x{agent_id}" name="ax{agent_id}"/>
        <motor gear="0.3 0 0 0 0 0" joint="y{agent_id}" name="ay{agent_id}"/>
        
"""
        
        xml += """    </actuator>
</mujoco>"""
        
        return xml

    @property
    def action_size(self) -> int:
        # Double integrator: 2 actions per controllable agent (ax, ay)
        # Fixed policy agents are not included in the action space
        return 2 * self.n_controllable

    @property
    def observation_size(self) -> int:
        return self.obs_size_per_agent * self.n_agents

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        if self.add_obstacles:
            return self._reset_with_obstacles(rng)
        else:
            return self._reset(rng)

    def _reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.init_q.copy()
        qvel_list = []
        
        # Initialize each agent with collision-free positions
        for i in range(self.n_agents):

            # split rng three times
            rng, rng_agent, rng_agent_1, rng_agent_2 = jax.random.split(rng, 4)

            # Sample position in walls
            x_final = jax.random.uniform(rng_agent, minval=-SAFETYGYM_RAA_BOX_CUSHION_RADIUS/1.1, maxval=SAFETYGYM_RAA_BOX_CUSHION_RADIUS/1.1)
            y_final = jax.random.uniform(rng_agent_1, minval=-SAFETYGYM_RAA_BOX_CUSHION_RADIUS/1.1, maxval=SAFETYGYM_RAA_BOX_CUSHION_RADIUS/1.1)

            # Set position for agent i (only x, y - no theta)
            qpos = qpos.at[2*i].set(x_final)
            qpos = qpos.at[2*i+1].set(y_final)

            # Set velocity
            qvel_agent = jax.random.uniform(rng_agent_2, (2,), minval=low, maxval=hi)
            qvel_list.append(qvel_agent)
        
        qvel = jnp.concatenate(qvel_list)
        pipeline_state = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(pipeline_state, jnp.zeros(2 * self.n_agents))
        reward, done, zero = jnp.zeros(3)
        metrics = {
            'forward_reward': zero,
            'reward_linvel': zero,
            'reward_quadctrl': zero,
            'reward_alive': zero,
        }
        # Add per-agent metrics
        for i in range(self.n_agents):
            metrics[f'x_position_{i}'] = zero
            metrics[f'y_position_{i}'] = zero
            metrics[f'distance_from_origin_{i}'] = zero
            metrics[f'x_velocity_{i}'] = zero
            metrics[f'y_velocity_{i}'] = zero
            metrics[f'x_acceleration_{i}'] = zero
            metrics[f'y_acceleration_{i}'] = zero
            
        return State(pipeline_state, obs, reward, done, metrics, info={'rng': rng, 'prev_qd': qvel})
    
    def _reset_with_obstacles(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.init_q.copy()
        qvel_list = []
        
        # Initialize each agent with collision-free positions
        for i in range(self.n_agents):
            rng, rng_agent = jax.random.split(rng)
            
            # Sample valid position for agent i
            def sample_valid_position(rng_key):
                """Sample a position that's not inside obstacles or too close to walls."""
                rng_key, rng_x, rng_y, rng_vel = jax.random.split(rng_key, 4)
                
                # Sample position uniformly within valid box bounds
                x = jax.random.uniform(rng_x, minval=-SAFETYGYM_RAA_BOX_CUSHION_RADIUS, 
                                      maxval=SAFETYGYM_RAA_BOX_CUSHION_RADIUS)
                y = jax.random.uniform(rng_y, minval=-SAFETYGYM_RAA_BOX_CUSHION_RADIUS, 
                                      maxval=SAFETYGYM_RAA_BOX_CUSHION_RADIUS)
                
                # Check if position is valid (not inside any obstacle's cushion)
                pos = jnp.array([x, y])
                distances_to_obstacles = jnp.linalg.norm(SAFETYGYM_OBSTACLE_SET - pos, axis=1)
                min_dist_to_obstacle = jnp.min(distances_to_obstacles)
                
                # Valid if minimum distance to any obstacle is greater than cushion radius
                is_valid = min_dist_to_obstacle > SAFETYGYM_RAA_OBSTACLE_CUSHION_RADIUS
                
                return is_valid, x, y, rng_vel
            
            # Rejection sampling to find valid position
            def sample_until_valid(carry):
                rng_key, is_valid, x, y, rng_vel = carry
                rng_key, rng_new = jax.random.split(rng_key)
                is_valid_new, x_new, y_new, rng_vel_new = sample_valid_position(rng_new)
                
                # Keep new sample if valid, otherwise keep trying
                x = jnp.where(is_valid, x, x_new)
                y = jnp.where(is_valid, y, y_new)
                is_valid = is_valid | is_valid_new
                
                return (rng_key, is_valid, x, y, rng_vel_new)
            
            def sample_condition(carry):
                _, is_valid, _, _, _ = carry
                return ~is_valid
            
            # Initialize with invalid state to start sampling
            init_carry = (rng_agent, False, 0.0, 0.0, rng_agent)
            
            # Run rejection sampling
            final_carry = jax.lax.while_loop(
                sample_condition, 
                sample_until_valid, 
                init_carry,
            )
            
            _, _, x_final, y_final, rng_vel = final_carry
            
            # Set position for agent i (only x, y - no theta)
            qpos = qpos.at[2*i].set(x_final)
            qpos = qpos.at[2*i+1].set(y_final)

            # Set velocity
            qvel_agent = jax.random.uniform(rng_vel, (2,), minval=low, maxval=hi)
            qvel_list.append(qvel_agent)
        
        qvel = jnp.concatenate(qvel_list)
        pipeline_state = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(pipeline_state, jnp.zeros(2 * self.n_agents))
        reward, done, zero = jnp.zeros(3)
        metrics = {
            'forward_reward': zero,
            'reward_linvel': zero,
            'reward_quadctrl': zero,
            'reward_alive': zero,
        }
        # Add per-agent metrics
        for i in range(self.n_agents):
            metrics[f'x_position_{i}'] = zero
            metrics[f'y_position_{i}'] = zero
            metrics[f'distance_from_origin_{i}'] = zero
            metrics[f'x_velocity_{i}'] = zero
            metrics[f'y_velocity_{i}'] = zero
            metrics[f'x_acceleration_{i}'] = zero
            metrics[f'y_acceleration_{i}'] = zero
            
        return State(pipeline_state, obs, reward, done, metrics, info={'rng': rng, 'prev_qd': qvel})

    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        # Construct full action array for all agents
        full_action = jnp.zeros(2 * self.n_agents)
        
        # Fill in actions for controllable agents
        for i, agent_idx in enumerate(self.controllable_agents):
            full_action = full_action.at[2*agent_idx].set(action[2*i])
            full_action = full_action.at[2*agent_idx+1].set(action[2*i+1])
        
        # Fill in actions for fixed policy agents
        if self.fixed_policy_fn is not None:
            obs = state.obs
            for agent_idx in self.fixed_policy_agents:
                # Get action from fixed policy
                fixed_action = self.fixed_policy_fn(obs, agent_idx)
                # Clip fixed action to prevent instability
                fixed_action = jnp.clip(fixed_action, -1.0, 1.0)
                full_action = full_action.at[2*agent_idx].set(fixed_action[0])
                full_action = full_action.at[2*agent_idx+1].set(fixed_action[1])
        
        # Clip and scale action to acceleration limits
        full_action = jnp.clip(full_action, -1.0, 1.0)
        scaled_action = full_action * self.rel_acel
        
        pipeline_state = self.pipeline_step(state.pipeline_state, scaled_action)

        # # Enforce position boundaries AFTER physics step
        # qpos = pipeline_state.q
        # qvel = pipeline_state.qd
        
        # for i in range(self.n_agents):
        #     x = qpos[2*i]
        #     y = qpos[2*i+1]
        #     vx = qvel[2*i]
        #     vy = qvel[2*i+1]
            
        #     # Clamp positions to boundaries
        #     x_clamped = jnp.clip(x, -SAFETYGYM_RAA_BOX_RADIUS, SAFETYGYM_RAA_BOX_RADIUS)
        #     y_clamped = jnp.clip(y, -SAFETYGYM_RAA_BOX_RADIUS, SAFETYGYM_RAA_BOX_RADIUS)
            
        #     # # If hit a wall, zero out the velocity component normal to the wall
        #     # vx = jnp.where((x != x_clamped), 0.0, vx)
        #     # vy = jnp.where((y != y_clamped), 0.0, vy)
            
        #     qpos = qpos.at[2*i].set(x_clamped)
        #     qpos = qpos.at[2*i+1].set(y_clamped)
        #     qvel = qvel.at[2*i].set(vx)
        #     qvel = qvel.at[2*i+1].set(vy)
        
        # pipeline_state = pipeline_state.replace(q=qpos, qd=qvel)

        # # Apply relative acceleration constraints per agent if specified
        # if self.rel_acel is not None:
        #     qd = pipeline_state.qd
        #     for i in range(self.n_agents):
        #         vx = qd[2*i]
        #         vy = qd[2*i+1]
        #         speed = jnp.sqrt(vx**2 + vy**2)
                
        #         # Clip speed if it exceeds max for this agent (with larger epsilon)
        #         scale = jnp.minimum(1.0, self.max_speed[i] / jnp.maximum(speed, 1e-6))
        #         qd = qd.at[2*i].set(vx * scale)
        #         qd = qd.at[2*i+1].set(vy * scale)
            
        #     pipeline_state = pipeline_state.replace(qd=qd)

        # Add noise BEFORE computing acceleration to avoid inf/nan
        rng = state.info.get('rng', None)
        if rng is not None:
            rng_qd = jax.random.split(rng)[0]
            qd_noise = jax.random.uniform(
                rng_qd, (self.sys.qd_size(),), minval=-self._reset_noise_scale, maxval=self._reset_noise_scale
            ) * self.qd_noise_std
            pipeline_state = pipeline_state.replace(qd=pipeline_state.qd + qd_noise)

        # Compute acceleration from velocity change with safe denominator
        prev_qd = state.info.get('prev_qd', pipeline_state.qd)
        dt_safe = jnp.maximum(self._dt, 1e-6)  # Prevent division by very small dt
        acceleration = (pipeline_state.qd - prev_qd) / dt_safe
        
        # # Clip acceleration to prevent extreme values
        # acceleration = jnp.clip(acceleration, -10.0 * self.max_acceleration, 10.0 * self.max_acceleration)

        obs = self._get_obs(pipeline_state, acceleration)
        # DEBUG FIXME do we even want accel in observation?

        return state.replace(
            pipeline_state=pipeline_state, 
            obs=obs, 
            reward=0., 
            done=0.,
            info={**state.info, 'rng': rng, 'prev_qd': pipeline_state.qd}
        )

    def _get_obs(self, pipeline_state: base.State, acceleration: jax.Array) -> jax.Array:
        """Observes body position, velocities, and accelerations for all agents."""
        position = pipeline_state.q
        velocity = pipeline_state.qd

        obs_list = []
        for i in range(self.n_agents):
            # Extract agent i's state: [x, y, vx, vy, ax, ay]
            agent_obs = jnp.array([
                position[2*i],      # x
                position[2*i+1],    # y
                velocity[2*i],      # vx
                velocity[2*i+1],    # vy
                acceleration[2*i],  # ax
                acceleration[2*i+1] # ay
            ])
            obs_list.append(agent_obs)
        
        # Concatenate all agent observations
        return jnp.concatenate(obs_list)
    
    def _default_policy_wrapper(self, obs_all_agents, agent_idx):
        """Wrapper to call default_evader_policy with environment parameters."""
        return default_evader_policy(
            obs_all_agents=obs_all_agents,
            agent_idx=agent_idx,
            n_agents=self.n_agents,
            obs_size_per_agent=self.obs_size_per_agent,
            alpha=10.0,
            n_samples=16,
            max_accel=1.0,
            controllable_agents=self.controllable_agents
        )
