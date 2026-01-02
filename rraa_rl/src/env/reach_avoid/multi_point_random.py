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

class MultiPointRandom(PipelineEnv):

    def __init__(
            self,
            n_agents=2,
            backend='mjx',
            reset_noise_scale=1e-2,
            qd_noise_std=0.0,
            fixed_velocity=None,  # Add this parameter (e.g., 0.5 for constant speed)
            **kwargs,
    ):
        """Multi-agent point robot environment.
        
        Args:
            n_agents: Number of agents
            backend: Physics backend ('mjx' or 'generalized')
            reset_noise_scale: Scale of noise for reset
            qd_noise_std: Standard deviation of velocity noise
            fixed_velocity: If not None, agents move at constant speed (only control rotation)
        """
        assert n_agents >= 1, "n_agents must be at least 1"
        
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
        self.obs_size_per_agent = 7  # x, y, sin(theta), cos(theta), vx, vy, vtheta
        self.fixed_velocity = fixed_velocity  # Store the fixed velocity

    def _generate_xml(self, n_agents):
        """Generate MuJoCo XML for n agents."""
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
        
        # Header
        xml = """<mujoco>
    <size njmax="3000" nconmax="1000"/>
    <option timestep="0.002"/>
    <default>
        <geom condim="3" density="1" rgba="1 0 0 1" />
        <joint damping=".001"/>
        <motor ctrlrange="-1 1" ctrllimited="true" forcerange="-.5 .5" forcelimited="true"/>
        <velocity ctrlrange="-1 1" ctrllimited="true" forcerange="-.5 .5" forcelimited="true"/>
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
        for idx, obstacle_pos in enumerate(SAFETYGYM_OBSTACLE_SET):
            x, y = float(obstacle_pos[0]), float(obstacle_pos[1])
            # Create box obstacles with size = SAFETYGYM_RAA_OBSTACLE_RADIUS
            xml += f"""        <geom name="obstacle_{idx}" pos="{x} {y} 0.1" size="{SAFETYGYM_RAA_OBSTACLE_RADIUS} {SAFETYGYM_RAA_OBSTACLE_RADIUS} 0.1" type="box" rgba="0.8 0.2 0.2 0.8" friction="1 0.1 0.1"/>\n"""
        
        xml += """        
"""
        
        # Add agents
        for i in range(n_agents):
            agent_id = i + 1
            color = colors[i % len(colors)]
            xml += f"""        <!-- Agent {agent_id} -->
        <body name="robot{agent_id}" pos="0 0 .1">
            <camera name="vision{agent_id}" pos="0 0 .15" xyaxes="0 -1 0 .4 0 1" fovy="90"/>
            <joint type="slide" axis="1 0 0" name="x{agent_id}" damping="0.01"/>
            <joint type="slide" axis="0 1 0" name="y{agent_id}" damping="0.01"/>
            <joint type="hinge" axis="0 0 1" name="z{agent_id}" damping="0.005"/>
            <geom name="robot{agent_id}" type="sphere" size=".1" friction="1 0.01 0.01" rgba="{color}"/>
            <geom name="pointarrow{agent_id}" pos="0.1 0 0" size="0.05 0.05 0.05" type="box" rgba="{color}"/>
            <site name="robot{agent_id}" rgba="{color[:-2]} .1"/>
        </body>
        
"""
        
        xml += """    </worldbody>
    <sensor>
"""
        
        # Add sensors
        for i in range(n_agents):
            agent_id = i + 1
            xml += f"""        <!-- Agent {agent_id} sensors -->
        <accelerometer site="robot{agent_id}" name="accelerometer{agent_id}"/>
        <velocimeter site="robot{agent_id}" name="velocimeter{agent_id}"/>
        <gyro site="robot{agent_id}" name="gyro{agent_id}"/>
        <magnetometer site="robot{agent_id}" name="magnetometer{agent_id}"/>
        <subtreecom body="robot{agent_id}" name="subtreecom{agent_id}"/>
        <subtreelinvel body="robot{agent_id}" name="subtreelinvel{agent_id}"/>
        <subtreeangmom body="robot{agent_id}" name="subtreeangmom{agent_id}"/>
        
"""
        
        xml += """    </sensor>
    <actuator>
"""
        
        # Add actuators
        for i in range(n_agents):
            agent_id = i + 1
            xml += f"""        <!-- Agent {agent_id} actuators -->
        <motor gear="0.3 0 0 0 0 0" site="robot{agent_id}" name="x{agent_id}"/>
        <motor gear="0.3 0 0 0 0 0" joint="z{agent_id}" name="z{agent_id}"/>
        
"""
        
        xml += """    </actuator>
</mujoco>"""
        
        return xml

    @property
    def action_size(self) -> int:
        # If fixed velocity, only control rotation (1 action per agent)
        # Otherwise, control both velocity and rotation (2 actions per agent)
        return self.n_agents if self.fixed_velocity is not None else 2 * self.n_agents

    @property
    def observation_size(self) -> int:
        return self.obs_size_per_agent * self.n_agents

    def reset(self, rng: jax.Array) -> State:
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
                rng_key, rng_x, rng_y, rng_theta, rng_vel = jax.random.split(rng_key, 5)
                
                # Sample position uniformly within valid box bounds
                x = jax.random.uniform(rng_x, minval=-SAFETYGYM_RAA_BOX_CUSHION_RADIUS, 
                                      maxval=SAFETYGYM_RAA_BOX_CUSHION_RADIUS)
                y = jax.random.uniform(rng_y, minval=-SAFETYGYM_RAA_BOX_CUSHION_RADIUS, 
                                      maxval=SAFETYGYM_RAA_BOX_CUSHION_RADIUS)
                theta = jax.random.uniform(rng_theta, minval=-jnp.pi, maxval=jnp.pi)
                
                # Check if position is valid (not inside any obstacle's cushion)
                pos = jnp.array([x, y])
                distances_to_obstacles = jnp.linalg.norm(SAFETYGYM_OBSTACLE_SET - pos, axis=1)
                min_dist_to_obstacle = jnp.min(distances_to_obstacles)
                
                # Valid if minimum distance to any obstacle is greater than cushion radius
                is_valid = min_dist_to_obstacle > SAFETYGYM_RAA_OBSTACLE_CUSHION_RADIUS
                
                return is_valid, x, y, theta, rng_vel
            
            # Rejection sampling to find valid position
            def sample_until_valid(carry):
                rng_key, is_valid, x, y, theta, rng_vel = carry
                rng_key, rng_new = jax.random.split(rng_key)
                is_valid_new, x_new, y_new, theta_new, rng_vel_new = sample_valid_position(rng_new)
                
                # Keep new sample if valid, otherwise keep trying
                x = jnp.where(is_valid, x, x_new)
                y = jnp.where(is_valid, y, y_new)
                theta = jnp.where(is_valid, theta, theta_new)
                is_valid = is_valid | is_valid_new
                
                return (rng_key, is_valid, x, y, theta, rng_vel_new)
            
            def sample_condition(carry):
                _, is_valid, _, _, _, _ = carry
                return ~is_valid
            
            # Initialize with invalid state to start sampling
            init_carry = (rng_agent, False, 0.0, 0.0, 0.0, rng_agent)
            
            # Run rejection sampling for max 100 iterations (should succeed much earlier)
            final_carry = jax.lax.while_loop(
                sample_condition, 
                sample_until_valid, 
                init_carry,
            )
            
            _, _, x_final, y_final, theta_final, rng_vel = final_carry
            
            # Set position for agent i
            qpos = qpos.at[3*i].set(x_final)
            qpos = qpos.at[3*i+1].set(y_final)
            qpos = qpos.at[3*i+2].set(theta_final)

            # Set velocity
            qvel_agent = jax.random.uniform(rng_vel, (3,), minval=low, maxval=hi)
            qvel_list.append(qvel_agent)
        
        qvel = jnp.concatenate(qvel_list)
        pipeline_state = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(pipeline_state)
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
            
        return State(pipeline_state, obs, reward, done, metrics, info={'rng': rng})

    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        # Convert action based on fixed_velocity mode
        if self.fixed_velocity is not None:
            # action is shape (n_agents,) - only rotation control
            # Create full action with fixed forward velocity
            full_action_list = []
            for i in range(self.n_agents):
                full_action_list.extend([self.fixed_velocity, action[i]])
            full_action = jnp.array(full_action_list)
        else:
            # action is already shape (2*n_agents,)
            full_action = action
            
        pipeline_state = self.pipeline_step(state.pipeline_state, full_action)

        rng = state.info.get('rng', None)
        if rng is not None:
            rng_qd = jax.random.split(rng)[0]
            qd_noise = jax.random.uniform(
                rng_qd, (self.sys.qd_size(),), minval=-self._reset_noise_scale, maxval=self._reset_noise_scale
            ) * self.qd_noise_std
            pipeline_state = pipeline_state.replace(qd=pipeline_state.qd + qd_noise)

        obs = self._get_obs(pipeline_state)

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=0., done=0.
        )

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observes body position, velocities, and angles for all agents."""
        position = pipeline_state.q
        velocity = pipeline_state.qd

        obs_list = []
        for i in range(self.n_agents):
            # Extract agent i's state: [x, y, sin(theta), cos(theta), vx, vy, vtheta]
            agent_obs = jnp.array([
                position[3*i],      # x
                position[3*i+1],    # y
                jnp.sin(position[3*i+2]),  # sin(theta)
                jnp.cos(position[3*i+2]),  # cos(theta)
                velocity[3*i],      # vx
                velocity[3*i+1],    # vy
                velocity[3*i+2]     # vtheta
            ])
            obs_list.append(agent_obs)
        
        # Concatenate all agent observations
        return jnp.concatenate(obs_list)
