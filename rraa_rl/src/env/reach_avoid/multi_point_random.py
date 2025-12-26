import jax
import jax.numpy as jnp
import mujoco
from io import StringIO

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

class MultiPointRandom(PipelineEnv):

    def __init__(
            self,
            n_agents=2,
            backend='mjx',
            reset_noise_scale=1e-2,
            qd_noise_std=0.0,
            **kwargs,
    ):
        """Multi-agent point robot environment.
        
        Args:
            n_agents: Number of agents
            backend: Physics backend ('mjx' or 'generalized')
            reset_noise_scale: Scale of noise for reset
            qd_noise_std: Standard deviation of velocity noise
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
        return 2 * self.n_agents

    @property
    def observation_size(self) -> int:
        return self.obs_size_per_agent * self.n_agents

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.init_q.copy()
        qvel_list = []
        
        # Initialize each agent
        for i in range(self.n_agents):
            rng, rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)
            # Position indices for agent i: [3*i, 3*i+1, 3*i+2] for [x, y, theta]
            qpos = qpos.at[3*i].set(qpos[3*i] + jax.random.uniform(rng1, minval=-2., maxval=2.))
            qpos = qpos.at[3*i+1].set(qpos[3*i+1] + jax.random.uniform(rng2, minval=-2., maxval=2.))
            qpos = qpos.at[3*i+2].set(qpos[3*i+2] + jax.random.uniform(rng3, minval=-jnp.pi, maxval=jnp.pi))

            qvel_agent = jax.random.uniform(rng4, (3,), minval=low, maxval=hi)
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
        # action should be shape (2*n_agents,)
        pipeline_state = self.pipeline_step(state.pipeline_state, action)

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
