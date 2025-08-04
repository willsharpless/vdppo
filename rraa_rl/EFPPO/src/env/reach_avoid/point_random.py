import jax
import jax.numpy as jnp
import mujoco

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

class PointRandom(PipelineEnv):

    def __init__(
            self,
            backend='mjx',
            reset_noise_scale=1e-2,
            **kwargs,
    ):

        path = "./rraa_rl/EFPPO/src/env/point.xml"
        sys = mjcf.load(path)

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

    @property
    def action_size(self) -> int:
        return 2

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.init_q
        qpos = qpos.at[0].set(qpos[0] + jax.random.uniform(rng1, minval=-2., maxval=2.))
        qpos = qpos.at[1].set(qpos[1] + jax.random.uniform(rng3, minval=-2., maxval=2.))
        qpos = qpos.at[2].set(qpos[2] + jax.random.uniform(rng4, minval=-jnp.pi, maxval=jnp.pi))
        qvel = jax.random.uniform(
            rng2, (self.sys.qd_size(),), minval=low, maxval=hi
        )
        pipeline_state = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(pipeline_state)
        reward, done, zero = jnp.zeros(3)
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
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""

        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_obs(pipeline_state)

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=0., done=0.
        )

    def _get_obs(
            self, pipeline_state: base.State
    ) -> jax.Array:
        """Observes body position, velocities, and angles."""
        position = pipeline_state.q
        velocity = pipeline_state.qd

        # external_contact_forces are excluded
        return jnp.array([
            position[0],
            position[1],
            jnp.sin(position[2]),
            jnp.cos(position[2]),
            velocity[0],
            velocity[1],
            velocity[2]
        ])

class PointRandomOuter(PipelineEnv):

    def __init__(
            self,
            backend='mjx',
            reset_noise_scale=1e-2,
            **kwargs,
    ):

        path = "./rraa_rl/EFPPO/src/env/point.xml"
        sys = mjcf.load(path)

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

    @property
    def action_size(self) -> int:
        return 2

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.init_q
        qpos = qpos.at[0].set(qpos[0] + jax.random.uniform(rng1, minval=-1., maxval=1.))
        qpos = qpos.at[1].set(qpos[1] + jax.random.uniform(rng3, minval=-1., maxval=1.))
        
        qpos = qpos.at[0].set(qpos[0] + 2. * jnp.sign(qpos[0]))
        qpos = qpos.at[1].set(qpos[1] + 2. * jnp.sign(qpos[1]))
        
        qpos = qpos.at[2].set(qpos[2] + jax.random.uniform(rng4, minval=-jnp.pi, maxval=jnp.pi))
        qvel = jax.random.uniform(
            rng2, (self.sys.qd_size(),), minval=low, maxval=hi
        )
        pipeline_state = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(pipeline_state)
        reward, done, zero = jnp.zeros(3)
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
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""

        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_obs(pipeline_state)

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=0., done=0.
        )

    def _get_obs(
            self, pipeline_state: base.State
    ) -> jax.Array:
        """Observes body position, velocities, and angles."""
        position = pipeline_state.q
        velocity = pipeline_state.qd

        # external_contact_forces are excluded
        return jnp.array([
            position[0],
            position[1],
            jnp.sin(position[2]),
            jnp.cos(position[2]),
            velocity[0],
            velocity[1],
            velocity[2]
        ])
