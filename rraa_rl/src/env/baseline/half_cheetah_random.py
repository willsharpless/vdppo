import jax
import jax.numpy as jnp

from brax.envs.base import State
from brax.envs.half_cheetah import Halfcheetah

class HalfCheetahRandom(Halfcheetah):

    def __init__(
        self,
        qd_noise_std=0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.qd_noise_std = qd_noise_std

    def reset(self, rng: jnp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qpos = qpos.at[0].set(qpos[0] + jax.random.uniform(rng3, minval=0.5, maxval=1.5))
        qvel = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        pipeline_state = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(pipeline_state)
        reward, done, zero = jnp.zeros(3)
        metrics = {
            'x_position': zero,
            'x_velocity': zero,
            'reward_ctrl': zero,
            'reward_run': zero,
        }
        return State(pipeline_state, obs, reward, done, metrics, info={'rng': rng})
        
    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics.

        WAS - Added noise to qd on each step. Defaults std to 0.
        """
        pipeline_state0 = state.pipeline_state
        assert pipeline_state0 is not None
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        rng = state.info.get('rng', None)
        if rng is not None:
            rng, rng_qd = jax.random.split(rng)
            qd_noise = jax.random.uniform(
                rng_qd, (self.sys.qd_size(),), minval=-self._reset_noise_scale, maxval=self._reset_noise_scale
            )  * self.qd_noise_std
            pipeline_state = pipeline_state.replace(qd=pipeline_state.qd + qd_noise)

        x_velocity = (
            pipeline_state.x.pos[0, 0] - pipeline_state0.x.pos[0, 0]
        ) / self.dt
        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))

        obs = self._get_obs(pipeline_state)
        reward = forward_reward - ctrl_cost
        state.metrics.update(
            x_position=pipeline_state.x.pos[0, 0],
            x_velocity=x_velocity,
            reward_run=forward_reward,
            reward_ctrl=-ctrl_cost,
        )

        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward)

