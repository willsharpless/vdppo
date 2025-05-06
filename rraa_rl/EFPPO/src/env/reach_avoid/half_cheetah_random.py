import jax
import jax.numpy as jnp

from brax.envs.base import State
from brax.envs.half_cheetah import Halfcheetah

class HalfCheetahRandom(Halfcheetah):

    def reset(self, rng: jnp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qpos = qpos.at[0].set(qpos[0] + jax.random.uniform(rng3, minval=0., maxval=4.0))
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
        return State(pipeline_state, obs, reward, done, metrics)
