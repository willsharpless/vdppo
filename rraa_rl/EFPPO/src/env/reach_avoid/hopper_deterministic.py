import jax
import jax.numpy as jnp

from brax.envs.base import State
from brax.envs.hopper import Hopper

class HopperDeterministic(Hopper):

    def reset(self, rng: jnp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

        # qpos = self.sys.init_q
        qpos = jnp.array([2., 0., 0., 0., 0., 0.], dtype=jnp.float32)
        qvel = jax.random.uniform(
            rng2, (self.sys.qd_size(),), minval=0, maxval=0
        )

        pipeline_state = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(pipeline_state)
        reward, done, zero = jnp.zeros(3)
        metrics = {
            'reward_forward': zero,
            'reward_ctrl': zero,
            'reward_healthy': zero,
            'x_position': zero,
            'x_velocity': zero,
        }
        return State(pipeline_state, obs, reward, done, metrics)
