import jax
import jax.numpy as jnp

from brax.envs.base import State
from brax.envs.humanoid import Humanoid

class HumanoidDeterministic(Humanoid):

    def reset(self, rng: jnp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1 = jax.random.split(rng, 2)

        ## Default brax reset
        qpos = self.sys.init_q
        qvel = 0. * jax.random.normal(rng1, (self.sys.qd_size(),))

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

class HumanoidRandom(Humanoid):

    def reset(self, rng: jnp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

        ## Default brax reset randomization
        # low, hi = -self._reset_noise_scale, self._reset_noise_scale
        # FIXME FIXME FIXME deterministic for now
        low, hi = -0. * self._reset_noise_scale, 0. * self._reset_noise_scale
        qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(
            rng2, (self.sys.qd_size(),), minval=low, maxval=hi
        )
        
        ## Random Offset (standard in Oswin's envs)
        # qpos = qpos.at[0].set(qpos[0] + jax.random.uniform(rng3, minval=0.5, maxval=1.5))
        # qvel = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        pipeline_state = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(pipeline_state, jnp.zeros(self.sys.act_size()))
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
