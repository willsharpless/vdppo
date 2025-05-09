import jax
import jax.numpy as jnp
from functools import partial
from gymnax.environments import spaces
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper
from flax import struct
from brax.envs.base import State

from .hopper_random import HopperRandom
from .hopper_deterministic import HopperDeterministic

@struct.dataclass
class EnvState:
    state: State
    energy: float
    reach: float
    avoid: int

@struct.dataclass
class EnvParams:
    min_energy: float = -400.0
    max_energy: float = 800.0
    torque_limit: float = 0.2
    max_torque: float = 1.0

# WAS: several below could be identical, but separated for safety

@struct.dataclass
class EnvStateR:
    state: State
    reach: float

@struct.dataclass
class EnvStateR1:
    state: State
    reach1: float

@struct.dataclass
class EnvStateR2:
    state: State
    reach2: float

@struct.dataclass
class EnvStateRR:
    state: State
    reach1: float
    reach2: float
    has_reached_1: float
    has_reached_2: float

@struct.dataclass
class EnvStateRAA:
    state: State
    avoid: float
    reach: float
    has_reached: float

@struct.dataclass
class EnvStateAvoidOnly:
    state: State
    avoid: float

@struct.dataclass
class EnvParamsEmpty:
    pass

class HopperAvoidCeiling:
    def __init__(self, backend="positional"):
        env = HopperRandom(backend=backend,
                           exclude_current_positions_from_observation=False,
                           terminate_when_unhealthy=False)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=2)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)
        self.default_params = EnvParams()

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)
        init_energy = jax.random.uniform(
            key, minval=params.min_energy, maxval=params.max_energy
        )
        head_pos, _, _, _, _, _ = self.calculate_position(state.obs)
        is_avoid = self.is_avoid(head_pos)
        avoid_value = jnp.where(is_avoid, -1, 1)
        reach_value = self.is_reach(head_pos, avoid_value)
        observation = jnp.concatenate([state.obs, jnp.array([avoid_value, init_energy])])
        env_state = EnvState(state, init_energy, reach_value, avoid_value)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.tanh(action)
        reach_limit_0 = jnp.fabs(u[0] * state.state.obs[-3] / 2.) > params.torque_limit
        energy_consumption_0 = jnp.where(reach_limit_0, (jnp.fabs(u[0] * state.state.obs[-3] / 2.) ** 2) * 0.6, 0.)
        reach_limit_1 = jnp.fabs(u[1] * state.state.obs[-2] / 2.) > params.torque_limit
        energy_consumption_1 = jnp.where(reach_limit_1, (jnp.fabs(u[1] * state.state.obs[-2] / 2.) ** 2) * 0.6, 0.)
        reach_limit_2 = jnp.fabs(u[2] * state.state.obs[-1] / 2.) > params.torque_limit
        energy_consumption_2 = jnp.where(reach_limit_2, (jnp.fabs(u[2] * state.state.obs[-1] / 2.) ** 2) * 0.6, 0.)
        energy_consumption = energy_consumption_0 + energy_consumption_1 + energy_consumption_2
        next_state = self._env.step(state.state, u)
        head_pos, _, _, _, _, _ = self.calculate_position(next_state.obs)
        is_avoid = self.is_avoid(head_pos)
        avoid_value = jnp.where(is_avoid, -1, state.avoid)
        reach_value = self.is_reach(head_pos, avoid_value)
        head_pos, jaw_pos, thg_pos, leg_pos, foot_front_pos, foot_back_pos = self.calculate_position(state.state.obs)
        pos_dict = {"head_pos": head_pos, "jaw_pos": jaw_pos, "thg_pos": thg_pos, "leg_pos": leg_pos,
                    "foot_front_pos": foot_front_pos, "foot_back_pos": foot_back_pos}
        next_energy = jnp.clip(state.energy - energy_consumption, params.min_energy, params.max_energy)
        observation = jnp.concatenate([next_state.obs, jnp.array([avoid_value, next_energy])])
        next_state_new = EnvState(next_state, next_energy, reach_value, avoid_value)

        return observation, next_state_new, energy_consumption, next_state.done > 0.5, pos_dict

    @partial(jax.jit, static_argnums=(0,))
    def calculate_position(self, obs):
        head_pos = jnp.array([obs[0] + 0.2 * jnp.sin(obs[2]),
                              obs[1] + 0.2 * jnp.cos(obs[2])])
        jaw_pos = jnp.array([obs[0] - 0.2 * jnp.sin(obs[2]),
                             obs[1] - 0.2 * jnp.cos(obs[2])])
        thg_pos = jnp.array([jaw_pos[0] - 0.45 * jnp.sin(obs[2] - obs[3]),
                             jaw_pos[1] - 0.45 * jnp.cos(obs[2] - obs[3])])
        leg_pos = jnp.array([thg_pos[0] - 0.5 * jnp.sin(obs[2] - obs[3] - obs[4]),
                             thg_pos[1] - 0.5 * jnp.cos(obs[2] - obs[3] - obs[4])])
        foot_back_pos = jnp.array([leg_pos[0] - 0.13 * jnp.cos(obs[2] - obs[3] - obs[4] - obs[5]),
                                    leg_pos[1] + 0.13 * jnp.sin(obs[2] - obs[3] - obs[4] - obs[5])])
        foot_front_pos = jnp.array([leg_pos[0] + 0.26 * jnp.cos(obs[2] - obs[3] - obs[4] - obs[5]),
                                   leg_pos[1] - 0.26 * jnp.sin(obs[2] - obs[3] - obs[4] - obs[5])])
        return head_pos, jaw_pos, thg_pos, leg_pos, foot_front_pos, foot_back_pos

    @partial(jax.jit, static_argnums=(0,))
    def is_reach(self, head_pos, avoid_value):
        reach = jnp.sqrt((head_pos[0] - 2.0) ** 2 + (head_pos[1] - 1.4) ** 2) - 0.1
        has_reached_goal = jnp.sqrt((head_pos[0] - 2.0) ** 2 + (head_pos[1] - 1.4) ** 2) < 0.1
        value = jnp.where(has_reached_goal, -2.5, reach)
        is_avoid = (avoid_value == -1)
        value = jnp.where(is_avoid, 3.0, value)
        return value * 100.0

    @partial(jax.jit, static_argnums=(0,))
    def is_avoid(self, head_pos):
        avoid_1 = (head_pos[1] >= 1.3) & (head_pos[0] >= 0.95) & (head_pos[0] <= 1.05)
        return avoid_1

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size + 2,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )

class HopperAvoidCeilingDeterministic:
    def __init__(self, backend="positional"):
        env = HopperDeterministic(backend=backend,
                           exclude_current_positions_from_observation=False,
                           terminate_when_unhealthy=False)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=2)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)
        self.default_params = EnvParams()

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)
        init_energy = jax.random.uniform(
            key, minval=0, maxval=params.max_energy
        )
        head_pos, _, _, _, _, _ = self.calculate_position(state.obs)
        is_avoid = self.is_avoid(head_pos)
        avoid_value = jnp.where(is_avoid, -1, 1)
        reach_value = self.is_reach(head_pos, avoid_value)
        observation = jnp.concatenate([state.obs, jnp.array([avoid_value, init_energy])])
        env_state = EnvState(state, init_energy, reach_value, avoid_value)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.tanh(action)
        reach_limit_0 = jnp.fabs(u[0] * state.state.obs[-3] / 2.) > params.torque_limit
        energy_consumption_0 = jnp.where(reach_limit_0, (jnp.fabs(u[0] * state.state.obs[-3] / 2.) ** 2) * 0.3, 0.)
        reach_limit_1 = jnp.fabs(u[1] * state.state.obs[-2] / 2.) > params.torque_limit
        energy_consumption_1 = jnp.where(reach_limit_1, (jnp.fabs(u[1] * state.state.obs[-2] / 2.) ** 2) * 0.3, 0.)
        reach_limit_2 = jnp.fabs(u[2] * state.state.obs[-1] / 2.) > params.torque_limit
        energy_consumption_2 = jnp.where(reach_limit_2, (jnp.fabs(u[2] * state.state.obs[-1] / 2.) ** 2) * 0.3, 0.)
        energy_consumption = energy_consumption_0 + energy_consumption_1 + energy_consumption_2
        next_state = self._env.step(state.state, u)
        head_pos, _, _, _, _, _ = self.calculate_position(next_state.obs)
        is_avoid = self.is_avoid(head_pos)
        avoid_value = jnp.where(is_avoid, -1, state.avoid)
        reach_value = self.is_reach(head_pos, avoid_value)
        head_pos, jaw_pos, thg_pos, leg_pos, foot_front_pos, foot_back_pos = self.calculate_position(state.state.obs)
        pos_dict = {"head_pos": head_pos, "jaw_pos": jaw_pos, "thg_pos": thg_pos, "leg_pos": leg_pos,
                    "foot_front_pos": foot_front_pos, "foot_back_pos": foot_back_pos}
        next_energy = jnp.clip(state.energy - energy_consumption, params.min_energy, params.max_energy)
        observation = jnp.concatenate([next_state.obs, jnp.array([avoid_value, next_energy])])
        next_state_new = EnvState(next_state, next_energy, reach_value, avoid_value)

        return observation, next_state_new, energy_consumption, next_state.done > 0.5, pos_dict

    @partial(jax.jit, static_argnums=(0,))
    def calculate_position(self, obs):
        head_pos = jnp.array([obs[0] + 0.2 * jnp.sin(obs[2]),
                              obs[1] + 0.2 * jnp.cos(obs[2])])
        jaw_pos = jnp.array([obs[0] - 0.2 * jnp.sin(obs[2]),
                             obs[1] - 0.2 * jnp.cos(obs[2])])
        thg_pos = jnp.array([jaw_pos[0] - 0.45 * jnp.sin(obs[2] - obs[3]),
                             jaw_pos[1] - 0.45 * jnp.cos(obs[2] - obs[3])])
        leg_pos = jnp.array([thg_pos[0] - 0.5 * jnp.sin(obs[2] - obs[3] - obs[4]),
                             thg_pos[1] - 0.5 * jnp.cos(obs[2] - obs[3] - obs[4])])
        foot_back_pos = jnp.array([leg_pos[0] - 0.13 * jnp.cos(obs[2] - obs[3] - obs[4] - obs[5]),
                                    leg_pos[1] + 0.13 * jnp.sin(obs[2] - obs[3] - obs[4] - obs[5])])
        foot_front_pos = jnp.array([leg_pos[0] + 0.26 * jnp.cos(obs[2] - obs[3] - obs[4] - obs[5]),
                                   leg_pos[1] - 0.26 * jnp.sin(obs[2] - obs[3] - obs[4] - obs[5])])
        return head_pos, jaw_pos, thg_pos, leg_pos, foot_front_pos, foot_back_pos

    @partial(jax.jit, static_argnums=(0,))
    def is_reach(self, head_pos, avoid_value):
        reach = jnp.sqrt((head_pos[0] - 2.0) ** 2 + (head_pos[1] - 1.4) ** 2) - 0.1
        has_reached_goal = jnp.sqrt((head_pos[0] - 2.0) ** 2 + (head_pos[1] - 1.4) ** 2) < 0.1
        value = jnp.where(has_reached_goal, -2.5, reach)
        is_avoid = (avoid_value == -1)
        value = jnp.where(is_avoid, 3.0, value)
        return value * 100.0

    @partial(jax.jit, static_argnums=(0,))
    def is_avoid(self, head_pos):
        avoid_1 = (head_pos[1] >= 1.3) & (head_pos[0] >= 0.95) & (head_pos[0] <= 1.05)
        return avoid_1
    @partial(jax.jit, static_argnums=(0,))
    def cross_product(self, array_1, array_2):
        return array_1[0] * array_2[1] - array_1[1] * array_2[0]

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size + 2,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )

class HopperAvoidCeilingWallDebug(HopperAvoidCeiling):
    def __init__(self, backend="positional"):
        super().__init__(backend=backend)

    @partial(jax.jit, static_argnums=(0,))
    def is_avoid(self, head_pos):
        avoid_1 = (head_pos[1] >= 1.3) & (head_pos[0] >= 0.95) & (head_pos[0] <= 1.05)
        avoid_2 = (head_pos[0] >= 2.35) # dont hit head on walls, bad dobby
        return avoid_1 | avoid_2

class HopperAvoidCeilingWall:
    def __init__(self, backend="positional"):
        env = HopperRandom(backend=backend,
                           exclude_current_positions_from_observation=False,
                           terminate_when_unhealthy=False)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=2)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)
        self.default_params = EnvParamsEmpty()

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)
        head_pos, _, _, _, _, _ = self.calculate_position(state.obs)
        is_avoid = self.is_avoid(head_pos)
        avoid_value = jnp.where(is_avoid, -1, 1)
        reach_value = self.is_reach(head_pos, avoid_value)
        observation = state.obs
        env_state = EnvStateR(state, reach_value)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.tanh(action)
        next_state = self._env.step(state.state, u)
        head_pos, _, _, _, _, _ = self.calculate_position(next_state.obs)
        is_avoid = self.is_avoid(head_pos)
        avoid_value = jnp.where(is_avoid, -1, state.avoid)
        reach_value = self.is_reach(head_pos, avoid_value)
        head_pos, jaw_pos, thg_pos, leg_pos, foot_front_pos, foot_back_pos = self.calculate_position(state.state.obs)
        pos_dict = {"head_pos": head_pos, "jaw_pos": jaw_pos, "thg_pos": thg_pos, "leg_pos": leg_pos,
                    "foot_front_pos": foot_front_pos, "foot_back_pos": foot_back_pos}
        observation = next_state.obs
        next_state_new = EnvStateR(next_state, reach_value)

        return observation, next_state_new, next_state.done > 0.5, pos_dict

    @partial(jax.jit, static_argnums=(0,))
    def calculate_position(self, obs):
        head_pos = jnp.array([obs[0] + 0.2 * jnp.sin(obs[2]),
                              obs[1] + 0.2 * jnp.cos(obs[2])])
        jaw_pos = jnp.array([obs[0] - 0.2 * jnp.sin(obs[2]),
                             obs[1] - 0.2 * jnp.cos(obs[2])])
        thg_pos = jnp.array([jaw_pos[0] - 0.45 * jnp.sin(obs[2] - obs[3]),
                             jaw_pos[1] - 0.45 * jnp.cos(obs[2] - obs[3])])
        leg_pos = jnp.array([thg_pos[0] - 0.5 * jnp.sin(obs[2] - obs[3] - obs[4]),
                             thg_pos[1] - 0.5 * jnp.cos(obs[2] - obs[3] - obs[4])])
        foot_back_pos = jnp.array([leg_pos[0] - 0.13 * jnp.cos(obs[2] - obs[3] - obs[4] - obs[5]),
                                    leg_pos[1] + 0.13 * jnp.sin(obs[2] - obs[3] - obs[4] - obs[5])])
        foot_front_pos = jnp.array([leg_pos[0] + 0.26 * jnp.cos(obs[2] - obs[3] - obs[4] - obs[5]),
                                   leg_pos[1] - 0.26 * jnp.sin(obs[2] - obs[3] - obs[4] - obs[5])])
        return head_pos, jaw_pos, thg_pos, leg_pos, foot_front_pos, foot_back_pos

    @partial(jax.jit, static_argnums=(0,))
    def is_reach(self, head_pos, avoid_value):
        reach = jnp.sqrt((head_pos[0] - 2.0) ** 2 + (head_pos[1] - 1.4) ** 2) - 0.1
        has_reached_goal = jnp.sqrt((head_pos[0] - 2.0) ** 2 + (head_pos[1] - 1.4) ** 2) < 0.1
        value = jnp.where(has_reached_goal, -2.5, reach)
        is_avoid = (avoid_value == -1)
        value = jnp.where(is_avoid, 3.0, value)
        return value * 100.0

    @partial(jax.jit, static_argnums=(0,))
    def is_avoid(self, head_pos):
        avoid_1 = (head_pos[1] >= 1.3) & (head_pos[0] >= 0.95) & (head_pos[0] <= 1.05)
        avoid_2 = (head_pos[0] >= 2.35) # dont hit head on walls, bad dobby
        return avoid_1 | avoid_2

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size + 2,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )
    
class HopperReachReach:
    def __init__(self, backend="positional"):
        env = HopperRandom(backend=backend,
                           exclude_current_positions_from_observation=False,
                           terminate_when_unhealthy=False)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=2)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)
        self.default_params = EnvParamsEmpty()

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)
        head_pos, _, _, _, _, _ = self.calculate_position(state.obs)

        reach1_value = self.is_reach1(head_pos)
        reach2_value = self.is_reach2(head_pos)
        
        has_reached_1 = reach1_value < 0
        has_reached_2 = reach2_value < 0

        observation = jnp.concatenate([state.obs, jnp.array([reach1_value, reach2_value])])
        env_state = EnvStateRR(state, reach1_value, reach2_value, has_reached_1, has_reached_2)
        
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.tanh(action)
        
        next_state = self._env.step(state.state, u)
        head_pos, _, _, _, _, _ = self.calculate_position(next_state.obs)
        
        reach1_value = self.is_reach1(head_pos)
        reach2_value = self.is_reach2(head_pos)

        has_reached_1 = jnp.logical_or(reach1_value < 0, state.has_reached_1)
        has_reached_2 = jnp.logical_or(reach2_value < 0, state.has_reached_2)

        head_pos, jaw_pos, thg_pos, leg_pos, foot_front_pos, foot_back_pos = self.calculate_position(state.state.obs)
        pos_dict = {"head_pos": head_pos, "jaw_pos": jaw_pos, "thg_pos": thg_pos, "leg_pos": leg_pos,
                    "foot_front_pos": foot_front_pos, "foot_back_pos": foot_back_pos}
        observation = jnp.concatenate([next_state.obs, jnp.array([reach1_value, reach2_value])])
        next_state_new = EnvStateRR(next_state, reach1_value, reach2_value, has_reached_1, has_reached_2)
        reward = 0. # used to be energy consumption? 0. works for HJR-RL I guess

        return observation, next_state_new, reward, next_state.done > 0.5, pos_dict

    @partial(jax.jit, static_argnums=(0,))
    def calculate_position(self, obs):
        head_pos = jnp.array([obs[0] + 0.2 * jnp.sin(obs[2]),
                              obs[1] + 0.2 * jnp.cos(obs[2])])
        jaw_pos = jnp.array([obs[0] - 0.2 * jnp.sin(obs[2]),
                             obs[1] - 0.2 * jnp.cos(obs[2])])
        thg_pos = jnp.array([jaw_pos[0] - 0.45 * jnp.sin(obs[2] - obs[3]),
                             jaw_pos[1] - 0.45 * jnp.cos(obs[2] - obs[3])])
        leg_pos = jnp.array([thg_pos[0] - 0.5 * jnp.sin(obs[2] - obs[3] - obs[4]),
                             thg_pos[1] - 0.5 * jnp.cos(obs[2] - obs[3] - obs[4])])
        foot_back_pos = jnp.array([leg_pos[0] - 0.13 * jnp.cos(obs[2] - obs[3] - obs[4] - obs[5]),
                                    leg_pos[1] + 0.13 * jnp.sin(obs[2] - obs[3] - obs[4] - obs[5])])
        foot_front_pos = jnp.array([leg_pos[0] + 0.26 * jnp.cos(obs[2] - obs[3] - obs[4] - obs[5]),
                                   leg_pos[1] - 0.26 * jnp.sin(obs[2] - obs[3] - obs[4] - obs[5])])
        return head_pos, jaw_pos, thg_pos, leg_pos, foot_front_pos, foot_back_pos

    @partial(jax.jit, static_argnums=(0,))
    def is_reach1(self, head_pos):
        target_center = [2., 1.4]
        reach = jnp.sqrt((head_pos[0] - target_center[0]) ** 2 + (head_pos[1] - target_center[1]) ** 2) - 0.1
        has_reached_goal = jnp.sqrt((head_pos[0] - target_center[0]) ** 2 + (head_pos[1] - target_center[1]) ** 2) < 0.1
        value = jnp.where(has_reached_goal, -2.5, reach)
        return value * 100.0
    
    @partial(jax.jit, static_argnums=(0,))
    def is_reach2(self, head_pos):
        target_center = [-2., 1.4]
        reach = jnp.sqrt((head_pos[0] - target_center[0]) ** 2 + (head_pos[1] - target_center[1]) ** 2) - 0.1
        has_reached_goal = jnp.sqrt((head_pos[0] - target_center[0]) ** 2 + (head_pos[1] - target_center[1]) ** 2) < 0.1
        value = jnp.where(has_reached_goal, -2.5, reach)
        return value * 100.0

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size + 2,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )
    
class HopperReach1:
    def __init__(self, backend="positional"):
        env = HopperRandom(backend=backend,
                           exclude_current_positions_from_observation=False,
                           terminate_when_unhealthy=False)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=2)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)
        self.default_params = EnvParamsEmpty()

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)
        head_pos, _, _, _, _, _ = self.calculate_position(state.obs)
        reach1_value = self.is_reach1(head_pos)
        reach2_value = self.is_reach2(head_pos)
        observation = jnp.concatenate([state.obs, jnp.array([reach1_value, reach2_value])])
        env_state = EnvStateR1(state, reach1_value)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.tanh(action)
        next_state = self._env.step(state.state, u)
        head_pos, _, _, _, _, _ = self.calculate_position(next_state.obs)
        reach1_value = self.is_reach1(head_pos)
        reach2_value = self.is_reach2(head_pos)
        head_pos, jaw_pos, thg_pos, leg_pos, foot_front_pos, foot_back_pos = self.calculate_position(state.state.obs)
        pos_dict = {"head_pos": head_pos, "jaw_pos": jaw_pos, "thg_pos": thg_pos, "leg_pos": leg_pos,
                    "foot_front_pos": foot_front_pos, "foot_back_pos": foot_back_pos}
        observation = jnp.concatenate([next_state.obs, jnp.array([reach1_value, reach2_value])]) 
        # STILL NEED REACH2 FOR LEARNING APPROPRIATE MAP
        next_state_new = EnvStateR1(next_state, reach1_value)
        reward = 0. # used to be energy consumption? FIXME

        return observation, next_state_new, reward, next_state.done > 0.5, pos_dict

    @partial(jax.jit, static_argnums=(0,))
    def calculate_position(self, obs):
        head_pos = jnp.array([obs[0] + 0.2 * jnp.sin(obs[2]),
                              obs[1] + 0.2 * jnp.cos(obs[2])])
        jaw_pos = jnp.array([obs[0] - 0.2 * jnp.sin(obs[2]),
                             obs[1] - 0.2 * jnp.cos(obs[2])])
        thg_pos = jnp.array([jaw_pos[0] - 0.45 * jnp.sin(obs[2] - obs[3]),
                             jaw_pos[1] - 0.45 * jnp.cos(obs[2] - obs[3])])
        leg_pos = jnp.array([thg_pos[0] - 0.5 * jnp.sin(obs[2] - obs[3] - obs[4]),
                             thg_pos[1] - 0.5 * jnp.cos(obs[2] - obs[3] - obs[4])])
        foot_back_pos = jnp.array([leg_pos[0] - 0.13 * jnp.cos(obs[2] - obs[3] - obs[4] - obs[5]),
                                    leg_pos[1] + 0.13 * jnp.sin(obs[2] - obs[3] - obs[4] - obs[5])])
        foot_front_pos = jnp.array([leg_pos[0] + 0.26 * jnp.cos(obs[2] - obs[3] - obs[4] - obs[5]),
                                   leg_pos[1] - 0.26 * jnp.sin(obs[2] - obs[3] - obs[4] - obs[5])])
        return head_pos, jaw_pos, thg_pos, leg_pos, foot_front_pos, foot_back_pos

    @partial(jax.jit, static_argnums=(0,))
    def is_reach1(self, head_pos):
        target_center = [2., 1.4]
        reach = jnp.sqrt((head_pos[0] - target_center[0]) ** 2 + (head_pos[1] - target_center[1]) ** 2) - 0.1
        has_reached_goal = jnp.sqrt((head_pos[0] - target_center[0]) ** 2 + (head_pos[1] - target_center[1]) ** 2) < 0.1
        value = jnp.where(has_reached_goal, -2.5, reach)
        return value * 100.0
    
    @partial(jax.jit, static_argnums=(0,))
    def is_reach2(self, head_pos):
        target_center = [-2., 1.4]
        reach = jnp.sqrt((head_pos[0] - target_center[0]) ** 2 + (head_pos[1] - target_center[1]) ** 2) - 0.1
        has_reached_goal = jnp.sqrt((head_pos[0] - target_center[0]) ** 2 + (head_pos[1] - target_center[1]) ** 2) < 0.1
        value = jnp.where(has_reached_goal, -2.5, reach)
        return value * 100.0

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size + 2,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )
    
class HopperReach2:
    def __init__(self, backend="positional"):
        env = HopperRandom(backend=backend,
                           exclude_current_positions_from_observation=False,
                           terminate_when_unhealthy=False)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=2)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)
        self.default_params = EnvParamsEmpty()

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)
        head_pos, _, _, _, _, _ = self.calculate_position(state.obs)
        reach1_value = self.is_reach1(head_pos)
        reach2_value = self.is_reach2(head_pos)
        observation = jnp.concatenate([state.obs, jnp.array([reach1_value, reach2_value])])
        env_state = EnvStateR2(state, reach2_value)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.tanh(action)
        next_state = self._env.step(state.state, u)
        head_pos, _, _, _, _, _ = self.calculate_position(next_state.obs)
        reach1_value = self.is_reach1(head_pos)
        reach2_value = self.is_reach2(head_pos)
        head_pos, jaw_pos, thg_pos, leg_pos, foot_front_pos, foot_back_pos = self.calculate_position(state.state.obs)
        pos_dict = {"head_pos": head_pos, "jaw_pos": jaw_pos, "thg_pos": thg_pos, "leg_pos": leg_pos,
                    "foot_front_pos": foot_front_pos, "foot_back_pos": foot_back_pos}
        observation = jnp.concatenate([next_state.obs, jnp.array([reach1_value, reach2_value])])
        next_state_new = EnvStateR2(next_state, reach2_value)
        reward = 0. # used to be energy consumption? FIXME

        return observation, next_state_new, reward, next_state.done > 0.5, pos_dict

    @partial(jax.jit, static_argnums=(0,))
    def calculate_position(self, obs):
        head_pos = jnp.array([obs[0] + 0.2 * jnp.sin(obs[2]),
                              obs[1] + 0.2 * jnp.cos(obs[2])])
        jaw_pos = jnp.array([obs[0] - 0.2 * jnp.sin(obs[2]),
                             obs[1] - 0.2 * jnp.cos(obs[2])])
        thg_pos = jnp.array([jaw_pos[0] - 0.45 * jnp.sin(obs[2] - obs[3]),
                             jaw_pos[1] - 0.45 * jnp.cos(obs[2] - obs[3])])
        leg_pos = jnp.array([thg_pos[0] - 0.5 * jnp.sin(obs[2] - obs[3] - obs[4]),
                             thg_pos[1] - 0.5 * jnp.cos(obs[2] - obs[3] - obs[4])])
        foot_back_pos = jnp.array([leg_pos[0] - 0.13 * jnp.cos(obs[2] - obs[3] - obs[4] - obs[5]),
                                    leg_pos[1] + 0.13 * jnp.sin(obs[2] - obs[3] - obs[4] - obs[5])])
        foot_front_pos = jnp.array([leg_pos[0] + 0.26 * jnp.cos(obs[2] - obs[3] - obs[4] - obs[5]),
                                   leg_pos[1] - 0.26 * jnp.sin(obs[2] - obs[3] - obs[4] - obs[5])])
        return head_pos, jaw_pos, thg_pos, leg_pos, foot_front_pos, foot_back_pos

    @partial(jax.jit, static_argnums=(0,))
    def is_reach1(self, head_pos):
        target_center = [2., 1.4]
        reach = jnp.sqrt((head_pos[0] - target_center[0]) ** 2 + (head_pos[1] - target_center[1]) ** 2) - 0.1
        has_reached_goal = jnp.sqrt((head_pos[0] - target_center[0]) ** 2 + (head_pos[1] - target_center[1]) ** 2) < 0.1
        value = jnp.where(has_reached_goal, -2.5, reach)
        return value * 100.0
    
    @partial(jax.jit, static_argnums=(0,))
    def is_reach2(self, head_pos):
        target_center = [-2., 1.4]
        reach = jnp.sqrt((head_pos[0] - target_center[0]) ** 2 + (head_pos[1] - target_center[1]) ** 2) - 0.1
        has_reached_goal = jnp.sqrt((head_pos[0] - target_center[0]) ** 2 + (head_pos[1] - target_center[1]) ** 2) < 0.1
        value = jnp.where(has_reached_goal, -2.5, reach)
        return value * 100.0

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size + 2,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )
    
class HopperReachReachDeterministic:
    def __init__(self, backend="positional"):
        env = HopperDeterministic(backend=backend,
                           exclude_current_positions_from_observation=False,
                           terminate_when_unhealthy=False)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=2)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)
        self.default_params = EnvParamsEmpty()

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)
        head_pos, _, _, _, _, _ = self.calculate_position(state.obs)

        reach1_value = self.is_reach1(head_pos)
        reach2_value = self.is_reach2(head_pos)
        
        has_reached_1 = reach1_value < 0
        has_reached_2 = reach2_value < 0

        observation = jnp.concatenate([state.obs, jnp.array([reach1_value, reach2_value])])
        env_state = EnvStateRR(state, reach1_value, reach2_value, has_reached_1, has_reached_2)
        
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.tanh(action)
        
        next_state = self._env.step(state.state, u)
        head_pos, _, _, _, _, _ = self.calculate_position(next_state.obs)
        
        reach1_value = self.is_reach1(head_pos)
        reach2_value = self.is_reach2(head_pos)

        has_reached_1 = jnp.logical_or(reach1_value < 0, state.has_reached_1)
        has_reached_2 = jnp.logical_or(reach2_value < 0, state.has_reached_2)

        head_pos, jaw_pos, thg_pos, leg_pos, foot_front_pos, foot_back_pos = self.calculate_position(state.state.obs)
        pos_dict = {"head_pos": head_pos, "jaw_pos": jaw_pos, "thg_pos": thg_pos, "leg_pos": leg_pos,
                    "foot_front_pos": foot_front_pos, "foot_back_pos": foot_back_pos}
        observation = jnp.concatenate([next_state.obs, jnp.array([reach1_value, reach2_value])])
        next_state_new = EnvStateRR(next_state, reach1_value, reach2_value, has_reached_1, has_reached_2)
        reward = 0. # used to be energy consumption? 0. works for HJR-RL I guess

        return observation, next_state_new, reward, next_state.done > 0.5, pos_dict

    @partial(jax.jit, static_argnums=(0,))
    def calculate_position(self, obs):
        head_pos = jnp.array([obs[0] + 0.2 * jnp.sin(obs[2]),
                              obs[1] + 0.2 * jnp.cos(obs[2])])
        jaw_pos = jnp.array([obs[0] - 0.2 * jnp.sin(obs[2]),
                             obs[1] - 0.2 * jnp.cos(obs[2])])
        thg_pos = jnp.array([jaw_pos[0] - 0.45 * jnp.sin(obs[2] - obs[3]),
                             jaw_pos[1] - 0.45 * jnp.cos(obs[2] - obs[3])])
        leg_pos = jnp.array([thg_pos[0] - 0.5 * jnp.sin(obs[2] - obs[3] - obs[4]),
                             thg_pos[1] - 0.5 * jnp.cos(obs[2] - obs[3] - obs[4])])
        foot_back_pos = jnp.array([leg_pos[0] - 0.13 * jnp.cos(obs[2] - obs[3] - obs[4] - obs[5]),
                                    leg_pos[1] + 0.13 * jnp.sin(obs[2] - obs[3] - obs[4] - obs[5])])
        foot_front_pos = jnp.array([leg_pos[0] + 0.26 * jnp.cos(obs[2] - obs[3] - obs[4] - obs[5]),
                                   leg_pos[1] - 0.26 * jnp.sin(obs[2] - obs[3] - obs[4] - obs[5])])
        return head_pos, jaw_pos, thg_pos, leg_pos, foot_front_pos, foot_back_pos

    @partial(jax.jit, static_argnums=(0,))
    def is_reach1(self, head_pos):
        target_center = [2., 1.4]
        reach = jnp.sqrt((head_pos[0] - target_center[0]) ** 2 + (head_pos[1] - target_center[1]) ** 2) - 0.1
        has_reached_goal = jnp.sqrt((head_pos[0] - target_center[0]) ** 2 + (head_pos[1] - target_center[1]) ** 2) < 0.1
        value = jnp.where(has_reached_goal, -2.5, reach)
        return value * 100.0
    
    @partial(jax.jit, static_argnums=(0,))
    def is_reach2(self, head_pos):
        target_center = [-2., 1.4]
        reach = jnp.sqrt((head_pos[0] - target_center[0]) ** 2 + (head_pos[1] - target_center[1]) ** 2) - 0.1
        has_reached_goal = jnp.sqrt((head_pos[0] - target_center[0]) ** 2 + (head_pos[1] - target_center[1]) ** 2) < 0.1
        value = jnp.where(has_reached_goal, -2.5, reach)
        return value * 100.0

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size + 2,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )
    
class HopperReach1Deterministic:
    def __init__(self, backend="positional"):
        env = HopperDeterministic(backend=backend,
                           exclude_current_positions_from_observation=False,
                           terminate_when_unhealthy=False)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=2)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)
        self.default_params = EnvParamsEmpty()

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)
        head_pos, _, _, _, _, _ = self.calculate_position(state.obs)
        reach1_value = self.is_reach1(head_pos)
        reach2_value = self.is_reach2(head_pos)
        observation = jnp.concatenate([state.obs, jnp.array([reach1_value, reach2_value])])
        env_state = EnvStateR1(state, reach1_value)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.tanh(action)
        next_state = self._env.step(state.state, u)
        head_pos, _, _, _, _, _ = self.calculate_position(next_state.obs)
        reach1_value = self.is_reach1(head_pos)
        reach2_value = self.is_reach2(head_pos)
        head_pos, jaw_pos, thg_pos, leg_pos, foot_front_pos, foot_back_pos = self.calculate_position(state.state.obs)
        pos_dict = {"head_pos": head_pos, "jaw_pos": jaw_pos, "thg_pos": thg_pos, "leg_pos": leg_pos,
                    "foot_front_pos": foot_front_pos, "foot_back_pos": foot_back_pos}
        observation = jnp.concatenate([next_state.obs, jnp.array([reach1_value, reach2_value])])
        next_state_new = EnvStateR1(next_state, reach1_value)
        reward = 0. # used to be energy consumption? FIXME

        return observation, next_state_new, reward, next_state.done > 0.5, pos_dict

    @partial(jax.jit, static_argnums=(0,))
    def calculate_position(self, obs):
        head_pos = jnp.array([obs[0] + 0.2 * jnp.sin(obs[2]),
                              obs[1] + 0.2 * jnp.cos(obs[2])])
        jaw_pos = jnp.array([obs[0] - 0.2 * jnp.sin(obs[2]),
                             obs[1] - 0.2 * jnp.cos(obs[2])])
        thg_pos = jnp.array([jaw_pos[0] - 0.45 * jnp.sin(obs[2] - obs[3]),
                             jaw_pos[1] - 0.45 * jnp.cos(obs[2] - obs[3])])
        leg_pos = jnp.array([thg_pos[0] - 0.5 * jnp.sin(obs[2] - obs[3] - obs[4]),
                             thg_pos[1] - 0.5 * jnp.cos(obs[2] - obs[3] - obs[4])])
        foot_back_pos = jnp.array([leg_pos[0] - 0.13 * jnp.cos(obs[2] - obs[3] - obs[4] - obs[5]),
                                    leg_pos[1] + 0.13 * jnp.sin(obs[2] - obs[3] - obs[4] - obs[5])])
        foot_front_pos = jnp.array([leg_pos[0] + 0.26 * jnp.cos(obs[2] - obs[3] - obs[4] - obs[5]),
                                   leg_pos[1] - 0.26 * jnp.sin(obs[2] - obs[3] - obs[4] - obs[5])])
        return head_pos, jaw_pos, thg_pos, leg_pos, foot_front_pos, foot_back_pos

    @partial(jax.jit, static_argnums=(0,))
    def is_reach1(self, head_pos):
        target_center = [2., 1.4]
        reach = jnp.sqrt((head_pos[0] - target_center[0]) ** 2 + (head_pos[1] - target_center[1]) ** 2) - 0.1
        has_reached_goal = jnp.sqrt((head_pos[0] - target_center[0]) ** 2 + (head_pos[1] - target_center[1]) ** 2) < 0.1
        value = jnp.where(has_reached_goal, -2.5, reach)
        return value * 100.0
    
    @partial(jax.jit, static_argnums=(0,))
    def is_reach2(self, head_pos):
        target_center = [-2., 1.4]
        reach = jnp.sqrt((head_pos[0] - target_center[0]) ** 2 + (head_pos[1] - target_center[1]) ** 2) - 0.1
        has_reached_goal = jnp.sqrt((head_pos[0] - target_center[0]) ** 2 + (head_pos[1] - target_center[1]) ** 2) < 0.1
        value = jnp.where(has_reached_goal, -2.5, reach)
        return value * 100.0

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size + 2,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )
    
class HopperReach2Deterministic:
    def __init__(self, backend="positional"):
        env = HopperDeterministic(backend=backend,
                           exclude_current_positions_from_observation=False,
                           terminate_when_unhealthy=False)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=2)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)
        self.default_params = EnvParamsEmpty()

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)
        head_pos, _, _, _, _, _ = self.calculate_position(state.obs)
        reach1_value = self.is_reach1(head_pos)
        reach2_value = self.is_reach2(head_pos)
        observation = jnp.concatenate([state.obs, jnp.array([reach1_value, reach2_value])])
        env_state = EnvStateR2(state, reach2_value)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.tanh(action)
        next_state = self._env.step(state.state, u)
        head_pos, _, _, _, _, _ = self.calculate_position(next_state.obs)
        reach1_value = self.is_reach1(head_pos)
        reach2_value = self.is_reach2(head_pos)
        head_pos, jaw_pos, thg_pos, leg_pos, foot_front_pos, foot_back_pos = self.calculate_position(state.state.obs)
        pos_dict = {"head_pos": head_pos, "jaw_pos": jaw_pos, "thg_pos": thg_pos, "leg_pos": leg_pos,
                    "foot_front_pos": foot_front_pos, "foot_back_pos": foot_back_pos}
        observation = jnp.concatenate([next_state.obs, jnp.array([reach1_value, reach2_value])])
        next_state_new = EnvStateR2(next_state, reach2_value)
        reward = 0. # used to be energy consumption? FIXME

        return observation, next_state_new, reward, next_state.done > 0.5, pos_dict

    @partial(jax.jit, static_argnums=(0,))
    def calculate_position(self, obs):
        head_pos = jnp.array([obs[0] + 0.2 * jnp.sin(obs[2]),
                              obs[1] + 0.2 * jnp.cos(obs[2])])
        jaw_pos = jnp.array([obs[0] - 0.2 * jnp.sin(obs[2]),
                             obs[1] - 0.2 * jnp.cos(obs[2])])
        thg_pos = jnp.array([jaw_pos[0] - 0.45 * jnp.sin(obs[2] - obs[3]),
                             jaw_pos[1] - 0.45 * jnp.cos(obs[2] - obs[3])])
        leg_pos = jnp.array([thg_pos[0] - 0.5 * jnp.sin(obs[2] - obs[3] - obs[4]),
                             thg_pos[1] - 0.5 * jnp.cos(obs[2] - obs[3] - obs[4])])
        foot_back_pos = jnp.array([leg_pos[0] - 0.13 * jnp.cos(obs[2] - obs[3] - obs[4] - obs[5]),
                                    leg_pos[1] + 0.13 * jnp.sin(obs[2] - obs[3] - obs[4] - obs[5])])
        foot_front_pos = jnp.array([leg_pos[0] + 0.26 * jnp.cos(obs[2] - obs[3] - obs[4] - obs[5]),
                                   leg_pos[1] - 0.26 * jnp.sin(obs[2] - obs[3] - obs[4] - obs[5])])
        return head_pos, jaw_pos, thg_pos, leg_pos, foot_front_pos, foot_back_pos

    @partial(jax.jit, static_argnums=(0,))
    def is_reach1(self, head_pos):
        target_center = [2., 1.4]
        reach = jnp.sqrt((head_pos[0] - target_center[0]) ** 2 + (head_pos[1] - target_center[1]) ** 2) - 0.1
        has_reached_goal = jnp.sqrt((head_pos[0] - target_center[0]) ** 2 + (head_pos[1] - target_center[1]) ** 2) < 0.1
        value = jnp.where(has_reached_goal, -2.5, reach)
        return value * 100.0
    
    @partial(jax.jit, static_argnums=(0,))
    def is_reach2(self, head_pos):
        target_center = [-2., 1.4]
        reach = jnp.sqrt((head_pos[0] - target_center[0]) ** 2 + (head_pos[1] - target_center[1]) ** 2) - 0.1
        has_reached_goal = jnp.sqrt((head_pos[0] - target_center[0]) ** 2 + (head_pos[1] - target_center[1]) ** 2) < 0.1
        value = jnp.where(has_reached_goal, -2.5, reach)
        return value * 100.0

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size + 2,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )


class HopperRAATemplate:
    def __init__(self, backend="positional"):
        env = HopperRandom(backend=backend,
                           exclude_current_positions_from_observation=False,
                           terminate_when_unhealthy=False)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=2)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)
        self.default_params = EnvParamsEmpty()   

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        raise NotImplementedError("reset() not implemented in base class")
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        raise NotImplementedError("step() not implemented in base class")

    @partial(jax.jit, static_argnums=(0,))
    def calculate_position(self, obs):
        head_pos = jnp.array([obs[0] + 0.2 * jnp.sin(obs[2]),
                              obs[1] + 0.2 * jnp.cos(obs[2])])
        jaw_pos = jnp.array([obs[0] - 0.2 * jnp.sin(obs[2]),
                             obs[1] - 0.2 * jnp.cos(obs[2])])
        thg_pos = jnp.array([jaw_pos[0] - 0.45 * jnp.sin(obs[2] - obs[3]),
                             jaw_pos[1] - 0.45 * jnp.cos(obs[2] - obs[3])])
        leg_pos = jnp.array([thg_pos[0] - 0.5 * jnp.sin(obs[2] - obs[3] - obs[4]),
                             thg_pos[1] - 0.5 * jnp.cos(obs[2] - obs[3] - obs[4])])
        foot_back_pos = jnp.array([leg_pos[0] - 0.13 * jnp.cos(obs[2] - obs[3] - obs[4] - obs[5]),
                                    leg_pos[1] + 0.13 * jnp.sin(obs[2] - obs[3] - obs[4] - obs[5])])
        foot_front_pos = jnp.array([leg_pos[0] + 0.26 * jnp.cos(obs[2] - obs[3] - obs[4] - obs[5]),
                                   leg_pos[1] - 0.26 * jnp.sin(obs[2] - obs[3] - obs[4] - obs[5])])
        return head_pos, jaw_pos, thg_pos, leg_pos, foot_front_pos, foot_back_pos

    @partial(jax.jit, static_argnums=(0,))
    def is_reach(self, head_pos):
        reach = jnp.sqrt((head_pos[0] - 2.0) ** 2 + (head_pos[1] - 1.4) ** 2) - 0.1
        return reach

    @partial(jax.jit, static_argnums=(0,))
    def is_avoid(self, head_pos):
        avoid_1 = (head_pos[1] >= 1.3) & (head_pos[0] >= 0.95) & (head_pos[0] <= 1.05)
        avoid_2 = (head_pos[0] >= 2.35) # dont hit head on walls, bad dobby
        return avoid_1 | avoid_2

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size + 2,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )


class HopperAvoidOnly(HopperRAATemplate):
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)
        head_pos, _, _, _, _, _ = self.calculate_position(state.obs)
        avoid_value = self.is_avoid(head_pos)
        reach_value = self.is_reach(head_pos)
        observation = jnp.concatenate([state.obs, jnp.array([avoid_value, reach_value])])
        env_state = EnvStateAvoidOnly(state, avoid_value)
        return observation, env_state
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.tanh(action)
        next_state = self._env.step(state.state, u)
        head_pos, _, _, _, _, _ = self.calculate_position(next_state.obs)
        avoid_value = self.is_avoid(head_pos)
        reach_value = self.is_reach(head_pos)
        # TODO: Do we need is_reach(head_pos) here?
        head_pos, jaw_pos, thg_pos, leg_pos, foot_front_pos, foot_back_pos = self.calculate_position(state.state.obs)
        pos_dict = {"head_pos": head_pos, "jaw_pos": jaw_pos, "thg_pos": thg_pos, "leg_pos": leg_pos,
                    "foot_front_pos": foot_front_pos, "foot_back_pos": foot_back_pos}
        observation = jnp.concatenate([next_state.obs, jnp.array([avoid_value, reach_value])])  # TODO: add reach_value?
        next_state_new = EnvStateAvoidOnly(next_state, avoid_value)
        reward = 0. # used to be energy consumption? FIXME

        return observation, next_state_new, reward, next_state.done > 0.5, pos_dict


class HopperReachAvoid(HopperRAATemplate):
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)
        head_pos, _, _, _, _, _ = self.calculate_position(state.obs)
        avoid_value = self.is_avoid(head_pos)
        reach_value = self.is_reach(head_pos)

        has_reached = reach_value < 0
        observation = jnp.concatenate([state.obs, jnp.array([avoid_value, reach_value])])
        env_state = EnvStateRAA(state, avoid_value, reach_value, has_reached)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.tanh(action)
        next_state = self._env.step(state.state, u)
        head_pos, _, _, _, _, _ = self.calculate_position(next_state.obs)
        avoid_value = self.is_avoid(head_pos)
        reach_value = self.is_reach(head_pos)

        has_reached = jnp.logical_or(reach_value < 0, state.has_reached)
        head_pos, jaw_pos, thg_pos, leg_pos, foot_front_pos, foot_back_pos = self.calculate_position(state.state.obs)
        pos_dict = {"head_pos": head_pos, "jaw_pos": jaw_pos, "thg_pos": thg_pos, "leg_pos": leg_pos,
                    "foot_front_pos": foot_front_pos, "foot_back_pos": foot_back_pos}
        observation = jnp.concatenate([next_state.obs, jnp.array([avoid_value, reach_value])])
        next_state_new = EnvStateRAA(next_state, avoid_value, reach_value, has_reached)
        reward = 0.

        return observation, next_state_new, reward, next_state.done > 0.5, pos_dict
