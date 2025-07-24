import jax
import jax.numpy as jnp
from functools import partial
from gymnax.environments import spaces
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper
from flax import struct
from brax.envs.base import State
from .half_cheetah_random import HalfCheetahRandom
from .half_cheetah_deterministic import HalfCheetahDeterministic
from copy import deepcopy 

@struct.dataclass
class EnvStateR1:
    state: jax.Array = struct.field(default_factory=jax.Array)
    reach1: float = 0.

@struct.dataclass
class EnvStateR2:
    state: jax.Array = struct.field(default_factory=jax.Array)
    reach2: float = 0.

@struct.dataclass
class EnvStateRR:
    state: jax.Array = struct.field(default_factory=jax.Array)
    reach1: float = 0.
    reach2: float = 0.
    has_reached_1: float = 0.
    has_reached_2: float = 0.

@struct.dataclass
class EnvParamsEmpty:
    pass

class HalfCheetahReachReachTemplate:
    def __init__(self, backend="positional"):
        env = HalfCheetahRandom(backend=backend,
                           exclude_current_positions_from_observation=False)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
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

        back_pos = jnp.array([obs[0] - 0.5 * jnp.cos(obs[2]),
                              obs[1] + 0.5 * jnp.sin(obs[2])])
        neck_pos = jnp.array([obs[0] + 0.5 * jnp.cos(obs[2]),
                              obs[1] - 0.5 * jnp.sin(obs[2])])
        head_pos = jnp.array([neck_pos[0] + 0.1 * jnp.cos(jnp.pi / 4 - obs[2]) +
                              0.15 * jnp.cos(jnp.pi / 2 - 0.87 - obs[2]),
                              neck_pos[1] + 0.1 * jnp.sin(jnp.pi / 4 - obs[2]) +
                              0.15 * jnp.sin(jnp.pi / 2 - 0.87 - obs[2])])
        front_thigh_pos = jnp.array([neck_pos[0] + 0.266 * jnp.cos(0.53 + jnp.pi / 2 + obs[2] + obs[6]),
                                     neck_pos[1] - 0.266 * jnp.sin(0.53 + jnp.pi / 2 + obs[2] + obs[6])])
        front_shin_pos = jnp.array([front_thigh_pos[0] + 0.212 * jnp.cos(-0.6 + jnp.pi / 2 + obs[2] + obs[6] + obs[7]),
                                    front_thigh_pos[1] - 0.212 * jnp.sin(-0.6 + jnp.pi / 2 + obs[2] + obs[6] + obs[7])])
        front_foot_pos = jnp.array([front_shin_pos[0] + 0.14 * jnp.cos(-0.6 + jnp.pi / 2 + obs[2] + obs[6] + obs[7] + obs[8]),
                                    front_shin_pos[1] - 0.14 * jnp.sin(-0.6 + jnp.pi / 2 + obs[2] + obs[6] + obs[7] + obs[8])])
        back_thigh_pos = jnp.array([back_pos[0] + 0.29 * jnp.cos(jnp.pi * 3 / 2 - 3.8 + obs[2] + obs[3]),
                                     back_pos[1] - 0.29 * jnp.sin(jnp.pi * 3 / 2 - 3.8 + obs[2] + obs[3])])
        back_shin_pos = jnp.array([back_thigh_pos[0] + 0.3 * jnp.cos(jnp.pi * 3 / 2 - 2.03 + obs[2] + obs[3] + obs[4]),
                                    back_thigh_pos[1] - 0.3 * jnp.sin(jnp.pi * 3 / 2 - 2.03 + obs[2] + obs[3] + obs[4])])
        back_foot_pos = jnp.array([back_shin_pos[0] + 0.188 * jnp.cos(jnp.pi / 2 - 0.27 + obs[2] + obs[3] + obs[4] + obs[5]),
                                    back_shin_pos[1] - 0.188 * jnp.sin(jnp.pi / 2 - 0.27 + obs[2] + obs[3] + obs[4] + obs[5])])

        vels = jnp.array([obs[9], obs[10], obs[11]])  # x_vel, z_vel, y_ang_vel
        
        return (head_pos, neck_pos, back_pos, front_thigh_pos, front_shin_pos, front_foot_pos,
                back_thigh_pos, back_shin_pos, back_foot_pos), vels

    # @partial(jax.jit, static_argnums=(0,))
    # def is_reach1(self, poses, vels):
    #     (head_pos, neck_pos, back_pos, front_thigh_pos, front_shin_pos,
    #       front_foot_pos, back_thigh_pos, back_shin_pos, back_foot_pos) = poses
    #     (x_vel, z_vel, y_ang_vel) = vels
        
    #     target_x_vel = 10.
    #     reach1_value = -(x_vel - target_x_vel) # negative when speed achieved
    #     return reach1_value * 10
    
    # @partial(jax.jit, static_argnums=(0,))
    # def is_reach2(self, poses, vels):
    #     (head_pos, neck_pos, back_pos, front_thigh_pos, front_shin_pos,
    #       front_foot_pos, back_thigh_pos, back_shin_pos, back_foot_pos) = poses
    #     (x_vel, z_vel, y_ang_vel) = vels
        
    #     target_ang_vel = 15.
    #     reach2_value = -(jnp.fabs(y_ang_vel) - target_ang_vel) # negative when speed achieved
    #     return reach2_value * 10

    @partial(jax.jit, static_argnums=(0,))
    def is_reach1(self, poses, vels):
        (head_pos, neck_pos, back_pos, front_thigh_pos, front_shin_pos,
          front_foot_pos, back_thigh_pos, back_shin_pos, back_foot_pos) = poses
        (x_vel, z_vel, y_ang_vel) = vels
        
        target_center, radius = [5., 1.], 0.1
        target_pos = front_foot_pos
        reach = jnp.sqrt((target_pos[..., 0] - target_center[0]) ** 2 + (target_pos[..., 1] - target_center[1]) ** 2) - radius
        has_reached_goal = jnp.sqrt((target_pos[..., 0] - target_center[0]) ** 2 + (target_pos[..., 1] - target_center[1]) ** 2) < radius
        value = jnp.where(has_reached_goal, -2.5, reach)
        return value * 100.0
    
    @partial(jax.jit, static_argnums=(0,))
    def is_reach2(self, poses, vels):
        (head_pos, neck_pos, back_pos, front_thigh_pos, front_shin_pos,
          front_foot_pos, back_thigh_pos, back_shin_pos, back_foot_pos) = poses
        (x_vel, z_vel, y_ang_vel) = vels
        
        target_center, radius = [-5., 1.], 0.1
        target_pos = back_foot_pos
        reach = jnp.sqrt((target_pos[..., 0] - target_center[0]) ** 2 + (target_pos[..., 1] - target_center[1]) ** 2) - radius
        has_reached_goal = jnp.sqrt((target_pos[..., 0] - target_center[0]) ** 2 + (target_pos[..., 1] - target_center[1]) ** 2) < radius
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

class HalfCheetahReachReach(HalfCheetahReachReachTemplate):
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)
        poses, vels = self.calculate_position(state.obs)
        reach1_value = self.is_reach1(poses, vels)
        reach2_value = self.is_reach2(poses, vels)
        has_reached_1 = reach1_value < 0
        has_reached_2 = reach2_value < 0
        observation = jnp.concatenate([state.obs, jnp.array([reach1_value, reach2_value])])
        env_state = EnvStateRR(state, reach1_value, reach2_value, has_reached_1, has_reached_2)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.tanh(action)
        next_state = self._env.step(state.state, u)
        poses, vels = self.calculate_position(next_state.obs)
        reach1_value = self.is_reach1(poses, vels)
        reach2_value = self.is_reach2(poses, vels)
        has_reached_1 = jnp.logical_or(reach1_value < 0, state.has_reached_1)
        has_reached_2 = jnp.logical_or(reach2_value < 0, state.has_reached_2)

        (head_pos, neck_pos, back_pos, front_thigh_pos, front_shin_pos,
         front_foot_pos, back_thigh_pos, back_shin_pos, back_foot_pos), vels = self.calculate_position(state.state.obs)
        pos_dict = {"head_pos": head_pos, "neck_pos": neck_pos, "back_pos": back_pos,
                    "front_thigh_pos": front_thigh_pos, "front_shin_pos": front_shin_pos, "front_foot_pos": front_foot_pos,
                    "back_thigh_pos": back_thigh_pos, "back_shin_pos": back_shin_pos, "back_foot_pos": back_foot_pos,
                    "x_vel": vels[0], "z_vel": vels[1], "y_ang_vel": vels[2]}
        
        observation = jnp.concatenate([next_state.obs, jnp.array([reach1_value, reach2_value])])
        next_state_new = EnvStateRR(next_state, reach1_value, reach2_value, has_reached_1, has_reached_2)
        reward = 0.
        # done = next_state.done > 0.5
        done = jnp.logical_or(has_reached_1, has_reached_2)

        return observation, next_state_new, reward, done, pos_dict
    
class HalfCheetahReach1(HalfCheetahReachReachTemplate):
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)
        poses, vels = self.calculate_position(state.obs)
        reach1_value = self.is_reach1(poses, vels)
        reach2_value = self.is_reach2(poses, vels)
        observation = jnp.concatenate([state.obs, jnp.array([reach1_value, reach2_value])])
        env_state = EnvStateR2(state, reach1_value)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.tanh(action)
        next_state = self._env.step(state.state, u)
        poses, vels = self.calculate_position(next_state.obs)
        reach1_value = self.is_reach1(poses, vels)
        reach2_value = self.is_reach2(poses, vels)

        (head_pos, neck_pos, back_pos, front_thigh_pos, front_shin_pos,
         front_foot_pos, back_thigh_pos, back_shin_pos, back_foot_pos), vels = self.calculate_position(state.state.obs)
        pos_dict = {"head_pos": head_pos, "neck_pos": neck_pos, "back_pos": back_pos,
                    "front_thigh_pos": front_thigh_pos, "front_shin_pos": front_shin_pos, "front_foot_pos": front_foot_pos,
                    "back_thigh_pos": back_thigh_pos, "back_shin_pos": back_shin_pos, "back_foot_pos": back_foot_pos,
                    "x_vel": vels[0], "z_vel": vels[1], "y_ang_vel": vels[2]}
        
        observation = jnp.concatenate([next_state.obs, jnp.array([reach1_value, reach2_value])])
        next_state_new = EnvStateR1(next_state, reach1_value)
        reward = 0.
        done = next_state.done > 0.5
        # done = has_reached_1

        return observation, next_state_new, reward, done, pos_dict
    
    @partial(jax.jit, static_argnums=(0,))
    def reset_toinput(self, key, reset_obs, params=None):
        # Derived from Reset function in: 
        # 1. brax.envs.hopper 
        # 2. brax.envs.wrappers.training (EpisodeWrapper)
        # 3. brax.envs.wrappers.auto_reset (AutoResetWrapper)
        reset_obs = deepcopy(reset_obs[:18])

        qpos = reset_obs[:9]
        qvel = reset_obs[9:18]
        pipeline_state = self._env.pipeline_init(qpos, qvel)
        obs = self._env._get_obs(pipeline_state)
        reward, done, zero = jnp.zeros(3)

        metrics = {
            'x_position': zero,
            'x_velocity': zero,
            'reward_ctrl': zero,
            'reward_run': zero,
        }
        state = State(pipeline_state, obs, reward, done, metrics)
        # Episode Metrics 
        rng = key 
        state.info['steps'] = jnp.zeros(rng.shape[:-1])
        state.info['truncation'] = jnp.zeros(rng.shape[:-1])
        # Keep separate record of episode done as state.info['done'] can be erased
        # by AutoResetWrapper
        state.info['episode_done'] = jnp.zeros(rng.shape[:-1])
        episode_metrics = dict()
        episode_metrics['sum_reward'] = jnp.zeros(rng.shape[:-1])
        episode_metrics['length'] = jnp.zeros(rng.shape[:-1])
        for metric_name in state.metrics.keys():
            episode_metrics[metric_name] = jnp.zeros(rng.shape[:-1])
        state.info['episode_metrics'] = episode_metrics
        state.info['first_pipeline_state'] = state.pipeline_state
        state.info['first_obs'] = state.obs

        poses, vels = self.calculate_position(state.obs)
        reach1_value = self.is_reach1(poses, vels)
        reach2_value = self.is_reach2(poses, vels)
        observation = jnp.concatenate([state.obs, jnp.array([reach1_value, reach2_value])])
        env_state = EnvStateR1(state, reach1_value)

        # FIXME: does the observation need to be transformed?
        # observation = self._env.transform_obs(observation)?

        return observation, env_state
    
class HalfCheetahReach2(HalfCheetahReachReachTemplate):
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)
        poses, vels = self.calculate_position(state.obs)
        reach1_value = self.is_reach1(poses, vels)
        reach2_value = self.is_reach2(poses, vels)
        observation = jnp.concatenate([state.obs, jnp.array([reach1_value, reach2_value])])
        env_state = EnvStateR2(state, reach2_value)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.tanh(action)
        next_state = self._env.step(state.state, u)
        poses, vels = self.calculate_position(next_state.obs)
        reach1_value = self.is_reach1(poses, vels)
        reach2_value = self.is_reach2(poses, vels)

        (head_pos, neck_pos, back_pos, front_thigh_pos, front_shin_pos,
         front_foot_pos, back_thigh_pos, back_shin_pos, back_foot_pos), vels = self.calculate_position(state.state.obs)
        pos_dict = {"head_pos": head_pos, "neck_pos": neck_pos, "back_pos": back_pos,
                    "front_thigh_pos": front_thigh_pos, "front_shin_pos": front_shin_pos, "front_foot_pos": front_foot_pos,
                    "back_thigh_pos": back_thigh_pos, "back_shin_pos": back_shin_pos, "back_foot_pos": back_foot_pos,
                    "x_vel": vels[0], "z_vel": vels[1], "y_ang_vel": vels[2]}
        
        observation = jnp.concatenate([next_state.obs, jnp.array([reach1_value, reach2_value])])
        next_state_new = EnvStateR2(next_state, reach2_value)
        reward = 0.
        done = next_state.done > 0.5
        # done = has_reached_1

        return observation, next_state_new, reward, done, pos_dict
    
    @partial(jax.jit, static_argnums=(0,))
    def reset_toinput(self, key, reset_obs, params=None):
        # Derived from Reset function in: 
        # 1. brax.envs.hopper 
        # 2. brax.envs.wrappers.training (EpisodeWrapper)
        # 3. brax.envs.wrappers.auto_reset (AutoResetWrapper)
        reset_obs = deepcopy(reset_obs[:18])

        qpos = reset_obs[:9]
        qvel = reset_obs[9:18]
        pipeline_state = self._env.pipeline_init(qpos, qvel)
        obs = self._env._get_obs(pipeline_state)
        reward, done, zero = jnp.zeros(3)

        metrics = {
            'x_position': zero,
            'x_velocity': zero,
            'reward_ctrl': zero,
            'reward_run': zero,
        }
        state = State(pipeline_state, obs, reward, done, metrics)
        # Episode Metrics 
        rng = key 
        state.info['steps'] = jnp.zeros(rng.shape[:-1])
        state.info['truncation'] = jnp.zeros(rng.shape[:-1])
        # Keep separate record of episode done as state.info['done'] can be erased
        # by AutoResetWrapper
        state.info['episode_done'] = jnp.zeros(rng.shape[:-1])
        episode_metrics = dict()
        episode_metrics['sum_reward'] = jnp.zeros(rng.shape[:-1])
        episode_metrics['length'] = jnp.zeros(rng.shape[:-1])
        for metric_name in state.metrics.keys():
            episode_metrics[metric_name] = jnp.zeros(rng.shape[:-1])
        state.info['episode_metrics'] = episode_metrics
        state.info['first_pipeline_state'] = state.pipeline_state
        state.info['first_obs'] = state.obs

        poses, vels = self.calculate_position(state.obs)
        reach1_value = self.is_reach1(poses, vels)
        reach2_value = self.is_reach2(poses, vels)
        observation = jnp.concatenate([state.obs, jnp.array([reach1_value, reach2_value])])
        env_state = EnvStateR2(state, reach2_value)

        # FIXME: does the observation need to be transformed?
        # observation = self._env.transform_obs(observation)?

        return observation, env_state