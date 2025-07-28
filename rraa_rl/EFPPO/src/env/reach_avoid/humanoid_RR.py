import jax
import jax.numpy as jnp
from functools import partial
from gymnax.environments import spaces
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper
from flax import struct
from brax.envs.base import State
from .humanoid_base import HumanoidDeterministic, HumanoidRandom
from copy import deepcopy 
from jax import lax

from jax.numpy import sin, cos
from jax.scipy.spatial.transform import Rotation as R

HUMANOID_TARGET_RIGHT, HUMANOID_TARGET_LEFT = [0., -2., 1.], [2., 0., 1.]
HUMANOID_TARGET_RADIUS = 0.1 
HUMANOID_TORSO_MIN_Z = 1.
HUMANOID_TORSO_MAX_Z = 2.

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

class HumanoidReachReachTemplate:
    def __init__(self, backend="positional"):
        env = HumanoidRandom(backend=backend, terminate_when_unhealthy=False,
                           exclude_current_positions_from_observation=False)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=2) # FIXME: do we want 2 action repeats (like hopper)?
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
    def calculate_position(self, state: State) -> dict:
        """
        Calculates the positions of the humanoid's body parts based on the (pipeline) state.
        """
        link_pos = {self._env.env.env.sys.link_names[i]: state.pipeline_state.x.pos[i] for i in range(len(self._env.env.env.sys.link_names))}
        link_rot = {self._env.env.env.sys.link_names[i]: state.pipeline_state.x.rot[i] for i in range(len(self._env.env.env.sys.link_names))}

        def extend_along_quat(pos, quat, offset_local):
            rot = R.from_quat(quat[jnp.array([1, 2, 3, 0])])  # (x, y, z, w)
            offset_world = rot.apply(offset_local)
            return pos + offset_world
        
        l_hand_offset = jnp.array([0.18, -0.18, 0.18])
        r_hand_offset = jnp.array([0.18, 0.18, 0.18])
        foot_offset = jnp.array([0.0, 0.0, -0.425])
        head_offset = jnp.array([0.0, 0.0, +0.25])
        
        l_hand_pos = extend_along_quat(link_pos["left_lower_arm"], link_rot["left_lower_arm"], l_hand_offset)
        r_hand_pos = extend_along_quat(link_pos["right_lower_arm"], link_rot["right_lower_arm"], r_hand_offset)
        l_foot_pos = extend_along_quat(link_pos["left_shin"], link_rot["left_shin"], foot_offset)
        r_foot_pos = extend_along_quat(link_pos["right_shin"], link_rot["right_shin"], foot_offset)
        head_pos = extend_along_quat(link_pos["torso"], link_rot["torso"], head_offset)

        link_pos['left_foot'] = l_foot_pos
        link_pos['right_foot'] = r_foot_pos
        link_pos['left_hand'] = l_hand_pos
        link_pos['right_hand'] = r_hand_pos
        link_pos['head_pos'] = head_pos

        return link_pos

    @partial(jax.jit, static_argnums=(0,))
    def is_reach1(self, poses):
        target_center, radius = HUMANOID_TARGET_RIGHT, HUMANOID_TARGET_RADIUS
        target_pos = poses["right_hand"]
        reach = jnp.sqrt((target_pos[..., 0] - target_center[0]) ** 2 + \
                         (target_pos[..., 1] - target_center[1]) ** 2 + \
                         (target_pos[..., 2] - target_center[2]) ** 2) - radius
        value = jnp.where(reach < 0., -2.5, reach)
        value = jnp.where(poses['torso'][..., 2] < HUMANOID_TORSO_MIN_Z, 10., value) # unhealthy (interal catch doesnt work with gae)
        value = jnp.where(poses['torso'][..., 2] > HUMANOID_TORSO_MAX_Z, 10., value) # unhealthy (interal catch doesnt work with gae)
        return value * 100.0
    
    @partial(jax.jit, static_argnums=(0,))
    def is_reach2(self, poses):
        target_center, radius = HUMANOID_TARGET_LEFT, HUMANOID_TARGET_RADIUS
        target_pos = poses["left_hand"]
        reach = jnp.sqrt((target_pos[..., 0] - target_center[0]) ** 2 + \
                         (target_pos[..., 1] - target_center[1]) ** 2 + \
                         (target_pos[..., 2] - target_center[2]) ** 2) - radius
        value = jnp.where(reach < 0., -2.5, reach)
        value = jnp.where(poses['torso'][..., 2] < HUMANOID_TORSO_MIN_Z, 10., value) # unhealthy (interal catch doesnt work with gae)
        value = jnp.where(poses['torso'][..., 2] > HUMANOID_TORSO_MAX_Z, 10., value) # unhealthy (interal catch doesnt work with gae)
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

class HumanoidReachReach(HumanoidReachReachTemplate):
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)
        poses = self.calculate_position(state)
        reach1_value = self.is_reach1(poses)
        reach2_value = self.is_reach2(poses)
        has_reached_1 = reach1_value < 0
        has_reached_2 = reach2_value < 0
        observation = jnp.concatenate([state.obs, jnp.array([reach1_value, reach2_value])])
        env_state = EnvStateRR(state, reach1_value, reach2_value, has_reached_1, has_reached_2)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.tanh(action)
        next_state_raw = self._env.step(state.state, u)

        ## Freeze if (current) state unhealthy
        unhealthy = jnp.logical_or(
            state.state.pipeline_state.x.pos[0, 2] < HUMANOID_TORSO_MIN_Z,
            state.state.pipeline_state.x.pos[0, 2] > HUMANOID_TORSO_MAX_Z
        )
        next_state = lax.cond(unhealthy, lambda _: state.state, lambda _: next_state_raw, operand=None,)        

        poses = self.calculate_position(next_state)
        reach1_value = self.is_reach1(poses)
        reach2_value = self.is_reach2(poses)
        has_reached_1 = jnp.logical_or(reach1_value < 0, state.has_reached_1)
        has_reached_2 = jnp.logical_or(reach2_value < 0, state.has_reached_2)
        
        observation = jnp.concatenate([next_state.obs, jnp.array([reach1_value, reach2_value])])
        next_state_new = EnvStateRR(next_state, reach1_value, reach2_value, has_reached_1, has_reached_2)
        reward = 0.
        # done = next_state.done > 0.5
        done = jnp.logical_or(next_state.done > 0.5, jnp.logical_and(has_reached_1, has_reached_2))
        # FIXME, does the logical_or (bug) exist if HalfCheetah classes??

        return observation, next_state_new, reward, done, poses
    
class HumanoidReach1(HumanoidReachReachTemplate):
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)
        poses = self.calculate_position(state)
        reach1_value = self.is_reach1(poses)
        reach2_value = self.is_reach2(poses)
        observation = jnp.concatenate([state.obs, jnp.array([reach1_value, reach2_value])])
        env_state = EnvStateR2(state, reach1_value)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.tanh(action)
        next_state_raw = self._env.step(state.state, u)

        ## Freeze if (current) state unhealthy
        unhealthy = jnp.logical_or(
            state.state.pipeline_state.x.pos[0, 2] < HUMANOID_TORSO_MIN_Z,
            state.state.pipeline_state.x.pos[0, 2] > HUMANOID_TORSO_MAX_Z
        )
        next_state = lax.cond(unhealthy, lambda _: state.state, lambda _: next_state_raw, operand=None,)  
        
        poses = self.calculate_position(next_state)
        reach1_value = self.is_reach1(poses)
        reach2_value = self.is_reach2(poses)
        
        observation = jnp.concatenate([next_state.obs, jnp.array([reach1_value, reach2_value])])
        next_state_new = EnvStateR1(next_state, reach1_value)
        reward = 0.
        done = next_state.done > 0.5
        # done = has_reached_1
        # FIXME, does the logical_or (bug) exist if HalfCheetah classes??

        return observation, next_state_new, reward, done, poses
    
    @partial(jax.jit, static_argnums=(0,))
    def reset_toinput(self, key, reset_obs, params=None):
        # reset_obs = deepcopy(reset_obs[:53])
        
        ## Remake Pipeline State
        qpos = reset_obs[:24]
        qvel = reset_obs[24:47]
        pipeline_state = self._env.pipeline_init(qpos, qvel)
        obs = self._env._get_obs(pipeline_state, jnp.zeros(self.action_size))
        # FIXME: humanoid obs need an action, meaning we would need to pass reset action too
        reward, done, zero = jnp.zeros(3)

        ## Define Metrics
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
        state = State(pipeline_state, obs, reward, done, metrics)
        
        ## Set Auxiliaries
        rng = key 
        for key_name in ['steps', 'truncation', 'episode_done']:
            state.info[key_name] = jnp.zeros(rng.shape[:-1])
        episode_metrics = dict()
        episode_metrics['sum_reward'] = jnp.zeros(rng.shape[:-1])
        episode_metrics['length'] = jnp.zeros(rng.shape[:-1])
        for metric_name in state.metrics.keys():
            episode_metrics[metric_name] = jnp.zeros(rng.shape[:-1])
        state.info['episode_metrics'] = episode_metrics
        state.info['first_pipeline_state'] = state.pipeline_state
        state.info['first_obs'] = state.obs

        ## Set Observation and EnvState
        poses = self.calculate_position(state)
        reach1_value = self.is_reach1(poses)
        reach2_value = self.is_reach2(poses)
        observation = jnp.concatenate([state.obs, jnp.array([reach1_value, reach2_value])])
        env_state = EnvStateR1(state, reach1_value)

        # FIXME: does the observation not need to be transformed?
        # observation = self._env.transform_obs(observation)?

        return observation, env_state
    
class HumanoidReach2(HumanoidReachReachTemplate):
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)
        poses = self.calculate_position(state)
        reach1_value = self.is_reach1(poses)
        reach2_value = self.is_reach2(poses)
        observation = jnp.concatenate([state.obs, jnp.array([reach1_value, reach2_value])])
        env_state = EnvStateR2(state, reach2_value)
        return observation, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.tanh(action)
        next_state_raw = self._env.step(state.state, u)

        ## Freeze if (current) state unhealthy
        unhealthy = jnp.logical_or(
            state.state.pipeline_state.x.pos[0, 2] < HUMANOID_TORSO_MIN_Z,
            state.state.pipeline_state.x.pos[0, 2] > HUMANOID_TORSO_MAX_Z
        )
        next_state = lax.cond(unhealthy, lambda _: state.state, lambda _: next_state_raw, operand=None,)  

        poses = self.calculate_position(next_state)
        reach1_value = self.is_reach1(poses)
        reach2_value = self.is_reach2(poses)
        
        observation = jnp.concatenate([next_state.obs, jnp.array([reach1_value, reach2_value])])
        next_state_new = EnvStateR2(next_state, reach2_value)
        reward = 0.
        done = next_state.done > 0.5
        # done = has_reached_1

        return observation, next_state_new, reward, done, poses
    
    @partial(jax.jit, static_argnums=(0,))
    def reset_toinput(self, key, reset_obs, params=None):
        # reset_obs = deepcopy(reset_obs[:53])
        
        ## Remake Pipeline State
        qpos = reset_obs[:24]
        qvel = reset_obs[24:47]
        pipeline_state = self._env.pipeline_init(qpos, qvel)
        obs = self._env._get_obs(pipeline_state, jnp.zeros(self.action_size))
        # FIXME: humanoid obs need an action, meaning we would need to pass reset action too
        reward, done, zero = jnp.zeros(3)

        ## Define Metrics
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
        state = State(pipeline_state, obs, reward, done, metrics)
        
        ## Set Auxiliaries
        rng = key 
        for key_name in ['steps', 'truncation', 'episode_done']:
            state.info[key_name] = jnp.zeros(rng.shape[:-1])
        episode_metrics = dict()
        episode_metrics['sum_reward'] = jnp.zeros(rng.shape[:-1])
        episode_metrics['length'] = jnp.zeros(rng.shape[:-1])
        for metric_name in state.metrics.keys():
            episode_metrics[metric_name] = jnp.zeros(rng.shape[:-1])
        state.info['episode_metrics'] = episode_metrics
        state.info['first_pipeline_state'] = state.pipeline_state
        state.info['first_obs'] = state.obs

        ## Set Observation and EnvState
        poses = self.calculate_position(state)
        reach1_value = self.is_reach1(poses)
        reach2_value = self.is_reach2(poses)
        observation = jnp.concatenate([state.obs, jnp.array([reach1_value, reach2_value])])
        env_state = EnvStateR2(state, reach2_value)

        # FIXME: does the observation not need to be transformed?
        # observation = self._env.transform_obs(observation)?

        return observation, env_state