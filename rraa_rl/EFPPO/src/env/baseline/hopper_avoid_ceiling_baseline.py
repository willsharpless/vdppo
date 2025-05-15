import jax
import jax.numpy as jnp
from functools import partial
from gymnax.environments import spaces
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper
from flax import struct
from brax.envs.base import State

from .hopper_random import HopperRandom

@struct.dataclass
class EnvState:
    state: State
    reach: float
    avoid: int
    cost: float

@struct.dataclass
class EnvStateRAA:
    state: State
    reach: float
    avoid: int
    min_reach: float # min reach value over trajectory - for state augmentation
    cost: float
    

@struct.dataclass
class EnvStateRR:
    state: State
    reach1: float
    reach2: float
    has_reached_1: float
    has_reached_2: float
    min_reach1: float # min reach value over trajectory - for state augmentation
    min_reach2: float # min reach value over trajectory - for state augmentation
    cost: float

@struct.dataclass
class EnvParams:
    gamma: float = 0.99
    torque_limit: float = 0.2
    max_torque: float = 1.0


class HopperAvoidCeilingBaseline:
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
        head_pos, _, _, _, _, _ = self.calculate_position(state.obs)
        is_avoid = self.is_avoid(head_pos)
        avoid_value = jnp.where(is_avoid, -1, 1)
        reach_value = self.is_reach(head_pos)
        observation = jnp.concatenate([state.obs, jnp.array([avoid_value])])
        env_state = EnvState(state, reach_value, avoid_value, 0.)
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
        reach_value = self.is_reach(head_pos)
        reward = params.gamma * reach_value - state.reach
        head_pos, jaw_pos, thg_pos, leg_pos, foot_front_pos, foot_back_pos = self.calculate_position(state.state.obs)
        pos_dict = {"head_pos": head_pos, "jaw_pos": jaw_pos, "thg_pos": thg_pos, "leg_pos": leg_pos,
                    "foot_front_pos": foot_front_pos, "foot_back_pos": foot_back_pos}
        observation = jnp.concatenate([next_state.obs, jnp.array([avoid_value])])
        next_state_new = EnvState(next_state, reach_value, avoid_value, energy_consumption)

        return observation, next_state_new, reward, (state.avoid == -1) | (state.reach < 0), pos_dict

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
        has_reached_goal = jnp.sqrt((head_pos[0] - 2.0) ** 2 + (head_pos[1] - 1.4) ** 2) < 0.1
        value = jnp.where(has_reached_goal, -2.5, reach)
        return value * 100.0

    @partial(jax.jit, static_argnums=(0,))
    def is_avoid(self, head_pos):
        avoid_1 = (head_pos[1] >= 1.3) & (head_pos[0] >= 0.95) & (head_pos[0] <= 1.05)
        return avoid_1

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size + 1),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )



    
###################### Baselines for CPPO - Reach Reach: RAA ######################

class HopperReachAlwaysAvoidBaseline_augmented: 
    """
    Hopper Avoid Ceiling Baseline environment for CPPO baseline
    Unaugmented state base

    reward format: gamma * (max (r1, r2)) - max(last r1, last r2)
    reward format: gamma * (r1 + r2) - (last r1 + last r2)
    """

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
    def compute_observation(self, state, last_state=None): 
        # Compute observation for constrained MDP
        return jnp.concatenate([state.state.obs, jnp.array([state.min_reach])])

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)
        head_pos, _, _, _, _, _ = self.calculate_position(state.obs)
        avoid_value = self.is_avoid(head_pos)
        reach_value = self.is_reach(head_pos)
        cost = avoid_value 

        min_reach = reach_value
        
        env_state = EnvStateRAA(state, reach_value, avoid_value, min_reach, cost)
        observation = self.compute_observation(state=env_state, last_state=None)

        return observation, env_state
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.tanh(action)
        next_state = self._env.step(state.state, u)
        head_pos, _, _, _, _, _ = self.calculate_position(next_state.obs)

        avoid_value = self.is_avoid(head_pos)
        reach_value = self.is_reach(head_pos)
        min_reach = jnp.minimum(state.min_reach, reach_value) 
        
        reward = params.gamma * reach_value - state.reach
        cost = avoid_value 

        head_pos, jaw_pos, thg_pos, leg_pos, foot_front_pos, foot_back_pos = self.calculate_position(state.state.obs)
        pos_dict = {"head_pos": head_pos, "jaw_pos": jaw_pos, "thg_pos": thg_pos, "leg_pos": leg_pos,
                    "foot_front_pos": foot_front_pos, "foot_back_pos": foot_back_pos}

        next_state_new = EnvStateRAA(next_state, reach_value, avoid_value, min_reach, cost)

        observation = self.compute_observation(state=next_state_new, last_state=state)

        done = False # NOTE: Force dones to false for always avoid - make last done true outside #(state.avoid > 0) | (state.reach < 0)
        return observation, next_state_new, reward, done, pos_dict

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
        # return reach # NOTE: OLD VERSION
        
        # try similar logic as the original 
        has_reached_goal = reach < 0 
        value = jnp.where(has_reached_goal, -2.5, reach)

        value = value * 100.0 
        return value

    @partial(jax.jit, static_argnums=(0,))
    def is_avoid(self, head_pos):
        def signed_dist_box(head_pos):
            x, y = head_pos

            inside_x = (x >= 0.95) & (x <= 1.05)
            inside_y = y >= 1.3
            is_inside = inside_x & inside_y

            # Inside: min distance to any boundary
            dist_left   = x - 0.95
            dist_right  = 1.05 - x
            dist_bottom = y - 1.3
            min_dist_inside = jnp.minimum(jnp.minimum(dist_left, dist_right), dist_bottom)

            # Outside: Euclidean distance to box
            dx_out = jnp.maximum(jnp.maximum(0.95 - x, x - 1.05), 0.0)
            dy_out = jnp.maximum(1.3 - y, 0)
            dist_outside = jnp.sqrt(dx_out ** 2 + dy_out ** 2)

            return jnp.where(is_inside, min_dist_inside, -dist_outside)
        dist_box = signed_dist_box(head_pos)
        dist_wall = head_pos[0] - 2.1
        dist_floor = 0.5 - head_pos[1]
        dist_wall_left = 0.0 - head_pos[0]
        return jnp.maximum(jnp.maximum(jnp.maximum(dist_box, dist_wall), dist_floor), dist_wall_left)
        # avoid_1 = (head_pos[1] >= 1.3) & (head_pos[0] >= 0.95) & (head_pos[0] <= 1.05)
        # avoid_2 = (head_pos[0] >= 2.35) # dont hit head on walls, bad dobby
        # avoid_3 = (head_pos[1] <= 0.5)
        # return avoid_1 | avoid_2 | avoid_3
    
    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size + 1),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )


###################### Baselines for CPPO - Reach Reach: RR ######################

class HopperReachReachBaseline_base: 
    """
    Hopper Avoid Ceiling Baseline environment for CPPO baseline
    Unaugmented state base

    reward format: gamma * (max (r1, r2)) - max(last r1, last r2)
    reward format: gamma * (r1 + r2) - (last r1 + last r2)
    """

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
    def compute_cost(self, curr_min_reach1_value, curr_min_reach2_value, prev_min_reach1_value=None, prev_min_reach2_value=None): 
        # Compute cost for constrained MDP 
        if prev_min_reach1_value is None or prev_min_reach2_value is None:
            cost = jnp.maximum(curr_min_reach1_value, curr_min_reach2_value)
        else: 
            cost = jnp.maximum(jnp.minimum(curr_min_reach1_value, prev_min_reach1_value), jnp.minimum(curr_min_reach2_value, prev_min_reach2_value))
        return cost 

    @partial(jax.jit, static_argnums=(0,))
    def compute_reward(self, state, last_state, params): 
        # Compute reward for constrained MDP 
        # Define this in the sub environments for different versions of reward functions - this env should be inherited from
        raise NotImplementedError("This function is not implemented for this environment.")
    
    @partial(jax.jit, static_argnums=(0,))
    def compute_observation(self, state):
        # Compute observation for constrained MDP 
        # Define this in the sub environments for different versions of reward functions - this env should be inherited from
        raise NotImplementedError("This function is not implemented for this environment.")
    
    def observation_space(self, params):
        # return spaces.Box(
        #     low=-jnp.inf,
        #     high=jnp.inf,
        #     shape=(self._env.observation_size + 2),
        # )
        raise NotImplementedError("This function is not implemented for this environment.")

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state = self._env.reset(key)
        head_pos, _, _, _, _, _ = self.calculate_position(state.obs)

        reach1_value = self.is_reach1(head_pos) 
        reach2_value = self.is_reach2(head_pos) 

        has_reached_1 = reach1_value < 0
        has_reached_2 = reach2_value < 0

        min_reach1 = reach1_value
        min_reach2 = reach2_value

        cost = self.compute_cost(min_reach1, min_reach2, prev_min_reach1_value=None, prev_min_reach2_value=None)

        env_state = EnvStateRR(state, reach1_value, reach2_value, has_reached_1, has_reached_2, min_reach1, min_reach2, cost)
        observation = self.compute_observation(env_state) 

        return observation, env_state
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        u = jnp.tanh(action)
        next_state = self._env.step(state.state, u)
        head_pos, _, _, _, _, _ = self.calculate_position(next_state.obs)

        reach1_value = self.is_reach1(head_pos)
        reach2_value = self.is_reach2(head_pos)
        has_reached_1 = jnp.logical_or(state.has_reached_1, reach1_value < 0)
        has_reached_2 = jnp.logical_or(state.has_reached_2, reach2_value < 0)
        min_reach1 = jnp.minimum(state.min_reach1, reach1_value)
        min_reach2 = jnp.minimum(state.min_reach2, reach2_value)
        
        cost = self.compute_cost(min_reach1, min_reach2, state.min_reach1, state.min_reach2)
        
        head_pos, jaw_pos, thg_pos, leg_pos, foot_front_pos, foot_back_pos = self.calculate_position(state.state.obs)
        pos_dict = {"head_pos": head_pos, "jaw_pos": jaw_pos, "thg_pos": thg_pos, "leg_pos": leg_pos,
                    "foot_front_pos": foot_front_pos, "foot_back_pos": foot_back_pos}

        next_state_new = EnvStateRR(next_state, reach1_value, reach2_value, has_reached_1, has_reached_2, min_reach1, min_reach2, cost)
        
        observation = self.compute_observation(next_state_new) #jnp.concatenate([state.obs, jnp.array([reach1_value]), jnp.array([reach2_value])])
        reward = self.compute_reward(state=next_state_new, last_state=state, params=params)

        # done = False # NOTE: Force dones to false for always avoid - make last done true outside 
        done = has_reached_1 & has_reached_2 # NOTE: Force dones to true for always avoid - make last done true outside #(state.avoid > 0) | (state.reach < 0)

        return observation, next_state_new, reward, done, pos_dict

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
        return value * 100
    
    @partial(jax.jit, static_argnums=(0,))
    def is_reach2(self, head_pos):
        target_center = [0., 1.4]
        reach = jnp.sqrt((head_pos[0] - target_center[0]) ** 2 + (head_pos[1] - target_center[1]) ** 2) - 0.1
        has_reached_goal = jnp.sqrt((head_pos[0] - target_center[0]) ** 2 + (head_pos[1] - target_center[1]) ** 2) < 0.1
        value = jnp.where(has_reached_goal, -2.5, reach)
        return value * 100
    

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )
    

class HopperReachReachBaseline_augmented_max(HopperReachReachBaseline_base): 
    """
    Hopper Avoid Ceiling Baseline environment for CPPO baseline
    
    Unaugmented state base: obs = [obs]
    reward format: gamma * (max (r1, r2)) - max(last r1, last r2)
    """

    
    @partial(jax.jit, static_argnums=(0,))
    def compute_reward(self, state, last_state, params): 
        # Compute reward for constrained MDP 
        # Max Reward: gamma * (max (r1, r2)) - max(last r1, last r2)
        return params.gamma * jnp.maximum(state.reach1, state.reach2) - jnp.maximum(last_state.reach1, last_state.reach2)
    
    @partial(jax.jit, static_argnums=(0,))
    def compute_observation(self, state):
        # Compute observation for constrained MDP 
        return jnp.concatenate([state.state.obs, jnp.array([state.min_reach1, state.min_reach2])])
    
    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size + 2),
        )
    
    

class HopperReachReachBaseline_augmented_sum(HopperReachReachBaseline_base):
    """
    Hopper Avoid Ceiling Baseline environment for CPPO baseline
    
    Unaugmented state base: obs = [obs]
    reward format: gamma * (r1 + r2) - (last r1 + last r2)
    """

    @partial(jax.jit, static_argnums=(0,))
    def compute_reward(self, state, last_state, params): 
        # Compute reward for constrained MDP 
        # Sum Reward: gamma * (r1 + r2) - (last r1 + last r2)
        return params.gamma * (state.reach1 + state.reach2) - (last_state.reach1 + last_state.reach2)
    
    @partial(jax.jit, static_argnums=(0,))
    def compute_observation(self, state):
        # Compute observation for constrained MDP 
        return jnp.concatenate([state.state.obs, jnp.array([state.min_reach1, state.min_reach2])])
    
    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size + 2),
        )
    

