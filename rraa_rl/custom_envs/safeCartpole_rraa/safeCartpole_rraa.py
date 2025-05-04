"""
Sander & Nikhil's Safe Cartpole environment
 - Will edited to remove HJR/CBF/DR
"""
# try: 
#   # When running inside module
#   from utils import CartPoleDeepreach, CartPoleHJR
# except: 
#   # When running from outside module
#   from custom_envs.utils import CartPoleDeepreach, CartPoleHJR

# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Safe Cartpole domain."""

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from lxml import etree
import numpy as np
import os 

# import hj_reachability as hj
# from torch2jax import t2j, j2t
# import jax 
# import jax.numpy as jnp 

import torch

_DEFAULT_TIME_LIMIT = 20 #10
SUITE = containers.TaggedTasks()
CURR_FILE_PATH = os.path.dirname(__file__)


def get_model_and_assets(num_poles=1):
  """Returns a tuple containing the model XML string and a dict of assets."""
  return _make_model(num_poles), common.ASSETS

def _make_model(n_poles):
  """Generates an xml string defining a cart with `n_poles` bodies."""
  custom_envs_folder = CURR_FILE_PATH 
  model_filename = os.path.join(custom_envs_folder, "safeCartpole_rraa.xml")
  xml_string = common.read_model(model_filename)
  if n_poles == 1:
    return xml_string
  mjcf = etree.fromstring(xml_string)
  parent = mjcf.find('./worldbody/body/body')  # Find first pole.
  # Make chain of poles.
  for pole_index in range(2, n_poles+1):
    child = etree.Element('body', name='pole_{}'.format(pole_index),
                          pos='0 0 1', childclass='pole')
    etree.SubElement(child, 'joint', name='hinge_{}'.format(pole_index))
    etree.SubElement(child, 'geom', name='pole_{}'.format(pole_index))
    parent.append(child)
    parent = child
  # Move plane down.
  floor = mjcf.find('./worldbody/geom')
  floor.set('pos', '0 0 {}'.format(1 - n_poles - .05))
  # Move cameras back.
  cameras = mjcf.findall('./worldbody/camera')
  cameras[0].set('pos', '0 {} 1'.format(-1 - 2*n_poles))
  cameras[1].set('pos', '0 {} 2'.format(-2*n_poles))
  return etree.tostring(mjcf, pretty_print=True)

@SUITE.add('benchmarking')
def balance(time_limit=_DEFAULT_TIME_LIMIT, random=None,
            environment_kwargs=None):
  """Returns the Cartpole Balance task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Balance(swing_up=False, sparse=False, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add('benchmarking')
def balance_sparse(time_limit=_DEFAULT_TIME_LIMIT, random=None,
                   environment_kwargs=None):
  """Returns the sparse reward variant of the Cartpole Balance task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Balance(swing_up=False, sparse=True, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add('benchmarking')
def swingup(time_limit=_DEFAULT_TIME_LIMIT, random=None,
            environment_kwargs=None):
  """Returns the Cartpole Swing-Up task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Balance(swing_up=True, sparse=False, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add('benchmarking')
def swingup_sparse(time_limit=_DEFAULT_TIME_LIMIT, random=None,
                   environment_kwargs=None):
  """Returns the sparse reward variant of the Cartpole Swing-Up task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Balance(swing_up=True, sparse=True, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add()
def two_poles(time_limit=_DEFAULT_TIME_LIMIT, random=None,
              environment_kwargs=None):
  """Returns the Cartpole Balance task with two poles."""
  physics = Physics.from_xml_string(*get_model_and_assets(num_poles=2))
  task = Balance(swing_up=True, sparse=False, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add()
def three_poles(time_limit=_DEFAULT_TIME_LIMIT, random=None, num_poles=3,
                sparse=False, environment_kwargs=None):
  """Returns the Cartpole Balance task with three or more poles."""
  physics = Physics.from_xml_string(*get_model_and_assets(num_poles=num_poles))
  task = Balance(swing_up=True, sparse=sparse, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)

class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Cartpole domain."""

  def cart_position(self):
    """Returns the position of the cart."""
    return self.named.data.qpos['slider'][0]

  def angular_vel(self):
    """Returns the angular velocity of the pole."""
    return self.data.qvel[1:]

  def pole_angle_cosine(self):
    """Returns the cosine of the pole angle."""
    return self.named.data.xmat[2:, 'zz']

  def bounded_position(self):
    """Returns the state, with pole angle split into sin/cos."""
    return np.hstack((self.cart_position(),
                      self.named.data.xmat[2:, ['zz', 'xz']].ravel()))


class Balance(base.Task):
  """A Cartpole `Task` to balance the pole.

  State is initialized either close to the target configuration or at a random
  configuration.
  """

  # Defines reward / l(x)
  _CART_RANGE = (-.25, .25)
  _ANGLE_COSINE_RANGE = (.995, 1)

  ################# RRAA Change #################
  _X_VEL_RANGE = (-0.1, 0.1)
  _THETA_VEL_RANGE = (0.1, 0.1)
  ################# RRAA Change #################

  def __init__(self, swing_up, sparse, problem_type="RA", use_velocity_target=True, train_type="baseline", random=None):
    """Initializes an instance of `Balance`.

    Args:
      swing_up: A `bool`, which if `True` sets the cart to the middle of the
        slider and the pole pointing towards the ground. Otherwise, sets the
        cart to a random position on the slider and the pole to a random
        near-vertical position.
      sparse: A `bool`, whether to return a sparse or a smooth reward.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
      ################# RRAA Change #################
      problem_type: A `str`, which if "RA" or "R" - changes the problem between reach_avoid and reach
        setting which in effect changes the reward function state augmentation: 
        - "R": reach: problem is just reaching the swingup position - take max l(x) over steps
        - "RA": reach_avoid: problem is to reach the swingup position while avoiding unsafe region
           takes max with min(max(reward(x_{t-1}), l(x)), g(x))
      use_velocity_target: A `bool`, whether to use the velocity targets in the reward function.
      train_type: A `str`, which if "RAA" or "baseline" - changes the training type - 
        this changes which rewards are used - if the min/max is done in the reward or outside
      ################# RRAA Change #################
    """

    # self.set_unsafe_region(unsafe_x_min=-10,
    #                        unsafe_x_max=10,
    #                        unsafe_vel_max=100,
    #                        unsafe_theta_min=0.15, # TODO: CHANGE
    #                        unsafe_theta_max=np.pi/2) # TODO: CHANGE
    
    unsafe_x_min     = -1.5 
    unsafe_x_max     = 1.5 
    unsafe_vel_max   = 20 
    
    unsafe_theta_min = np.pi/8
    unsafe_theta_max =  np.pi/4
    unsafe_theta_in_range = True # True = specified theta range is unsafe

    self.set_unsafe_region(unsafe_x_min=unsafe_x_min, unsafe_x_max=unsafe_x_max, unsafe_vel_max=unsafe_vel_max, unsafe_theta_min=unsafe_theta_min, unsafe_theta_max=unsafe_theta_max, 
                        unsafe_theta_in_range=unsafe_theta_in_range)
    self._sparse = sparse
    self._swing_up = swing_up
    # self.setup_hj_reachability()

    ################# RRAA Change #################
    self.problem_type = problem_type
    self.last_lofx = None 
    self.last_gofx = None 
    self.use_velocity_target = use_velocity_target
    self.train_type = train_type # ["RAA", "baseline"]
    ################# RRAA Change #################

    super().__init__(random=random)

  def set_unsafe_region(self, unsafe_x_min, unsafe_x_max, unsafe_vel_max, unsafe_theta_min, unsafe_theta_max, unsafe_theta_in_range): 
    """
    Set the unsafe region: 
    """
    self.unsafe_x_min = unsafe_x_min 
    self.unsafe_x_max = unsafe_x_max 

    self.unsafe_vel_max = unsafe_vel_max

    self.unsafe_theta_min = unsafe_theta_min 
    self.unsafe_theta_max = unsafe_theta_max 

    self.unsafe_theta_in_range = unsafe_theta_in_range # Default should be True!!!!!

    self.use_unsafe_theta = True
    if self.unsafe_theta_min == self.unsafe_theta_max: 
      self.use_unsafe_theta = False  
    return 
  
  def is_unsafe(self, physics): 
    """
    Returns boolean if the cartpole is in the unsafe region
    """
    x = physics.named.data.qpos[0]
    theta = (physics.named.data.qpos[1] + np.pi)%(2*np.pi) - np.pi
    xdot = physics.named.data.qvel[0]
    thetadot = physics.named.data.qvel[1]

    # return self.cartpole_deepreach.is_unsafe(state=np.array([x, theta, xdot, thetadot]))
    # Will: FROM DEEPREACH DYNAMICS

    if x < self.unsafe_x_min or self.unsafe_x_max < x: 
        return True 
    elif xdot < -self.unsafe_vel_max or self.unsafe_vel_max < xdot: 
        return True 
    elif self.use_unsafe_theta:
        if self.unsafe_theta_in_range: 
            return (self.unsafe_theta_min < theta and theta < self.unsafe_theta_max)
        else: 
            return (theta < self.unsafe_theta_min or self.unsafe_theta_max < theta)

    return False 

  ################# RRAA Change #################
  ##################################### g(x) #####################################
  
  def get_gofx(self, obs, sparse=False):
    # Computes g(x): avoid function
    # CONVENTIONS: NEGATIVE IFF UNSAFE 
    """
    A penalty function (ie g(x)) for given obs. Mainly to be used in custom PPO/SAC algorithms.
    Defined to be _negative_ iff unsafe.
    
    NOTE WAS:
    For now, computed directly from obs and may diverge from is_unsafe, eg. if not sparse. 
    (We should probably make target/obstacle sdf's for future envs)
    
    Doing this bc chat says to convert to physics (and use is_unsafe), one would need to either:
      1. cache each state physics during roll-out 
        - requires subclassing on_policy_algorithm + defining custom collect_rollouts method (SB3)
        - slow if not-vectorized
        - hefty memory
      2. "approximate" w/ smth like,
        def observation_to_physics(obs):
          return {"pos": obs[..., 0], "vel": obs[..., 1]}
    """
    if isinstance(obs, torch.Tensor):
      obs_cpu = obs.clone().cpu().numpy()
    else: 
      obs_cpu = np.array(obs)

    x = obs_cpu[..., 0]
    theta = (np.arctan2(obs_cpu[..., 2], obs_cpu[..., 1]) + np.pi) % (2*np.pi) - np.pi
    xdot = obs_cpu[..., 3]
    thetadot = obs_cpu[..., 4]

    if sparse:
      # Matches is_unsafe logic

      unsafe_x = (x < self.unsafe_x_min) | (x > self.unsafe_x_max)
      unsafe_vel = (xdot < -self.unsafe_vel_max) | (xdot > self.unsafe_vel_max)

      if self.use_unsafe_theta:
          if self.unsafe_theta_in_range:
              unsafe_theta = (theta > self.unsafe_theta_min) & (theta < self.unsafe_theta_max)
          else:
              unsafe_theta = (theta < self.unsafe_theta_min) | (theta > self.unsafe_theta_max)
      else:
          unsafe_theta = np.zeros_like(x, dtype=bool)

      penalty = -(unsafe_x | unsafe_vel | unsafe_theta).astype(float)

    else:
      # continuous versions of the above, w cos sim for angles

      x_penalty = np.abs(x - (self.unsafe_x_min + self.unsafe_x_max)/2) - (self.unsafe_x_max - self.unsafe_x_min) / 2
      v_penalty = np.abs(xdot - ((-self.unsafe_vel_max) + self.unsafe_vel_max)/2) - (self.unsafe_vel_max - (-self.unsafe_vel_max)) / 2\

      x_penalty = - x_penalty # Conventions: negative iff unsafe
      v_penalty = - v_penalty
      
      if self.use_unsafe_theta:
        theta_min = ((self.unsafe_theta_min + np.pi) % (2 * np.pi)) - np.pi
        theta_max = ((self.unsafe_theta_max + np.pi) % (2 * np.pi)) - np.pi

        safe_theta_center = ((theta_min + theta_max) / 2 + np.pi) % (2 * np.pi) - np.pi
        safe_theta_halfwidth = (theta_max - theta_min) / 2
        safe_theta_halfwidth += (safe_theta_halfwidth < 0) * np.pi

        angle_diff = (theta - safe_theta_center + np.pi) % (2 * np.pi) - np.pi
        theta_penalty = np.cos(angle_diff) - np.cos(safe_theta_halfwidth)
        if not self.unsafe_theta_in_range: 
          theta_penalty = theta_penalty * -1 
      else:
        raise NotImplementedError("Unsafe theta not implemented for smooth penalty")
      
      penalty = np.min([x_penalty, v_penalty, theta_penalty], axis=0)
      
    return penalty # negative in obstacle!
  
  def get_penalty(self, obs):
    """Returns a sparse or a smooth penalty, as specified in the constructor."""
    return self.get_gofx(obs, sparse=self._sparse)

  ##################################### g(x) #####################################
  ##################################### l(x) #####################################

  def get_lofx(self, obs, sparse): 
    # Computes l(x): target function

    # NOTE: l(x) is changed to just be a swingup task (NO BALANCE)
    # Get reward l(x) from observation

    
    if isinstance(obs, torch.Tensor):
      obs_cpu = obs.clone().cpu().numpy()
    else: 
      obs_cpu = np.array(obs)

    x = obs_cpu[..., 0]
    theta = (np.arctan2(obs_cpu[..., 2], obs_cpu[..., 1]) + np.pi) % (2*np.pi) - np.pi
    xdot = obs_cpu[..., 3]
    thetadot = obs_cpu[..., 4]

    # FIXME: Validate that this is the correct theta - think it might be sine instead of cosine with our theta convention
    physics_pole_angle_cosine = np.sin(theta) # physics.pole_angle_cosine()

    if sparse:
      cart_in_bounds = rewards.tolerance(x,
                                         self._CART_RANGE)
      angle_in_bounds = rewards.tolerance(physics_pole_angle_cosine, #physics.pole_angle_cosine() # FIXME: This might be correct - I think with our theta it is a sine and not cosine
                                          self._ANGLE_COSINE_RANGE).prod()

      xdot_in_bounds = rewards.tolerance(xdot,
                                          self._X_VEL_RANGE)

      thetadot_in_bounds = rewards.tolerance(thetadot,  
                                          self._THETA_VEL_RANGE)

      return cart_in_bounds * angle_in_bounds * xdot_in_bounds * thetadot_in_bounds
    else:
      upright = (physics_pole_angle_cosine + 1) / 2
      centered = rewards.tolerance(x, margin=2)
      centered = (1 + centered) / 2

      small_velocity = rewards.tolerance(thetadot, margin=5).min()
      small_velocity = (1 + small_velocity) / 2

      x_sdf = np.abs(x - (self._CART_RANGE[0] + self._CART_RANGE[1]) / 2) - (self._CART_RANGE[1] - self._CART_RANGE[0]) / 2
      
      # FIXME: might want to change this to be properly centered
      theta_sdf = np.abs(physics_pole_angle_cosine - (self._ANGLE_COSINE_RANGE[0] + self._ANGLE_COSINE_RANGE[1]) / 2) - (self._ANGLE_COSINE_RANGE[1] - self._ANGLE_COSINE_RANGE[0]) / 2

      xdot_sdf = np.abs(xdot - (self._X_VEL_RANGE[0] + self._X_VEL_RANGE[1]) / 2) - (self._X_VEL_RANGE[1] - self._X_VEL_RANGE[0]) / 2

      thetadot_sdf = np.abs(thetadot - (self._THETA_VEL_RANGE[0] + self._THETA_VEL_RANGE[1]) / 2) - (self._THETA_VEL_RANGE[1] - self._THETA_VEL_RANGE[0]) / 2

      x_sdf = -x_sdf # Conventions: positive iff in target 
      theta_sdf = -theta_sdf
      xdot_sdf = -xdot_sdf
      thetadot_sdf = -thetadot_sdf
      
      reward = np.min([x_sdf, theta_sdf, xdot_sdf, thetadot_sdf], axis=0)

    return reward

  ##################################### l(x) #####################################
  ################# RRAA Change #################

  def obs_to_cbfstate(self, obs):
    if isinstance(obs, torch.Tensor):
      obs_cpu = obs.cpu().numpy()
    else: 
      obs_cpu = np.array(obs)

    x = obs_cpu[..., 0]
    theta = (np.arctan2(obs_cpu[..., 2], obs_cpu[..., 1]) + np.pi) % (2*np.pi) - np.pi
    xdot = obs_cpu[..., 3]
    thetadot = obs_cpu[..., 4]
    
    try:
      state = np.array([x[0], theta[0],xdot[0], thetadot[0]])
    except: 
      state = np.array([x, theta, xdot, thetadot])
    return state 
  
  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.

    Initializes the cart and pole according to `swing_up`, and in both cases
    adds a small random initial velocity to break symmetry.

    Args:
      physics: An instance of `Physics`.
    """
    nv = physics.model.nv
    
    max_counter = 1000 
    counter = 0 
    found_start = False

    # NOTE: right now only support for one pole
    while not found_start: 
      if self._swing_up: 
        x = .01*self.random.randn()
        theta = np.pi + .01*self.random.randn()
      else: 
        x = self.random.uniform(-.1, .1)
        theta = self.random.uniform(-.034, .034, nv - 1)

      xdot = 0.01 * self.random.randn()
      thetadot = 0.01 * self.random.randn()

      start_state = np.array([x, theta, xdot, thetadot])

      # Will: what happens if we remove this?
      found_start = True

      # start_state_value = self.hjr_state_to_value(start_state)

      # if start_state_value >= 0:
      #   # safe 
      #   found_start = True
      # else: 
      #   counter += 1
      #   if counter > max_counter: 
      #     # Force 0 and print that it occured
      #     start_state = np.array([0.0, np.pi, 0.0, 0.0])      
      #     start_state_val = self.hjr_state_to_value(start_state)
      #     print("\n\n\n\nMax counter exceeded: forcing to ", start_state)
      #     print("Start state value: ", start_state_val)
      #     print("\n\n\n")
      #     found_start = True 

    physics.named.data.qpos['slider'] = start_state[0]
    physics.named.data.qpos['hinge_1'] = start_state[1]
    physics.named.data.qvel[0] = start_state[2]
    physics.named.data.qvel[1] = start_state[3]

    # if self._swing_up:
    #   physics.named.data.qpos['slider'] = .01*self.random.randn()
    #   physics.named.data.qpos['hinge_1'] = np.pi + .01*self.random.randn()
    #   physics.named.data.qpos[2:] = .1*self.random.randn(nv - 2)
    # else:
    #   physics.named.data.qpos['slider'] = self.random.uniform(-.1, .1)
    #   physics.named.data.qpos[1:] = self.random.uniform(-.034, .034, nv - 1)
    # physics.named.data.qvel[:] = 0.01 * self.random.randn(physics.model.nv)

    super().initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation of the (bounded) physics state."""
    obs = collections.OrderedDict()
    obs['position'] = physics.bounded_position()
    obs['velocity'] = physics.velocity()

    if self.is_unsafe(physics=physics): 
      physics.named.model.geom_rgba['pole_1'] = [1, 0, 0, 1] # force red for now
      physics.named.model.geom_rgba['cart'] = [1, 0, 0, 1] # force red for now
    else: 
      physics.named.model.geom_rgba['pole_1'] = [0.5, 0.5, 0.5, 1] # default back to beige
      physics.named.model.geom_rgba['cart'] = [0.5, 0.5, 0.5, 1] # default back to beige
    return obs

  ################# RRAA Change #################
  def get_reward(self, physics):
    """Returns a sparse or a smooth reward, as specified in the constructor."""
    obs = self.get_observation(physics)
    
    # reshape obs 
    obs_array = np.concatenate([obs['position'], obs['velocity']]).reshape(1, -1)
    
    if self.train_type == "baseline":
      # Baseline: does the min/max with avoid and target for PPO baseline
      if self.last_lofx is None or self.last_gofx is None: 
        if self.problem_type == "R": 
          lofx = self.get_lofx(obs_array, sparse=self._sparse)
          self.last_lofx = lofx
          reward = lofx 
        elif self.problem_type == "RA":
          lofx = self.get_lofx(obs_array, sparse=self._sparse)
          gofx = self.get_gofx(obs_array, sparse=self._sparse)
          self.last_lofx = lofx
          self.last_gofx = gofx
          reward = min(lofx, gofx)
        else:
          raise ValueError("Problem type not recognized")

      else: 
        if self.problem_type == "R": 
          lofx = self.get_lofx(obs_array, sparse=self._sparse)
          reward = max(lofx, self.last_lofx)
          self.last_lofx = lofx
        elif self.problem_type == "RA":
          lofx = self.get_lofx(obs_array, sparse=self._sparse)
          gofx = self.get_gofx(obs_array, sparse=self._sparse)

          max_lofx = max(lofx, self.last_lofx)
          min_gofx = min(gofx, self.last_gofx)

          self.last_lofx = lofx
          self.last_gofx = gofx

          reward = min(max_lofx, min_gofx)
        else:
          raise ValueError("Problem type not recognized")

      reward = reward.item()
    elif self.train_type == "RAA": 
      lofx = self.get_lofx(obs_array, sparse=self._sparse)
      gofx = self.get_gofx(obs_array, sparse=self._sparse)

      reward = min(lofx, gofx)  
      
      reward = reward.item()
    return reward #self._get_reward(physics, sparse=self._sparse)
    ################# RRAA Change #################