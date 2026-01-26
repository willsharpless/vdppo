"""JAX-compatible Scene environment using MJX.

This module provides a JAX-compatible version of the Scene environment
that uses MuJoCo's MJX backend for hardware-accelerated physics simulation.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Tuple

import ipdb
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from dm_control import mjcf
from loguru import logger
from mujoco import mjx

from rraa_rl.envs import mjx_patch
from rraa_rl.envs.manipspace import mjcf_utils
from rraa_rl.envs.manipspace.diff_ik_jax import DiffIKControllerJax
from rraa_rl.envs.manipspace.lie.se3_jax import SE3
from rraa_rl.envs.manipspace.lie.so3_jax import SO3, mat_to_quat, quat_multiply, quat_rotate, quat_to_mat, so3_exp


class SceneEnvState(NamedTuple):
    """State of the Scene environment for JAX compatibility."""

    mjx_data: mjx.Data
    step_count: jax.Array
    prev_qpos: jax.Array
    prev_qvel: jax.Array
    prev_button_pos: jax.Array  # Previous button joint positions for detecting presses
    button_states: jax.Array  # Current button states (binary)
    target_button_states: jax.Array  # Target button states
    target_drawer_pos: jax.Array  # Target drawer position
    target_window_pos: jax.Array  # Target window position
    target_cube_pos: jax.Array  # Target cube position
    target_cube_quat: jax.Array  # Target cube quaternion
    rng: jax.Array


class SceneEnvConfig(NamedTuple):
    """Static configuration for the Scene environment."""

    physics_timestep: float
    control_timestep: float
    n_steps: int
    workspace_bounds: jax.Array
    arm_sampling_bounds: jax.Array
    object_sampling_bounds: jax.Array
    target_sampling_bounds: jax.Array
    action_low: jax.Array
    action_high: jax.Array
    home_qpos: jax.Array
    effector_down_quat: jax.Array
    drawer_center: jax.Array
    num_cubes: int
    num_buttons: int
    num_button_states: int
    # Scalers for observation normalization
    xyz_center: jax.Array
    xyz_scaler: float
    gripper_scaler: float
    button_scaler: float
    drawer_scaler: float
    window_scaler: float


class ManipStep(NamedTuple):
    """Result of a manipulation environment step."""

    next_state: SceneEnvState
    observation: jax.Array
    reward: jax.Array
    done: jax.Array
    info: Dict[str, Any]


class SceneEnvJax:
    """JAX-compatible Scene environment using MJX.

    This environment consists of a cube, two buttons, a drawer, and a window.
    The goal is to manipulate the objects to a target configuration.
    The buttons toggle the lock state of the drawer and window.

    Uses MuJoCo's MJX backend for hardware-accelerated physics simulation.
    Supports JIT compilation and vectorization for efficient training.

    The default control mode is relative end-effector control. The 5-D action
    space corresponds to:
    - 3-D relative end-effector position (x, y, z)
    - 1-D relative end-effector yaw
    - 1-D relative gripper opening

    Attributes:
        mj_model: MuJoCo model
        mjx_model: MJX model (JAX-compatible)
        config: Static environment configuration
    """

    def __init__(
        self,
        physics_timestep: float = 0.002,
        control_timestep: float = 0.05,
    ):
        """Initialize a new SceneEnvJax instance.

        Args:
            physics_timestep: Physics timestep for simulation
            control_timestep: Control timestep (action frequency)
        """
        # Build the MJCF model
        self._desc_dir = Path(__file__).resolve().parent / "descriptions"
        mjcf_model = self._build_mjcf_model(self._desc_dir)

        # Compile to MuJoCo model
        xml_str = mjcf_utils.to_string(mjcf_model)
        assets = mjcf_utils.get_assets(mjcf_model)
        self.mj_model = mujoco.MjModel.from_xml_string(xml_str, assets)
        self.mj_model.opt.timestep = physics_timestep

        self.impl = "warp"

        # Convert to MJX model
        self.mjx_model = mjx.put_model(self.mj_model, impl=self.impl)

        # Compute timesteps
        n_steps = int(round(control_timestep / physics_timestep))

        # Define configuration
        home_qpos = jnp.asarray([-jnp.pi / 2, -jnp.pi / 2, jnp.pi / 2, -jnp.pi / 2, -jnp.pi / 2, 0])
        effector_down_quat = jnp.asarray([0.0, 1.0, 0.0, 0.0])  # wxyz
        workspace_bounds = jnp.asarray([[0.25, -0.35, 0.02], [0.6, 0.35, 0.35]])
        arm_sampling_bounds = jnp.asarray([[0.25, -0.2, 0.20], [0.6, 0.2, 0.35]])
        object_sampling_bounds = jnp.asarray([[0.3, -0.07], [0.45, 0.18]])
        target_sampling_bounds = object_sampling_bounds
        action_range = jnp.array([0.05, 0.05, 0.05, 0.3, 1.0])
        drawer_center = jnp.array([0.33, -0.24, 0.066])

        self.config = SceneEnvConfig(
            physics_timestep=physics_timestep,
            control_timestep=control_timestep,
            n_steps=n_steps,
            workspace_bounds=workspace_bounds,
            arm_sampling_bounds=arm_sampling_bounds,
            object_sampling_bounds=object_sampling_bounds,
            target_sampling_bounds=target_sampling_bounds,
            action_low=-action_range,
            action_high=action_range,
            home_qpos=home_qpos,
            effector_down_quat=effector_down_quat,
            drawer_center=drawer_center,
            num_cubes=1,
            num_buttons=2,
            num_button_states=2,
            xyz_center=jnp.array([0.425, 0.0, 0.0]),
            xyz_scaler=10.0,
            gripper_scaler=3.0,
            button_scaler=120.0,
            drawer_scaler=18.0,
            window_scaler=15.0,
        )

        # Get joint and actuator IDs
        arm_joint_names = [
            "ur5e/shoulder_pan_joint",
            "ur5e/shoulder_lift_joint",
            "ur5e/elbow_joint",
            "ur5e/wrist_1_joint",
            "ur5e/wrist_2_joint",
            "ur5e/wrist_3_joint",
        ]
        self._arm_joint_ids = [self.mj_model.joint(name).id for name in arm_joint_names]

        arm_actuator_names = [
            "ur5e/shoulder_pan",
            "ur5e/shoulder_lift",
            "ur5e/elbow",
            "ur5e/wrist_1",
            "ur5e/wrist_2",
            "ur5e/wrist_3",
        ]
        self._arm_actuator_ids = np.asarray([self.mj_model.actuator(name).id for name in arm_actuator_names])

        gripper_actuator_names = ["ur5e/robotiq/fingers_actuator"]
        self._gripper_actuator_ids = np.asarray([self.mj_model.actuator(name).id for name in gripper_actuator_names])

        self._gripper_opening_joint_id = self.mj_model.joint("ur5e/robotiq/right_driver_joint").id
        self._pinch_site_id = self.mj_model.site("ur5e/robotiq/pinch").id
        self._attach_site_id = self.mj_model.site("ur5e/attachment_site").id

        # Object joint IDs
        self._cube_joint_ids = [self.mj_model.joint(f"object_joint_{i}").id for i in range(self.config.num_cubes)]
        self._cube_joint_qpos_addrs = [
            self.mj_model.joint(f"object_joint_{i}").qposadr.item() for i in range(self.config.num_cubes)
        ]

        # Button joint IDs
        self._button_joint_ids = [
            self.mj_model.joint(f"buttonbox_joint_{i}").id for i in range(self.config.num_buttons)
        ]
        self._button_joint_qpos_addrs = [
            self.mj_model.joint(f"buttonbox_joint_{i}").qposadr.item() for i in range(self.config.num_buttons)
        ]
        self._button_joint_dof_addrs = [
            self.mj_model.joint(f"buttonbox_joint_{i}").dofadr.item() for i in range(self.config.num_buttons)
        ]

        # Drawer and window joint IDs
        self._drawer_joint_id = self.mj_model.joint("drawer_slide").id
        self._drawer_joint_qpos_addr = self.mj_model.joint("drawer_slide").qposadr.item()
        self._drawer_joint_dof_addr = self.mj_model.joint("drawer_slide").dofadr.item()
        self._window_joint_id = self.mj_model.joint("window_slide").id
        self._window_joint_qpos_addr = self.mj_model.joint("window_slide").qposadr.item()
        self._window_joint_dof_addr = self.mj_model.joint("window_slide").dofadr.item()

        # Site IDs
        self._drawer_site_id = self.mj_model.site("drawer_handle_center").id
        self._window_site_id = self.mj_model.site("window_handle_center").id
        self._button_site_ids = [self.mj_model.site(f"btntop_{i}").id for i in range(self.config.num_buttons)]

        # Compute T_pa (pinch to attach transformation) at home position
        mj_data = mujoco.MjData(self.mj_model)
        mj_data.qpos[:6] = np.asarray(home_qpos)
        mujoco.mj_forward(self.mj_model, mj_data)

        pinch_pos = mj_data.site_xpos[self._pinch_site_id].copy()
        pinch_mat = mj_data.site_xmat[self._pinch_site_id].reshape(3, 3).copy()
        attach_pos = mj_data.site_xpos[self._attach_site_id].copy()
        attach_mat = mj_data.site_xmat[self._attach_site_id].reshape(3, 3).copy()

        pinch_quat = np.zeros(4)
        attach_quat = np.zeros(4)
        mujoco.mju_mat2Quat(pinch_quat, pinch_mat.ravel())
        mujoco.mju_mat2Quat(attach_quat, attach_mat.ravel())

        # T_pa = T_wp^-1 @ T_wa
        pinch_pose = SE3.from_rotation_and_translation(SO3(wxyz=jnp.asarray(pinch_quat)), jnp.asarray(pinch_pos))
        attach_pose = SE3.from_rotation_and_translation(SO3(wxyz=jnp.asarray(attach_quat)), jnp.asarray(attach_pos))
        T_pa = pinch_pose.inverse().multiply(attach_pose)
        T_pa = T_pa.fix_quat_sign()
        self._T_pa_wxyz_xyz = T_pa.wxyz_xyz

        # Initialize inverse kinematics controller (same approach as ManipSpaceEnv)
        ik_mjcf = mjcf.from_path((self._desc_dir / "universal_robots_ur5e" / "ur5e.xml"), escape_separators=True)
        ik_xml_str = mjcf_utils.to_string(ik_mjcf)
        ik_assets = mjcf_utils.get_assets(ik_mjcf)
        ik_model = mujoco.MjModel.from_xml_string(ik_xml_str, ik_assets)
        self._ik = DiffIKControllerJax(
            model=ik_model,
            sites=["attachment_site"],
            # qpos0=home_qpos,
        )

    @staticmethod
    def _build_mjcf_model(desc_dir: Path) -> mjcf.RootElement:
        """Build the MJCF model for the environment."""
        # Set scene
        arena_mjcf = mjcf.from_path((desc_dir / "floor_wall.xml").as_posix())
        arena_mjcf.model = "ur5e_arena"

        arena_mjcf.statistic.center = (0.3, 0, 0.15)
        arena_mjcf.statistic.extent = 0.7
        getattr(arena_mjcf.visual, "global").elevation = -20
        getattr(arena_mjcf.visual, "global").azimuth = 180
        arena_mjcf.statistic.meansize = 0.04
        arena_mjcf.visual.map.znear = 0.1
        arena_mjcf.visual.map.zfar = 10.0

        # Add UR5e robot arm
        ur5e_mjcf = mjcf.from_path((desc_dir / "universal_robots_ur5e" / "ur5e.xml"), escape_separators=True)
        ur5e_mjcf.model = "ur5e"

        for light in ur5e_mjcf.find_all("light"):
            light.remove()
            del light

        # Attach the robotiq gripper to the UR5e flange
        gripper_mjcf = mjcf.from_path((desc_dir / "robotiq_2f85" / "2f85.xml"), escape_separators=True)
        gripper_mjcf.model = "robotiq"
        mjcf_utils.attach(ur5e_mjcf, gripper_mjcf, "attachment_site")

        # Attach UR5e to the scene
        mjcf_utils.attach(arena_mjcf, ur5e_mjcf)

        # Add objects
        cube_mjcf = mjcf.from_path((desc_dir / "cube.xml").as_posix())
        arena_mjcf.include_copy(cube_mjcf)
        button_mjcf = mjcf.from_path((desc_dir / "buttons.xml").as_posix())
        arena_mjcf.include_copy(button_mjcf)
        drawer_mjcf = mjcf.from_path((desc_dir / "drawer.xml").as_posix())
        arena_mjcf.include_copy(drawer_mjcf)
        window_mjcf = mjcf.from_path((desc_dir / "window.xml").as_posix())
        arena_mjcf.include_copy(window_mjcf)

        # Add cameras
        cameras = {
            "front": {
                "pos": (1.139, 0.000, 0.821),
                "xyaxes": (0.000, 1.000, 0.000, -0.627, 0.000, 0.779),
            },
            "front_pixels": {
                "pos": (0.905, 0.000, 0.762),
                "xyaxes": (0.000, 1.000, 0.000, -0.771, 0.000, 0.637),
            },
        }
        for camera_name, camera_kwargs in cameras.items():
            arena_mjcf.worldbody.add("camera", name=camera_name, **camera_kwargs)

        return arena_mjcf

    @property
    def action_size(self) -> int:
        """Return the action dimension."""
        return 5

    @property
    def observation_size(self) -> int:
        """Return the observation dimension.

        Components:
        - joint_pos: 6
        - joint_vel: 6
        - effector_pos: 3
        - cos/sin_yaw: 2
        - gripper_opening: 1
        - gripper_contact: 1
        - per cube (1 cube): pos(3) + quat(4) + cos/sin_yaw(2) = 9
        - per button (2 buttons): state(2) + pos(1) + vel(1) = 4 each = 8
        - drawer: pos(1) + vel(1) = 2
        - window: pos(1) + vel(1) = 2
        Total: 6+6+3+2+1+1+9+8+2+2 = 40
        """
        return 40

    def reset(self, rng: jax.Array) -> Tuple[SceneEnvState, jax.Array]:
        """Reset the environment.

        Args:
            rng: JAX random key

        Returns:
            Tuple of (state, observation)
        """
        rng, rng_arm, rng_yaw, rng_cube_xy, rng_cube_yaw, rng_buttons, rng_drawer, rng_window = jax.random.split(rng, 8)

        # Create initial MJX data
        mjx_data = mjx.make_data(self.mj_model, impl=self.impl)

        # Sample initial arm end-effector position
        eff_pos = jax.random.uniform(
            rng_arm,
            shape=(3,),
            minval=self.config.arm_sampling_bounds[0],
            maxval=self.config.arm_sampling_bounds[1],
        )

        # Sample initial arm yaw
        yaw = jax.random.uniform(rng_yaw, minval=-jnp.pi, maxval=jnp.pi)

        # Compute target orientation
        yaw_quat = quat_from_z_radians(yaw)
        eff_ori = quat_multiply(yaw_quat, self.config.effector_down_quat)

        # Solve IK for initial arm joint positions
        qpos_arm = self._solve_ik(eff_pos, eff_ori, self.config.home_qpos)

        # Set arm joint positions
        qpos = mjx_data.qpos.at[:6].set(qpos_arm)

        # Sample cube position and orientation
        cube_xy = jax.random.uniform(
            rng_cube_xy,
            shape=(2,),
            minval=self.config.object_sampling_bounds[0],
            maxval=self.config.object_sampling_bounds[1],
        )
        cube_pos = jnp.array([cube_xy[0], cube_xy[1], 0.02])
        cube_yaw = jax.random.uniform(rng_cube_yaw, minval=0.0, maxval=2 * jnp.pi)
        cube_quat = quat_from_z_radians(cube_yaw)

        # Set cube position (qpos address for object_joint_0)
        cube_qpos_addr = self._cube_joint_qpos_addrs[0]
        qpos = qpos.at[cube_qpos_addr : cube_qpos_addr + 3].set(cube_pos)
        qpos = qpos.at[cube_qpos_addr + 3 : cube_qpos_addr + 7].set(cube_quat)

        # Sample button states (0 or 1 for each button)
        button_states = jax.random.randint(
            rng_buttons, shape=(self.config.num_buttons,), minval=0, maxval=self.config.num_button_states
        )

        # Sample drawer position
        drawer_pos = jax.random.uniform(rng_drawer, minval=-0.16, maxval=0.0)
        qpos = qpos.at[self._drawer_joint_qpos_addr].set(drawer_pos)

        # Sample window position
        window_pos = jax.random.uniform(rng_window, minval=0.0, maxval=0.2)
        qpos = qpos.at[self._window_joint_qpos_addr].set(window_pos)

        # Update mjx_data with new qpos
        mjx_data = mjx_data.replace(qpos=qpos)

        # Forward kinematics
        mjx_data = mjx.forward(self.mjx_model, mjx_data)

        # Get initial button positions
        prev_button_pos = jnp.array(
            [mjx_data.qpos[self._button_joint_qpos_addrs[i]] for i in range(self.config.num_buttons)]
        )

        # Initialize targets (same as current for simplicity - can be overridden)
        target_button_states = button_states.copy()
        target_drawer_pos = jnp.array(drawer_pos)
        target_window_pos = jnp.array(window_pos)
        target_cube_pos = cube_pos.copy()
        target_cube_quat = cube_quat.copy()

        # Make sure prev_qpos and prev_qvel are backed by different arrays so we can donate.
        prev_qpos = mjx_data.qpos.copy()
        prev_qvel = mjx_data.qvel.copy()

        state = SceneEnvState(
            mjx_data=mjx_data,
            step_count=jnp.array(0),
            prev_qpos=prev_qpos,
            prev_qvel=prev_qvel,
            prev_button_pos=prev_button_pos,
            button_states=button_states,
            target_button_states=target_button_states,
            target_drawer_pos=target_drawer_pos,
            target_window_pos=target_window_pos,
            target_cube_pos=target_cube_pos,
            target_cube_quat=target_cube_quat,
            rng=rng,
        )

        obs = self._get_observation(state)
        return state, obs

    def step(
        self,
        state: SceneEnvState,
        action: jax.Array,
    ) -> ManipStep[SceneEnvState]:
        """Take a step in the environment.

        Args:
            state: Current environment state
            action: Action to take (normalized to [-1, 1])

        Returns:
            Tuple of (next_state, observation, reward, done, info)
        """
        # Unnormalize action
        action = self._unnormalize_action(action)
        a_pos, a_ori, a_gripper = action[:3], action[3], action[4]

        # Get current effector state
        mjx_data = state.mjx_data
        effector_pos = mjx_data.site_xpos[self._pinch_site_id]
        effector_mat = mjx_data.site_xmat[self._pinch_site_id].reshape(3, 3)
        effector_quat = mat_to_quat(effector_mat)
        effector_yaw = compute_yaw_from_quat(effector_quat)
        gripper_opening = jnp.clip(mjx_data.qpos[self._gripper_opening_joint_id] / 0.8, 0, 1)

        # Compute target pose
        target_pos = effector_pos + a_pos
        target_yaw = effector_yaw + a_ori
        target_gripper = gripper_opening + a_gripper

        # logger.debug(f"tgt_eff tran: {target_pos}")

        # Clip to workspace bounds
        target_pos = jnp.clip(
            target_pos,
            self.config.workspace_bounds[0],
            self.config.workspace_bounds[1],
        )
        target_yaw = jnp.clip(target_yaw, -jnp.pi, jnp.pi)
        target_gripper = jnp.clip(target_gripper, 0.0, 1.0)

        # Compute target orientation
        target_ori = quat_multiply(quat_from_z_radians(target_yaw), self.config.effector_down_quat)

        # Solve IK for target joint positions
        curr_qpos = mjx_data.qpos[:6]
        # logger.debug("jax curr_qpos: {}".format(curr_qpos))
        qpos_target = self._solve_ik(target_pos, target_ori, curr_qpos)

        # Set control
        ctrl = mjx_data.ctrl
        ctrl = ctrl.at[self._arm_actuator_ids].set(qpos_target)
        ctrl = ctrl.at[self._gripper_actuator_ids].set(255.0 * target_gripper)
        # logger.debug("jax ctrl: {}".format(ctrl))
        mjx_data = mjx_data.replace(ctrl=ctrl)

        # Save previous state
        prev_qpos = mjx_data.qpos
        prev_qvel = mjx_data.qvel
        prev_button_pos = jnp.array(
            [mjx_data.qpos[self._button_joint_qpos_addrs[i]] for i in range(self.config.num_buttons)]
        )

        # Step physics
        mjx_data = self._step_physics(mjx_data)

        # Update button states based on button presses
        new_button_states = self._update_button_states(
            state.button_states,
            prev_button_pos,
            mjx_data.qpos,
        )

        # Compute success
        success = self._compute_success(
            mjx_data,
            new_button_states,
            state.target_button_states,
            state.target_drawer_pos,
            state.target_window_pos,
            state.target_cube_pos,
        )

        # Create new state
        new_state = SceneEnvState(
            mjx_data=mjx_data,
            step_count=state.step_count + 1,
            prev_qpos=prev_qpos,
            prev_qvel=prev_qvel,
            prev_button_pos=prev_button_pos,
            button_states=new_button_states,
            target_button_states=state.target_button_states,
            target_drawer_pos=state.target_drawer_pos,
            target_window_pos=state.target_window_pos,
            target_cube_pos=state.target_cube_pos,
            target_cube_quat=state.target_cube_quat,
            rng=state.rng,
        )

        obs = self._get_observation(new_state)

        # Compute reward
        reward = jnp.where(success, 1.0, 0.0)

        # Done when successful
        done = success

        info = {"success": success}

        return ManipStep(new_state, obs, reward, done, info)

    def _step_physics(self, mjx_data: mjx.Data) -> mjx.Data:
        """Step the physics simulation."""

        # logger.debug(f"jax0 qpos: {mjx_data.qpos}")
        # logger.debug(f"jax0 qvel: {mjx_data.qvel}")
        # mjx_data = mjx.step(self.mjx_model, mjx_data)
        # logger.debug(f"jax1 qpos: {mjx_data.qpos}")
        # logger.debug(f"jax1 qvel: {mjx_data.qvel}")

        # for ii in range(self.config.n_steps - 1):
        #     mjx_data = mjx.step(self.mjx_model, mjx_data)
        #     # logger.debug(f"jax{ii+2} qpos: {mjx_data.qpos}")
        #     # logger.debug(f"jax{ii+2} qvel: {mjx_data.qvel}")

        def step_fn(data, _):
            return mjx.step(self.mjx_model, data), None

        mjx_data, _ = jax.lax.scan(step_fn, mjx_data, None, length=self.config.n_steps)

        # for ii in range(self.config.n_steps):
        #     logger.debug("ii={}, qpos={}".format(ii, mjx_data.qpos))
        #     mjx_data_new = mjx.step(self.mjx_model, mjx_data)
        #     if jnp.any(jnp.isnan(mjx_data_new.qpos)):
        #         ipdb.set_trace()
        #     mjx_data = mjx_data_new
        # logger.debug("done, qpos={}".format(mjx_data.qpos))

        # logger.debug(f"jaxn qpos: {mjx_data.qpos}")
        # logger.debug(f"jaxn qvel: {mjx_data.qvel}")

        return mjx_data

    def _update_button_states(
        self,
        button_states: jax.Array,
        prev_button_pos: jax.Array,
        qpos: jax.Array,
    ) -> jax.Array:
        """Update button states based on button presses."""
        new_states = []
        for i in range(self.config.num_buttons):
            cur_pos = qpos[self._button_joint_qpos_addrs[i]]
            prev_pos = prev_button_pos[i]
            # Button pressed when crossing -0.02 threshold
            pressed = (prev_pos > -0.02) & (cur_pos <= -0.02)
            # Toggle state on press
            new_state = jnp.where(pressed, (button_states[i] + 1) % self.config.num_button_states, button_states[i])
            new_states.append(new_state)
        return jnp.array(new_states)

    def _compute_success(
        self,
        mjx_data: mjx.Data,
        button_states: jax.Array,
        target_button_states: jax.Array,
        target_drawer_pos: jax.Array,
        target_window_pos: jax.Array,
        target_cube_pos: jax.Array,
    ) -> jax.Array:
        """Compute whether the task is successful."""
        # Cube success
        cube_qpos_addr = self._cube_joint_qpos_addrs[0]
        cube_pos = mjx_data.qpos[cube_qpos_addr : cube_qpos_addr + 3]
        cube_success = jnp.linalg.norm(cube_pos - target_cube_pos) <= 0.04

        # Button success
        button_success = jnp.all(button_states == target_button_states)

        # Drawer success
        drawer_pos = mjx_data.qpos[self._drawer_joint_qpos_addr]
        drawer_success = jnp.abs(drawer_pos - target_drawer_pos) <= 0.04

        # Window success
        window_pos = mjx_data.qpos[self._window_joint_qpos_addr]
        window_success = jnp.abs(window_pos - target_window_pos) <= 0.04

        return cube_success & button_success & drawer_success & window_success

    def _solve_ik(
        self,
        target_pos: jax.Array,
        target_quat: jax.Array,
        curr_qpos: jax.Array,
    ) -> jax.Array:
        """Solve inverse kinematics for target end-effector pose.

        Converts from pinch site target to attach site target, then delegates
        to DiffIKControllerJax (same approach as ManipSpaceEnv).

        Args:
            target_pos: Target position for the pinch site
            target_quat: Target orientation (wxyz quaternion) for the pinch site
            curr_qpos: Current arm joint positions (6 DOF)

        Returns:
            Joint positions that achieve the target pose
        """
        # Convert pinch target to attach target using T_pa
        T_pa = SE3(wxyz_xyz=self._T_pa_wxyz_xyz)
        T_wp = SE3.from_rotation_and_translation(SO3(wxyz=target_quat), target_pos)
        T_wa = T_wp.multiply(T_pa)

        target_attach_pos = T_wa.translation()
        target_attach_quat = T_wa.rotation().wxyz

        logger.debug(f"jax T_pa     : {T_pa}")

        # Solve IK using the controller
        return self._ik.solve(
            pos=target_attach_pos,
            quat=target_attach_quat,
            curr_qpos=curr_qpos,
        )

    def _get_observation(self, state: SceneEnvState) -> jax.Array:
        """Compute observation from state."""
        mjx_data = state.mjx_data
        config = self.config

        # Joint positions and velocities
        joint_pos = mjx_data.qpos[:6]
        joint_vel = mjx_data.qvel[:6]

        # End-effector position
        effector_pos = mjx_data.site_xpos[self._pinch_site_id]

        # End-effector yaw
        effector_mat = mjx_data.site_xmat[self._pinch_site_id].reshape(3, 3)
        effector_quat = mat_to_quat(effector_mat)
        effector_yaw = compute_yaw_from_quat(effector_quat)

        # Gripper state
        gripper_opening = jnp.clip(mjx_data.qpos[self._gripper_opening_joint_id] / 0.8, 0, 1)
        gripper_contact = jnp.array([0.0])  # Simplified

        # Build observation
        obs_parts = [
            joint_pos,
            joint_vel,
            (effector_pos - config.xyz_center) * config.xyz_scaler,
            jnp.array([jnp.cos(effector_yaw)]),
            jnp.array([jnp.sin(effector_yaw)]),
            jnp.array([gripper_opening * config.gripper_scaler]),
            gripper_contact,
        ]

        # Cube observations
        for i in range(config.num_cubes):
            cube_qpos_addr = self._cube_joint_qpos_addrs[i]
            cube_pos = mjx_data.qpos[cube_qpos_addr : cube_qpos_addr + 3]
            cube_quat = mjx_data.qpos[cube_qpos_addr + 3 : cube_qpos_addr + 7]
            cube_yaw = compute_yaw_from_quat(cube_quat)
            obs_parts.extend(
                [
                    (cube_pos - config.xyz_center) * config.xyz_scaler,
                    cube_quat,
                    jnp.array([jnp.cos(cube_yaw)]),
                    jnp.array([jnp.sin(cube_yaw)]),
                ]
            )

        # Button observations
        for i in range(config.num_buttons):
            button_state_onehot = jax.nn.one_hot(state.button_states[i], config.num_button_states)
            button_pos = mjx_data.qpos[self._button_joint_qpos_addrs[i] : self._button_joint_qpos_addrs[i] + 1]
            button_vel = mjx_data.qvel[self._button_joint_dof_addrs[i] : self._button_joint_dof_addrs[i] + 1]
            obs_parts.extend(
                [
                    button_state_onehot,
                    button_pos * config.button_scaler,
                    button_vel,
                ]
            )

        # Drawer observations
        drawer_pos = mjx_data.qpos[self._drawer_joint_qpos_addr : self._drawer_joint_qpos_addr + 1]
        drawer_vel = mjx_data.qvel[self._drawer_joint_dof_addr : self._drawer_joint_dof_addr + 1]
        obs_parts.extend(
            [
                drawer_pos * config.drawer_scaler,
                drawer_vel,
            ]
        )

        # Window observations
        window_pos = mjx_data.qpos[self._window_joint_qpos_addr : self._window_joint_qpos_addr + 1]
        window_vel = mjx_data.qvel[self._window_joint_dof_addr : self._window_joint_dof_addr + 1]
        obs_parts.extend(
            [
                window_pos * config.window_scaler,
                window_vel,
            ]
        )

        return jnp.concatenate(obs_parts)

    def _unnormalize_action(self, action: jax.Array) -> jax.Array:
        """Unnormalize action from [-1, 1] to action bounds."""
        return 0.5 * (action + 1) * (self.config.action_high - self.config.action_low) + self.config.action_low

    def _normalize_action(self, action: jax.Array) -> jax.Array:
        """Normalize action to [-1, 1]."""
        action = 2 * (action - self.config.action_low) / (self.config.action_high - self.config.action_low) - 1
        return jnp.clip(action, -1, 1)

    def set_target(
        self,
        state: SceneEnvState,
        target_cube_pos: Optional[jax.Array] = None,
        target_cube_quat: Optional[jax.Array] = None,
        target_button_states: Optional[jax.Array] = None,
        target_drawer_pos: Optional[jax.Array] = None,
        target_window_pos: Optional[jax.Array] = None,
    ) -> SceneEnvState:
        """Set new target for the environment.

        Args:
            state: Current state
            target_cube_pos: Target cube position (optional)
            target_cube_quat: Target cube quaternion (optional)
            target_button_states: Target button states (optional)
            target_drawer_pos: Target drawer position (optional)
            target_window_pos: Target window position (optional)

        Returns:
            Updated state with new targets
        """
        new_target_cube_pos = target_cube_pos if target_cube_pos is not None else state.target_cube_pos
        new_target_cube_quat = target_cube_quat if target_cube_quat is not None else state.target_cube_quat
        new_target_button_states = (
            target_button_states if target_button_states is not None else state.target_button_states
        )
        new_target_drawer_pos = target_drawer_pos if target_drawer_pos is not None else state.target_drawer_pos
        new_target_window_pos = target_window_pos if target_window_pos is not None else state.target_window_pos

        return state._replace(
            target_cube_pos=new_target_cube_pos,
            target_cube_quat=new_target_cube_quat,
            target_button_states=new_target_button_states,
            target_drawer_pos=new_target_drawer_pos,
            target_window_pos=new_target_window_pos,
        )


# Helper functions
def quat_from_z_radians(theta: jax.Array) -> jax.Array:
    """Create quaternion from rotation around z-axis."""
    half_theta = theta / 2.0
    return jnp.array([jnp.cos(half_theta), 0.0, 0.0, jnp.sin(half_theta)])


def compute_yaw_from_quat(quat: jax.Array) -> jax.Array:
    """Compute yaw angle from quaternion."""
    w, x, y, z = quat
    return jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _quat_to_axis_angle(quat: jax.Array) -> jax.Array:
    """Convert quaternion to axis-angle representation."""
    w = quat[0]
    xyz = quat[1:]

    # Handle sign ambiguity
    sign = jnp.where(w < 0, -1.0, 1.0)
    xyz = xyz * sign
    w = w * sign

    norm_xyz = jnp.linalg.norm(xyz)
    angle = 2.0 * jnp.arctan2(norm_xyz, w)

    # Normalize axis
    axis = jnp.where(norm_xyz > 1e-10, xyz / norm_xyz, jnp.array([0.0, 0.0, 1.0]))
    return angle * axis
