"""JAX-compatible ManipSpace environment using MJX.

This module provides a JAX-compatible version of the ManipSpace environment
that uses MuJoCo's MJX backend for hardware-accelerated physics simulation.
The environment is designed for use with JAX-based reinforcement learning
algorithms and supports JIT compilation and vectorization.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from dm_control import mjcf
from mujoco import mjx

from rraa_rl.env.manipspace import mjcf_utils
from rraa_rl.env.manipspace.lie.se3_jax import SE3
from rraa_rl.env.manipspace.lie.so3_jax import SO3, mat_to_quat, quat_multiply, quat_rotate, quat_to_mat


class EnvState(NamedTuple):
    """State of the environment for JAX compatibility."""

    mjx_data: mjx.Data
    step_count: jax.Array
    prev_qpos: jax.Array
    prev_qvel: jax.Array
    rng: jax.Array


class EnvConfig(NamedTuple):
    """Static configuration for the environment."""

    physics_timestep: float
    control_timestep: float
    n_steps: int
    workspace_bounds: jax.Array
    arm_sampling_bounds: jax.Array
    action_low: jax.Array
    action_high: jax.Array
    home_qpos: jax.Array
    effector_down_quat: jax.Array


@dataclass
class ManipSpaceEnvJax:
    """JAX-compatible ManipSpace environment using MJX.

    This environment uses MuJoCo's MJX backend for hardware-accelerated physics
    simulation. It supports JIT compilation and vectorization for efficient
    training with JAX-based RL algorithms.

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

    mj_model: mujoco.MjModel
    mjx_model: mjx.Model
    config: EnvConfig
    _arm_joint_ids: jax.Array
    _arm_actuator_ids: jax.Array
    _gripper_actuator_ids: jax.Array
    _gripper_opening_joint_id: int
    _pinch_site_id: int
    _attach_site_id: int
    _T_pa_wxyz_xyz: jax.Array  # Pinch to attach transformation

    @classmethod
    def create(
        cls,
        physics_timestep: float = 0.002,
        control_timestep: float = 0.05,
    ) -> "ManipSpaceEnvJax":
        """Create a new ManipSpaceEnvJax instance.

        Args:
            physics_timestep: Physics timestep for simulation
            control_timestep: Control timestep (action frequency)

        Returns:
            Initialized ManipSpaceEnvJax instance
        """
        # Build the MJCF model
        desc_dir = Path(__file__).resolve().parent / "descriptions"
        mjcf_model = cls._build_mjcf_model(desc_dir)

        # Compile to MuJoCo model
        xml_str = mjcf_utils.to_string(mjcf_model)
        assets = mjcf_utils.get_assets(mjcf_model)
        mj_model = mujoco.MjModel.from_xml_string(xml_str, assets)
        mj_model.opt.timestep = physics_timestep

        # Convert to MJX model
        mjx_model = mjx.put_model(mj_model, impl="warp")

        # Compute timesteps
        n_steps = int(round(control_timestep / physics_timestep))

        # Define configuration
        home_qpos = jnp.asarray([-jnp.pi / 2, -jnp.pi / 2, jnp.pi / 2, -jnp.pi / 2, -jnp.pi / 2, 0])
        effector_down_quat = jnp.asarray([0.0, 1.0, 0.0, 0.0])  # wxyz
        workspace_bounds = jnp.asarray([[0.25, -0.35, 0.02], [0.6, 0.35, 0.35]])
        arm_sampling_bounds = jnp.asarray([[0.25, -0.35, 0.20], [0.6, 0.35, 0.35]])
        action_range = jnp.array([0.05, 0.05, 0.05, 0.3, 1.0])

        config = EnvConfig(
            physics_timestep=physics_timestep,
            control_timestep=control_timestep,
            n_steps=n_steps,
            workspace_bounds=workspace_bounds,
            arm_sampling_bounds=arm_sampling_bounds,
            action_low=-action_range,
            action_high=action_range,
            home_qpos=home_qpos,
            effector_down_quat=effector_down_quat,
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
        arm_joint_ids = jnp.asarray([mj_model.joint(name).id for name in arm_joint_names])

        arm_actuator_names = [
            "ur5e/shoulder_pan",
            "ur5e/shoulder_lift",
            "ur5e/elbow",
            "ur5e/wrist_1",
            "ur5e/wrist_2",
            "ur5e/wrist_3",
        ]
        arm_actuator_ids = jnp.asarray([mj_model.actuator(name).id for name in arm_actuator_names])

        gripper_actuator_names = ["ur5e/robotiq/fingers_actuator"]
        gripper_actuator_ids = jnp.asarray([mj_model.actuator(name).id for name in gripper_actuator_names])

        gripper_opening_joint_id = mj_model.joint("ur5e/robotiq/right_driver_joint").id
        pinch_site_id = mj_model.site("ur5e/robotiq/pinch").id
        attach_site_id = mj_model.site("ur5e/attachment_site").id

        # Compute T_pa (pinch to attach transformation) at home position
        mj_data = mujoco.MjData(mj_model)
        mj_data.qpos[:6] = np.asarray(home_qpos)
        mujoco.mj_forward(mj_model, mj_data)

        pinch_pos = mj_data.site_xpos[pinch_site_id].copy()
        pinch_mat = mj_data.site_xmat[pinch_site_id].reshape(3, 3).copy()
        attach_pos = mj_data.site_xpos[attach_site_id].copy()
        attach_mat = mj_data.site_xmat[attach_site_id].reshape(3, 3).copy()

        pinch_quat = np.zeros(4)
        attach_quat = np.zeros(4)
        mujoco.mju_mat2Quat(pinch_quat, pinch_mat.ravel())
        mujoco.mju_mat2Quat(attach_quat, attach_mat.ravel())

        # T_pa = T_wp^-1 @ T_wa
        pinch_pose = SE3.from_rotation_and_translation(SO3(wxyz=jnp.asarray(pinch_quat)), jnp.asarray(pinch_pos))
        attach_pose = SE3.from_rotation_and_translation(SO3(wxyz=jnp.asarray(attach_quat)), jnp.asarray(attach_pos))
        T_pa = pinch_pose.inverse().multiply(attach_pose)
        T_pa_wxyz_xyz = T_pa.wxyz_xyz

        return cls(
            mj_model=mj_model,
            mjx_model=mjx_model,
            config=config,
            _arm_joint_ids=arm_joint_ids,
            _arm_actuator_ids=arm_actuator_ids,
            _gripper_actuator_ids=gripper_actuator_ids,
            _gripper_opening_joint_id=gripper_opening_joint_id,
            _pinch_site_id=pinch_site_id,
            _attach_site_id=attach_site_id,
            _T_pa_wxyz_xyz=T_pa_wxyz_xyz,
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

        return arena_mjcf

    @property
    def action_size(self) -> int:
        """Return the action dimension."""
        return 5

    @property
    def observation_size(self) -> int:
        """Return the observation dimension."""
        # joint_pos(6) + joint_vel(6) + effector_pos(3) + cos/sin_yaw(2) + gripper(2)
        return 19

    def reset(self, rng: jax.Array) -> Tuple[EnvState, jax.Array]:
        """Reset the environment.

        Args:
            rng: JAX random key

        Returns:
            Tuple of (state, observation)
        """
        rng, rng_pos, rng_yaw = jax.random.split(rng, 3)

        # Create initial MJX data
        mjx_data = mjx.make_data(self.mjx_model)

        # Sample initial end-effector position
        eff_pos = jax.random.uniform(
            rng_pos,
            shape=(3,),
            minval=self.config.arm_sampling_bounds[0],
            maxval=self.config.arm_sampling_bounds[1],
        )

        # Sample initial yaw
        yaw = jax.random.uniform(rng_yaw, minval=-jnp.pi, maxval=jnp.pi)

        # Compute target orientation
        yaw_quat = _quat_from_z_radians(yaw)
        eff_ori = quat_multiply(yaw_quat, self.config.effector_down_quat)

        # Solve IK for initial joint positions
        qpos_init = self._solve_ik(eff_pos, eff_ori, self.config.home_qpos)

        # Set initial state
        qpos = mjx_data.qpos.at[:6].set(qpos_init)
        mjx_data = mjx_data.replace(qpos=qpos)

        # Forward kinematics
        mjx_data = mjx.forward(self.mjx_model, mjx_data)

        state = EnvState(
            mjx_data=mjx_data,
            step_count=jnp.array(0),
            prev_qpos=mjx_data.qpos,
            prev_qvel=mjx_data.qvel,
            rng=rng,
        )

        obs = self._get_observation(state)
        return state, obs

    def step(
        self,
        state: EnvState,
        action: jax.Array,
    ) -> Tuple[EnvState, jax.Array, jax.Array, jax.Array, Dict[str, Any]]:
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
        effector_yaw = _compute_yaw_from_quat(effector_quat)
        gripper_opening = jnp.clip(mjx_data.qpos[self._gripper_opening_joint_id] / 0.8, 0, 1)

        # Compute target pose
        target_pos = effector_pos + a_pos
        target_yaw = effector_yaw + a_ori
        target_gripper = gripper_opening + a_gripper

        # Clip to workspace bounds
        target_pos = jnp.clip(
            target_pos,
            self.config.workspace_bounds[0],
            self.config.workspace_bounds[1],
        )
        target_yaw = jnp.clip(target_yaw, -jnp.pi, jnp.pi)
        target_gripper = jnp.clip(target_gripper, 0.0, 1.0)

        # Compute target orientation
        target_ori = quat_multiply(_quat_from_z_radians(target_yaw), self.config.effector_down_quat)

        # Solve IK for target joint positions
        curr_qpos = mjx_data.qpos[:6]
        qpos_target = self._solve_ik(target_pos, target_ori, curr_qpos)

        # Set control
        ctrl = mjx_data.ctrl
        ctrl = ctrl.at[self._arm_actuator_ids].set(qpos_target)
        ctrl = ctrl.at[self._gripper_actuator_ids].set(255.0 * target_gripper)
        mjx_data = mjx_data.replace(ctrl=ctrl)

        # Save previous state
        prev_qpos = mjx_data.qpos
        prev_qvel = mjx_data.qvel

        # Step physics
        mjx_data = self._step_physics(mjx_data)

        # Compute observation
        new_state = EnvState(
            mjx_data=mjx_data,
            step_count=state.step_count + 1,
            prev_qpos=prev_qpos,
            prev_qvel=prev_qvel,
            rng=state.rng,
        )
        obs = self._get_observation(new_state)

        # Compute reward (default: 0)
        reward = jnp.array(0.0)

        # Done flag (default: False)
        done = jnp.array(False)

        info = {}

        return new_state, obs, reward, done, info

    def _step_physics(self, mjx_data: mjx.Data) -> mjx.Data:
        """Step the physics simulation."""

        def step_fn(data, _):
            return mjx.step(self.mjx_model, data), None

        mjx_data, _ = jax.lax.scan(step_fn, mjx_data, None, length=self.config.n_steps)
        return mjx_data

    def _solve_ik(
        self,
        target_pos: jax.Array,
        target_quat: jax.Array,
        curr_qpos: jax.Array,
    ) -> jax.Array:
        """Solve inverse kinematics for target end-effector pose.

        Uses a simple iterative IK solver with damped least squares.
        """
        # Convert pinch target to attach target using T_pa
        T_pa = SE3(wxyz_xyz=self._T_pa_wxyz_xyz)
        T_wp = SE3.from_rotation_and_translation(SO3(wxyz=target_quat), target_pos)
        T_wa = T_wp.multiply(T_pa)

        target_attach_pos = T_wa.translation()
        target_attach_quat = T_wa.rotation().wxyz

        # Simple damped least squares IK
        def ik_step(qpos, _):
            # Create temporary data
            mjx_data = mjx.make_data(self.mjx_model)
            qpos_full = mjx_data.qpos.at[:6].set(qpos)
            mjx_data = mjx_data.replace(qpos=qpos_full)

            # Forward kinematics
            mjx_data = mjx.kinematics(self.mjx_model, mjx_data)
            mjx_data = mjx.com_pos(self.mjx_model, mjx_data)

            # Get current attach site pose
            attach_pos = mjx_data.site_xpos[self._attach_site_id]
            attach_mat = mjx_data.site_xmat[self._attach_site_id].reshape(3, 3)
            attach_quat = mat_to_quat(attach_mat)

            # Position error
            pos_err = target_attach_pos - attach_pos

            # Orientation error
            attach_quat_inv = attach_quat * jnp.array([1.0, -1.0, -1.0, -1.0])
            err_quat = quat_multiply(target_attach_quat, attach_quat_inv)
            ori_err = _quat_to_axis_angle(err_quat)

            err = jnp.concatenate([pos_err, ori_err])

            # Compute Jacobian
            jacp, jacr = mjx.jac_site(self.mjx_model, mjx_data, self._attach_site_id)
            jac = jnp.vstack([jacp[:, :6], jacr[:, :6]])

            # Damped least squares
            damping = 1e-6 * jnp.eye(6)
            H = jac @ jac.T + damping
            update = jac.T @ jnp.linalg.solve(H, err)

            # Scale update
            max_change = jnp.radians(45)
            update_max = jnp.max(jnp.abs(update))
            scale = jnp.where(update_max > max_change, max_change / update_max, 1.0)
            update = update * scale

            new_qpos = qpos + update
            return new_qpos, None

        # Run IK iterations
        qpos, _ = jax.lax.scan(ik_step, curr_qpos, None, length=10)
        return qpos

    def _get_observation(self, state: EnvState) -> jax.Array:
        """Compute observation from state."""
        mjx_data = state.mjx_data

        # Joint positions and velocities
        joint_pos = mjx_data.qpos[:6]
        joint_vel = mjx_data.qvel[:6]

        # End-effector position
        effector_pos = mjx_data.site_xpos[self._pinch_site_id]

        # End-effector yaw
        effector_mat = mjx_data.site_xmat[self._pinch_site_id].reshape(3, 3)
        effector_quat = mat_to_quat(effector_mat)
        effector_yaw = _compute_yaw_from_quat(effector_quat)

        # Gripper state
        gripper_opening = jnp.clip(mjx_data.qpos[self._gripper_opening_joint_id] / 0.8, 0, 1)
        gripper_contact = jnp.array([0.0])  # Simplified - contact detection is complex in MJX

        # Normalize
        xyz_center = jnp.array([0.425, 0.0, 0.0])
        xyz_scaler = 10.0
        gripper_scaler = 3.0

        obs = jnp.concatenate(
            [
                joint_pos,
                joint_vel,
                (effector_pos - xyz_center) * xyz_scaler,
                jnp.array([jnp.cos(effector_yaw)]),
                jnp.array([jnp.sin(effector_yaw)]),
                jnp.array([gripper_opening * gripper_scaler]),
                gripper_contact,
            ]
        )
        return obs

    def _unnormalize_action(self, action: jax.Array) -> jax.Array:
        """Unnormalize action from [-1, 1] to action bounds."""
        return 0.5 * (action + 1) * (self.config.action_high - self.config.action_low) + self.config.action_low

    def _normalize_action(self, action: jax.Array) -> jax.Array:
        """Normalize action to [-1, 1]."""
        action = 2 * (action - self.config.action_low) / (self.config.action_high - self.config.action_low) - 1
        return jnp.clip(action, -1, 1)


# Helper functions
def _quat_from_z_radians(theta: jax.Array) -> jax.Array:
    """Create quaternion from rotation around z-axis."""
    half_theta = theta / 2.0
    return jnp.array([jnp.cos(half_theta), 0.0, 0.0, jnp.sin(half_theta)])


def _compute_yaw_from_quat(quat: jax.Array) -> jax.Array:
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
