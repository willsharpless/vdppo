from __future__ import annotations

import functools as ft
import pathlib
import pickle
from functools import partial
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Tuple

import einops as ei
import flax.linen as nn
import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax_dataclasses as jdc
import matplotlib.pyplot as plt
import mujoco
import mujoco.mjx.third_party.mujoco_warp as mjw
import numpy as np
from attrs import define
from dm_control import mjcf
from jaxtyping import PRNGKeyArray
from loguru import logger
from matplotlib.colors import to_rgba
from mujoco import mjx

from rraa_rl.emoji_util import plot_emoji
from rraa_rl.envs import mjx_patch
from rraa_rl.envs.manipspace import mjcf_utils
from rraa_rl.envs.manipspace.diff_ik_jax import DiffIKControllerJax
from rraa_rl.envs.manipspace.lie.se3_jax import SE3
from rraa_rl.envs.manipspace.lie.so3_jax import SO3, mat_to_quat, quat_multiply, quat_rotate, quat_to_mat, so3_exp
from rraa_rl.envs.manipspace.scene_env_jax import (SceneEnvConfig, SceneEnvJax, compute_yaw_from_quat,
                                                   quat_from_z_radians)
from rraa_rl.geometry import AABB, LineSegment, dist_pt_to_aabb, segment_intersects_aabb
from rraa_rl.jax_types import BoolScalar
from rraa_rl.jax_utils import softmaximum, softminimum, tree_stack
from rraa_rl.src.env.general_task.env import (AugObs, BaseEnv, EnvCfg, EnvStep, EnvUsingBase, StateWithTemporalNode,
                                              StaticTemporalNodeMixin, StaticTemporalNodeMixinCfg)


@define(slots=False)
class SceneBaseCfg(EnvCfg, StaticTemporalNodeMixinCfg):
    specification: str = "F( drawer_open && F( cube_in_drawer )) && G F( drawer_closed )"

    n_actions_px: int = 7
    n_actions_py: int = 7
    n_actions_pz: int = 7
    n_actions_rot: int = 5
    n_actions_grip: int = 3

    trunc_steps: int = 400


@jdc.pytree_dataclass
class SceneBaseState:
    mjx_data: mjx.Data

    prev_button_pos: jax.Array  # Previous button joint positions for detecting presses

    # Current button states (binary), 0 is locked.
    button_states: jax.Array

    # target_button_states: jax.Array  # Target button states
    # target_drawer_pos: jax.Array  # Target drawer position
    # target_window_pos: jax.Array  # Target window position
    # target_cube_pos: jax.Array  # Target cube position
    # target_cube_quat: jax.Array  # Target cube quaternion

    steps: jnp.ndarray


@jdc.pytree_dataclass
class SceneBaseMinState:
    """SceneBaseState but with only the minimal necessary fields. For saving memory."""

    qpos: jnp.ndarray
    qvel: jnp.ndarray
    prev_button_pos: jax.Array  # Previous button joint positions for detecting presses
    button_states: jax.Array
    steps: jnp.ndarray


@jdc.pytree_dataclass
class SceneBaseResetMinimal:
    qpos: jnp.ndarray
    qvel: jnp.ndarray
    button_states: jax.Array


_data_source = None


class SceneData:
    def __init__(self) -> None:
        # Load data of qpos and qvel.
        pkl_path = pathlib.Path(__file__).parent / "manipspace/collected_states.pkl"
        with open(pkl_path, "rb") as f:
            data: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = pickle.load(f)

        # Key is (agent_name, motion_label)
        # agent_labels_to_save = {
        #     "drawer": ["approach", "grasp_start", "grasp_end", "move", "release"],
        #     "cube": ["pick_start", "pick_end", "place", "place_start", "place_end"],
        # }
        # Contains (qpos, qvel) tuples.
        self.data = data

        # Stack the qpos and qvel for each (agent, label).
        self.stacked_data: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        for key, lst in data.items():
            qpos_list, qvel_list = zip(*lst)
            qpos_stack = np.stack(qpos_list, axis=0)
            qvel_stack = np.stack(qvel_list, axis=0)
            self.stacked_data[key] = (qpos_stack, qvel_stack)

        # Concatenate everything together.
        qpos_all = []
        qvel_all = []
        for key, (qpos_stack, qvel_stack) in self.stacked_data.items():
            qpos_all.append(qpos_stack)
            qvel_all.append(qvel_stack)
        self.qpos_all = np.concatenate(qpos_all, axis=0)
        self.qvel_all = np.concatenate(qvel_all, axis=0)

    @property
    def n_total(self) -> int:
        return self.qpos_all.shape[0]

    @property
    def qpos_all_jax(self) -> jax.Array:
        return jnp.asarray(self.qpos_all)

    @property
    def qvel_all_jax(self) -> jax.Array:
        return jnp.asarray(self.qvel_all)

    @staticmethod
    def get():
        global _data_source
        if _data_source is None:
            _data_source = SceneData()
        return _data_source


class SceneBase(BaseEnv):
    def __init__(self, cfg: SceneBaseCfg = SceneBaseCfg()):
        super().__init__()

        # physics_timestep: float = 0.002
        physics_timestep: float = 0.001
        control_timestep: float = 0.05

        self.cfg = cfg

        # Build the MJCF model
        self._desc_dir = Path(__file__).resolve().parent / "manipspace/descriptions"
        mjcf_model = SceneEnvJax._build_mjcf_model(self._desc_dir)

        # Compile to MuJoCo model
        xml_str = mjcf_utils.to_string(mjcf_model)
        assets = mjcf_utils.get_assets(mjcf_model)
        self.mj_model = mujoco.MjModel.from_xml_string(xml_str, assets)
        self.mj_model.opt.timestep = physics_timestep

        self.make_data_args = dict(njmax=500)

        #
        # # mujoco.mj_saveLastXML("scene_base.xml", self.mj_model)
        # buffer_size = mujoco.mj_sizeModel(self.mj_model)
        # buffer = np.empty(shape=buffer_size, dtype=np.uint8)
        # mujoco.mj_saveModel(self.mj_model, None, buffer)
        # # Save buffer to file.
        # with open("scene_base_model.mjb", "wb") as f:
        #     f.write(buffer)
        #
        # logger.success("Saved scene_base_model.mjb")
        # exit(0)

        # Unlock the drawer joint, lock the window joint.
        self.mj_model.joint("drawer_slide").damping[0] = 2.0
        self.mj_model.joint("window_slide").damping[0] = 1e6

        self.impl = "warp"

        # Convert to MJX model
        self.mjx_model = mjx.put_model(self.mj_model, impl=self.impl)

        m = self.mjx_model
        opt = m.opt

        broadphase, filter = (
            mjw.BroadphaseType(opt._impl.broadphase).name,
            mjw.BroadphaseFilter(opt._impl.broadphase_filter).name,
        )
        solver, cone = mjw.SolverType(opt.solver).name, mjw.ConeType(m.opt.cone).name
        integrator = mjw.IntegratorType(m.opt.integrator).name
        iterations, ls_iterations = m.opt.iterations, m.opt.ls_iterations
        ls_str = f"{'parallel' if m.opt._impl.ls_parallel else 'iterative'} linesearch iterations: {ls_iterations}"

        print(
            f"  nbody: {m.nbody} nv: {m.nv} ngeom: {m.ngeom} nu: {m.nu} is_sparse: {m.opt._impl.is_sparse}\n"
            f"  broadphase: {broadphase} broadphase_filter: {filter}\n"
            f"  solver: {solver} cone: {cone} iterations: {iterations} {ls_str}\n"
            f"  integrator: {integrator} graph_conditional: {m.opt._impl.graph_conditional}"
        )

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
        )

        self.scene_data = SceneData.get()

    @property
    def n_agents(self) -> int:
        return 1

    @property
    def value_lims(self):
        return -1, 1

    @property
    def n_actions_per_agent(self) -> list[list[int]]:
        """
        Original: [ Δpx, Δpy, Δpz, Δrot, Δgrip]
        """
        cfg = self.cfg
        n_actions = [cfg.n_actions_px, cfg.n_actions_py, cfg.n_actions_pz, cfg.n_actions_grip, cfg.n_actions_rot]
        return [n_actions]

    def make_data(self, qpos=None, qvel=None, forward: bool = True):
        data = mjx.make_data(self.mj_model, impl=self.impl, **self.make_data_args)

        modified = False
        if qpos is not None:
            data = data.replace(qpos=qpos)
            modified = True

        if qvel is not None:
            data = data.replace(qvel=qvel)
            modified = True

        if modified and forward:
            data = mjx.forward(self.mjx_model, data)

        return data

    def to_minstate(self, state: SceneBaseState) -> SceneBaseMinState:
        return SceneBaseMinState(
            qpos=state.mjx_data.qpos,
            qvel=state.mjx_data.qvel,
            prev_button_pos=state.prev_button_pos,
            button_states=state.button_states,
            steps=state.steps,
        )

    def from_minstate(self, minstate: SceneBaseMinState) -> SceneBaseState:
        mjx_data = self.make_data()
        mjx_data = mjx_data.replace(
            qpos=minstate.qpos,
            qvel=minstate.qvel,
        )
        # Forward the model to compute derived quantities
        mjx_data = mjx.forward(self.mjx_model, mjx_data)
        return SceneBaseState(
            mjx_data=mjx_data,
            prev_button_pos=minstate.prev_button_pos,
            button_states=minstate.button_states,
            steps=minstate.steps,
        )

    def _action_to_controls(self, action_list: list[jnp.ndarray]):
        assert len(action_list) == 1
        action = action_list[0]
        assert action.shape == (5,) and action.dtype == jnp.int32

        # Convert it to a normalized continuous control in [-1, 1]
        cfg = self.cfg
        action_px = jnp.linspace(-1, 1, cfg.n_actions_px)[action[0]]
        action_py = jnp.linspace(-1, 1, cfg.n_actions_py)[action[1]]
        action_pz = jnp.linspace(-1, 1, cfg.n_actions_pz)[action[2]]
        action_rot = jnp.linspace(-1, 1, cfg.n_actions_rot)[action[3]]
        action_grip = jnp.linspace(-1, 1, cfg.n_actions_grip)[action[4]]
        action_continuous = jnp.array([action_px, action_py, action_pz, action_rot, action_grip])

        # Scale to actual control ranges
        control_range = self.config.action_high - self.config.action_low
        control = self.config.action_low + 0.5 * (action_continuous + 1.0) * control_range

        return control

    def get_effector_state(self, mjx_data: mjx.Data):
        effector_pos = mjx_data.site_xpos[self._pinch_site_id]
        effector_mat = mjx_data.site_xmat[self._pinch_site_id].reshape(3, 3)
        effector_quat = mat_to_quat(effector_mat)
        effector_yaw = compute_yaw_from_quat(effector_quat)
        gripper_opening = jnp.clip(mjx_data.qpos[self._gripper_opening_joint_id] / 0.8, 0, 1)

        return effector_pos, effector_yaw, gripper_opening

    def set_control(self, mjx_data: mjx.Data, qpos_target: jnp.ndarray, target_gripper: jnp.ndarray):
        ctrl = mjx_data.ctrl
        ctrl = ctrl.at[self._arm_actuator_ids].set(qpos_target)
        ctrl = ctrl.at[self._gripper_actuator_ids].set(255.0 * target_gripper)
        return mjx_data.replace(ctrl=ctrl)

    def _solve_ik(
        self,
        target_pos: jax.Array,
        target_quat: jax.Array,
        curr_qpos: jax.Array,
    ) -> jax.Array:
        # Convert pinch target to attach target using T_pa
        T_pa = SE3(wxyz_xyz=self._T_pa_wxyz_xyz)
        T_wp = SE3.from_rotation_and_translation(SO3(wxyz=target_quat), target_pos)
        T_wa = T_wp.multiply(T_pa)

        target_attach_pos = T_wa.translation()
        target_attach_quat = T_wa.rotation().wxyz

        # Solve IK using the controller
        return self._ik.solve(
            pos=target_attach_pos,
            quat=target_attach_quat,
            curr_qpos=curr_qpos,
        )

    def _step_physics(self, mjx_data: mjx.Data) -> mjx.Data:
        def step_fn(data, _):
            return mjx.step(self.mjx_model, data), None

        mjx_data, _ = jax.lax.scan(step_fn, mjx_data, None, length=self.config.n_steps)
        return mjx_data

    def get_is_button_pressed(self, qpos: jnp.ndarray, prev_button_pos: jnp.ndarray) -> jnp.ndarray:
        pressed = []
        for i in range(self.config.num_buttons):
            cur_pos = qpos[self._button_joint_qpos_addrs[i]]
            prev_pos = prev_button_pos[i]
            is_pressed = (prev_pos > -0.02) & (cur_pos <= -0.02)
            pressed.append(is_pressed)
        return jnp.array(pressed)

    def _update_button_states(
        self,
        button_states: jax.Array,
        prev_button_pos: jax.Array,
        qpos: jax.Array,
    ) -> jax.Array:
        b_is_pressed = self.get_is_button_pressed(qpos, prev_button_pos)
        b_state_toggle = (button_states + 1) % self.config.num_button_states
        # If pressed, toggle the button state.
        b_state_new = jnp.where(b_is_pressed, b_state_toggle, button_states)
        return b_state_new
        # new_states = []
        # for i in range(self.config.num_buttons):
        #     cur_pos = qpos[self._button_joint_qpos_addrs[i]]
        #     prev_pos = prev_button_pos[i]
        #     # Button pressed when crossing -0.02 threshold
        #     pressed = (prev_pos > -0.02) & (cur_pos <= -0.02)
        #     # Toggle state on press
        #     new_state = jnp.where(pressed, (button_states[i] + 1) % self.config.num_button_states, button_states[i])
        #     new_states.append(new_state)
        # return jnp.array(new_states)

    def next_state(self, state: SceneBaseState, control: jnp.ndarray):
        assert control.shape == (5,)
        delta_pos, delta_ori, delta_gripper = control[:3], control[3], control[4]

        effector_pos, effector_yaw, gripper_opening = self.get_effector_state(state.mjx_data)

        # Compute target pose
        target_pos = effector_pos + delta_pos
        target_yaw = effector_yaw + delta_ori
        target_gripper = gripper_opening + delta_gripper

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
        curr_qpos = state.mjx_data.qpos[:6]
        qpos_target = self._solve_ik(target_pos, target_ori, curr_qpos)

        # Set control
        mjx_data = self.set_control(state.mjx_data, qpos_target, target_gripper)

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

        # Update state
        with jdc.copy_and_mutate(state) as state_new:
            state_new.mjx_data = mjx_data
            state_new.prev_button_pos = prev_button_pos
            state_new.button_states = new_button_states
            state_new.steps += 1

        info_dyn = {}
        return state_new, info_dyn

    def step(self, state: SceneBaseState, action: list[jnp.ndarray]) -> EnvStep:
        controls = self._action_to_controls(action)
        state_new, info_dyn = self.next_state(state, controls)
        obs_new = self.get_obs(state_new)

        predicates = self.get_predicates(state_new)
        term = False
        trunc = state_new.steps >= self.cfg.trunc_steps

        info = {"age": state_new.steps} | info_dyn
        return EnvStep(state_new, obs_new, predicates, term, trunc, info)

    def get_obs_and_names(self, state: SceneBaseState):
        def fl(lst: list[list[str]]) -> list[str]:
            return [item for sublist in lst for item in sublist]

        mjx_data = state.mjx_data
        config = self.config

        # Joint positions and velocities
        joint_pos = mjx_data.qpos[:6]
        joint_vel = mjx_data.qvel[:6]
        joint_pos_names = [f"joint_pos_{ii}" for ii in range(6)]
        joint_vel_names = [f"joint_vel_{ii}" for ii in range(6)]

        # End-effector position
        effector_pos = mjx_data.site_xpos[self._pinch_site_id]
        effector_pos_names = [f"eff_p{axis}" for axis in ["x", "y", "z"]]

        # End-effector yaw
        effector_mat = mjx_data.site_xmat[self._pinch_site_id].reshape(3, 3)
        effector_quat = mat_to_quat(effector_mat)
        effector_yaw = compute_yaw_from_quat(effector_quat)
        effector_yaw_names = ["eff_yaw"]

        # Gripper state
        gripper_opening = jnp.clip(mjx_data.qpos[self._gripper_opening_joint_id] / 0.8, 0, 1)
        gripper_opening_names = ["gripper_opening"]

        gripper_contact = jnp.array([0.0])  # Simplified
        gripper_contact_names = ["gripper_contact"]

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
        obs_names = (
            joint_pos_names
            + joint_vel_names
            + effector_pos_names
            + effector_yaw_names
            + gripper_opening_names
            + gripper_contact_names
        )

        # Cube observations
        assert config.num_cubes == 1
        for i in range(config.num_cubes):
            cube_qpos_addr = self._cube_joint_qpos_addrs[i]
            cube_pos = mjx_data.qpos[cube_qpos_addr : cube_qpos_addr + 3]
            cube_pos_names = [f"cube{i}_p{axis}" for axis in ["x", "y", "z"]]
            cube_quat = mjx_data.qpos[cube_qpos_addr + 3 : cube_qpos_addr + 7]
            cube_quat_names = [f"cube{i}_qw", f"cube{i}_qx", f"cube{i}_qy", f"cube{i}_qz"]
            cube_yaw = compute_yaw_from_quat(cube_quat)
            cube_yaw_names = [f"cube{i}_yaw"]
            obs_parts.extend(
                [
                    (cube_pos - config.xyz_center) * config.xyz_scaler,
                    cube_quat,
                    jnp.array([jnp.cos(cube_yaw)]),
                    jnp.array([jnp.sin(cube_yaw)]),
                ]
            )
            obs_names.extend(cube_pos_names + cube_quat_names + cube_yaw_names)

        # # Button observations
        # for i in range(config.num_buttons):
        #     button_state_onehot = jax.nn.one_hot(state.button_states[i], config.num_button_states)
        #     button_state_names = [f"btn{i}_state_{s}" for s in range(config.num_button_states)]
        #     button_pos = mjx_data.qpos[self._button_joint_qpos_addrs[i] : self._button_joint_qpos_addrs[i] + 1]
        #     button_pos_names = [f"btn{i}_pos"]
        #     button_vel = mjx_data.qvel[self._button_joint_dof_addrs[i] : self._button_joint_dof_addrs[i] + 1]
        #     button_vel_names = [f"btn{i}_vel"]
        #     obs_parts.extend(
        #         [
        #             button_state_onehot,
        #             button_pos * config.button_scaler,
        #             button_vel,
        #         ]
        #     )
        #     obs_names.extend(button_state_names + button_pos_names + button_vel_names)

        # Drawer observations
        drawer_pos = mjx_data.qpos[self._drawer_joint_qpos_addr : self._drawer_joint_qpos_addr + 1]
        drawer_pos_names = ["drawer_pos"]
        drawer_vel = mjx_data.qvel[self._drawer_joint_dof_addr : self._drawer_joint_dof_addr + 1]
        drawer_vel_names = ["drawer_vel"]
        obs_parts.extend(
            [
                drawer_pos * config.drawer_scaler,
                drawer_vel,
            ]
        )
        obs_names.extend(drawer_pos_names + drawer_vel_names)

        # Window observations
        window_pos = mjx_data.qpos[self._window_joint_qpos_addr : self._window_joint_qpos_addr + 1]
        window_pos_names = ["window_pos"]
        window_vel = mjx_data.qvel[self._window_joint_dof_addr : self._window_joint_dof_addr + 1]
        window_vel_names = ["window_vel"]
        obs_parts.extend(
            [
                window_pos * config.window_scaler,
                window_vel,
            ]
        )
        obs_names.extend(window_pos_names + window_vel_names)

        return jnp.concatenate(obs_parts), obs_names

    def is_in_drawer(self, mjx_data: mjx.Data, obj_pos: jnp.ndarray):
        """Check if the object is in the drawer."""
        drawer_pos_y = mjx_data.site_xpos[self._drawer_site_id][1]
        drawer_low = jnp.array([0.21, drawer_pos_y - 0.27, 0.0])
        drawer_high = jnp.array([0.45, drawer_pos_y - 0.07, 0.15])
        return jnp.all(drawer_low <= obj_pos) & jnp.all(obj_pos <= drawer_high)

    def get_cube_pos(self, mjx_data: mjx.Data) -> jnp.ndarray:
        # jid = self.mjx_model.name2id("object_joint_0", mujoco.mjtObj.mjOBJ_JOINT)
        jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "object_joint_0")
        qpos_adr = self.mjx_model.jnt_qposadr[jid]
        # cube_pos = mjx_data.joint("object_joint_0").qpos[:3]
        cube_pos = mjx_data.qpos[qpos_adr : qpos_adr + 3]
        return cube_pos

    def is_cube_in_drawer(self, mjx_data: mjx.Data) -> BoolScalar:
        cube_pos = self.get_cube_pos(mjx_data)
        return self.is_in_drawer(mjx_data, cube_pos)

    def get_drawer_predicates(self, mjx_data: mjx.Data):
        drawer_pos = mjx_data.qpos[self._drawer_joint_qpos_addr]
        is_drawer_closed = drawer_pos >= -0.04
        is_drawer_open = drawer_pos < -0.12
        return is_drawer_open, is_drawer_closed

        # if force_drawer_open == "1" or self._data.joint("drawer_slide").qpos[0] >= -0.08:  # Drawer closed.
        #     logger.info("drawer closed, opening!")
        #     self._target_drawer_pos = -0.16
        # else:  # Drawer open.
        #     logger.info("drawer open, closing!")
        #     self._target_drawer_pos = 0.0
        # self._model.site("drawer_handle_center_target").pos[1] = self._target_drawer_pos

    # def is_cube_grasped(self, mjx_data: mjx.Data) -> BoolScalar:

    def get_predicates_bool(self, state: SceneBaseState) -> dict[str, BoolScalar]:
        is_drawer_open, is_drawer_closed = self.get_drawer_predicates(state.mjx_data)
        predicates = {
            # "cube_grasped": self.is_cube_grasped(state.mjx_data),
            "cube_in_drawer": self.is_cube_in_drawer(state.mjx_data),
            "drawer_open": is_drawer_open,
            "drawer_closed": is_drawer_closed,
        }
        return predicates

    def get_predicates_float(self, state: SceneBaseState) -> dict[str, jnp.ndarray]:
        return {}

    def get_predicates(self, state: SceneBaseState) -> dict:
        predicates_bool = self.get_predicates_bool(state)
        predicates = {k: jnp.where(v, 1.0, -1.0) for k, v in predicates_bool.items()}
        predicates_float = self.get_predicates_float(state)
        return predicates | predicates_float

    def get_button_pos(self, mjx_data: mjx.Data) -> jnp.ndarray:
        button_pos = jnp.array(
            [mjx_data.qpos[self._button_joint_qpos_addrs[i]] for i in range(self.config.num_buttons)]
        )
        return button_pos

    def _reset_uniform(self, key: PRNGKeyArray):
        rng_arm, rng_yaw, rng_cube_xy, rng_cube_yaw, rng_buttons, rng_drawer = jr.split(key, 6)

        qpos = jnp.zeros(self.mjx_model.nq)
        qvel = jnp.zeros(self.mjx_model.nv)

        # Sample initial arm end-effector position
        eff_pos = jr.uniform(
            rng_arm,
            shape=(3,),
            minval=self.config.arm_sampling_bounds[0],
            maxval=self.config.arm_sampling_bounds[1],
        )

        # Sample initial arm yaw
        yaw = jr.uniform(rng_yaw, minval=-jnp.pi, maxval=jnp.pi)

        # Compute target orientation
        yaw_quat = quat_from_z_radians(yaw)
        eff_ori = quat_multiply(yaw_quat, self.config.effector_down_quat)

        # Solve IK for initial arm joint positions
        qpos_arm = self._solve_ik(eff_pos, eff_ori, self.config.home_qpos)

        # Set arm joint positions
        qpos = qpos.at[:6].set(qpos_arm)

        # Sample cube position and orientation
        cube_xy = jr.uniform(
            rng_cube_xy,
            shape=(2,),
            minval=self.config.object_sampling_bounds[0],
            maxval=self.config.object_sampling_bounds[1],
        )
        cube_pos = jnp.array([cube_xy[0], cube_xy[1], 0.02])
        cube_yaw = jr.uniform(rng_cube_yaw, minval=0.0, maxval=2 * jnp.pi)
        cube_quat = quat_from_z_radians(cube_yaw)

        # Set cube position (qpos address for object_joint_0)
        cube_qpos_addr = self._cube_joint_qpos_addrs[0]
        qpos = qpos.at[cube_qpos_addr : cube_qpos_addr + 3].set(cube_pos)
        qpos = qpos.at[cube_qpos_addr + 3 : cube_qpos_addr + 7].set(cube_quat)

        # Sample button states (0 or 1 for each button)
        # button_states = jr.randint(
        #     rng_buttons, shape=(self.config.num_buttons,), minval=0, maxval=self.config.num_button_states
        # )
        button_states = jnp.array([0, 1])

        # Sample drawer position
        drawer_pos = jr.uniform(rng_drawer, minval=-0.16, maxval=0.0)
        qpos = qpos.at[self._drawer_joint_qpos_addr].set(drawer_pos)

        # Sample window position
        # window_pos = jr.uniform(rng_window, minval=0.0, maxval=0.2)
        window_pos = 0.0
        qpos = qpos.at[self._window_joint_qpos_addr].set(window_pos)

        # # Update mjx_data with new qpos
        # mjx_data = mjx_data.replace(qpos=qpos)

        # # Initialize targets (same as current for simplicity - can be overridden)
        # target_button_states = button_states.copy()
        # target_drawer_pos = jnp.array(drawer_pos)
        # target_window_pos = jnp.array(window_pos)
        # target_cube_pos = cube_pos.copy()
        # target_cube_quat = cube_quat.copy()

        # Make sure prev_qpos and prev_qvel are backed by different arrays so we can donate.
        state = SceneBaseResetMinimal(
            qpos=qpos,
            qvel=qvel,
            button_states=button_states,
        )
        # state = SceneBaseState(
        #     mjx_data=mjx_data,
        #     prev_button_pos=prev_button_pos,
        #     button_states=button_states,
        #     # target_button_states=target_button_states,
        #     # target_drawer_pos=target_drawer_pos,
        #     # target_window_pos=target_window_pos,
        #     # target_cube_pos=target_cube_pos,
        #     # target_cube_quat=target_cube_quat,
        #     steps=jnp.array(0),
        # )
        return state

    def _reset_from_clean_data(self, key: PRNGKeyArray):
        # Sample a random qpos and qvel from the clean data.
        n_total = self.scene_data.n_total
        idx = jr.randint(key, shape=(), minval=0, maxval=n_total)

        qpos_sample = self.scene_data.qpos_all_jax[idx]
        qvel_sample = self.scene_data.qvel_all_jax[idx]

        # # Create initial MJX data
        # mjx_data = self.make_data()
        # mjx_data = mjx_data.replace(qpos=qpos_sample, qvel=qvel_sample)
        #
        # if forward:
        #     # Forward kinematics to populate.
        #     mjx_data = mjx.forward(self.mjx_model, mjx_data)

        # Get initial button positions
        button_states = jnp.array([0, 1])

        state = SceneBaseResetMinimal(
            qpos=qpos_sample,
            qvel=qvel_sample,
            button_states=button_states,
        )
        return state

    def _reset_from_noisy_data(self, key: PRNGKeyArray):
        # First sample from clean data.
        key_data, key_noise = jr.split(key)
        state = self._reset_from_clean_data(key_data)

        # Create the mjx, since we need to know where the end effector is.
        mjx_data = self.make_data(state.qpos, state.qvel, forward=True)

        # If the end effector is not close to the cube, add noise to the cube position.
        effector_pos, _, _ = self.get_effector_state(mjx_data)
        cube_pos = self.get_cube_pos(mjx_data)
        dist_ee_cube = jnp.linalg.norm(effector_pos - cube_pos)
        is_ee_far = dist_ee_cube > 0.3

        # Add noise to qpos and qvel.
        std_qpos = jnp.full(state.qpos.shape, 1e-3)

        # Larger noise to cube position if end effector is far.
        cube_pos_xy_noise = jnp.where(is_ee_far, 0.1, 1e-3)
        for i in range(self.config.num_cubes):
            cube_qpos_addr = self._cube_joint_qpos_addrs[i]
            std_qpos = std_qpos.at[cube_qpos_addr : cube_qpos_addr + 2].set(cube_pos_xy_noise)
        noise_qpos = jr.normal(key_noise, shape=mjx_data.qpos.shape) * std_qpos

        std_qvel = jnp.full(mjx_data.qvel.shape, 1e-2)
        # Larger noise to drawer joint vel.
        std_qvel = std_qvel.at[self._drawer_joint_dof_addr].set(1e-1)
        noise_qvel = jr.normal(key_noise, shape=mjx_data.qvel.shape) * std_qvel

        qpos = state.qpos + noise_qpos
        qvel = state.qvel + noise_qvel

        with jdc.copy_and_mutate(state) as state_new:
            state_new.qpos = qpos
            state_new.qvel = qvel

        return state_new

    def from_reset_minimal(self, state_min: SceneBaseResetMinimal) -> SceneBaseState:
        mjx_data = self.make_data(state_min.qpos, state_min.qvel, forward=True)
        return SceneBaseState(
            mjx_data=mjx_data,
            prev_button_pos=self.get_button_pos(mjx_data),
            button_states=state_min.button_states,
            steps=jnp.array(0),
        )

    def reset(self, key: PRNGKeyArray):
        p_reset_data_clean = 0.25
        p_reset_data_noisy = 0.5
        p_reset_uniform = 1 - p_reset_data_clean - p_reset_data_noisy

        key_which, key_reset = jr.split(key)
        probs = jnp.array([p_reset_data_clean, p_reset_data_noisy, p_reset_uniform])
        which_reset = jr.choice(key_which, a=3, p=probs)

        state_clean = self._reset_from_clean_data(key_reset)
        state_noisy = self._reset_from_noisy_data(key_reset)
        state_uniform = self._reset_uniform(key_reset)
        stack_list = [state_clean, state_noisy, state_uniform]
        assert len(probs) == len(stack_list)

        state_stack = tree_stack(stack_list)
        state_minimal: SceneBaseResetMinimal = jtu.tree_map(lambda x: x[which_reset], state_stack)

        # We do this to avoid having to index into warp data.
        state_full = self.from_reset_minimal(state_minimal)
        return state_full

        # return state


class ManipScene(StaticTemporalNodeMixin, EnvUsingBase):
    Cfg = SceneBaseCfg
    State = StateWithTemporalNode[SceneBaseState]

    def __init__(self, cfg: SceneBaseCfg):
        self.cfg = cfg
        base_env = SceneBase(cfg)
        EnvUsingBase.__init__(self, cfg, self.specification, base_env)
        StaticTemporalNodeMixin.__init__(self, cfg)
        self.base = base_env

    @ft.partial(jax.jit, static_argnames=("self", "batch_size", "init"))
    def reset_batch(self, key: PRNGKeyArray, batch_size: int, init: bool = False) -> StateWithTemporalNode:
        key_reset, key_steps = jr.split(key)
        b_state: StateWithTemporalNode[SceneBaseState] = super().reset_batch(key, batch_size)

        if init:
            # Randomize the initial timestep.
            with jdc.copy_and_mutate(b_state) as b_state_new:
                b_state_new.base.steps = jr.randint(key_steps, (batch_size,), 0, self.base.cfg.trunc_steps)
        else:
            b_state_new = b_state

        return b_state_new

    @property
    def specification(self):
        return self.cfg.specification
