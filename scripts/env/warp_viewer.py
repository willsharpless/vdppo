# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""mjwarp-viewer: load and simulate an MJCF with MuJoCo Warp.

Usage: mjwarp-viewer <mjcf XML path> [flags]

Example:
  mjwarp-viewer benchmark/humanoid/humanoid.xml -o "opt.solver=cg"
"""

import copy
import enum
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

import ipdb

import mujoco
import mujoco.mjx.third_party.mujoco_warp as mjw
import mujoco.viewer
import numpy as np
import warp as wp
from absl import app, flags
from dm_control import mjcf
from etils import epath
# mjwarp-viewer has priviledged access to a few internal methods
from mujoco.mjx.third_party.mujoco_warp._src.io import find_keys, make_trajectory, override_model
from ogbench.manipspace import lie, mjcf_utils

from vdppo.env.manipspace.diff_ik import DiffIKController


class EngineOptions(enum.IntEnum):
    """Engine option."""

    WARP = 0
    C = 1


_CLEAR_KERNEL_CACHE = flags.DEFINE_bool("clear_kernel_cache", False, "Clear kernel cache (to calculate full JIT time)")
_ENGINE = flags.DEFINE_enum_class("engine", EngineOptions.WARP, EngineOptions, "Simulation engine")
_NCONMAX = flags.DEFINE_integer("nconmax", None, "Maximum number of contacts.")
_NJMAX = flags.DEFINE_integer("njmax", None, "Maximum number of constraints per world.")
_OVERRIDE = flags.DEFINE_multi_string("override", [], "Model overrides (notation: foo.bar = baz)", short_name="o")
_KEYFRAME = flags.DEFINE_integer("keyframe", 0, "keyframe to initialize simulation.")
_DEVICE = flags.DEFINE_string("device", None, "override the default Warp device")
_REPLAY = flags.DEFINE_string("replay", None, "keyframe sequence to replay, keyframe name must prefix match")
_IK_CONTROL = flags.DEFINE_bool("ik_control", False, "Enable IK control using keyboard")

_VIEWER_GLOBAL_STATE = {
    "running": True,
    "step_once": False,
    # IK control deltas (relative movements per step)
    "ik_delta_pos": np.zeros(3),  # (x, y, z)
    "ik_delta_yaw": 0.0,
    "ik_delta_gripper": 0.0,
    "ik_running": False,
}


def key_callback(key: int) -> None:
    print("key: {}".format(key))

    if key == 32:  # Space bar
        _VIEWER_GLOBAL_STATE["running"] = not _VIEWER_GLOBAL_STATE["running"]
        logging.info("RUNNING = %s", _VIEWER_GLOBAL_STATE["running"])
    elif key == 46:  # period
        _VIEWER_GLOBAL_STATE["step_once"] = True
    # IK control keys
    # Position: W/S (forward/backward X), A/D (left/right Y), Q/E (up/down Z)
    # Yaw: Z/C (rotate left/right)
    # Gripper: R/F (open/close)
    elif key == 265:  # W - move forward (+X)
        _VIEWER_GLOBAL_STATE["ik_delta_pos"][0] = 0.02
    elif key == 264:  # S - move backward (-X)
        _VIEWER_GLOBAL_STATE["ik_delta_pos"][0] = -0.02
    elif key == 263:  # A - move left (+Y)
        _VIEWER_GLOBAL_STATE["ik_delta_pos"][1] = 0.02
    elif key == 262:  # D - move right (-Y)
        _VIEWER_GLOBAL_STATE["ik_delta_pos"][1] = -0.02
    elif key == 81:  # Q - move up (+Z)
        _VIEWER_GLOBAL_STATE["ik_delta_pos"][2] = 0.02
    elif key == 69:  # E - move down (-Z)
        _VIEWER_GLOBAL_STATE["ik_delta_pos"][2] = -0.02
    elif key == 90:  # Z - rotate left (+ yaw)
        _VIEWER_GLOBAL_STATE["ik_delta_yaw"] = 0.1
    elif key == 67:  # C - rotate right (- yaw)
        _VIEWER_GLOBAL_STATE["ik_delta_yaw"] = -0.1
    elif key == 82:  # R - open gripper
        _VIEWER_GLOBAL_STATE["ik_delta_gripper"] = 0.1
    elif key == 70:  # F - close gripper
        _VIEWER_GLOBAL_STATE["ik_delta_gripper"] = -0.1
    elif key == 88:
        # X - reset deltas
        _VIEWER_GLOBAL_STATE["ik_delta_pos"][:] = 0.0
    elif key == 80:  # P - pause/resume
        _VIEWER_GLOBAL_STATE["ik_running"] = not _VIEWER_GLOBAL_STATE["ik_running"]
        print("ik running: {}".format(_VIEWER_GLOBAL_STATE["ik_running"]))


class IKController:
    """IK controller for keyboard control of end-effector position."""

    def __init__(self, mjm: mujoco.MjModel, mjd: mujoco.MjData):
        self._mjm = mjm
        self._mjd = mjd

        # Find site and joint IDs
        self._pinch_site_id = mjm.site("ur5e/robotiq/pinch").id
        self._attach_site_id = mjm.site("ur5e/attachment_site").id

        # Find arm joint IDs (UR5e has 6 joints)
        arm_joint_names = [
            "ur5e/shoulder_pan_joint",
            "ur5e/shoulder_lift_joint",
            "ur5e/elbow_joint",
            "ur5e/wrist_1_joint",
            "ur5e/wrist_2_joint",
            "ur5e/wrist_3_joint",
        ]
        self._arm_joint_ids = np.asarray([mjm.joint(name).id for name in arm_joint_names])

        # Find arm actuator IDs
        arm_actuator_names = [
            "ur5e/shoulder_pan",
            "ur5e/shoulder_lift",
            "ur5e/elbow",
            "ur5e/wrist_1",
            "ur5e/wrist_2",
            "ur5e/wrist_3",
        ]
        self._arm_actuator_ids = np.asarray([mjm.actuator(name).id for name in arm_actuator_names])

        # Gripper actuator
        self._gripper_actuator_ids = np.asarray([mjm.actuator("ur5e/robotiq/fingers_actuator").id])
        self._gripper_opening_joint_id = mjm.joint("ur5e/robotiq/right_driver_joint").id

        # Constants
        self._home_qpos = np.asarray([-np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])
        self._effector_down_rotation = lie.SO3(np.asarray([0.0, 1.0, 0.0, 0.0]))
        self._workspace_bounds = np.asarray([[0.25, -0.35, 0.02], [0.6, 0.35, 0.35]])

        # Initialize IK controller using UR5e model
        self._desc_dir = (
            Path(__file__).resolve().parent.parent.parent / "src" / "vdppo" / "env" / "manipspace" / "descriptions"
        )
        ik_mjcf = mjcf.from_path(
            (self._desc_dir / "universal_robots_ur5e" / "ur5e.xml").as_posix(), escape_separators=True
        )
        xml_str = mjcf_utils.to_string(ik_mjcf)
        assets = mjcf_utils.get_assets(ik_mjcf)
        ik_model = mujoco.MjModel.from_xml_string(xml_str, assets)
        self._ik = DiffIKController(model=ik_model, sites=["attachment_site"])

        # Compute T_pa (transform from pinch to attach)
        mujoco.mj_forward(mjm, mjd)
        self._update_T_pa()

        # Current gripper opening state
        self._gripper_opening = 0.0

    def _update_T_pa(self):
        """Update the transform from pinch to attach site."""
        pinch_pose = lie.SE3.from_rotation_and_translation(
            rotation=lie.SO3.from_matrix(self._mjd.site_xmat[self._pinch_site_id].reshape(3, 3)),
            translation=self._mjd.site_xpos[self._pinch_site_id],
        )
        attach_pose = lie.SE3.from_rotation_and_translation(
            rotation=lie.SO3.from_matrix(self._mjd.site_xmat[self._attach_site_id].reshape(3, 3)),
            translation=self._mjd.site_xpos[self._attach_site_id],
        )
        self._T_pa = pinch_pose.inverse() @ attach_pose

    def apply_ik_control(self, delta_pos: np.ndarray, delta_yaw: float, delta_gripper: float):
        """Apply IK control based on relative end-effector movements.

        Args:
            delta_pos: Relative position change (x, y, z)
            delta_yaw: Relative yaw change
            delta_gripper: Relative gripper opening change
        """
        # Get current effector state
        effector_pos = self._mjd.site_xpos[self._pinch_site_id].copy()
        effector_yaw = lie.SO3.from_matrix(self._mjd.site_xmat[self._pinch_site_id].reshape(3, 3)).compute_yaw_radians()

        # Compute target effector pose
        target_effector_translation = effector_pos + delta_pos
        target_effector_orientation = (
            lie.SO3.from_z_radians(delta_yaw)
            @ lie.SO3.from_z_radians(effector_yaw)
            @ self._effector_down_rotation.inverse()
        )
        self._gripper_opening = np.clip(self._gripper_opening + delta_gripper, 0.0, 1.0)

        # Clip to workspace bounds
        np.clip(target_effector_translation, *self._workspace_bounds, out=target_effector_translation)
        yaw = np.clip(target_effector_orientation.compute_yaw_radians(), -np.pi, +np.pi)
        target_effector_orientation = lie.SO3.from_z_radians(yaw) @ self._effector_down_rotation

        # Compute attach frame target
        target_effector_pose = lie.SE3.from_rotation_and_translation(
            rotation=target_effector_orientation,
            translation=target_effector_translation,
        )
        T_wa = target_effector_pose @ self._T_pa

        # Solve IK
        curr_qpos = self._mjd.qpos[self._arm_joint_ids]
        qpos_target = self._ik.solve(
            pos=T_wa.translation(),
            quat=T_wa.rotation().wxyz,
            curr_qpos=curr_qpos,
        )

        # Set control
        self._mjd.ctrl[self._arm_actuator_ids] = qpos_target
        self._mjd.ctrl[self._gripper_actuator_ids] = 255.0 * self._gripper_opening


def _load_model(path: epath.Path) -> mujoco.MjModel:
    if not path.exists():
        resource_path = epath.resource_path("mjx") / "third_party/mujoco_warp" / path
        if not resource_path.exists():
            raise FileNotFoundError(f"file not found: {path}\nalso tried: {resource_path}")
        path = resource_path

    print(f"Loading model from: {path}...")
    if path.suffix == ".mjb":
        return mujoco.MjModel.from_binary_path(path.as_posix())

    spec = mujoco.MjSpec.from_file(path.as_posix())
    # check if the file has any mujoco.sdf test plugins
    if any(p.plugin_name.startswith("mujoco.sdf") for p in spec.plugins):
        from mujoco.mjx.third_party.mujoco_warp.test_data.collision_sdf.utils import \
            register_sdf_plugins as register_sdf_plugins

        register_sdf_plugins(mjw)
    return spec.compile()


def _compile_step(m, d):
    print("Compiling physics step...", end="", flush=True)
    start = time.time()
    # capture the whole step function as a CUDA graph
    with wp.ScopedCapture() as capture:
        mjw.step(m, d)
    elapsed = time.time() - start
    print(f"done ({elapsed:0.2g}s).")
    return capture.graph


def _main(argv: Sequence[str]) -> None:
    """Runs viewer app."""
    if len(argv) < 2:
        raise app.UsageError("Missing required input: mjcf path.")
    elif len(argv) > 2:
        raise app.UsageError("Too many command-line arguments.")

    mjm = _load_model(epath.Path(argv[1]))
    mjd = mujoco.MjData(mjm)
    ctrls = None
    ctrlid = 0
    if _REPLAY.value:
        keys = find_keys(mjm, _REPLAY.value)
        if not keys:
            raise app.UsageError(f"Key prefix not find: {_REPLAY.value}")
        ctrls = make_trajectory(mjm, keys)
        mujoco.mj_resetDataKeyframe(mjm, mjd, keys[0])
    elif mjm.nkey > 0 and _KEYFRAME.value > -1:
        mujoco.mj_resetDataKeyframe(mjm, mjd, _KEYFRAME.value)

    if _ENGINE.value == EngineOptions.C:
        override_model(mjm, _OVERRIDE.value)
        print(
            f"  nbody: {mjm.nbody} nv: {mjm.nv} ngeom: {mjm.ngeom} nu: {mjm.nu}\n"
            f"  solver: {mujoco.mjtSolver(mjm.opt.solver).name} cone: {mujoco.mjtCone(mjm.opt.cone).name}"
            f" iterations: {mjm.opt.iterations} ls_iterations: {mjm.opt.ls_iterations}\n"
            f"  integrator: {mujoco.mjtIntegrator(mjm.opt.integrator).name}\n"
        )
        print(f"MuJoCo C simulating with dt = {mjm.opt.timestep:.3f}...")
    else:
        wp.config.quiet = flags.FLAGS["verbosity"].value < 1
        wp.init()
        if _CLEAR_KERNEL_CACHE.value:
            wp.clear_kernel_cache()

        with wp.ScopedDevice(_DEVICE.value):
            m = mjw.put_model(mjm)
            override_model(m, _OVERRIDE.value)
            broadphase, filter = (
                mjw.BroadphaseType(m.opt.broadphase).name,
                mjw.BroadphaseFilter(m.opt.broadphase_filter).name,
            )
            solver, cone = mjw.SolverType(m.opt.solver).name, mjw.ConeType(m.opt.cone).name
            integrator = mjw.IntegratorType(m.opt.integrator).name
            iterations, ls_iterations = m.opt.iterations, m.opt.ls_iterations
            ls_str = f"{'parallel' if m.opt.ls_parallel else 'iterative'} linesearch iterations: {ls_iterations}"
            print(
                f"  nbody: {m.nbody} nv: {m.nv} ngeom: {m.ngeom} nu: {m.nu} is_sparse: {m.opt.is_sparse}\n"
                f"  broadphase: {broadphase} broadphase_filter: {filter}\n"
                f"  solver: {solver} cone: {cone} iterations: {iterations} {ls_str}\n"
                f"  integrator: {integrator} graph_conditional: {m.opt.graph_conditional}"
            )
            ipdb.set_trace()
            d = mjw.put_data(mjm, mjd, nconmax=_NCONMAX.value, njmax=_NJMAX.value)
            print("nconmax: {},  naconmax: {}".format(_NCONMAX.value, d.naconmax))
            print(f"Data\n  nworld: {d.nworld} nconmax: {d.naconmax / d.nworld} njmax: {d.njmax}\n")
            graph = _compile_step(m, d)
            print(f"MuJoCo Warp simulating with dt = {m.opt.timestep.numpy()[0]:.3f}...")

    # Initialize IK controller if enabled
    ik_controller = None
    if _IK_CONTROL.value:
        try:
            ik_controller = IKController(mjm, mjd)
            print("IK control enabled. Use keyboard to control end-effector:")
            print("  W/S: Forward/Backward (X)")
            print("  A/D: Left/Right (Y)")
            print("  Q/E: Up/Down (Z)")
            print("  Z/C: Rotate Left/Right (Yaw)")
            print("  R/F: Open/Close Gripper")
            print("  Space: Pause/Resume simulation")
        except Exception as e:
            print(f"Warning: Could not initialize IK controller: {e}")
            print("IK control disabled.")

    with mujoco.viewer.launch_passive(mjm, mjd, key_callback=key_callback) as viewer:
        opt = copy.copy(mjm.opt)

        while True:
            start = time.time()

            if ctrls is not None and ctrlid < len(ctrls):
                mjd.ctrl[:] = ctrls[ctrlid]
                ctrlid += 1
            elif ik_controller is not None:
                if _VIEWER_GLOBAL_STATE["ik_running"]:
                  # Apply IK control based on keyboard input
                  delta_pos = _VIEWER_GLOBAL_STATE["ik_delta_pos"].copy()
                  delta_yaw = _VIEWER_GLOBAL_STATE["ik_delta_yaw"]
                  delta_gripper = _VIEWER_GLOBAL_STATE["ik_delta_gripper"]
                  # print("pos: {} | yaw: {:.2f} | gripper: {:.2f}".format(delta_pos, delta_yaw, delta_gripper))

                  # # Reset deltas after reading
                  # _VIEWER_GLOBAL_STATE["ik_delta_pos"][:] = 0
                  # _VIEWER_GLOBAL_STATE["ik_delta_yaw"] = 0.0
                  # _VIEWER_GLOBAL_STATE["ik_delta_gripper"] = 0.0

                  # Apply IK control
                  ik_controller.apply_ik_control(delta_pos, delta_yaw, delta_gripper)

            if _ENGINE.value == EngineOptions.C:
                mujoco.mj_step(mjm, mjd)
            else:  # mjwarp
                wp.copy(d.ctrl, wp.array([mjd.ctrl.astype(np.float32)]))
                wp.copy(d.act, wp.array([mjd.act.astype(np.float32)]))
                wp.copy(d.xfrc_applied, wp.array([mjd.xfrc_applied.astype(np.float32)]))
                wp.copy(d.qpos, wp.array([mjd.qpos.astype(np.float32)]))
                wp.copy(d.qvel, wp.array([mjd.qvel.astype(np.float32)]))
                wp.copy(d.time, wp.array([mjd.time], dtype=wp.float32))

                # if the user changed an option in the MuJoCo Simulate UI, go ahead and recompile the step
                # TODO: update memory tied to option max iterations
                if mjm.opt != opt:
                    opt = copy.copy(mjm.opt)
                    m = mjw.put_model(mjm)
                    graph = _compile_step(m, d)

                if _VIEWER_GLOBAL_STATE["running"]:
                    wp.capture_launch(graph)
                    wp.synchronize()
                elif _VIEWER_GLOBAL_STATE["step_once"]:
                    _VIEWER_GLOBAL_STATE["step_once"] = False
                    wp.capture_launch(graph)
                    wp.synchronize()

                mjw.get_data_into(mjd, mjm, d)

            viewer.sync()

            elapsed = time.time() - start
            if elapsed < mjm.opt.timestep:
                time.sleep(mjm.opt.timestep - elapsed)


def main():
    # absl flags assumes __main__ is the main running module for printing usage documentation
    # pyproject bin scripts break this assumption, so manually set argv and docstring
    sys.argv[0] = "mujoco_warp.viewer"
    sys.modules["__main__"].__doc__ = __doc__
    app.run(_main)


if __name__ == "__main__":
    main()
