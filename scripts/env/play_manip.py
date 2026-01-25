import os

import ipdb
import jax.numpy as jnp
import numpy as np
from loguru import logger

from rraa_rl.envs.manipspace.scene_env_jax import ManipStep, SceneEnvJax, SceneEnvState
from rraa_rl.envs.manipspace.scene_env import SceneEnv

os.environ["XLA_FLAGS"] = "--xla_gpu_graph_min_graph_size=1"

import copy
import logging
import time
from typing import Sequence, Dict, Any, Tuple

import jax
import jax.random as jr
import mujoco
import mujoco.viewer
# import mujoco_warp as mjw
import warp as wp
from absl import app, flags
from jax import numpy as jp
from mujoco import mjx


_VIEWER_GLOBAL_STATE = {
    "running": True,
}


def key_callback(key: int) -> None:
    if key == 32:  # Space bar
        _VIEWER_GLOBAL_STATE["running"] = not _VIEWER_GLOBAL_STATE["running"]
        logging.info("RUNNING = %s", _VIEWER_GLOBAL_STATE["running"])


def compare_states(
    env_jax: SceneEnvJax,
    state_jax: SceneEnvState,
    obs_jax: jnp.ndarray,
    env_np: SceneEnv,
    obs_np: np.ndarray,
    step_count: int,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> Dict[str, Any]:
    """Compare states and observations between JAX and NumPy environments.
    
    Returns a dictionary of differences found.
    """
    diffs = {}
    
    # Get JAX data
    mjx_data = state_jax.mjx_data
    jax_qpos = np.asarray(jax.device_get(mjx_data.qpos))
    jax_qvel = np.asarray(jax.device_get(mjx_data.qvel))
    jax_ctrl = np.asarray(jax.device_get(mjx_data.ctrl))
    jax_time = float(jax.device_get(mjx_data.time))
    jax_obs = np.asarray(jax.device_get(obs_jax))
    
    # Get NumPy data
    np_qpos = env_np._data.qpos.copy()
    np_qvel = env_np._data.qvel.copy()
    np_ctrl = env_np._data.ctrl.copy()
    np_time = env_np._data.time
    np_obs = obs_np
    
    # Compare qpos
    qpos_diff = np.abs(jax_qpos - np_qpos)
    if np.any(qpos_diff > atol):
        max_idx = int(np.argmax(qpos_diff))
        diffs["qpos"] = {
            "max_diff": float(np.max(qpos_diff)),
            "max_idx": max_idx,
            "jax_val": float(jax_qpos[max_idx]),
            "np_val": float(np_qpos[max_idx]),
            "jax": jax_qpos.tolist(),
            "np": np_qpos.tolist(),
        }
        logger.warning(f"[Step {step_count}] qpos mismatch: max_diff={np.max(qpos_diff):.6f} at idx={max_idx}, jax={jax_qpos[max_idx]:.6f}, np={np_qpos[max_idx]:.6f}")
    
    # Compare qvel
    qvel_diff = np.abs(jax_qvel - np_qvel)
    if np.any(qvel_diff > atol):
        max_idx = int(np.argmax(qvel_diff))
        diffs["qvel"] = {
            "max_diff": float(np.max(qvel_diff)),
            "max_idx": max_idx,
            "jax_val": float(jax_qvel[max_idx]),
            "np_val": float(np_qvel[max_idx]),
            "jax": jax_qvel.tolist(),
            "np": np_qvel.tolist(),
        }
        logger.warning(f"[Step {step_count}] qvel mismatch: max_diff={np.max(qvel_diff):.6f} at idx={max_idx}, jax={jax_qvel[max_idx]:.6f}, np={np_qvel[max_idx]:.6f}")
    
    # Compare ctrl
    ctrl_diff = np.abs(jax_ctrl - np_ctrl)
    if np.any(ctrl_diff > atol):
        max_idx = int(np.argmax(ctrl_diff))
        diffs["ctrl"] = {
            "max_diff": float(np.max(ctrl_diff)),
            "max_idx": max_idx,
            "jax_val": float(jax_ctrl[max_idx]),
            "np_val": float(np_ctrl[max_idx]),
            "jax": jax_ctrl.tolist(),
            "np": np_ctrl.tolist(),
        }
        logger.warning(f"[Step {step_count}] ctrl mismatch: max_diff={np.max(ctrl_diff):.6f} at idx={max_idx}, jax={jax_ctrl[max_idx]:.6f}, np={np_ctrl[max_idx]:.6f}")
    
    # Compare time
    time_diff = abs(jax_time - np_time)
    if time_diff > atol:
        diffs["time"] = {
            "diff": time_diff,
            "jax": jax_time,
            "np": np_time,
        }
        logger.warning(f"[Step {step_count}] time mismatch: jax={jax_time:.6f}, np={np_time:.6f}")
    
    # Compare observations element by element with labels
    jax_obs_labels = _get_obs_labels_jax(env_jax)
    np_obs_labels = _get_obs_labels_np(env_np)
    
    # Check if observation shapes match
    if jax_obs.shape != np_obs.shape:
        diffs["obs_shape"] = {
            "jax_shape": jax_obs.shape,
            "np_shape": np_obs.shape,
        }
        logger.error(f"[Step {step_count}] OBSERVATION SHAPE MISMATCH: jax={jax_obs.shape}, np={np_obs.shape}")
        
        # Print detailed breakdown of each observation
        logger.info("JAX observation breakdown:")
        _print_obs_breakdown_jax(env_jax, state_jax)
        logger.info("NumPy observation breakdown:")
        _print_obs_breakdown_np(env_np)
        
        # Still try to compare common prefix
        min_len = min(len(jax_obs), len(np_obs))
        obs_diff = np.abs(jax_obs[:min_len] - np_obs[:min_len])
        if np.any(obs_diff > atol):
            max_idx = int(np.argmax(obs_diff))
            jax_label = jax_obs_labels[max_idx] if max_idx < len(jax_obs_labels) else f"jax_idx_{max_idx}"
            np_label = np_obs_labels[max_idx] if max_idx < len(np_obs_labels) else f"np_idx_{max_idx}"
            diffs["obs"] = {
                "max_diff": float(np.max(obs_diff)),
                "max_idx": max_idx,
                "jax_label": jax_label,
                "np_label": np_label,
                "jax_val": float(jax_obs[max_idx]),
                "np_val": float(np_obs[max_idx]),
            }
            logger.warning(f"[Step {step_count}] obs mismatch in common prefix: max_diff={np.max(obs_diff):.6f} at idx={max_idx}")
    else:
        obs_diff = np.abs(jax_obs - np_obs)
        if np.any(obs_diff > atol):
            max_idx = int(np.argmax(obs_diff))
            obs_label = jax_obs_labels[max_idx] if max_idx < len(jax_obs_labels) else f"idx_{max_idx}"
            diffs["obs"] = {
                "max_diff": float(np.max(obs_diff)),
                "max_idx": max_idx,
                "label": obs_label,
                "jax_val": float(jax_obs[max_idx]),
                "np_val": float(np_obs[max_idx]),
                "jax": jax_obs.tolist(),
                "np": np_obs.tolist(),
            }
            logger.warning(f"[Step {step_count}] obs mismatch: max_diff={np.max(obs_diff):.6f} at idx={max_idx} ({obs_label}), jax={jax_obs[max_idx]:.6f}, np={np_obs[max_idx]:.6f}")
            
            # Log all mismatched observation elements
            mismatch_mask = obs_diff > atol
            mismatch_indices = np.where(mismatch_mask)[0]
            if len(mismatch_indices) <= 10:
                for idx in mismatch_indices:
                    label = jax_obs_labels[idx] if idx < len(jax_obs_labels) else f"idx_{idx}"
                    logger.info(f"  obs[{idx}] ({label}): jax={jax_obs[idx]:.6f}, np={np_obs[idx]:.6f}, diff={obs_diff[idx]:.6f}")
    
    # Compare button states
    jax_button_states = np.asarray(jax.device_get(state_jax.button_states))
    np_button_states = env_np._cur_button_states
    if not np.array_equal(jax_button_states, np_button_states):
        diffs["button_states"] = {
            "jax": jax_button_states.tolist(),
            "np": np_button_states.tolist(),
        }
        logger.warning(f"[Step {step_count}] button_states mismatch: jax={jax_button_states}, np={np_button_states}")
    
    # Compare site positions (effector)
    jax_effector_pos = np.asarray(jax.device_get(mjx_data.site_xpos[env_jax._pinch_site_id]))
    np_effector_pos = env_np._data.site_xpos[env_np._pinch_site_id].copy()
    effector_pos_diff = np.abs(jax_effector_pos - np_effector_pos)
    if np.any(effector_pos_diff > atol):
        diffs["effector_pos"] = {
            "max_diff": float(np.max(effector_pos_diff)),
            "jax": jax_effector_pos.tolist(),
            "np": np_effector_pos.tolist(),
        }
        logger.warning(f"[Step {step_count}] effector_pos mismatch: jax={jax_effector_pos}, np={np_effector_pos}, diff={effector_pos_diff}")
    
    # Compare effector orientation (site_xmat)
    jax_effector_mat = np.asarray(jax.device_get(mjx_data.site_xmat[env_jax._pinch_site_id]))
    np_effector_mat = env_np._data.site_xmat[env_np._pinch_site_id].copy()

    assert jax_effector_mat.shape == (3,3)
    assert np_effector_mat.shape == (9,)
    jax_effector_mat = jax_effector_mat.flatten()

    effector_mat_diff = np.abs(jax_effector_mat - np_effector_mat)
    if np.any(effector_mat_diff > atol):
        diffs["effector_mat"] = {
            "max_diff": float(np.max(effector_mat_diff)),
            "jax": jax_effector_mat.tolist(),
            "np": np_effector_mat.tolist(),
        }
        logger.warning(f"[Step {step_count}] effector_mat mismatch: max_diff={np.max(effector_mat_diff):.6f}")
    
    if diffs:
        logger.info(f"[Step {step_count}] Found {len(diffs)} differences: {list(diffs.keys())}")
    else:
        logger.debug(f"[Step {step_count}] States match")
    
    return diffs


def _get_obs_labels_jax(env_jax: SceneEnvJax) -> list:
    """Generate labels for each JAX observation dimension."""
    labels = []
    config = env_jax.config
    
    # Joint positions (6)
    for i in range(6):
        labels.append(f"joint_pos_{i}")
    
    # Joint velocities (6)
    for i in range(6):
        labels.append(f"joint_vel_{i}")
    
    # Effector position (3)
    for dim in ["x", "y", "z"]:
        labels.append(f"effector_pos_{dim}")
    
    # Effector yaw cos/sin (2)
    labels.append("effector_yaw_cos")
    labels.append("effector_yaw_sin")
    
    # Gripper opening (1)
    labels.append("gripper_opening")
    
    # Gripper contact (1)
    labels.append("gripper_contact")
    
    # Cubes
    for i in range(config.num_cubes):
        for dim in ["x", "y", "z"]:
            labels.append(f"cube_{i}_pos_{dim}")
        for j in range(4):
            labels.append(f"cube_{i}_quat_{j}")
        labels.append(f"cube_{i}_yaw_cos")
        labels.append(f"cube_{i}_yaw_sin")
    
    # Buttons
    for i in range(config.num_buttons):
        for j in range(config.num_button_states):
            labels.append(f"button_{i}_state_onehot_{j}")
        labels.append(f"button_{i}_pos")
        labels.append(f"button_{i}_vel")
    
    # Drawer (2)
    labels.append("drawer_pos")
    labels.append("drawer_vel")
    
    # Window (2)
    labels.append("window_pos")
    labels.append("window_vel")
    
    return labels


def _get_obs_labels_np(env_np: SceneEnv) -> list:
    """Generate labels for each NumPy observation dimension."""
    labels = []
    
    # Joint positions (6)
    for i in range(6):
        labels.append(f"joint_pos_{i}")
    
    # Joint velocities (6)
    for i in range(6):
        labels.append(f"joint_vel_{i}")
    
    # Effector position (3)
    for dim in ["x", "y", "z"]:
        labels.append(f"effector_pos_{dim}")
    
    # Effector yaw cos/sin (2)
    labels.append("effector_yaw_cos")
    labels.append("effector_yaw_sin")
    
    # Gripper opening (1)
    labels.append("gripper_opening")
    
    # Gripper contact (1)
    labels.append("gripper_contact")
    
    # Cubes
    for i in range(env_np._num_cubes):
        for dim in ["x", "y", "z"]:
            labels.append(f"cube_{i}_pos_{dim}")
        for j in range(4):
            labels.append(f"cube_{i}_quat_{j}")
        labels.append(f"cube_{i}_yaw_cos")
        labels.append(f"cube_{i}_yaw_sin")
    
    # Buttons
    for i in range(env_np._num_buttons):
        for j in range(env_np._num_button_states):
            labels.append(f"button_{i}_state_onehot_{j}")
        labels.append(f"button_{i}_pos")
        labels.append(f"button_{i}_vel")
    
    # Drawer (2)
    labels.append("drawer_pos")
    labels.append("drawer_vel")
    
    # Window (2)
    labels.append("window_pos")
    labels.append("window_vel")
    
    return labels


def _print_obs_breakdown_jax(env_jax: SceneEnvJax, state_jax: SceneEnvState) -> None:
    """Print detailed breakdown of JAX observation."""
    mjx_data = state_jax.mjx_data
    config = env_jax.config
    
    obs_parts = []
    
    # Joint positions and velocities
    joint_pos = np.asarray(jax.device_get(mjx_data.qpos[:6]))
    joint_vel = np.asarray(jax.device_get(mjx_data.qvel[:6]))
    logger.info(f"  joint_pos: shape={joint_pos.shape}")
    logger.info(f"  joint_vel: shape={joint_vel.shape}")
    obs_parts.append(("joint_pos", joint_pos.shape[0]))
    obs_parts.append(("joint_vel", joint_vel.shape[0]))
    
    # Effector position (3)
    effector_pos = np.asarray(jax.device_get(mjx_data.site_xpos[env_jax._pinch_site_id]))
    logger.info(f"  effector_pos: shape={effector_pos.shape}")
    obs_parts.append(("effector_pos", effector_pos.shape[0]))
    
    # Effector yaw cos/sin (2)
    logger.info(f"  effector_yaw_cos: shape=(1,)")
    logger.info(f"  effector_yaw_sin: shape=(1,)")
    obs_parts.append(("effector_yaw_cos", 1))
    obs_parts.append(("effector_yaw_sin", 1))
    
    # Gripper (2)
    logger.info(f"  gripper_opening: shape=(1,)")
    logger.info(f"  gripper_contact: shape=(1,)")
    obs_parts.append(("gripper_opening", 1))
    obs_parts.append(("gripper_contact", 1))
    
    # Cubes
    for i in range(config.num_cubes):
        cube_qpos_addr = env_jax._cube_joint_qpos_addrs[i]
        cube_pos = np.asarray(jax.device_get(mjx_data.qpos[cube_qpos_addr : cube_qpos_addr + 3]))
        cube_quat = np.asarray(jax.device_get(mjx_data.qpos[cube_qpos_addr + 3 : cube_qpos_addr + 7]))
        logger.info(f"  cube_{i}_pos: shape={cube_pos.shape}")
        logger.info(f"  cube_{i}_quat: shape={cube_quat.shape}")
        logger.info(f"  cube_{i}_yaw_cos: shape=(1,)")
        logger.info(f"  cube_{i}_yaw_sin: shape=(1,)")
        obs_parts.append((f"cube_{i}_pos", cube_pos.shape[0]))
        obs_parts.append((f"cube_{i}_quat", cube_quat.shape[0]))
        obs_parts.append((f"cube_{i}_yaw_cos", 1))
        obs_parts.append((f"cube_{i}_yaw_sin", 1))
    
    # Buttons  
    for i in range(config.num_buttons):
        logger.info(f"  button_{i}_state_onehot: shape=({config.num_button_states},)")
        button_pos = np.asarray(jax.device_get(mjx_data.qpos[env_jax._button_joint_qpos_addrs[i] : env_jax._button_joint_qpos_addrs[i] + 1]))
        button_vel = np.asarray(jax.device_get(mjx_data.qvel[env_jax._button_joint_qpos_addrs[i] : env_jax._button_joint_qpos_addrs[i] + 1]))
        logger.info(f"  button_{i}_pos: shape={button_pos.shape}, using qpos_addr={env_jax._button_joint_qpos_addrs[i]}")
        logger.info(f"  button_{i}_vel: shape={button_vel.shape}, using qpos_addr={env_jax._button_joint_qpos_addrs[i]} (SHOULD USE DOFADR!)")
        obs_parts.append((f"button_{i}_state_onehot", config.num_button_states))
        obs_parts.append((f"button_{i}_pos", button_pos.shape[0]))
        obs_parts.append((f"button_{i}_vel", button_vel.shape[0]))
    
    # Drawer
    drawer_pos = np.asarray(jax.device_get(mjx_data.qpos[env_jax._drawer_joint_qpos_addr : env_jax._drawer_joint_qpos_addr + 1]))
    drawer_vel = np.asarray(jax.device_get(mjx_data.qvel[env_jax._drawer_joint_qpos_addr : env_jax._drawer_joint_qpos_addr + 1]))
    logger.info(f"  drawer_pos: shape={drawer_pos.shape}, using qpos_addr={env_jax._drawer_joint_qpos_addr}")
    logger.info(f"  drawer_vel: shape={drawer_vel.shape}, using qpos_addr={env_jax._drawer_joint_qpos_addr} (SHOULD USE DOFADR!)")
    obs_parts.append(("drawer_pos", drawer_pos.shape[0]))
    obs_parts.append(("drawer_vel", drawer_vel.shape[0]))
    
    # Window
    window_pos = np.asarray(jax.device_get(mjx_data.qpos[env_jax._window_joint_qpos_addr : env_jax._window_joint_qpos_addr + 1]))
    window_vel = np.asarray(jax.device_get(mjx_data.qvel[env_jax._window_joint_qpos_addr : env_jax._window_joint_qpos_addr + 1]))
    logger.info(f"  window_pos: shape={window_pos.shape}, using qpos_addr={env_jax._window_joint_qpos_addr}")
    logger.info(f"  window_vel: shape={window_vel.shape}, using qpos_addr={env_jax._window_joint_qpos_addr} (SHOULD USE DOFADR!)")
    obs_parts.append(("window_pos", window_pos.shape[0]))
    obs_parts.append(("window_vel", window_vel.shape[0]))
    
    total = sum(size for _, size in obs_parts)
    logger.info(f"  TOTAL JAX obs size: {total}")
    

def _print_obs_breakdown_np(env_np: SceneEnv) -> None:
    """Print detailed breakdown of NumPy observation."""
    from ogbench.manipspace import lie
    
    ob_info = env_np.compute_ob_info()
    obs_parts = []
    
    # Joint positions and velocities
    joint_pos = ob_info["proprio/joint_pos"]
    joint_vel = ob_info["proprio/joint_vel"]
    logger.info(f"  joint_pos: shape={joint_pos.shape}")
    logger.info(f"  joint_vel: shape={joint_vel.shape}")
    obs_parts.append(("joint_pos", joint_pos.shape[0]))
    obs_parts.append(("joint_vel", joint_vel.shape[0]))
    
    # Effector position (3)
    effector_pos = ob_info["proprio/effector_pos"]
    logger.info(f"  effector_pos: shape={effector_pos.shape}")
    obs_parts.append(("effector_pos", effector_pos.shape[0]))
    
    # Effector yaw cos/sin
    effector_yaw = ob_info["proprio/effector_yaw"]
    logger.info(f"  effector_yaw_cos: shape={np.cos(effector_yaw).shape}")
    logger.info(f"  effector_yaw_sin: shape={np.sin(effector_yaw).shape}")
    obs_parts.append(("effector_yaw_cos", np.cos(effector_yaw).shape[0]))
    obs_parts.append(("effector_yaw_sin", np.sin(effector_yaw).shape[0]))
    
    # Gripper
    gripper_opening = ob_info["proprio/gripper_opening"]
    gripper_contact = ob_info["proprio/gripper_contact"]
    logger.info(f"  gripper_opening: shape={gripper_opening.shape}")
    logger.info(f"  gripper_contact: shape={gripper_contact.shape}")
    obs_parts.append(("gripper_opening", gripper_opening.shape[0]))
    obs_parts.append(("gripper_contact", gripper_contact.shape[0]))
    
    # Cubes
    for i in range(env_np._num_cubes):
        cube_pos = ob_info[f"privileged/block_{i}_pos"]
        cube_quat = ob_info[f"privileged/block_{i}_quat"]
        cube_yaw = ob_info[f"privileged/block_{i}_yaw"]
        logger.info(f"  cube_{i}_pos: shape={cube_pos.shape}")
        logger.info(f"  cube_{i}_quat: shape={cube_quat.shape}")
        logger.info(f"  cube_{i}_yaw_cos: shape={np.cos(cube_yaw).shape}")
        logger.info(f"  cube_{i}_yaw_sin: shape={np.sin(cube_yaw).shape}")
        obs_parts.append((f"cube_{i}_pos", cube_pos.shape[0]))
        obs_parts.append((f"cube_{i}_quat", cube_quat.shape[0]))
        obs_parts.append((f"cube_{i}_yaw_cos", np.cos(cube_yaw).shape[0]))
        obs_parts.append((f"cube_{i}_yaw_sin", np.sin(cube_yaw).shape[0]))
    
    # Buttons
    for i in range(env_np._num_buttons):
        button_state = np.eye(env_np._num_button_states)[env_np._cur_button_states[i]]
        button_pos = ob_info[f"privileged/button_{i}_pos"]
        button_vel = ob_info[f"privileged/button_{i}_vel"]
        logger.info(f"  button_{i}_state_onehot: shape={button_state.shape}")
        logger.info(f"  button_{i}_pos: shape={button_pos.shape}")
        logger.info(f"  button_{i}_vel: shape={button_vel.shape}")
        obs_parts.append((f"button_{i}_state_onehot", button_state.shape[0]))
        obs_parts.append((f"button_{i}_pos", button_pos.shape[0]))
        obs_parts.append((f"button_{i}_vel", button_vel.shape[0]))
    
    # Drawer
    drawer_pos = ob_info["privileged/drawer_pos"]
    drawer_vel = ob_info["privileged/drawer_vel"]
    logger.info(f"  drawer_pos: shape={drawer_pos.shape}")
    logger.info(f"  drawer_vel: shape={drawer_vel.shape}")
    obs_parts.append(("drawer_pos", drawer_pos.shape[0]))
    obs_parts.append(("drawer_vel", drawer_vel.shape[0]))
    
    # Window
    window_pos = ob_info["privileged/window_pos"]
    window_vel = ob_info["privileged/window_vel"]
    logger.info(f"  window_pos: shape={window_pos.shape}")
    logger.info(f"  window_vel: shape={window_vel.shape}")
    obs_parts.append(("window_pos", window_pos.shape[0]))
    obs_parts.append(("window_vel", window_vel.shape[0]))
    
    total = sum(size for _, size in obs_parts)
    logger.info(f"  TOTAL NumPy obs size: {total}")


def sync_np_to_jax(
    env_jax: SceneEnvJax,
    state_jax: SceneEnvState,
    env_np: SceneEnv,
) -> Tuple[SceneEnvState, jnp.ndarray]:
    """Set the state of the JAX environment to match the NumPy environment.
    
    Returns:
        Tuple of (new_state_jax, obs_jax)
    """
    # Get qpos, qvel, ctrl, and button_states from NumPy environment
    np_qpos = env_np._data.qpos.copy()
    np_qvel = env_np._data.qvel.copy()
    np_ctrl = env_np._data.ctrl.copy()
    np_button_states = env_np._cur_button_states.copy()
    np_time = env_np._data.time
    
    # Update mjx_data with NumPy values
    mjx_data = state_jax.mjx_data
    mjx_data = mjx_data.replace(
        qpos=jnp.array(np_qpos),
        qvel=jnp.array(np_qvel),
        ctrl=jnp.array(np_ctrl),
        time=jnp.array(np_time),
    )
    
    # Run forward kinematics to update site positions etc.
    mjx_data = mjx.forward(env_jax.mjx_model, mjx_data)
    
    # Get button positions for prev_button_pos
    prev_button_pos = jnp.array(
        [mjx_data.qpos[env_jax._button_joint_qpos_addrs[i]] for i in range(env_jax.config.num_buttons)]
    )
    
    # Create new state with updated values
    new_state_jax = state_jax._replace(
        mjx_data=mjx_data,
        button_states=jnp.array(np_button_states),
        prev_button_pos=prev_button_pos,
        prev_qpos=jnp.array(np_qpos),
        prev_qvel=jnp.array(np_qvel),
    )
    
    # Compute observation
    obs_jax = env_jax._get_observation(new_state_jax)
    
    logger.info("Synchronized NumPy state to JAX environment")
    return new_state_jax, obs_jax


def compare_action_processing(
    env_jax: SceneEnvJax,
    state_jax: SceneEnvState,
    action: jnp.ndarray,
    env_np: SceneEnv,
    action_np_normalized: np.ndarray,
    step_count: int,
    atol: float = 1e-4,
) -> Dict[str, Any]:
    """Compare action processing between JAX and NumPy environments.
    
    This function compares the intermediate computations in set_control/step
    before physics stepping to help identify where differences originate.
    """
    from ogbench.manipspace import lie
    
    diffs = {}
    
    mjx_data = state_jax.mjx_data
    
    # === JAX side: replicate the action processing logic ===
    # Unnormalize action
    jax_action_unnorm = np.asarray(jax.device_get(
        0.5 * (action + 1) * (env_jax.config.action_high - env_jax.config.action_low) + env_jax.config.action_low
    ))
    jax_a_pos = jax_action_unnorm[:3]
    jax_a_ori = jax_action_unnorm[3]
    jax_a_gripper = jax_action_unnorm[4]
    
    # Get current effector state from JAX
    jax_effector_pos = np.asarray(jax.device_get(mjx_data.site_xpos[env_jax._pinch_site_id]))
    jax_effector_mat = np.asarray(jax.device_get(mjx_data.site_xmat[env_jax._pinch_site_id])).reshape(3, 3)
    jax_gripper_opening = float(jax.device_get(
        jnp.clip(mjx_data.qpos[env_jax._gripper_opening_joint_id] / 0.8, 0, 1)
    ))
    
    # === NumPy side: replicate the action processing logic ===
    np_action_unnorm = env_np.unnormalize_action(action_np_normalized)
    np_a_pos = np_action_unnorm[:3]
    np_a_ori = np_action_unnorm[3]
    np_a_gripper = np_action_unnorm[4]
    
    # Get current effector state from NumPy
    np_effector_pos = env_np._data.site_xpos[env_np._pinch_site_id].copy()
    np_effector_mat = env_np._data.site_xmat[env_np._pinch_site_id].copy().reshape(3, 3)
    np_gripper_opening = float(np.clip(env_np._data.qpos[env_np._gripper_opening_joint_id] / 0.8, 0, 1))
    
    # Compare unnormalized actions
    action_diff = np.abs(jax_action_unnorm - np_action_unnorm)
    if np.any(action_diff > atol):
        diffs["action_unnorm"] = {
            "max_diff": float(np.max(action_diff)),
            "jax": jax_action_unnorm.tolist(),
            "np": np_action_unnorm.tolist(),
        }
        logger.warning(f"[Step {step_count}] action_unnorm mismatch: max_diff={np.max(action_diff):.6f}")
    
    # Compare effector positions (before action)
    eff_pos_diff = np.abs(jax_effector_pos - np_effector_pos)
    if np.any(eff_pos_diff > atol):
        diffs["pre_effector_pos"] = {
            "max_diff": float(np.max(eff_pos_diff)),
            "jax": jax_effector_pos.tolist(),
            "np": np_effector_pos.tolist(),
        }
        logger.warning(f"[Step {step_count}] pre_effector_pos mismatch: max_diff={np.max(eff_pos_diff):.6f}")
    
    # Compare effector orientations
    eff_mat_diff = np.abs(jax_effector_mat - np_effector_mat)
    if np.any(eff_mat_diff > atol):
        diffs["pre_effector_mat"] = {
            "max_diff": float(np.max(eff_mat_diff)),
            "jax": jax_effector_mat.tolist(),
            "np": np_effector_mat.tolist(),
        }
        logger.warning(f"[Step {step_count}] pre_effector_mat mismatch: max_diff={np.max(eff_mat_diff):.6f}")
    
    # Compare gripper openings
    gripper_diff = abs(jax_gripper_opening - np_gripper_opening)
    if gripper_diff > atol:
        diffs["pre_gripper_opening"] = {
            "diff": gripper_diff,
            "jax": jax_gripper_opening,
            "np": np_gripper_opening,
        }
        logger.warning(f"[Step {step_count}] pre_gripper_opening mismatch: jax={jax_gripper_opening:.6f}, np={np_gripper_opening:.6f}")
    
    # Compare yaw extraction
    # JAX yaw extraction (from scene_env_jax.py)
    from rraa_rl.envs.manipspace.lie.so3_jax import mat_to_quat
    jax_quat = np.asarray(jax.device_get(mat_to_quat(jnp.array(jax_effector_mat))))
    # w, x, y, z = jax_quat
    # jax_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    jax_yaw = float(jax.device_get(_compute_yaw_from_quat_np(jax_quat)))
    
    # NumPy yaw extraction (from manipspace_env.py)
    np_yaw = lie.SO3.from_matrix(np_effector_mat).compute_yaw_radians()
    
    yaw_diff = abs(jax_yaw - np_yaw)
    if yaw_diff > atol:
        diffs["pre_effector_yaw"] = {
            "diff": yaw_diff,
            "jax": jax_yaw,
            "np": np_yaw,
        }
        logger.warning(f"[Step {step_count}] pre_effector_yaw mismatch: jax={jax_yaw:.6f}, np={np_yaw:.6f}")
    
    if diffs:
        logger.info(f"[Step {step_count}] Action processing has {len(diffs)} differences: {list(diffs.keys())}")
    
    return diffs


def _compute_yaw_from_quat_np(quat: np.ndarray) -> float:
    """Compute yaw angle from quaternion (wxyz format)."""
    w, x, y, z = quat
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main():
    jax.config.update('jax_debug_nans', True)

    # Create both environments
    logger.info("Creating SceneEnvJax...")
    env_jax = SceneEnvJax()
    
    logger.info("Creating SceneEnv...")
    env_np = SceneEnv(env_type=None, mode='data_collection')
    
    # Reset NumPy environment first (this will be the source of truth)
    logger.info("Resetting NumPy environment...")
    obs_np_initial, _ = env_np.reset()
    
    # Create initial JAX state (will be overwritten by sync)
    logger.info("Creating initial JAX state...")
    state_jax, _ = env_jax.reset(jr.PRNGKey(0))

    print(f"Default backend: {jax.default_backend()}")
    step_fn = mjx.step

    def set_data_fn(dx, ctrl, act, xfrc_applied, qpos, qvel, time_):
        return dx.tree_replace(
            {
                "ctrl": jp.array(ctrl),
                "act": jp.array(act),
                "xfrc_applied": jp.array(xfrc_applied),
                "qpos": jp.array(qpos),
                "qvel": jp.array(qvel),
                "time": jp.array(time_),
            }
        )

    m = env_jax.mj_model
    mx = env_jax.mjx_model

    d = mujoco.MjData(m)
    dx = state_jax.mjx_data

    start = time.time()

    action = jnp.array(env_jax.config.action_high)
    env_step_fn = env_jax.step
    env_step_fn = jax.jit(env_step_fn, keep_unused=True).lower(state_jax, action).compile()
    elapsed = time.time() - start
    print(f"Compilation took {elapsed}s.")

    set_data_fn = (
        jax.jit(set_data_fn, donate_argnums=(0,), keep_unused=True)
        .lower(dx, d.ctrl, d.act, d.xfrc_applied, d.qpos, d.qvel, d.time)
        .compile()
    )

    # Sync NumPy state to JAX environment initially
    logger.info("Syncing initial NumPy state to JAX environment...")
    state_jax, obs_jax = sync_np_to_jax(env_jax, state_jax, env_np)
    dx = state_jax.mjx_data
    
    # Get initial NumPy observation
    obs_np = env_np.compute_observation()
    
    # Compare initial states
    logger.info("Comparing initial states...")
    compare_states(env_jax, state_jax, obs_jax, env_np, obs_np, step_count=0)

    viewer = mujoco.viewer.launch_passive(m, d, key_callback=key_callback)

    idx = 2
    # Use a simple constant normalized action for testing (both envs expect normalized actions in [-1, 1])
    # SceneEnvJax.step and SceneEnv.step both expect normalized actions and unnormalize internally
    action_normalized = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
    action_normalized = action_normalized.at[idx].set(0.01)  # Small normalized action
    action_np_normalized = np.asarray(jax.device_get(action_normalized))

    with viewer:
        opt = copy.copy(m.opt)
        n_iters = 0
        all_diffs = []
        
        while True:
            logger.debug("Iteration {}".format(n_iters))
            start = time.time()

            dx = set_data_fn(dx, d.ctrl, d.act, d.xfrc_applied, d.qpos, d.qvel, d.time)
            state_jax = state_jax._replace(mjx_data=dx)

            if _VIEWER_GLOBAL_STATE["running"]:
                print(f"\n=== Step {n_iters + 1} ===")
                
                # Compare action processing before stepping
                action_diffs = compare_action_processing(
                    env_jax, state_jax, action_normalized,
                    env_np, action_np_normalized,
                    step_count=n_iters + 1
                )

                # Step JAX environment (expects normalized action)
                logger.debug("Stepping JAX environment...")
                out_jax: ManipStep = env_step_fn(state_jax, action_normalized)
                dx = out_jax.next_state.mjx_data
                state_jax = out_jax.next_state
                obs_jax = out_jax.observation
                
                # Step NumPy environment (expects normalized action)
                logger.debug("Stepping NumPy environment...")
                obs_np, reward_np, terminated_np, truncated_np, info_np = env_np.step(action_np_normalized)
                
                # Compare states after step
                diffs = compare_states(
                    env_jax, state_jax, obs_jax, 
                    env_np, obs_np, 
                    step_count=n_iters + 1
                )
                
                if diffs or action_diffs:
                    all_diffs.append({
                        "step": n_iters + 1,
                        "action_diffs": action_diffs,
                        "state_diffs": diffs,
                    })
                    
                    # Log detailed info for first few differences
                    if len(all_diffs) <= 3:
                        if action_diffs:
                            logger.info(f"Action diff info: {action_diffs}")
                        if diffs:
                            logger.info(f"State diff info: {diffs}")

            # Copy only the fields needed for visualization to avoid shape mismatch
            # errors with sparse flex fields (flexedge_J has shape (0, nv) in MJX
            # but (0,) in MuJoCo).
            d.qpos[:] = jax.device_get(dx.qpos)
            d.qvel[:] = jax.device_get(dx.qvel)
            d.time = jax.device_get(dx.time)
            mujoco.mj_forward(m, d)
            viewer.sync()

            elapsed = time.time() - start
            if elapsed < m.opt.timestep:
                time.sleep(5 * m.opt.timestep - elapsed)

            n_iters += 1
            
            # Stop after a certain number of iterations for debugging
            if n_iters >= 1:
                logger.info(f"Stopped after {n_iters} iterations")
                logger.info(f"Total steps with differences: {len(all_diffs)}")
                if all_diffs:
                    logger.info(f"First difference at step: {all_diffs[0]['step']}")
                    print_diff_summary(all_diffs)
                break


def print_diff_summary(all_diffs: list) -> None:
    """Print a summary of all differences found."""
    logger.info("\n" + "=" * 60)
    logger.info("DIFFERENCE SUMMARY")
    logger.info("=" * 60)
    
    # Count occurrences of each type of difference
    action_diff_counts = {}
    state_diff_counts = {}
    
    for diff_entry in all_diffs:
        for key in diff_entry.get("action_diffs", {}):
            action_diff_counts[key] = action_diff_counts.get(key, 0) + 1
        for key in diff_entry.get("state_diffs", {}):
            state_diff_counts[key] = state_diff_counts.get(key, 0) + 1
    
    if action_diff_counts:
        logger.info("\nAction processing differences:")
        for key, count in sorted(action_diff_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {key}: {count} occurrences")
    
    if state_diff_counts:
        logger.info("\nState/observation differences:")
        for key, count in sorted(state_diff_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {key}: {count} occurrences")
    
    # Show the first occurrence of each type of difference
    logger.info("\nFirst occurrence details:")
    shown_keys = set()
    for diff_entry in all_diffs:
        step = diff_entry["step"]
        for key, val in diff_entry.get("action_diffs", {}).items():
            if key not in shown_keys:
                logger.info(f"\n  {key} (first at step {step}):")
                if "max_diff" in val:
                    logger.info(f"    max_diff: {val['max_diff']:.6f}")
                if "jax" in val and "np" in val:
                    if isinstance(val["jax"], (int, float)):
                        logger.info(f"    jax: {val['jax']:.6f}")
                        logger.info(f"    np: {val['np']:.6f}")
                shown_keys.add(key)
        
        for key, val in diff_entry.get("state_diffs", {}).items():
            if key not in shown_keys:
                logger.info(f"\n  {key} (first at step {step}):")
                if "max_diff" in val:
                    logger.info(f"    max_diff: {val['max_diff']:.6f}")
                if "max_idx" in val:
                    logger.info(f"    max_idx: {val['max_idx']}")
                if "label" in val:
                    logger.info(f"    label: {val['label']}")
                if "jax_val" in val and "np_val" in val:
                    logger.info(f"    jax_val: {val['jax_val']:.6f}")
                    logger.info(f"    np_val: {val['np_val']:.6f}")
                shown_keys.add(key)
    
    logger.info("\n" + "=" * 60)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
