"""JAX-compatible differential inverse kinematics controller for MJX environments."""

from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from rraa_rl.envs import mjx_patch

PI = jnp.pi
PI_2 = 2 * jnp.pi


def angle_diff(q1: jax.Array, q2: jax.Array) -> jax.Array:
    """Compute angular difference between two angles."""
    return jnp.mod(q1 - q2 + PI, PI_2) - PI


class DiffIKControllerJax:
    """Differential inverse kinematics controller (JAX/MJX version).

    This controller uses JAX for JIT compilation and vectorization.
    It requires an MJX model for forward kinematics.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        sites: list,
        qpos0: jax.Array = None,
        damping_coeff: float = 1e-12,
        max_angle_change: float = jnp.radians(45),
    ):
        """Initialize the differential IK controller.

        Args:
            model: MuJoCo model (will be converted to MJX model internally)
            sites: List of site names to control
            qpos0: Default joint positions (for null-space control)
            damping_coeff: Damping coefficient for regularization
            max_angle_change: Maximum joint angle change per iteration
        """
        self.impl = "warp"
        self._mj_model = model
        self._mjx_model = mjx.put_model(model, impl=self.impl)
        self._qpos0 = qpos0
        self._max_angle_change = max_angle_change

        # Number of sites
        self._ns = len(sites)
        self._site_ids = [model.site(s).id for s in sites]

        # Precompute damping matrix
        self._damping = damping_coeff * jnp.eye(6 * self._ns)
        self._nv = model.nv

        # JIT compile the solve function
        # self._solve_jit = jax.jit(partial(self._solve_step, self._mj_model, self._mjx_model))
        self._solve_jit = partial(self._solve_step, self._mj_model, self._mjx_model)

    def _forward_kinematics(self, mjx_model: mjx.Model, mjx_data: mjx.Data) -> mjx.Data:
        """Perform forward kinematics to update site positions."""
        mjx_data = mjx.kinematics(mjx_model, mjx_data)
        # mjx_data = mjx.com_pos(mjx_model, mjx_data)
        mjx_data = mjx_patch.com_pos(mjx_model, mjx_data)
        return mjx_data

    def _compute_error(
        self,
        mjx_data: mjx.Data,
        target_pos: jax.Array,
        target_quat: jax.Array,
    ) -> jax.Array:
        """Compute position and orientation error for all sites."""
        errors = []
        for i, site_id in enumerate(self._site_ids):
            # Position error
            pos_err = target_pos[i] - mjx_data.site_xpos[site_id]

            # Orientation error
            site_mat = mjx_data.site_xmat[site_id].reshape(3, 3)
            site_quat = mat_to_quat_jax(site_mat)
            site_quat_inv = site_quat * jnp.array([1.0, -1.0, -1.0, -1.0])
            err_quat = quat_mul_jax(target_quat[i], site_quat_inv)
            ori_err = quat_to_vel_jax(err_quat)

            errors.append(jnp.concatenate([pos_err, ori_err]))

        return jnp.stack(errors).ravel()

    def _compute_jacobian(
        self,
        mjx_model: mjx.Model,
        mjx_data: mjx.Data,
    ) -> jax.Array:
        """Compute site Jacobians for all controlled sites."""
        jacs = []
        for site_id in self._site_ids:
            jacp, jacr = mjx_patch.jac_site(mjx_model, mjx_data, site_id)
            assert jacp.shape == (self._nv, 3)
            assert jacr.shape == (self._nv, 3)
            # mjx.jac_site returns (nv, 3) shaped arrays, need to transpose to (3, nv)
            jac = jnp.vstack([jacp.T, jacr.T])
            # jac = jnp.vstack([jacp, jacr])
            jacs.append(jac)
        return jnp.vstack(jacs)

    def _solve_step(
        self,
        mj_model: mujoco.MjModel,
        mjx_model: mjx.Model,
        qpos: jax.Array,
        target_pos: jax.Array,
        target_quat: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """Single IK iteration step.

        Returns:
            Updated qpos and error magnitude
        """
        # Create data and set qpos
        mjx_data = mjx.make_data(mj_model, impl=self.impl)
        mjx_data = mjx_data.replace(qpos=qpos)

        # Forward kinematics
        mjx_data = self._forward_kinematics(mjx_model, mjx_data)

        # Compute error
        err = self._compute_error(mjx_data, target_pos, target_quat)
        err_reshaped = err.reshape(self._ns, 6)

        # Compute Jacobian
        jac = self._compute_jacobian(mjx_model, mjx_data)

        # Solve using damped least squares
        H = jac @ jac.T + self._damping
        update = jac.T @ jnp.linalg.solve(H, err)

        # Apply null-space control if qpos0 is specified
        if self._qpos0 is not None:
            jac_pinv = jnp.linalg.pinv(H)
            q_err = angle_diff(self._qpos0, qpos)
            null_proj = jnp.eye(self._nv) - (jac.T @ jac_pinv) @ jac
            update = update + null_proj @ q_err

        # Scale update to respect max angle change
        update_max = jnp.max(jnp.abs(update))
        scale = jnp.where(update_max > self._max_angle_change, self._max_angle_change / update_max, 1.0)
        update = update * scale

        # Integrate update
        new_qpos = qpos + update

        # Compute error magnitude
        pos_err = jnp.linalg.norm(err_reshaped[:, :3])
        ori_err = jnp.linalg.norm(err_reshaped[:, 3:])

        return new_qpos, jnp.array([pos_err, ori_err])

    def solve(
        self,
        pos: jax.Array,
        quat: jax.Array,
        curr_qpos: jax.Array,
        max_iters: int = 20,
        pos_thresh: float = 1e-4,
        ori_thresh: float = 1e-4,
    ) -> jax.Array:
        """Solve IK for target position and orientation.

        Args:
            pos: Target position(s) of shape (n_sites, 3) or (3,)
            quat: Target quaternion(s) (wxyz) of shape (n_sites, 4) or (4,)
            curr_qpos: Current joint positions
            max_iters: Maximum number of iterations
            pos_thresh: Position error threshold
            ori_thresh: Orientation error threshold

        Returns:
            Joint positions that achieve the target pose
        """
        pos = jnp.atleast_2d(pos)
        quat = jnp.atleast_2d(quat)

        qpos = curr_qpos

        # def body_fn(carry):
        #     qpos, i = carry
        #     new_qpos, errs = self._solve_jit(qpos, pos, quat)
        #     return (new_qpos, i + 1)
        #
        # def cond_fn(carry):
        #     qpos, i = carry
        #     _, errs = self._solve_jit(qpos, pos, quat)
        #     converged = (errs[0] <= pos_thresh) & (errs[1] <= ori_thresh)
        #     return ~converged & (i < max_iters)
        #
        # # Run IK loop
        # qpos, _ = jax.lax.while_loop(cond_fn, body_fn, (qpos, 0))

        qpos, _ = self._solve_jit(qpos, pos, quat)

        return qpos

    def solve_numpy(
        self,
        pos,
        quat,
        curr_qpos,
        max_iters: int = 20,
        pos_thresh: float = 1e-4,
        ori_thresh: float = 1e-4,
    ):
        """Solve IK and return numpy array (convenience wrapper)."""
        import numpy as np

        pos = jnp.asarray(pos)
        quat = jnp.asarray(quat)
        curr_qpos = jnp.asarray(curr_qpos)

        result = self.solve(pos, quat, curr_qpos, max_iters, pos_thresh, ori_thresh)
        return np.asarray(result)


# Helper functions for quaternion operations
def mat_to_quat_jax(mat: jax.Array) -> jax.Array:
    """Convert 3x3 rotation matrix to quaternion (wxyz format)."""
    mat = mat.reshape(3, 3)
    trace = jnp.trace(mat)

    def case0():
        s = 0.5 / jnp.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (mat[2, 1] - mat[1, 2]) * s
        y = (mat[0, 2] - mat[2, 0]) * s
        z = (mat[1, 0] - mat[0, 1]) * s
        return jnp.array([w, x, y, z])

    def case1():
        s = 2.0 * jnp.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2])
        w = (mat[2, 1] - mat[1, 2]) / s
        x = 0.25 * s
        y = (mat[0, 1] + mat[1, 0]) / s
        z = (mat[0, 2] + mat[2, 0]) / s
        return jnp.array([w, x, y, z])

    def case2():
        s = 2.0 * jnp.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2])
        w = (mat[0, 2] - mat[2, 0]) / s
        x = (mat[0, 1] + mat[1, 0]) / s
        y = 0.25 * s
        z = (mat[1, 2] + mat[2, 1]) / s
        return jnp.array([w, x, y, z])

    def case3():
        s = 2.0 * jnp.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1])
        w = (mat[1, 0] - mat[0, 1]) / s
        x = (mat[0, 2] + mat[2, 0]) / s
        y = (mat[1, 2] + mat[2, 1]) / s
        z = 0.25 * s
        return jnp.array([w, x, y, z])

    diag = jnp.array([trace, mat[0, 0], mat[1, 1], mat[2, 2]])
    idx = jnp.argmax(diag)
    result = jax.lax.switch(idx, [case0, case1, case2, case3])
    return jnp.where(result[0] < 0, -result, result)


def quat_mul_jax(q1: jax.Array, q2: jax.Array) -> jax.Array:
    """Multiply two quaternions (wxyz format)."""
    w0, x0, y0, z0 = q1
    w1, x1, y1, z1 = q2
    return jnp.array(
        [
            -x0 * x1 - y0 * y1 - z0 * z1 + w0 * w1,
            x0 * w1 + y0 * z1 - z0 * y1 + w0 * x1,
            -x0 * z1 + y0 * w1 + z0 * x1 + w0 * y1,
            x0 * y1 - y0 * x1 + z0 * w1 + w0 * z1,
        ]
    )


def quat_to_vel_jax(quat: jax.Array) -> jax.Array:
    """Convert quaternion to angular velocity (axis-angle representation)."""
    # For small angles: omega ≈ 2 * (qx, qy, qz) / dt, with dt=1
    # This is equivalent to log(q) for unit quaternions
    xyz = quat[1:]
    w = quat[0]

    # Handle the case where w is negative (quaternion double cover)
    sign = jnp.where(w < 0, -1.0, 1.0)
    xyz = xyz * sign
    w = w * sign

    # Compute angle
    norm_xyz = jnp.linalg.norm(xyz)
    angle = 2.0 * jnp.arctan2(norm_xyz, w)

    # Normalize axis and scale by angle
    axis = jnp.where(norm_xyz > 1e-10, xyz / norm_xyz, jnp.array([0.0, 0.0, 1.0]))
    return angle * axis
