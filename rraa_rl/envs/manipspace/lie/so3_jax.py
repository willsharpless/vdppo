"""JAX-compatible SO3 implementation for MJX environments."""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

_IDENTITY_WXYZ = jnp.array([1.0, 0.0, 0.0, 0.0])
_INVERT_QUAT_SIGN = jnp.array([1.0, -1.0, -1.0, -1.0])


def get_epsilon(dtype: jnp.dtype = jnp.float32) -> float:
    """Get epsilon for numerical stability based on dtype."""
    return jnp.where(dtype == jnp.float64, 1e-10, 1e-5)


class SO3(NamedTuple):
    """Special orthogonal group for 3D rotations (JAX version).

    Internal parameterization is (qw, qx, qy, qz).
    """
    wxyz: jax.Array

    @staticmethod
    def identity() -> SO3:
        return SO3(wxyz=_IDENTITY_WXYZ)

    @staticmethod
    def from_wxyz(wxyz: jax.Array) -> SO3:
        return SO3(wxyz=wxyz)

    @staticmethod
    def from_x_radians(theta: float) -> SO3:
        return SO3.exp(jnp.array([theta, 0.0, 0.0]))

    @staticmethod
    def from_y_radians(theta: float) -> SO3:
        return SO3.exp(jnp.array([0.0, theta, 0.0]))

    @staticmethod
    def from_z_radians(theta: float) -> SO3:
        return SO3.exp(jnp.array([0.0, 0.0, theta]))

    @staticmethod
    def from_rpy_radians(roll: float, pitch: float, yaw: float) -> SO3:
        return so3_multiply(
            so3_multiply(SO3.from_z_radians(yaw), SO3.from_y_radians(pitch)),
            SO3.from_x_radians(roll)
        )

    @staticmethod
    def from_matrix(matrix: jax.Array) -> SO3:
        """Convert a 3x3 rotation matrix to quaternion."""
        return SO3(wxyz=mat_to_quat(matrix))

    @staticmethod
    def exp(tangent: jax.Array) -> SO3:
        """Exponential map from tangent space to SO3."""
        return SO3(wxyz=so3_exp(tangent))

    def as_matrix(self) -> jax.Array:
        """Convert quaternion to 3x3 rotation matrix."""
        return quat_to_mat(self.wxyz)

    def compute_yaw_radians(self) -> jax.Array:
        """Compute yaw angle from quaternion."""
        q0, q1, q2, q3 = self.wxyz
        return jnp.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))

    def compute_roll_radians(self) -> jax.Array:
        q0, q1, q2, q3 = self.wxyz
        return jnp.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))

    def compute_pitch_radians(self) -> jax.Array:
        q0, q1, q2, q3 = self.wxyz
        return jnp.arcsin(2 * (q0 * q2 - q3 * q1))

    def log(self) -> jax.Array:
        """Logarithmic map from SO3 to tangent space."""
        return so3_log(self.wxyz)

    def inverse(self) -> SO3:
        return SO3(wxyz=self.wxyz * _INVERT_QUAT_SIGN)

    def normalize(self) -> SO3:
        return SO3(wxyz=self.wxyz / jnp.linalg.norm(self.wxyz))

    def apply(self, target: jax.Array) -> jax.Array:
        """Rotate a 3D vector by this quaternion."""
        return quat_rotate(self.wxyz, target)

    def multiply(self, other: SO3) -> SO3:
        return so3_multiply(self, other)


# Functional versions for use with JAX transformations
def so3_exp(tangent: jax.Array) -> jax.Array:
    """Exponential map from tangent space to quaternion."""
    theta_squared = jnp.dot(tangent, tangent)
    theta_pow_4 = theta_squared * theta_squared
    eps = get_epsilon(tangent.dtype)

    # Use Taylor expansion for small angles
    safe_theta = jnp.where(theta_squared < eps, 1.0, jnp.sqrt(theta_squared))
    safe_half_theta = 0.5 * safe_theta

    # Taylor expansion coefficients
    real_taylor = 1.0 - theta_squared / 8.0 + theta_pow_4 / 384.0
    imag_taylor = 0.5 - theta_squared / 48.0 + theta_pow_4 / 3840.0

    # Regular formula
    real_regular = jnp.cos(safe_half_theta)
    imag_regular = jnp.sin(safe_half_theta) / safe_theta

    # Select based on angle magnitude
    use_taylor = theta_squared < eps
    real = jnp.where(use_taylor, real_taylor, real_regular)
    imaginary = jnp.where(use_taylor, imag_taylor, imag_regular)

    return jnp.concatenate([jnp.array([real]), imaginary * tangent])


def so3_log(wxyz: jax.Array) -> jax.Array:
    """Logarithmic map from quaternion to tangent space."""
    w = wxyz[0]
    xyz = wxyz[1:]
    norm_sq = jnp.dot(xyz, xyz)
    eps = get_epsilon(wxyz.dtype)

    norm_safe = jnp.where(norm_sq < eps, 1.0, jnp.sqrt(norm_sq))
    w_safe = jnp.where(norm_sq < eps, w, 1.0)

    atan_n_over_w = jnp.arctan2(
        jnp.where(w < 0, -norm_safe, norm_safe),
        jnp.abs(w)
    )

    # Taylor expansion for small angles
    atan_factor_taylor = 2.0 / w_safe - 2.0 / 3.0 * norm_sq / (w_safe ** 3)

    # Regular formula
    atan_factor_regular = jnp.where(
        jnp.abs(w) < eps,
        jnp.where(w > 0, 1.0, -1.0) * jnp.pi / norm_safe,
        2.0 * atan_n_over_w / norm_safe
    )

    use_taylor = norm_sq < eps
    atan_factor = jnp.where(use_taylor, atan_factor_taylor, atan_factor_regular)

    return atan_factor * xyz


def so3_multiply(a: SO3, b: SO3) -> SO3:
    """Multiply two SO3 rotations."""
    return SO3(wxyz=quat_multiply(a.wxyz, b.wxyz))


def quat_multiply(q1: jax.Array, q2: jax.Array) -> jax.Array:
    """Multiply two quaternions (wxyz format)."""
    w0, x0, y0, z0 = q1
    w1, x1, y1, z1 = q2
    return jnp.array([
        -x0 * x1 - y0 * y1 - z0 * z1 + w0 * w1,
        x0 * w1 + y0 * z1 - z0 * y1 + w0 * x1,
        -x0 * z1 + y0 * w1 + z0 * x1 + w0 * y1,
        x0 * y1 - y0 * x1 + z0 * w1 + w0 * z1,
    ])


def quat_rotate(quat: jax.Array, vec: jax.Array) -> jax.Array:
    """Rotate a 3D vector by a quaternion."""
    # q * v * q^-1 where v is treated as a pure quaternion (0, vec)
    q_inv = quat * _INVERT_QUAT_SIGN
    v_quat = jnp.concatenate([jnp.zeros(1), vec])
    result = quat_multiply(quat_multiply(quat, v_quat), q_inv)
    return result[1:]


def quat_to_mat(wxyz: jax.Array) -> jax.Array:
    """Convert quaternion to 3x3 rotation matrix."""
    w, x, y, z = wxyz

    # Normalize quaternion
    norm = jnp.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    return jnp.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])


def mat_to_quat(mat: jax.Array) -> jax.Array:
    """Convert 3x3 rotation matrix to quaternion (wxyz format).
    
    Uses Shepperd's method for numerical stability.
    """
    mat = mat.reshape(3, 3)
    trace = jnp.trace(mat)

    # Four possible formulations - choose based on largest diagonal element
    def case0():
        # trace is largest
        s = 0.5 / jnp.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (mat[2, 1] - mat[1, 2]) * s
        y = (mat[0, 2] - mat[2, 0]) * s
        z = (mat[1, 0] - mat[0, 1]) * s
        return jnp.array([w, x, y, z])

    def case1():
        # mat[0,0] is largest
        s = 2.0 * jnp.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2])
        w = (mat[2, 1] - mat[1, 2]) / s
        x = 0.25 * s
        y = (mat[0, 1] + mat[1, 0]) / s
        z = (mat[0, 2] + mat[2, 0]) / s
        return jnp.array([w, x, y, z])

    def case2():
        # mat[1,1] is largest
        s = 2.0 * jnp.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2])
        w = (mat[0, 2] - mat[2, 0]) / s
        x = (mat[0, 1] + mat[1, 0]) / s
        y = 0.25 * s
        z = (mat[1, 2] + mat[2, 1]) / s
        return jnp.array([w, x, y, z])

    def case3():
        # mat[2,2] is largest
        s = 2.0 * jnp.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1])
        w = (mat[1, 0] - mat[0, 1]) / s
        x = (mat[0, 2] + mat[2, 0]) / s
        y = (mat[1, 2] + mat[2, 1]) / s
        z = 0.25 * s
        return jnp.array([w, x, y, z])

    # Select case based on which is largest
    diag = jnp.array([trace, mat[0, 0], mat[1, 1], mat[2, 2]])
    idx = jnp.argmax(diag)

    result = jax.lax.switch(idx, [case0, case1, case2, case3])

    # Ensure positive w (canonical form)
    return jnp.where(result[0] < 0, -result, result)


def skew(x: jax.Array) -> jax.Array:
    """Compute skew-symmetric matrix from a 3D vector."""
    wx, wy, wz = x
    return jnp.array([
        [0.0, -wz, wy],
        [wz, 0.0, -wx],
        [-wy, wx, 0.0],
    ])
