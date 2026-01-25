"""JAX-compatible SE3 implementation for MJX environments."""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from rraa_rl.envs.manipspace.lie.so3_jax import SO3, get_epsilon, skew, so3_exp, so3_log, quat_multiply, quat_rotate

_IDENTITY_WXYZ_XYZ = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


class SE3(NamedTuple):
    """Special Euclidean group for proper rigid transforms in 3D (JAX version).

    Internal parameterization is (qw, qx, qy, qz, x, y, z).
    Tangent parameterization is (vx, vy, vz, omega_x, omega_y, omega_z).
    """
    wxyz_xyz: jax.Array

    @staticmethod
    def identity() -> SE3:
        return SE3(wxyz_xyz=_IDENTITY_WXYZ_XYZ)

    @staticmethod
    def from_rotation_and_translation(rotation: SO3, translation: jax.Array) -> SE3:
        return SE3(wxyz_xyz=jnp.concatenate([rotation.wxyz, translation]))

    @staticmethod
    def from_matrix(matrix: jax.Array) -> SE3:
        """Create SE3 from a 4x4 homogeneous transformation matrix."""
        rotation = SO3.from_matrix(matrix[:3, :3])
        translation = matrix[:3, 3]
        return SE3.from_rotation_and_translation(rotation, translation)

    def rotation(self) -> SO3:
        return SO3(wxyz=self.wxyz_xyz[:4])

    def translation(self) -> jax.Array:
        return self.wxyz_xyz[4:]

    def as_matrix(self) -> jax.Array:
        """Convert to 4x4 homogeneous transformation matrix."""
        hmat = jnp.eye(4)
        hmat = hmat.at[:3, :3].set(self.rotation().as_matrix())
        hmat = hmat.at[:3, 3].set(self.translation())
        return hmat

    @staticmethod
    def exp(tangent: jax.Array) -> SE3:
        """Exponential map from tangent space to SE3."""
        return SE3(wxyz_xyz=se3_exp(tangent))

    def log(self) -> jax.Array:
        """Logarithmic map from SE3 to tangent space."""
        return se3_log(self.wxyz_xyz)

    def inverse(self) -> SE3:
        """Compute inverse transformation."""
        R_inv = self.rotation().inverse()
        t_inv = -quat_rotate(R_inv.wxyz, self.translation())
        return SE3.from_rotation_and_translation(R_inv, t_inv)

    def normalize(self) -> SE3:
        """Normalize the rotation component."""
        return SE3.from_rotation_and_translation(
            self.rotation().normalize(),
            self.translation()
        )

    def apply(self, target: jax.Array) -> jax.Array:
        """Apply transformation to a 3D point."""
        return quat_rotate(self.rotation().wxyz, target) + self.translation()

    def multiply(self, other: SE3) -> SE3:
        """Compose two SE3 transformations."""
        return se3_multiply(self, other)


def se3_exp(tangent: jax.Array) -> jax.Array:
    """Exponential map from tangent space to SE3 (wxyz_xyz format)."""
    v = tangent[:3]  # linear velocity
    omega = tangent[3:]  # angular velocity

    theta_squared = jnp.dot(omega, omega)
    eps = get_epsilon(tangent.dtype)

    # Rotation part
    rot_wxyz = so3_exp(omega)

    # Translation part using V matrix
    theta_squared_safe = jnp.where(theta_squared < eps, 1.0, theta_squared)
    theta_safe = jnp.sqrt(theta_squared_safe)
    skew_omega = skew(omega)

    # V matrix for Taylor expansion (small angle)
    rot_mat = SO3(wxyz=rot_wxyz).as_matrix()

    # V matrix for regular case
    V_regular = (
        jnp.eye(3)
        + (1.0 - jnp.cos(theta_safe)) / theta_squared_safe * skew_omega
        + (theta_safe - jnp.sin(theta_safe)) / (theta_squared_safe * theta_safe) * (skew_omega @ skew_omega)
    )

    V = jnp.where(theta_squared < eps, rot_mat, V_regular)
    translation = V @ v

    return jnp.concatenate([rot_wxyz, translation])


def se3_log(wxyz_xyz: jax.Array) -> jax.Array:
    """Logarithmic map from SE3 to tangent space."""
    rot_wxyz = wxyz_xyz[:4]
    translation = wxyz_xyz[4:]

    omega = so3_log(rot_wxyz)
    theta_squared = jnp.dot(omega, omega)
    eps = get_epsilon(wxyz_xyz.dtype)

    theta_squared_safe = jnp.where(theta_squared < eps, 1.0, theta_squared)
    theta_safe = jnp.sqrt(theta_squared_safe)
    half_theta_safe = 0.5 * theta_safe
    skew_omega = skew(omega)

    # V_inv for Taylor expansion
    V_inv_taylor = jnp.eye(3) - 0.5 * skew_omega + (skew_omega @ skew_omega) / 12.0

    # V_inv for regular case
    V_inv_regular = (
        jnp.eye(3)
        - 0.5 * skew_omega
        + (1.0 - theta_safe * jnp.cos(half_theta_safe) / (2.0 * jnp.sin(half_theta_safe)))
        / theta_squared_safe * (skew_omega @ skew_omega)
    )

    V_inv = jnp.where(theta_squared < eps, V_inv_taylor, V_inv_regular)
    v = V_inv @ translation

    return jnp.concatenate([v, omega])


def se3_multiply(a: SE3, b: SE3) -> SE3:
    """Multiply two SE3 transformations."""
    rot_a = a.rotation()
    rot_b = b.rotation()
    rot_result = SO3(wxyz=quat_multiply(rot_a.wxyz, rot_b.wxyz))
    trans_result = quat_rotate(rot_a.wxyz, b.translation()) + a.translation()
    return SE3.from_rotation_and_translation(rot_result, trans_result)
