import jax.numpy as jnp
import jax_dataclasses as jdc

from vdppo.common.jax_types import BoolScalar


@jdc.pytree_dataclass
class AABB:
    """Axis-Aligned Bounding Box (AABB) in 2D space."""

    minpos: jnp.ndarray
    maxpos: jnp.ndarray


@jdc.pytree_dataclass
class RectCenterExtent:
    center: jnp.ndarray
    extent: jnp.ndarray


@jdc.pytree_dataclass
class LineSegment:
    p0: jnp.ndarray
    p1: jnp.ndarray


def dist_pt_to_rect(pt: jnp.ndarray, rect: RectCenterExtent) -> jnp.ndarray:
    """Compute the squared distance from a point to a rectangle defined by center and extent."""
    delta = jnp.abs(pt - rect.center) - rect.extent
    delta_clamped = jnp.maximum(delta, 0.0)
    return jnp.linalg.norm(delta_clamped, axis=-1)


def dist_pt_to_aabb(pt: jnp.ndarray, aabb: AABB) -> jnp.ndarray:
    """Compute the squared distance from a point to an AABB."""
    q = jnp.minimum(jnp.maximum(pt, aabb.minpos), aabb.maxpos)
    return jnp.linalg.norm(pt - q, axis=-1)

    # delta_lower = jnp.maximum(0.0, aabb.minpos - pt)
    # delta_upper = jnp.maximum(0.0, pt - aabb.maxpos)
    # delta = delta_lower + delta_upper
    # return jnp.sum(delta**2)


def segment_intersects_aabb(seg: LineSegment, aabb: AABB, eps: float = 1e-12) -> BoolScalar:
    """Slab method to test if a line segment intersects an AABB."""
    p0 = jnp.asarray(seg.p0)
    p1 = jnp.asarray(seg.p1)
    mn = jnp.asarray(aabb.minpos)
    mx = jnp.asarray(aabb.maxpos)

    d = p1 - p0  # (2,)
    parallel = jnp.abs(d) < eps  # (2,)

    # If parallel to an axis, p0 must be inside that slab.
    inside_slab = (p0 >= mn) & (p0 <= mx)  # (2,)
    parallel_ok = jnp.all(~parallel | inside_slab)  # ()

    # Safe division; then neutralize the t-interval for parallel axes.
    safe_d = jnp.where(parallel, 1.0, d)  # (2,)
    inv_d = 1.0 / safe_d  # (2,)

    t0 = (mn - p0) * inv_d  # (2,)
    t1 = (mx - p0) * inv_d  # (2,)

    t_enter_axis = jnp.minimum(t0, t1)  # (2,)
    t_exit_axis = jnp.maximum(t0, t1)  # (2,)

    # Parallel axes impose no constraint on t: [-inf, +inf]
    t_enter_axis = jnp.where(parallel, -jnp.inf, t_enter_axis)
    t_exit_axis = jnp.where(parallel, jnp.inf, t_exit_axis)

    tmin = jnp.max(t_enter_axis)  # ()
    tmax = jnp.min(t_exit_axis)  # ()

    # Overlap with segment parameter range [0, 1]
    hit = parallel_ok & (tmax >= tmin) & (tmax >= 0.0) & (tmin <= 1.0)
    return hit
