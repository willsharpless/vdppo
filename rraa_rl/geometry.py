import jax_dataclasses as jdc
import jax.numpy as jnp

@jdc.pytree_dataclass
class AABB:
    """Axis-Aligned Bounding Box (AABB) in 2D space."""

    min_pos: jnp.ndarray
    max_pos: jnp.ndarray

def dist_pt_to_aabb(pt: jnp.ndarray, aabb: AABB) -> jnp.ndarray:
    """Compute the squared distance from a point to an AABB."""
    delta_lower = jnp.maximum(0.0, aabb.min_pos - pt)
    delta_upper = jnp.maximum(0.0, pt - aabb.max_pos)
    delta = delta_lower + delta_upper
    return jnp.sum(delta**2)