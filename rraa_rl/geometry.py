import jax_dataclasses as jdc
import jax.numpy as jnp

@jdc.pytree_dataclass
class AABB:
    """Axis-Aligned Bounding Box (AABB) in 2D space."""

    minpos: jnp.ndarray
    maxpos: jnp.ndarray

def dist_pt_to_aabb(pt: jnp.ndarray, aabb: AABB) -> jnp.ndarray:
    """Compute the squared distance from a point to an AABB."""
    q = jnp.minimum(jnp.maximum(pt, aabb.minpos), aabb.maxpos)
    return jnp.linalg.norm(pt - q, axis=-1)

    # delta_lower = jnp.maximum(0.0, aabb.minpos - pt)
    # delta_upper = jnp.maximum(0.0, pt - aabb.maxpos)
    # delta = delta_lower + delta_upper
    # return jnp.sum(delta**2)