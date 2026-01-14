import jax.numpy as jnp


def softminimum(x, axis=-1, temperature=1.0):
    """Compute the soft minimum of a tensor along a specified axis.

    Args:
        x: A JAX array.
        axis: The axis along which to compute the soft minimum.
        temperature: A positive scalar that controls the "softness" of the minimum.

    Returns:
        A JAX array containing the soft minimum values.
    """
    scaled_x = -x / temperature
    log_sum_exp = jnp.log(jnp.sum(jnp.exp(scaled_x), axis=axis, keepdims=True))
    softmin = -temperature * log_sum_exp
    return jnp.squeeze(softmin, axis=axis)
