import einops as ei
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np


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


def tree_where_dim0(cond, x_tree, y_tree, which=jnp):
    def tree_where_inner(x, y):
        # x: (b, ...)
        # y: (b, ...)
        # cond: (b, )

        # Get the full shape by broadcasting x and y.
        full_shape = np.broadcast_shapes(x.shape, y.shape)

        cond_reshaped = which.reshape(cond, (cond.shape[0],) + (1,) * (len(full_shape) - 1))
        return which.where(cond_reshaped, x, y)

    return jtu.tree_map(tree_where_inner, x_tree, y_tree)


def tree_cat(trees, axis=0, which=jnp):
    def tree_cat_inner(*args):
        return which.concatenate(args, axis=axis)

    return jtu.tree_map(tree_cat_inner, *trees)


def switch01(arr: jnp.ndarray):
    # Switch the first two axes of an array.
    assert arr.ndim >= 2
    return ei.rearrange(arr, "b0 b1 ... -> b1 b0 ...")
