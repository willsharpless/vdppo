from typing import Any, Callable, Sequence, TypeVar

import einops as ei
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

_F = TypeVar("_F", bound=Callable)


def softmaximum(x, axis=-1, temperature=1.0):
    """Compute the soft maximum of a tensor along a specified axis.

    Args:
        x: A JAX array.
        axis: The axis along which to compute the soft maximum.
        temperature: A positive scalar that controls the "softness" of the maximum.

    Returns:
        A JAX array containing the soft maximum values.
    """
    scaled_x = x / temperature
    log_sum_exp = jax.scipy.special.logsumexp(scaled_x, axis=axis)
    softmax = temperature * log_sum_exp
    return softmax


def softminimum(x, axis=-1, temperature=1.0):
    return -softmaximum(-x, axis=axis, temperature=temperature)


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


def tree_stack(trees, axis=0, which=jnp):
    def tree_stack_inner(*args):
        return which.stack(args, axis=axis)

    return jtu.tree_map(tree_stack_inner, *trees)


def switch01(arr: jnp.ndarray):
    # Switch the first two axes of an array.
    assert arr.ndim >= 2
    return ei.rearrange(arr, "b0 b1 ... -> b1 b0 ...")


def rep_vmap(fn: _F, rep: int, in_axes: int | Sequence[Any] = 0, **kwargs) -> _F:
    for ii in range(rep):
        fn = jax.vmap(fn, in_axes=in_axes, **kwargs)
    return fn


def jax_vmap(fn: _F, in_axes: int | Sequence[Any] = 0, out_axes: Any = 0, rep: int = None) -> _F:
    if rep is not None:
        return rep_vmap(fn, rep=rep, in_axes=in_axes, out_axes=out_axes)

    return jax.vmap(fn, in_axes, out_axes)
