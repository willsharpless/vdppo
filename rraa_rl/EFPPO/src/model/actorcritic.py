from typing import Sequence, Type

import distrax
import jax
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp

import einops as ei
import flax.linen as nn
from flax.linen import initializers
from flax.linen.initializers import constant, orthogonal

class ActorCritic_Continuous(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class ActorCritic_Discrete(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Policy_Network(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        return pi


class Policy_Network_Discrete(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        return pi


class Value_Network(nn.Module):
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        critic = nn.Dense(
            256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return jnp.squeeze(critic, axis=-1)


class Actor_Network_SAC(nn.Module):
    action_dim: Sequence[int]
    action_scale: Sequence[int]
    action_bias: Sequence[int]

    @nn.compact
    def __call__(self, state, rng):
        log_std_max = 2
        log_std_min = -6

        output = nn.Dense(
            256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(state)
        output = nn.tanh(output)
        output = nn.Dense(
            256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(output)
        output = nn.tanh(output)

        mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(output)

        output = nn.Dense(
            self.action_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(output)

        log_std = nn.tanh(output)

        # Rescale log_std to ensure it is within range [log_std_min, log_std_max].
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = jnp.exp(log_std)

        x_t = self._sample(mean, std, rng)
        y_t = nn.tanh(x_t)

        action = y_t * self.action_scale + self.action_bias

        log_prob = self._log_prob(mean, std, x_t)

        log_prob -= jnp.log(self.action_scale * (1 - y_t**2) + 1e-6)
        log_prob = jnp.sum(log_prob, axis=-1, keepdims=True)

        mean = nn.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob.squeeze()

    def _log_prob(self, mean, std, value):
        var = std**2
        log_scale = jnp.log(std)
        return -((value - mean) ** 2) / (2 * var) - log_scale - jnp.log(jnp.sqrt(2 * jnp.pi))

    def _sample(self, mean, std, rng):
        rng, _rng = jax.random.split(rng)
        return jax.random.normal(_rng, shape=mean.shape) * std + mean

class Representation_Network_SAC(nn.Module):
    @nn.compact
    def __call__(self, state, action, future_state):
        output_1 = jnp.concatenate([state, action], axis=-1)
        output_1 = nn.Dense(
            256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(output_1)
        output_1 = nn.tanh(output_1)
        output_1 = nn.Dense(
            256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(output_1)
        output_1 = nn.tanh(output_1)
        output_1 = nn.Dense(
            64, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(output_1)

        output_2 = nn.Dense(
            256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(future_state)
        output_2 = nn.tanh(output_2)
        output_2 = nn.Dense(
            256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(output_2)
        output_2 = nn.tanh(output_2)
        output_2 = nn.Dense(
            64, kernel_init=orthogonal(jnp.sqrt(0.01)), bias_init=constant(0.0)
        )(output_2)

        return output_1.squeeze(), output_2.squeeze()

class IQE(nn.Module):
    dim_per_component: int

    encoder: type[nn.Module]

    @nn.compact
    def __call__(self, x, y):
        x, y = jnp.broadcast_arrays(x, y)

        encoder = self.encoder()
        feat_x, feat_y = jax.vmap(encoder)(jnp.stack([x, y], axis=0))
        assert feat_x.shape[-1] == feat_y.shape[-1]
        input_size = feat_x.shape[-1]

        assert input_size % self.dim_per_component == 0
        n_components = input_size // self.dim_per_component

        # 1: Interpret as matrices.
        def reshape_to_mat(arr):
            return ei.rearrange(arr, "... (nc dim) -> ... nc dim", nc=n_components, dim=self.dim_per_component)

        x_mat, y_mat = reshape_to_mat(feat_x), reshape_to_mat(feat_y)

        # 2: Run IQE
        b_d = iqe(x_mat, y_mat)

        # 3: Use MaxMean to get a scalar.
        dist = MaxMean()(b_d[None]).squeeze(0)
        assert dist.shape == x.shape[:-1]

        return dist

def iqe(cD_x, cD_y):
    """Compute the IQE. We do that by first sorting, then merging all intervals,
    then summing the remaining disjoint intervals.
    """
    batch_shape = cD_x.shape[:-2]
    n_components, D = cD_x.shape[-2:]  # D: dim_per_component
    cD_valid = cD_x < cD_y

    # Sort the endpoints so we can merge intervals.
    cD2_xy = jnp.concatenate([cD_x, cD_y], axis=-1)
    iota = lax.broadcasted_iota(jnp.int_, cD2_xy.shape, dimension=cD2_xy.ndim - 1)
    cD2_vals, cD2_idxs = lax.sort_key_val(cD2_xy, iota, dimension=-1)

    # Mask out invalid entries in the sorted array, and identify whether it is the left or right end of the interval.
    cD2_isvalid = jnp.take_along_axis(cD_valid, cD2_idxs % D, axis=-1)
    cD2_endpoint_direction = cD2_isvalid * jnp.where(cD2_idxs < D, -1, 1)

    # Merge nested intervals.
    cD2_endpoint_depth = jnp.cumsum(cD2_endpoint_direction, axis=-1)
    cD2_flat_endpoints = (cD2_endpoint_depth < 0) * (-1.0)
    cD2_merged_endpoints = jnp.concatenate(
        [cD2_flat_endpoints[..., 0:1], jnp.diff(cD2_flat_endpoints, axis=-1)], axis=-1
    )

    # Sum of right endpoint - left endpoint.
    c_sum_of_unions = jnp.sum(cD2_vals * cD2_merged_endpoints, axis=-1)
    assert c_sum_of_unions.shape == batch_shape + (n_components,)

    return c_sum_of_unions


class MaxMean(nn.Module):
    @nn.compact
    def __call__(self, d):
        raw_alpha_init = initializers.constant(-1)
        raw_alpha = self.param("raw_alpha", raw_alpha_init, (1,))
        alpha = jnn.sigmoid(raw_alpha)
        return (1 - alpha) * d.mean(axis=-1) + alpha * d.max(axis=-1)