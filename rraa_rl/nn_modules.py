from typing import Sequence

import flax.linen as nn
import jax.nn as jnn
import jax.numpy as jnp
import jax.tree_util as jtu

from rraa_rl.distribution import BlockwiseWithMode, tfd
from rraa_rl.mlp import MLP
from rraa_rl.nn_utils import default_nn_init, scaled_init
from rraa_rl.src.env.general_task.env import AugObs, AugObsAutomata


class MAMultiDiscretePolicy(nn.Module):
    """Each agent has a multi-discrete policy."""

    hidden_dims: Sequence[int]
    # For each agent, a list of number of actions for each action dimension.
    n_actions_per_agent: list[list[int]]

    scale_final: float = 0.01

    p_max: float | None = 0.999
    """Prevent extreme probabilities, analogous to min_std for Gaussian policies.
    Do this by doing a mixture of the probabilities, which is equiv to logaddexp the logits."""

    @property
    def total_n_logits(self) -> int:
        """Total number of logits across all agents and action dimensions."""
        return sum(sum(agent_n_actions_multi) for agent_n_actions_multi in self.n_actions_per_agent)

    @nn.compact
    def __call__(self, obs) -> tfd.Distribution:
        x = MLP(hid_sizes=self.hidden_dims, act=nn.tanh, act_final=True)(obs)

        last_layer_init = scaled_init(default_nn_init(), self.scale_final)
        logits = nn.Dense(self.total_n_logits, kernel_init=last_layer_init, name="LogitsHead")(x)

        # Split logits for each agent and action dimension.
        dists = []
        start_idx = 0
        for agent_n_actions_multi in self.n_actions_per_agent:
            agent_dists = []
            for n_actions in agent_n_actions_multi:
                end_idx = start_idx + n_actions
                logits_this_action_dim = logits[..., start_idx:end_idx]

                if self.p_max is not None:
                    # Adjust the logits to prevent extreme probabilities.
                    eps = 1 - self.p_max
                    logp_this_dim = jnn.log_softmax(logits_this_action_dim, axis=-1)
                    assert logp_this_dim.shape == logits_this_action_dim.shape

                    # log( (1-eps) * p) = log( 1-eps) + logp
                    logp_a = jnp.log1p(-eps) + logp_this_dim
                    # log( eps * 1 / n_actions ) = log(eps) - log(n_actions)
                    logp_b = jnp.log(eps) - jnp.log(n_actions)

                    # log p' = log( (1-eps) * p + eps * 1/n_actions )
                    logits_this_action_dim_new = jnp.logaddexp(logp_a, logp_b)
                    assert logits_this_action_dim_new.shape == logits_this_action_dim.shape
                    logits_this_action_dim = logits_this_action_dim_new

                dist = tfd.Categorical(logits=logits_this_action_dim)
                agent_dists.append(dist)
                start_idx = end_idx

            # Combine distributions for each action dimension, for this action.
            agent_multi_dist = BlockwiseWithMode(agent_dists)
            dists.append(agent_multi_dist)

        # When sampled, this returns a list of actions, one per agent.
        dist = tfd.JointDistributionSequential(dists)
        return dist


class SeparateMAMultiDiscretePolicy(nn.Module):
    """MAMultiDiscretePolicy, but vmap'ed n_out times."""

    hidden_dims: Sequence[int]
    n_actions_per_agent: list[list[int]]
    n_out: int
    p_max: float | None = 0.999
    out_axes: int = -1

    @nn.compact
    def __call__(self, obs) -> tfd.Distribution:
        BatchPolicy = nn.vmap(
            MAMultiDiscretePolicy,
            in_axes=None,
            out_axes=self.out_axes,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.n_out,
        )
        policy = BatchPolicy(self.hidden_dims, self.n_actions_per_agent, p_max=self.p_max)(obs)
        return policy


class ScalarValue(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, obs):
        x = MLP(hid_sizes=self.hidden_dims, act=nn.tanh, act_final=True)(obs)
        v = nn.Dense(1, kernel_init=default_nn_init(), name="ValueHead")(x)
        return jnp.squeeze(v, axis=-1)


class VDValue(nn.Module):
    hidden_dims: Sequence[int]
    n_out: int

    @nn.compact
    def __call__(self, obs):
        BatchValue = nn.vmap(
            ScalarValue,
            in_axes=None,
            out_axes=-1,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.n_out,
        )
        v = BatchValue(self.hidden_dims)(obs)
        assert v.shape[-1] == self.n_out

        return v


class VDValueShared(nn.Module):
    hidden_dims: Sequence[int]
    n_out: int

    @nn.compact
    def __call__(self, obs):
        x = MLP(hid_sizes=self.hidden_dims, act=nn.tanh, act_final=True)(obs)
        v = nn.Dense(self.n_out, kernel_init=default_nn_init(), name="ValueHead")(x)
        return v


class BaseObsOnly(nn.Module):
    nn: nn.Module

    @nn.compact
    def __call__(self, obs: AugObs):
        return self.nn(obs.base)


class BothObs(nn.Module):
    nn: nn.Module

    @nn.compact
    def __call__(self, obs: AugObs):
        combined_obs = obs.combine()
        return self.nn(combined_obs)


class LearnTemporalEmbedding(nn.Module):
    nn: nn.Module
    n_temporal_nodes: int
    n_temporal_types: int
    n_embed_feats: int

    @nn.compact
    def __call__(self, obs: AugObs):
        # Learn an embedding for obs.temporal_node_idx and obs.temporal_type_idx, then concatenate to obs.base
        node_idx_embed = nn.Embed(
            num_embeddings=self.n_temporal_nodes, features=self.n_embed_feats, name="NodeIdxEmbed"
        )(obs.temporal_node_idx)
        node_type_embed = nn.Embed(
            num_embeddings=self.n_temporal_types, features=self.n_embed_feats, name="NodeTypeEmbed"
        )(obs.temporal_node_type)
        obs = jnp.concatenate([obs.base, node_idx_embed, node_type_embed], axis=-1)
        return self.nn(obs)


class IndexAtEnd(nn.Module):
    """Use the base obs only. The nn will output n_out outputs, and we index using the temporal_node_idx."""

    nn: nn.Module
    n_out: int
    index_pos: int = -1

    @nn.compact
    def __call__(self, obs: AugObs | AugObsAutomata):
        n_out = self.nn(obs.base)

        def check_dim(arr):
            assert arr.shape[self.index_pos] == self.n_out
            return arr

        jtu.tree_map(check_dim, n_out)

        match obs:
            case AugObsAutomata(automata_idx=automata_index):
                index = automata_index
            case AugObs(temporal_node_idx=temporal_node_index):
                index = temporal_node_index
            case _:
                raise ValueError(f"Unexpected obs type: {type(obs)}")

        if self.index_pos == -1:
            out = jtu.tree_map(lambda arr: arr[..., index], n_out)
        elif self.index_pos == -2:
            out = jtu.tree_map(lambda arr: arr[..., index, :], n_out)
        else:
            raise NotImplementedError("")

        return out
