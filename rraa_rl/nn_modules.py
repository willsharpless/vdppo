from typing import Sequence

import flax.linen as nn

from rraa_rl.distribution import tfd
from rraa_rl.mlp import MLP
from rraa_rl.nn_utils import default_nn_init, scaled_init


class MAMultiDiscretePolicy(nn.Module):
    """Each agent has a multi-discrete policy."""

    hidden_dims: Sequence[int]
    # For each agent, a list of number of actions for each action dimension.
    n_actions_per_agent: list[list[int]]

    scale_final: float = 0.01

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
                dist = tfd.Categorical(logits=logits_this_action_dim)
                agent_dists.append(dist)
                start_idx = end_idx

            # Combine distributions for each action dimension, for this action.
            agent_multi_dist = tfd.Blockwise(agent_dists)
            dists.append(agent_multi_dist)

        # When sampled, this returns a list of actions, one per agent.
        dist = tfd.JointDistributionSequential(dists)
        return dist


class VDValue(nn.Module):
    hidden_dims: Sequence[int]
    n_out: int

    @nn.compact
    def __call__(self, obs):
        x = MLP(hid_sizes=self.hidden_dims, act=nn.tanh, act_final=True)(obs)
        v = nn.Dense(self.n_out, kernel_init=default_nn_init(), name="ValueHead")(x)
        return v
