import functools as ft
from typing import Any, NamedTuple

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from attrs import define
from flax import struct
from jaxtyping import PRNGKeyArray
from valtr.reachability import (DAGId, DAGNode, collect_predicate_info, extract_trigger_predicate_map,
                                temporal_nodes_topological)
from valtr.valtr import to_dag

from rraa_rl.jax_utils import softminimum
from rraa_rl.src.env.general_task.env import Env, EnvStep
from rraa_rl.src.env.general_task.herd_base import HerdBase, HerdBaseCfg, HerdBaseState


class HerdOsState(NamedTuple):
    temporal_node_idx: int
    base: HerdBaseState


@define
class HerdOsCfg:
    base: HerdBaseCfg = HerdBaseCfg()
    # What fraction of the batch is which temporal node at reset. This is in reverse topological order.
    temporal_node_fracs: list[float] = [0.6, 0.4]


class HerdOs(Env):
    """Herding environment with one or more herders and a herd of agents. The herd moves according to some fixed policy.
    The herders can influence the herd by moving around them.

    Each herd agent is a single-integrator that minimizes the soft minimum distance to the herders, the obstacles,
    and other herd agents, where the distances are scaled such that herders have larger influence.
    If the distance is large enough, the herd agents stay still.

    In the discrete action setup, each herder is a double-integrator that can accelerate / decelerate in either axis.

    action: (n_herders, 2): int, {0, 1, 2} for each axis, where 0 = -accel, 1 = no accel, 2 = accel
    """
    State = HerdOsState

    def __init__(self, cfg: HerdOsCfg = HerdOsCfg()):
        self.cfg = cfg
        self.base = HerdBase(cfg.base)

        dag_builder, dag_root = to_dag(self.specification)
        self.dag_nodes = dag_builder.nodes
        self.dag_root = dag_root
        self.pred_info = collect_predicate_info(self.dag_nodes, self.dag_root)
        # self.dag_info = extract_trigger_predicate_map(self.dag_nodes, self.dag_root)

        # root first.
        self.temporal_nodes = temporal_nodes_topological(self.dag_nodes, self.dag_root)[::-1]

    @property
    def n_agents(self) -> int:
        return self.base.n_agents

    @property
    def n_actions_per_agent(self) -> list[list[int]]:
        return self.base.n_actions_per_agent

    @property
    def n_temporal_nodes(self):
        return len(self.temporal_nodes)

    @property
    def specification(self):
        # return "F(G(herd_herded)) && G( !herder_collide ) && G( !herder_oob )"
        return "( !herder_collide && ! herder_oob ) U ( G(herd_herded && !herder_collide && ! herder_oob) )"

    def _temporal_node_idx_to_node_type(self):
        """Map from temporal node idx to node type (as index)"""
        temporal_node_types = DAGNode.get_temporal_classes_sorted()
        temporal_node_type_list = []
        for idx, node_id in enumerate(self.temporal_nodes):
            node = self.dag_nodes[node_id]
            node_fullname = f"{type(node).__module__}.{type(node).__qualname__}"
            node_type = temporal_node_types.index(node_fullname)
            temporal_node_type_list.append(node_type)

        return jnp.array(temporal_node_type_list)

    def _augment_obs(self, state: HerdOsState, base_obs: jnp.ndarray):
        """Augment the base observation with the (one hot) temporal node idx and (one hot) node type."""
        obs_node_idx = jnn.one_hot(state.temporal_node_idx, self.n_temporal_nodes)

        temporal_node_idx_to_node_type = self._temporal_node_idx_to_node_type()
        node_type = temporal_node_idx_to_node_type[state.temporal_node_idx]

        obs_node_type = jnn.one_hot(node_type, DAGNode.n_temporal_classes())
        obs = jnp.concatenate([base_obs, obs_node_idx, obs_node_type], axis=-1)

        return obs

    def step(self, state: HerdOsState, action: jnp.ndarray):
        base_step: EnvStep = self.base.step(state.base, action)
        state_new = state._replace(base=base_step.envstate)
        obs = self._augment_obs(state_new, base_step.obs)
        step = base_step._replace(envstate=state_new, obs=obs)

        return step

    def get_obs(self, state: Any) -> Any:
        base_obs = self.base.get_obs(state.base)
        obs = self._augment_obs(state, base_obs)
        return obs

    def reset(self, key: PRNGKeyArray) -> HerdOsState:
        key_base, key_node = jr.split(key)
        base_state = self.base.reset(key_base)

        # Sample temporal node idx according to configured fractions.
        node_fracs = jnp.array(self.cfg.temporal_node_fracs)
        node_fracs = node_fracs / jnp.sum(node_fracs)
        temporal_node_idx = jr.choice(key_node, a=self.n_temporal_nodes, p=node_fracs)
        state = HerdOsState(
            temporal_node_idx=temporal_node_idx,
            base=base_state,
        )
        return state
