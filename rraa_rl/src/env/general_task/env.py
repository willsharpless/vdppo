from typing import Any, NamedTuple

import jax
import jax.nn as jnn
import jax.random as jr
from jax import numpy as jnp
from jaxtyping import PRNGKeyArray
from valtr.reachability import (DAGAvoid, DAGConst, DAGGUMinN, DAGGUSingle, DAGId, DAGMaxN, DAGMinN, DAGNegate, DAGNode,
                                DAGReach, DAGReachAvoid, DAGVar, collect_predicate_info, get_node_parent_dict,
                                has_temporal_children, temporal_nodes_topological)
from valtr.valtr import to_dag


class EnvStep(NamedTuple):
    envstate: Any
    obs: Any
    predicates: dict
    term: bool
    trunc: bool
    info: dict


class BaseEnv:
    def reset(self, key: PRNGKeyArray) -> Any:
        raise NotImplementedError("")

    def reset_batch(self, key: PRNGKeyArray, batch_size: int) -> Any:
        b_key = jr.split(key, batch_size)
        return jax.vmap(self.reset)(b_key)


class Env:
    def __init__(self, specification: str):
        dag_builder, dag_root = to_dag(specification, dag_filename="herd_os_dag.pdf")
        self.dag_nodes = dag_builder.nodes
        self.dag_root = dag_root
        self.pred_info = collect_predicate_info(self.dag_nodes, self.dag_root)
        # self.dag_info = extract_trigger_predicate_map(self.dag_nodes, self.dag_root)

        # root first.
        self.temporal_nodes: list[DAGId] = temporal_nodes_topological(self.dag_nodes, self.dag_root)[::-1]
        self._augment_obs_names = None

        self.node_parent_dict: dict[DAGId, DAGId] = get_node_parent_dict(self.dag_nodes, self.dag_root)

    def step(self, state: Any, action: Any) -> EnvStep:
        raise NotImplementedError("")

    @property
    def temporal_node_names(self) -> list[str]:
        names = []
        for node_id in self.temporal_nodes:
            node = self.dag_nodes[node_id]
            name = f"{type(node).__name__} (%{node_id})"
            names.append(name)
        return names

    @property
    def specification(self):
        raise NotImplementedError("")

    @property
    def n_temporal_nodes(self):
        return len(self.temporal_nodes)

    @property
    def value_lims(self):
        raise NotImplementedError("")

    def reset(self, key: PRNGKeyArray) -> Any:
        raise NotImplementedError("")

    def reset_batch(self, key: PRNGKeyArray, batch_size: int) -> Any:
        b_key = jr.split(key, batch_size)
        return jax.vmap(self.reset)(b_key)

    def get_obs(self, state: Any) -> Any:
        raise NotImplementedError("")

    def get_dummy_obs(self) -> Any:
        state = self.reset(jr.key(0))
        return self.get_obs(state)

    @property
    def n_agents(self) -> int:
        """Number of controlled agents."""
        raise NotImplementedError("")

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

    def augment_obs_names(self) -> list[str]:
        if self._augment_obs_names is None:
            dummy_state = self.reset(jr.PRNGKey(0))
            _, obs_names = self._augment_obs_and_names(dummy_state)
            self._augment_obs_names = obs_names
        return self._augment_obs_names

    def _augment_obs_and_names(self, state: Any):
        """Augment the base observation with the (one hot) temporal node idx and (one hot) node type."""
        if self.n_temporal_nodes > 1:
            obs_node_idx = jnn.one_hot(state.temporal_node_idx, self.n_temporal_nodes)
            obs_node_idx_names = [f"nodeidx1h_{ii}" for ii in range(self.n_temporal_nodes)]

            temporal_node_idx_to_node_type = self._temporal_node_idx_to_node_type()
            node_type = temporal_node_idx_to_node_type[state.temporal_node_idx]

            obs_node_type = jnn.one_hot(node_type, DAGNode.n_temporal_classes())
            obs_node_type_names = [f"nodetype1h_{ii}" for ii in range(DAGNode.n_temporal_classes())]

            obs_aug = jnp.concatenate([obs_node_idx, obs_node_type], axis=-1)
            obs_names = [*obs_node_idx_names, *obs_node_type_names]
        else:
            obs_aug = jnp.array([], dtype=jnp.float32)
            obs_names = []

        return obs_aug, obs_names

    def _get_augment_obs(self, state: Any):
        obs, _ = self._augment_obs_and_names(state)
        return obs

    def _augment_obs(self, state: Any, obs: jnp.ndarray):
        obs_aug = self._get_augment_obs(state)
        return AugObs(base=obs, temporal=obs_aug)

    @property
    def eval_T(self) -> int:
        raise NotImplementedError("")


class AugObs(NamedTuple):
    """Separate the "base" observation and the observation of the temporal node."""

    base: jnp.ndarray
    temporal: jnp.ndarray

    def combine(self, which=jnp):
        return which.concatenate([self.base, self.temporal], axis=-1)


class DAGTransition(NamedTuple):
    parent: DAGId
    child: DAGId
    condition: jnp.ndarray


class TemporalNodeTransition(NamedTuple):
    parent: int
    child: int
    condition: jnp.ndarray


def evaluate_dag(
    dag_nodes: list[DAGNode],
    node_idx: DAGId,
    predicates: dict[str, jnp.ndarray],
    scratch: dict[DAGId, jnp.ndarray] | None = None,
    which=jnp,
) -> jnp.ndarray:
    if scratch is None:
        scratch: dict[DAGId, jnp.ndarray] = {}

    # Check if already computed.
    if node_idx in scratch:
        return scratch[node_idx]

    dag_node = dag_nodes[node_idx]
    match dag_node:
        case DAGConst(value=value):
            raise ValueError("Const nodes should have been removed")
        case DAGVar(name=name):
            out = predicates[name]
        case DAGNegate(arg=arg):
            out = -evaluate_dag(dag_nodes, arg, predicates, scratch)
        case DAGMinN(args=args):
            args_vals = which.stack(
                [evaluate_dag(dag_nodes, arg, predicates, scratch) for arg in args],
                axis=0,
            )
            out = which.min(args_vals, axis=0)
        case DAGMaxN(args=args):
            args_vals = which.stack(
                [evaluate_dag(dag_nodes, arg, predicates, scratch) for arg in args],
                axis=0,
            )
            out = which.max(args_vals, axis=0)
        case _:
            raise ValueError("Shouldn't have any temporal nodes.")

    scratch[node_idx] = out
    return out


def get_triggers(
    dag_nodes: list[DAGNode],
    temporal_nodes: list[DAGId],
    parent_idx: DAGId,
    node_idx: DAGId,
    predicates: dict[str, jnp.ndarray],
    t_value: jnp.ndarray,
    scratch: dict[DAGId, jnp.ndarray],
    which=jnp,
) -> list[DAGTransition]:
    node = dag_nodes[node_idx]

    # Either:
    # - max_i min_j (..., temporal)
    # -       min_j (..., temporal)
    # - a temporal node (e.g., G)

    if isinstance(node, DAGMaxN):
        min_nodes = []
        for child_id in node.args:
            child = dag_nodes[child_id]
            if isinstance(child, DAGMinN):
                min_nodes.append(child)
            else:
                raise ValueError("Expected max to have min children")
    elif isinstance(node, DAGMinN):
        min_nodes = [node]
    elif isinstance(node, DAGAvoid):
        # Ga  transition if a AND G a
        temporal_node_idx = temporal_nodes.index(node_idx)
        value = t_value[temporal_node_idx]

        avoid_dag_id = node.avoid
        # TODO: Modify evaluate_dag to allow temporal nodes.
        stay_value = evaluate_dag(dag_nodes, avoid_dag_id, predicates, scratch, which=which)

        value = jnp.minimum(value, stay_value)
        return [DAGTransition(parent_idx, node_idx, value)]

    elif not has_temporal_children(node_idx, dag_nodes):
        # No temporal children, so doesn't transition to anything.
        return []
    else:
        raise NotImplementedError("")

    triggers = []
    for min_node in min_nodes:
        # Should be at least 1 temporal, 1 nontemporal.
        assert len(min_node.args) >= 2
        temporal_children = []
        nontemporal_children = []
        for child_id in min_node.args:
            child = dag_nodes[child_id]
            if child.is_temporal():
                temporal_children.append(child_id)
            elif isinstance(child, DAGGUMinN):
                # If we reach a GUminN, transition to the first temporal child.
                temporal_children.append(child.args[0])
            else:
                dag_value = evaluate_dag(dag_nodes, child_id, predicates, scratch, which=which)
                nontemporal_children.append(dag_value)

        assert len(temporal_children) == 1, "There should only be one temporal child per min node."
        temporal_idx = temporal_children[0]

        assert len(nontemporal_children) >= 1, "There should be at least one non-temporal child per min node"

        # Compute the min over the nontemporal children
        nontemporal_value = which.stack(nontemporal_children, axis=0).min(axis=0)
        triggers.append(DAGTransition(parent_idx, temporal_idx, nontemporal_value))

    return triggers


def get_gu_triggers(
    dag_nodes: list[DAGNode],
    GU_min_node_idx: DAGId,
    predicates: dict[str, jnp.ndarray],
    t_value: jnp.ndarray,
    scratch: dict[DAGId, jnp.ndarray],
    which=jnp,
) -> list[DAGTransition]:
    """Get triggers for GU."""
    gu_min_node = dag_nodes[GU_min_node_idx]
    assert isinstance(gu_min_node, DAGGUMinN)
    gu_single_dag_ids: list[DAGId] = list(gu_min_node.args)
    n_GU = len(gu_single_dag_ids)

    triggers: list[DAGTransition] = []

    # For each GUSingle, when triggered, move to the next GUSingle.
    for gu_idx, gu_single_dag_id in enumerate(gu_single_dag_ids):
        gu_single = dag_nodes[gu_single_dag_id]
        if not isinstance(gu_single, DAGGUSingle):
            raise ValueError(f"Expected GUminN to have only GUsingle children, found {type(gu_single)}")

        gu_single_reach_dag_id = gu_single.reach
        condition_val = evaluate_dag(dag_nodes, gu_single_reach_dag_id, predicates, scratch, which=which)

        gu_idx_next = (gu_idx + 1) % n_GU
        gu_single_next_dag_id = gu_single_dag_ids[gu_idx_next]
        triggers.append(DAGTransition(gu_single_dag_id, gu_single_next_dag_id, condition_val))

    return triggers


def get_rules(temporal_nodes, dag_nodes, predicates: dict[str, jnp.ndarray], t_value: jnp.ndarray, which=jnp):
    """Get the temporal node transition rules."""
    scratch: dict[DAGId, jnp.ndarray] = {}

    all_dag_triggers: list[DAGTransition] = []
    for temporal_node_idx, node_idx in enumerate(temporal_nodes):
        node = dag_nodes[node_idx]
        match node:
            case DAGReach(reach=reach_id):
                triggers = get_triggers(
                    dag_nodes,
                    temporal_nodes,
                    node_idx,
                    reach_id,
                    predicates,
                    t_value,
                    scratch,
                    which=which,
                )
                all_dag_triggers.extend(triggers)
            case DAGAvoid(avoid=avoid_id):
                # Avoid nodes are at the very bottom and don't transition to other temporal nodes.
                pass
            case DAGReachAvoid(reach=reach_id, avoid=avoid_id):
                triggers = get_triggers(
                    dag_nodes,
                    temporal_nodes,
                    node_idx,
                    reach_id,
                    predicates,
                    t_value,
                    scratch,
                    which=which,
                )
                all_dag_triggers.extend(triggers)
            case DAGGUSingle():
                # Handle GU separately below
                pass
            case _:
                raise ValueError(f"Unexpected temporal node type: {type(node)}")

    for node_idx, node in enumerate(dag_nodes):
        if not isinstance(node, DAGGUMinN):
            continue
        triggers = get_gu_triggers(
            dag_nodes,
            DAGId(node_idx),
            predicates,
            t_value,
            scratch,
            which=which,
        )
        all_dag_triggers.extend(triggers)

    # Convert to TemporalNodeTransition by converting from DAGId to temporal node idx.
    all_triggers: list[TemporalNodeTransition] = []
    for trigger in all_dag_triggers:
        parent_idx = temporal_nodes.index(trigger.parent)
        child_idx = temporal_nodes.index(trigger.child)
        all_triggers.append(TemporalNodeTransition(parent_idx, child_idx, trigger.condition))

    return all_triggers
