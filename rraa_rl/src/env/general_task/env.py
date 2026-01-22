import copy
import functools as ft
from typing import Any, Generic, NamedTuple, Protocol, Self, TypeVar

import jax
import jax.nn as jnn
import jax.random as jr
import jax.tree_util as jtu
import jax_dataclasses as jdc
import matplotlib.pyplot as plt
import numpy as np
from attrs import define
from jax import numpy as jnp
from jaxtyping import PRNGKeyArray
from valtr.reachability import (DAGAvoid, DAGConst, DAGGUMinN, DAGGUSingle, DAGId, DAGMaxN, DAGMinN, DAGNegate, DAGNode,
                                DAGReach, DAGReachAvoid, DAGVar, collect_predicate_info, get_node_parent_dict,
                                has_temporal_children, temporal_nodes_topological)
from valtr.valtr import to_dag

from rraa_rl.evaluate_dag import evaluate_dag
from rraa_rl.jax_utils import tree_cat

_EnvState = TypeVar("_EnvState")
_Obs = TypeVar("_Obs")


class EnvStep(NamedTuple, Generic[_EnvState, _Obs]):
    envstate: _EnvState
    obs: _Obs
    predicates: dict
    term: bool
    trunc: bool
    info: dict


class BaseEnv(Generic[_EnvState, _Obs]):
    def __init__(self):
        self._obs_names = None
        self.active_predicates: list[str] | None = None

    def is_predicate_active(self, predicate_name: str) -> bool:
        if self.active_predicates is None:
            return True

        return predicate_name in self.active_predicates

    def step(self, state: _EnvState, action: jnp.ndarray) -> EnvStep[_EnvState, _Obs]:
        raise NotImplementedError("")

    def reset(self, key: PRNGKeyArray) -> Any:
        raise NotImplementedError("")

    @ft.partial(
        jax.jit,
        static_argnames=(
            "self",
            "batch_size",
        ),
    )
    def reset_batch(self, key: PRNGKeyArray, batch_size: int) -> Any:
        b_key = jr.split(key, batch_size)
        return jax.vmap(self.reset)(b_key)

    @property
    def n_agents(self) -> int:
        raise NotImplementedError("")

    @property
    def value_lims(self) -> tuple[float, float]:
        raise NotImplementedError("")

    @property
    def n_actions_per_agent(self) -> list[list[int]]:
        raise NotImplementedError("")

    @property
    def max_entropy(self) -> float:
        # Sum of log of number of actions, per dimension, per agent.
        n_actions_per_agent = self.n_actions_per_agent
        agent_entropies = []
        for actions_per_agent in n_actions_per_agent:
            actions_per_agent = np.array(actions_per_agent)
            agent_entropy = np.log(actions_per_agent).sum()
            agent_entropies.append(agent_entropy)

        return np.sum(np.array(agent_entropies))

    @property
    def max_entropy(self) -> float:
        raise NotImplementedError("")

    def get_obs(self, state: _EnvState):
        obs, _ = self.get_obs_and_names(state)
        return obs

    def get_obs_and_names(self, state: Any) -> tuple[jnp.ndarray, list[str]]:
        raise NotImplementedError("")

    def get_obs_names(self):
        if self._obs_names is None:
            dummy_state = self.reset(jr.PRNGKey(0))
            _, obs_names = self.get_obs_and_names(dummy_state)
            self._obs_names = obs_names
        return self._obs_names

    def get_predicates(self, state: Any) -> dict[str, jnp.ndarray]:
        raise NotImplementedError("")


@define(slots=False)
class EnvCfg:
    eval_T: int = 200


class Env:
    def __init__(self, cfg: EnvCfg, specification: str):
        self.cfg = cfg
        dag_builder, dag_root = to_dag(specification, ir_filename="dags/herd_os_ir", dag_filename="dags/herd_os_dag")
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

    def reset_batch(self, key: PRNGKeyArray, batch_size: int, init: bool = False) -> Any:
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

    @property
    def n_actions_per_agent(self) -> list[list[int]]:
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
        raise NotImplementedError("")
        # obs_aug = self._get_augment_obs(state)
        # return AugObs(base=obs, temporal=obs_aug)

    @property
    def eval_T(self) -> int:
        return self.cfg.eval_T

    def transition_temporal_node(
        self, predicates: dict[str, jnp.ndarray], t_value: jnp.ndarray, temporal_node_idx: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute the new temporal node idx after applying the DAG transitions."""
        all_triggers = get_rules(self.temporal_nodes, self.dag_nodes, predicates, t_value)
        t_parents = jnp.stack([trigger.parent for trigger in all_triggers])
        t_conditions = jnp.stack([trigger.condition for trigger in all_triggers])
        t_children = jnp.stack([trigger.child for trigger in all_triggers])

        t_is_valid = (t_parents == temporal_node_idx) & (t_conditions > 0.0)

        # If there are multiple valid transitions, pick the one with the highest condition value.
        t_has_valid = jnp.any(t_is_valid)
        t_conditions_masked = jnp.where(t_is_valid, t_conditions, -jnp.inf)
        transition_idx = jnp.argmax(t_conditions_masked)

        temporal_node_idx_new = jnp.where(t_has_valid, t_children[transition_idx], temporal_node_idx)

        # jd.print("-----------", ordered=True)
        # jd.print("t_parents: {}", t_parents, ordered=True)
        # jd.print("t_conditions: {}", t_conditions, ordered=True)
        # jd.print("t_children: {}", t_children, ordered=True)
        # jd.print("{} -> {}", temporal_node_idx, temporal_node_idx_new, ordered=True)

        return temporal_node_idx_new


BaseClass = TypeVar("BaseClass")


@jdc.pytree_dataclass
class StateWithTemporalNode(Generic[BaseClass]):
    temporal_node_idx: int
    base: BaseClass


class EnvUsingBase(Env):
    def __init__(self, cfg: EnvCfg, specification: str, base_env: BaseEnv):
        super().__init__(cfg, specification)
        self.base = base_env

    @property
    def n_agents(self) -> int:
        return self.base.n_agents

    @property
    def value_lims(self):
        return self.base.value_lims

    @property
    def n_actions_per_agent(self) -> list[list[int]]:
        return self.base.n_actions_per_agent

    @property
    def max_entropy(self) -> float:
        return self.base.max_entropy

    @property
    def n_temporal_nodes(self):
        return len(self.temporal_nodes)

    def step(self, state: Any, action: jnp.ndarray):
        base_step: EnvStep = self.base.step(state.base, action)

        temporal_node_idx = state.temporal_node_idx

        state_new = jdc.replace(state, temporal_node_idx=temporal_node_idx, base=base_step.envstate)
        obs = self._augment_obs(state_new, base_step.obs)
        step = base_step._replace(envstate=state_new, obs=obs)

        return step

    def get_obs(self, state: Any) -> Any:
        base_obs = self.base.get_obs(state.base)
        return self._augment_obs(state, base_obs)

    def get_obs_names(self) -> list[str]:
        return self.base.get_obs_names() + self.augment_obs_names()

    def get_predicates(self, state: Any) -> dict[str, jnp.ndarray]:
        return self.base.get_predicates(state.base)

    def setup_ax(self, ax: plt.Axes):
        return self.base.setup_ax(ax)


@define(slots=False)
class StaticTemporalNodeMixinCfg:
    temporal_node_fracs: list[float] | None = None
    """Fractions for sampling each temporal node. This is reverse topological order. If None, split evenly."""

    @property
    def root_only(self):
        return self.temporal_node_fracs[0] == 1.0


class StaticTemporalNodeMixinProtocol(Protocol):
    base: BaseEnv
    cfg: StaticTemporalNodeMixinCfg
    n_temporal_nodes: int
    temporal_nodes: list[DAGId]

    def _get_augment_obs(self, state: StateWithTemporalNode) -> jnp.ndarray: ...

    def _temporal_node_idx_to_node_type(self) -> jnp.ndarray: ...


class StaticTemporalNodeMixin:
    def __init__(self: StaticTemporalNodeMixinProtocol, cfg: StaticTemporalNodeMixinCfg, **kwargs):
        self.cfg = cfg

        if self.cfg.temporal_node_fracs is None:
            # Split it evenly.
            self.cfg.temporal_node_fracs = np.full(self.n_temporal_nodes, 1.0 / self.n_temporal_nodes).tolist()

        assert len(self.cfg.temporal_node_fracs) == len(self.temporal_nodes)

    @ft.partial(jax.jit, static_argnames=("self",))
    def reset(self: StaticTemporalNodeMixinProtocol, key: PRNGKeyArray) -> StateWithTemporalNode:
        key_base, key_node = jr.split(key)
        base_state = self.base.reset(key_base)

        # Sample temporal node idx according to configured fractions.
        node_fracs = jnp.array(self.cfg.temporal_node_fracs)
        node_fracs = node_fracs / jnp.sum(node_fracs)
        temporal_node_idx = jr.choice(key_node, a=self.n_temporal_nodes, p=node_fracs)
        state = StateWithTemporalNode(temporal_node_idx=temporal_node_idx, base=base_state)
        return state

    @ft.partial(jax.jit, static_argnames=("self", "batch_size"))
    def reset_batch(self: StaticTemporalNodeMixinProtocol, key: PRNGKeyArray, batch_size: int) -> Any:
        # Instead of randomly sampling temporal nodes, we assign fixed fractions of the batch to each temporal node.
        base_state = self.base.reset_batch(key, batch_size)

        n_per_temporal_node = np.round(np.array(self.cfg.temporal_node_fracs) * batch_size).astype(int)
        n_per_temporal_node[-1] = batch_size - n_per_temporal_node[:-1].sum()

        temporal_node_idxs = jnp.concatenate([jnp.full((n,), idx) for idx, n in enumerate(n_per_temporal_node)], axis=0)
        state = StateWithTemporalNode(
            temporal_node_idx=temporal_node_idxs,
            base=base_state,
        )
        return state

    def get_eval_states(self: StaticTemporalNodeMixinProtocol, n_envs: int) -> StateWithTemporalNode:
        # Assign envs evenly to each temporal node.
        n_envs_per_node = np.full((self.n_temporal_nodes,), n_envs // self.n_temporal_nodes)
        n_envs_per_node[0] = n_envs - n_envs_per_node[1:].sum()

        key = jr.PRNGKey(seed=12345)
        max_n_envs_per_node = n_envs_per_node.max()
        m_state_base = self.base.reset_batch(key, max_n_envs_per_node)

        states = []
        for ii, n_envs_this in enumerate(n_envs_per_node):
            state_base = jtu.tree_map(lambda x: x[:n_envs_this], m_state_base)
            state = StateWithTemporalNode(
                temporal_node_idx=jnp.full((n_envs_this,), ii),
                base=state_base,
            )
            states.append(state)

        b_state0 = tree_cat(states, axis=0)
        return b_state0

    def _augment_obs(self: StaticTemporalNodeMixinProtocol, state: StateWithTemporalNode, obs: jnp.ndarray):
        obs_aug = self._get_augment_obs(state)
        if self.n_temporal_nodes == 1:
            temporal_node_idx = 0
            temporal_node_type = 0
        else:
            temporal_node_idx = state.temporal_node_idx
            temporal_node_idx_to_node_type = self._temporal_node_idx_to_node_type()
            temporal_node_type = temporal_node_idx_to_node_type[state.temporal_node_idx]

        return AugObs(
            temporal_node_idx=temporal_node_idx, temporal_node_type=temporal_node_type, base=obs, temporal=obs_aug
        )


class AugObs(NamedTuple):
    """Separate the "base" observation and the observation of the temporal node."""

    temporal_node_idx: int
    temporal_node_type: int
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


def get_triggers(
    dag_nodes: list[DAGNode],
    parent_idx: DAGId,
    node_idx: DAGId,
    predicates: dict[str, jnp.ndarray],
    V_dict: dict[DAGId, jnp.ndarray],
    scratch: dict[DAGId, jnp.ndarray],
    which=jnp,
) -> list[DAGTransition]:
    node = dag_nodes[node_idx]

    # Either:
    # - max_i min_j (..., temporal)
    # -       min_j (..., temporal)
    # - a temporal node (e.g., G)
    #
    # # Convert t_value to V_dict.
    # V_dict = {}
    # for temporal_node_idx, dag_id in enumerate(temporal_nodes):
    #     V_dict[dag_id] = t_value[temporal_node_idx]

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
        value = V_dict[node_idx]

        avoid_dag_id = node.avoid
        # TODO: Modify evaluate_dag to allow temporal nodes.
        stay_value = evaluate_dag(dag_nodes, avoid_dag_id, predicates, V_dict, scratch, which=which)

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
                dag_value = evaluate_dag(dag_nodes, child_id, predicates, V_dict, scratch, which=which)
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
    V_dict: dict[DAGId, jnp.ndarray],
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
        condition_val = evaluate_dag(
            dag_nodes, gu_single_reach_dag_id, predicates, V_dict, scratch=scratch, which=which
        )

        gu_idx_next = (gu_idx + 1) % n_GU
        gu_single_next_dag_id = gu_single_dag_ids[gu_idx_next]
        triggers.append(DAGTransition(gu_single_dag_id, gu_single_next_dag_id, condition_val))

    return triggers


def get_rules(
    temporal_nodes: list[DAGId],
    dag_nodes: list[DAGNode],
    predicates: dict[str, jnp.ndarray],
    t_value: jnp.ndarray,
    which=jnp,
):
    """Get the temporal node transition rules."""
    scratch: dict[DAGId, jnp.ndarray] = {}

    # Convert t_value into V_dict.
    V_dict = {}
    for temporal_node_idx, dag_id in enumerate(temporal_nodes):
        V_dict[dag_id] = t_value[temporal_node_idx]

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
                    node_idx,
                    reach_id,
                    predicates,
                    V_dict,
                    scratch=scratch,
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
            V_dict,
            scratch=scratch,
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
