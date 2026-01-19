import copy
import functools as ft
from typing import Any, NamedTuple

import ipdb
import jax
import jax.debug as jd
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax_dataclasses as jdc
import numpy as np
from attrs import define
from flax import struct
from jaxtyping import PRNGKeyArray
from loguru import logger
from valtr.reachability import (DAGAvoid, DAGConst, DAGGUMinN, DAGGUSingle, DAGId, DAGMaxN, DAGMinN, DAGNegate, DAGNode,
                                DAGReach, DAGReachAvoid, DAGVar, collect_predicate_info, extract_trigger_predicate_map,
                                get_node_parent_dict, has_temporal_children, temporal_nodes_topological)
from valtr.valtr import to_dag

from rraa_rl.jax_types import BoolScalar
from rraa_rl.jax_utils import softminimum, tree_cat
from rraa_rl.src.env.general_task.env import Env, EnvStep
from rraa_rl.src.env.general_task.herd_base import HerdBase, HerdBaseCfg, HerdBaseState


class DAGTransition(NamedTuple):
    parent: DAGId
    child: DAGId
    condition: jnp.ndarray


class TemporalNodeTransition(NamedTuple):
    parent: int
    child: int
    condition: jnp.ndarray


@jdc.pytree_dataclass
class HerdOsState:
    temporal_node_idx: int
    base: HerdBaseState


@define
class HerdOsCfg:
    base: HerdBaseCfg = HerdBaseCfg()
    # What fraction of the batch is which temporal node at reset. This is in reverse topological order.
    # temporal_node_fracs: list[float] = [0.6, 0.4]
    # temporal_node_fracs: list[float] = [0.4, 0.3, 0.3]
    temporal_node_fracs: list[float] | None = None

    do_temporal_transition: bool = False
    """If true (e.g., eval), then change the temporal node according to the DAG transitions."""

    eval_T: int = 200

    @property
    def root_only(self):
        return self.temporal_node_fracs[0] == 1.0


class AugObs(NamedTuple):
    """Separate the "base" observation and the observation of the temporal node."""

    base: jnp.ndarray
    temporal: jnp.ndarray

    def combine(self, which=jnp):
        return which.concatenate([self.base, self.temporal], axis=-1)


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
        self.base = HerdBase(cfg.base, should_term_fn=self.should_terminate)

        dag_builder, dag_root = to_dag(self.specification, dag_filename="herd_os_dag.pdf")
        self.dag_nodes = dag_builder.nodes
        self.dag_root = dag_root
        self.pred_info = collect_predicate_info(self.dag_nodes, self.dag_root)
        # self.dag_info = extract_trigger_predicate_map(self.dag_nodes, self.dag_root)

        # root first.
        self.temporal_nodes: list[DAGId] = temporal_nodes_topological(self.dag_nodes, self.dag_root)[::-1]

        if self.cfg.temporal_node_fracs is None:
            # Split it evenly.
            self.cfg.temporal_node_fracs = np.full(self.n_temporal_nodes, 1.0 / self.n_temporal_nodes).tolist()

        assert len(self.cfg.temporal_node_fracs) == len(self.temporal_nodes)
        self._augment_obs_names = None

        self.node_parent_dict: dict[DAGId, DAGId] = get_node_parent_dict(self.dag_nodes, self.dag_root)

    @property
    def temporal_node_names(self) -> list[str]:
        names = []
        for node_id in self.temporal_nodes:
            node = self.dag_nodes[node_id]
            name = f"{type(node).__name__} (%{node_id})"
            names.append(name)
        return names

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

    @property
    def specification(self):
        # return "F(G(herd_herded)) && G( !herder_collide ) && G( !herder_oob )"
        # return "( !herder_collide && ! herder_oob ) U ( G(herd_herded && !herder_collide && ! herder_oob) )"
        # return "F herder_c1 && F herder_c2"
        # return "F G (herder_c1 || herder_c2)"
        # return "F herder_c1"
        # return "!herder_collide U herder_c1"
        return "G F herder_c1 && G F herder_c2"

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

    def _augment_obs_and_names(self, state: HerdOsState):
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

    def _get_augment_obs(self, state: HerdOsState):
        obs, _ = self._augment_obs_and_names(state)
        return obs

    def _augment_obs(self, state: HerdOsState, obs: jnp.ndarray):
        obs_aug = self._get_augment_obs(state)
        return AugObs(base=obs, temporal=obs_aug)

    def should_terminate(self, predicates: dict[str, jnp.ndarray]) -> BoolScalar:
        eps = 0.1

        # Terminate when leaving the allowed area.
        is_oob = predicates["herder_oob"] > eps
        should_term = is_oob

        # Terminate when reaching the goal, or leaving the allowed area.
        # is_goal = predicates["herder_c1"] > eps
        # should_term = is_goal | is_oob

        return should_term

    def step(self, state: HerdOsState, action: jnp.ndarray):
        base_step: EnvStep = self.base.step(state.base, action)

        temporal_node_idx = state.temporal_node_idx

        # Temporal transitions with G require the value function, so we pass that in as a callback.
        # if self.cfg.do_temporal_transition:
        #     logger.debug("Doing temporal transition")
        #     predicates = base_step.predicates
        #     temporal_node_idx = self.transition_temporal_node(predicates, temporal_node_idx)

        state_new = jdc.replace(state, temporal_node_idx=temporal_node_idx, base=base_step.envstate)
        obs = self._augment_obs(state_new, base_step.obs)
        step = base_step._replace(envstate=state_new, obs=obs)

        return step

    def transition_temporal_node(
        self, predicates: dict[str, jnp.ndarray], t_value: jnp.ndarray, temporal_node_idx: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute the new temporal node idx after applying the DAG transitions."""
        all_triggers = self.get_rules(predicates, t_value)
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

    def get_obs(self, state: Any) -> Any:
        base_obs = self.base.get_obs(state.base)
        return self._augment_obs(state, base_obs)

    def get_obs_names(self) -> list[str]:
        return self.base.get_obs_names() + self.augment_obs_names()

    def augment_obs_names(self) -> list[str]:
        if self._augment_obs_names is None:
            dummy_state = self.reset(jr.PRNGKey(0))
            _, obs_names = self._augment_obs_and_names(dummy_state)
            self._augment_obs_names = obs_names
        return self._augment_obs_names

    def get_predicates(self, state: HerdOsState) -> dict[str, jnp.ndarray]:
        return self.base.get_predicates(state.base)

    def reset(self, key: PRNGKeyArray) -> HerdOsState:
        key_base, key_node = jr.split(key)
        base_state = self.base.reset(key_base)

        # Sample temporal node idx according to configured fractions.
        node_fracs = jnp.array(self.cfg.temporal_node_fracs)
        node_fracs = node_fracs / jnp.sum(node_fracs)
        temporal_node_idx = jr.choice(key_node, a=self.n_temporal_nodes, p=node_fracs)
        state = HerdOsState(temporal_node_idx=temporal_node_idx, base=base_state)
        return state

    def reset_batch(self, key: PRNGKeyArray, batch_size: int) -> Any:
        # Instead of randomly sampling temporal nodes, we assign fixed fractions of the batch to each temporal node.
        base_state = self.base.reset_batch(key, batch_size)

        n_per_temporal_node = np.round(np.array(self.cfg.temporal_node_fracs) * batch_size).astype(int)
        n_per_temporal_node[-1] = batch_size - n_per_temporal_node[:-1].sum()

        temporal_node_idxs = jnp.concatenate([jnp.full((n,), idx) for idx, n in enumerate(n_per_temporal_node)], axis=0)
        state = HerdOsState(
            temporal_node_idx=temporal_node_idxs,
            base=base_state,
        )
        return state

    def get_rules(self, predicates: dict[str, jnp.ndarray], t_value: jnp.ndarray, which=jnp):
        """Get the temporal node transition rules."""
        scratch: dict[DAGId, jnp.ndarray] = {}

        all_dag_triggers: list[DAGTransition] = []
        for temporal_node_idx, node_idx in enumerate(self.temporal_nodes):
            node = self.dag_nodes[node_idx]
            match node:
                case DAGReach(reach=reach_id):
                    triggers = get_triggers(
                        self.dag_nodes,
                        self.temporal_nodes,
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
                        self.dag_nodes,
                        self.temporal_nodes,
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

        for node_idx, node in enumerate(self.dag_nodes):
            if not isinstance(node, DAGGUMinN):
                continue
            triggers = get_gu_triggers(
                self.dag_nodes,
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
            parent_idx = self.temporal_nodes.index(trigger.parent)
            child_idx = self.temporal_nodes.index(trigger.child)
            all_triggers.append(TemporalNodeTransition(parent_idx, child_idx, trigger.condition))

        return all_triggers

    def get_eval_states(self, n_envs: int) -> HerdOsState:
        # Assign envs evenly to each temporal node.
        n_envs_per_node = np.full((self.n_temporal_nodes,), n_envs // self.n_temporal_nodes)
        n_envs_per_node[0] = n_envs - n_envs_per_node[1:].sum()

        key = jr.PRNGKey(seed=12345)
        max_n_envs_per_node = n_envs_per_node.max()
        m_state_base = self.base.reset_batch(key, max_n_envs_per_node)

        states = []
        for ii, n_envs_this in enumerate(n_envs_per_node):
            state_base = jtu.tree_map(lambda x: x[:n_envs_this], m_state_base)
            state = HerdOsState(
                temporal_node_idx=jnp.full((n_envs_this,), ii),
                base=state_base,
            )
            states.append(state)

        b_state0 = tree_cat(states, axis=0)
        return b_state0

    @property
    def eval_T(self) -> int:
        return self.cfg.eval_T

    def with_temporal_transitions(self) -> "HerdOs":
        """Return a copy of this environment that does temporal transitions on step."""
        cfg_new = copy.deepcopy(self.cfg)
        cfg_new.do_temporal_transition = True
        return HerdOs(cfg_new)


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
