from attrs import define, field, frozen
import copy
import functools as ft
from typing import Any, Generic, NamedTuple, Protocol, TypeVar, Union

import flax.linen as nn
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
from typing_extensions import Self
from valtr.reachability import (DAGAvoid, DAGConst, DAGGUMinN, DAGGUSingle, DAGId, DAGMaxN, DAGMinN, DAGNegate, DAGNode,
                                DAGReach, DAGReachAvoid, DAGVar, collect_predicate_info, get_node_parent_dict,
                                has_temporal_children, temporal_nodes_topological)

from rraa_rl.evaluate_dag import evaluate_dag
from rraa_rl.jax_utils import tree_cat
from valtr.valtr import to_dag
from valtr.reachability import DAGId, DAGMinN, DAGReach, DAGAvoid, has_temporal_children, DAGNode, DAGReachAvoid
from valtr.valtr import to_dag, to_dag_notransform

from rraa_rl.src.env.general_task.env import Env, StaticTemporalNodeMixin, EnvStep, StateWithTemporalNode, EnvUsingBase

_EnvState = TypeVar("_EnvState")
_Obs = TypeVar("_Obs")


@frozen
class CMDPOperation:
    pass


@frozen
class CMDPAvoid(CMDPOperation):
    avoid: DAGId  # A(arg). Arg should be purely propositional.

@frozen
class CMDPWeakUntil(CMDPOperation):
    stay: DAGId # Should be purely propositional.
    reach: DAGId # Should be purely propositional.

@frozen
class CMDPReachChain(CMDPOperation):
    """Represents F(r_1 & F(r_2 & F(r_3 ...)))"""

    reach: DAGId  # Should be purely propositional.
    condition: list[DAGId] # Previous reaches that need to have occurred for this reach to occur.


@frozen
class CMDPFG(CMDPOperation):
    """Represents F G r. Requires an epsilon move."""

    stay: DAGId  # Should be purely propositional.


@frozen
class CMDPGF(CMDPOperation):
    """Represents G F r."""

    reach: DAGId  # Should be purely propositional.

@define
class CMDPInfo:
    operations: list[CMDPOperation]
    """List of operations. Each corresponds to a separate value function."""

    @property
    def reach_flags(self) -> list[DAGId]:
        """Returns the list of DAGIds that need to be tracked (as reach flags) for the ReachChain."""
        reach_ids: list[DAGId] = []
        for op in self.operations:
            match op:
                case CMDPReachChain(condition=condition):
                    for c in condition:
                        if c not in reach_ids:
                            reach_ids.append(c)
                case CMDPWeakUntil(reach=reach_id):
                    if reach_id not in reach_ids:
                        reach_ids.append(reach_id)
                case _:
                    pass
        return reach_ids

    @property
    def n_reach_flags(self) -> int:
        return len(self.reach_flags)

    @property
    def n_epsilon_moves(self) -> int:
        return sum(1 for op in self.operations if isinstance(op, CMDPFG))

    @property
    def has_epsilon_move(self) -> bool:
        return any(isinstance(op, CMDPFG) for op in self.operations)

def parse_reach_chain(nodes: list[DAGNode], root: DAGId) -> list[CMDPOperation]:
    """Parses a reach chain starting from the given root node.

    F(r_1 & F(r_2 & F(r_3)))

    gives us the following CMDP operations:
    - CMDPReachChain(reach=r_1, condition=[])
    - CMDPReachChain(reach=r_2, condition=[r_1])
    - CMDPReachChain(reach=r_3, condition=[r_1, r_2])

    """
    operations: list[CMDPOperation] = []
    current_node_id = root
    condition_chain: list[DAGId] = []

    while True:
        current_node = nodes[current_node_id]
        match current_node:
            case DAGReach(reach=reach_id):
                operations.append(CMDPReachChain(reach=reach_id, condition=condition_chain.copy()))
                condition_chain.append(reach_id)

                # Move to the next node in the chain.
                if isinstance(next_node := nodes[reach_id], DAGMinN):
                    # End of the chain.
                    break
                elif isinstance(next_node, DAGReach):
                    current_node_id = reach_id
                else:
                    raise NotImplementedError(f"Unsupported DAG node type in reach chain: {type(next_node)}")
            case _:
                raise NotImplementedError(f"Unsupported DAG node type in reach chain: {type(current_node)}")

    return operations

def parse_dag(nodes: list[DAGNode], root: DAGId) -> CMDPInfo:
    """Parses a DAG into a list of CMDP operations.

    """
    root_node = nodes[root]

    # The root node should be AND of multiple operations.
    if isinstance(root_node, DAGMinN):
        children = root_node.children
    else:
        children = [root]

    operations: list[CMDPOperation] = []

    for child in children:
        child_node = nodes[child]
        match child_node:
            case DAGAvoid(avoid=avoid) if isinstance(reach_node := nodes[avoid], DAGReach):
                # GF r
                assert not has_temporal_children(reach_node.reach, nodes)
                operations.append(CMDPGF(reach=reach_node.reach))
            case DAGReach(reach=reach) if isinstance(avoid_node := nodes[reach], DAGAvoid):
                # FG q
                assert not has_temporal_children(avoid_node.avoid, nodes)
                operations.append(CMDPFG(stay=avoid_node.avoid))
            case DAGReach(reach=reach):
                # F chain.
                reach_chain_operations = parse_reach_chain(nodes, child)
                operations.extend(reach_chain_operations)
            case DAGAvoid(avoid=avoid_id):
                # G(q)
                assert not has_temporal_children(avoid_id, nodes)
                operations.append(CMDPAvoid(avoid_id))
            case DAGReachAvoid(reach=reach, avoid=avoid):
                ra_operations = [CMDPReachChain(reach=reach, condition=[]), CMDPAvoid(avoid=avoid)]
                operations.extend(ra_operations)
            case _:
                raise NotImplementedError(f"Unsupported DAG node type: {type(child_node)}")

    return CMDPInfo(operations=operations)

BaseClassState = TypeVar("BaseClassState")

@jdc.pytree_dataclass
class CMDPAugState(Generic[BaseClassState]):
    reach_flags: dict[int, jnp.ndarray]
    """Dictionary mapping from the DAGId to a boolean flag indicating whether that reach condition has been met."""

    # ( num_FG, )
    epsilon_moved: jnp.ndarray
    """Indicates whether an epsilon move has been made for each FG condition."""

    base: BaseClassState

class CMDPObs(NamedTuple):
    base: jnp.ndarray
    cmdp: jnp.ndarray

    def combine(self, which=jnp):
        return which.concatenate([self.base, self.cmdp], axis=-1)

@define
class CMDPCfg:
    random_reset: bool = True
    """If true, randomly initialize the reach flags and epsilon moved flags on reset."""

class CMDPEnvWrapper(Env):
    def __init__(self, cfg: CMDPCfg, env: StaticTemporalNodeMixin | EnvUsingBase):
        super().__init__(env.cfg, env.specification)
        self._env = env
        self.cfg = cfg

        dag_builder, dag_root = to_dag_notransform(
            env.specification, ir_filename="dags/herd_os_ir", dag_filename="dags/cmdp"
        )
        self.dag_nodes_notrans = dag_builder.nodes
        self.dag_root_notrans = dag_root
        self.cmdp_info = parse_dag(self.dag_nodes_notrans, self.dag_root_notrans)

    @property
    def specification(self):
        return self._env.specification

    @property
    def n_conjunctions(self) -> int:
        return len(self.cmdp_info.operations)

    @property
    def n_agents(self) -> int:
        """Add one for the epsilon move."""
        if self.cmdp_info.has_epsilon_move:
            return self._env.n_agents + 1
        else:
            return self._env.n_agents

    def to_minstate(self, state: CMDPAugState) -> CMDPAugState:
        # validate=False because we are changing the structure.
        with jdc.copy_and_mutate(state, validate=False) as state_new:
            state_new.base = self._env.base.to_minstate(state.base)
        return state_new

    def from_minstate(self, minstate: CMDPAugState) -> CMDPAugState:
        # validate=False because we are changing the structure.
        with jdc.copy_and_mutate(minstate, validate=False) as state_new:
            state_new.base = self._env.base.from_minstate(minstate.base)
        return state_new

    def step(self, state: CMDPAugState, action: list[jnp.ndarray]):
        assert len(action) == self.n_agents
        epsilon_action = None
        if self.cmdp_info.has_epsilon_move:
            epsilon_action = action[-1]
            action = action[:-1]

        step: EnvStep[Any] = self._env.base.step(state.base, action)
        state_base_new = step.envstate

        state_new = self.step_cmdp(state, state_base_new, step.predicates, epsilon_action)

        obs = self.get_obs(state_new)
        step = step._replace(state=state_new, obs=obs)
        return step

    def step_cmdp(self, state: CMDPAugState, state_base_new: Any, predicates: dict[str, jnp.ndarray], epsilon_action: jnp.ndarray | None = None) -> CMDPAugState:
        # 1: Update reach flags.
        reach_flags = state.reach_flags
        scratch = {}
        for dag_id in self.cmdp_info.reach_flags:
            dag_value = evaluate_dag(self.dag_nodes_notrans, dag_id, predicates, scratch=scratch)
            reached = dag_value > 0.0
            reach_flags[dag_id] = reach_flags[dag_id] | reached

        # 2: Update epsilon moves for FG conditions.
        epsilon_moved = state.epsilon_moved
        if self.cmdp_info.has_epsilon_move:
            assert epsilon_action is not None
            # Integer action: 0 = no epsilon move, ii = epsilon move for ii-1 FG.
            assert epsilon_action.shape == ()
            raise NotImplementedError("")

        return CMDPAugState(reach_flags, epsilon_moved, state_base_new)

    def get_obs(self, state: CMDPAugState) -> CMDPObs:
        base_obs = self._env.base.get_obs(state.base)
        return self._augment_obs(state, base_obs)

    def _get_augment_obs(self, state: Any):
        obs, _ = self._augment_obs_and_names(state)
        return obs

    def _augment_obs(self, state: CMDPAugState, obs: jnp.ndarray) -> CMDPObs:
        obs_aug = self._get_augment_obs(state)
        return CMDPObs(base=obs, cmdp=obs_aug)

    def _augment_obs_and_names(self, state: CMDPAugState):
        """Augment the base observation with the reach flags and the eps_moved."""
        obs_aug = jnp.concatenate([state.reach_flags, state.epsilon_moved])
        assert obs_aug.shape == (self.cmdp_info.n_reach_flags + self.cmdp_info.n_epsilon_moves,)

        reach_names = [f"reachflag_{i}" for i in range(self.cmdp_info.n_reach_flags)]
        eps_names = [f"eps_moved_{i}" for i in range(self.cmdp_info.n_epsilon_moves)]
        obs_names = reach_names + eps_names

        return obs_aug, obs_names

    @ft.partial(jax.jit, static_argnames=("self",))
    def reset(self, key: PRNGKeyArray) -> CMDPAugState:
        key_base, key_reachflag, key_eps = jr.split(key, 3)

        base_state = self._env.base.reset(key)

        if self.cfg.random_reset:
            n_reach_flags = self.cmdp_info.n_reach_flags
            reach_flags_array = jr.bernoulli(key_reachflag, p=0.5, shape=(n_reach_flags,))
            reach_flags = {dag_id: reach_flags_array[i] for i, dag_id in enumerate(self.cmdp_info.reach_flags)}

            epsilon_moved = jr.bernoulli(key_eps, p=0.5, shape=(self.cmdp_info.n_epsilon_moves,))
        else:
            reach_flags = {dag_id: jnp.array(False) for dag_id in self.cmdp_info.reach_flags}
            epsilon_moved = jnp.zeros((self.cmdp_info.n_epsilon_moves,), dtype=jnp.bool_)

        return CMDPAugState(reach_flags, epsilon_moved, base_state)

    @ft.partial(jax.jit, static_argnames=("self", "batch_size"))
    def reset_batch(self, key: PRNGKeyArray, batch_size: int, init: bool = False) -> Any:
        return jax.vmap(self.reset)(jr.split(key, batch_size))