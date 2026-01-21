from jax import numpy as jnp
from valtr.reachability import (DAGConst, DAGId, DAGMaxN, DAGMinN, DAGNegate, DAGNode,
                                DAGVar, DAGGUMinN)


def evaluate_dag(
    dag_nodes: list[DAGNode],
    node_idx: DAGId,
    predicates: dict[str, jnp.ndarray],
    V_dict: dict[DAGId, jnp.ndarray] | None = None,
    scratch: dict[DAGId, jnp.ndarray] | None = None,
    allow_const: bool = False,
    which=jnp
):
    if scratch is None:
        scratch = {}

    if node_idx in scratch:
        return scratch[node_idx]

    dag_node = dag_nodes[node_idx]
    match dag_node:
        case DAGConst(value=value):
            if allow_const:
                big_float = 6.969e7
                value_float = {True: big_float, False: -big_float}[value]
                out = which.full((), value_float)
            else:
                raise ValueError("Const nodes should have been removed")
        case DAGVar(name=name):
            out = predicates[name]
            # logger.debug("Var(%{}) = {}".format(node_idx, out))
        case DAGNegate(arg=arg):
            out = -evaluate_dag(dag_nodes, arg, predicates, V_dict, scratch, which=which)
            # logger.debug("Negate(%{}) = {}".format(node_idx, out))
        case DAGMinN(args=args):
            args_vals = which.stack(
                [evaluate_dag(dag_nodes, arg, predicates, V_dict, scratch, which=which) for arg in args],
                axis=-1,
            )
            out = which.min(args_vals, axis=-1)
        case DAGMaxN(args=args):
            args_vals = which.stack(
                [evaluate_dag(dag_nodes, arg, predicates, V_dict, scratch, which=which) for arg in args],
                axis=-1,
            )
            out = which.max(args_vals, axis=-1)
        case DAGGUMinN(args=args):
            # Treat it as a normal Min.
            args_vals = which.stack(
                [evaluate_dag(dag_nodes, arg, predicates, V_dict, scratch, which=which) for arg in args],
                axis=-1,
            )
            out = which.min(args_vals, axis=-1)
        case _:
            if V_dict is None:
                raise ValueError("Trying to evaluate temporal node without value function")

            out = V_dict[node_idx]

    scratch[node_idx] = out
    return out
