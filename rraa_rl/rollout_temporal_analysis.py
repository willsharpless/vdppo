import ipdb
import jax
import jax.numpy as jnp
import numpy as np
from valtr.reachability import (DAGGU, DAGAvoid, DAGConst, DAGId, DAGMaxN, DAGMinN, DAGNegate, DAGNode, DAGReach,
                                DAGReachAvoid, DAGVar)

from rraa_rl.collector import RolloutOutput
from rraa_rl.src.env.general_task.herd_os import DAGTransition, HerdOs


def evaluate_triggers(env: HerdOs, trajs: list[RolloutOutput]) -> dict:
    triggers_dict = {}
    for ii, traj in enumerate(trajs):
        triggers = env.get_rules(traj.predicates)
        # triggers = [DAGTransition(trigger.parent, trigger.child, np.any(trigger.condition)) for trigger in triggers]

        for trigger in triggers:
            key = (trigger.parent, trigger.child)
            triggers_dict[key] = triggers_dict.get(key, [])
            triggers_dict[key].append(np.any(trigger.condition))

    triggers_dict = {k: np.array(v) for k, v in triggers_dict.items()}
    return triggers_dict


def evaluate_ltl_finite(env: HerdOs, T_pred: dict[str, np.ndarray], which=jnp):
    """Evaluate whether the LTL formula (when treated as finite) holds over the finite trace.
    Solve using dynamic programming."""

    dag_nodes = env.dag_nodes
    dag_root = env.dag_root

    tmp_key = list(T_pred.keys())[0]
    (T,) = T_pred[tmp_key].shape

    # 1. Obtain the final value.
    pred_final = {k: v[-1] for k, v in T_pred.items()}
    dag_values = {}
    get_values(dag_nodes, dag_root, pred_final, next_values=None, which=which, values=dag_values)
    assert len(dag_values) == len(dag_nodes)
    dag_values_curr = dag_values

    if which is np:
        # 2. Move backwards through time.
        for kk in range(T - 2, -1, -1):  # T-2, T-3, ..., 0
            pred = {k: v[kk] for k, v in T_pred.items()}

            dag_values_next = dag_values_curr
            dag_values_curr = {}
            get_values(dag_nodes, dag_root, pred, next_values=dag_values_next, which=np, values=dag_values_curr)
    else:
        # 2. Move backwards using scan.
        def step(dag_values_next, pred):
            dag_values_curr_ = {}
            get_values(dag_nodes, dag_root, pred, next_values=dag_values_next, which=which, values=dag_values_curr_)
            return dag_values_curr_, None

        T_pred_prefix_reversed = {k: v[:-1][::-1] for k, v in T_pred.items()}
        dag_values_curr, _ = jax.lax.scan(step, dag_values_curr, T_pred_prefix_reversed)

    return dag_values_curr


def get_values(
    dag_nodes: list[DAGNode],
    dag_root: DAGId,
    predicates: dict[str, np.ndarray],
    next_values: dict[DAGId, np.ndarray] | None,
    which=jnp,
    values: dict[DAGId, np.ndarray] | None = None,
):
    is_terminal = next_values is None

    if values is None:
        values: dict[DAGId, np.ndarray] = {}

    node = dag_nodes[dag_root]
    match node:
        case DAGConst(value=value):
            raise ValueError("Const should have been simplified away.")
        case DAGVar(name=name):
            val = predicates[name]
        case DAGNegate(arg=arg_id):
            arg_val = get_values(dag_nodes, arg_id, predicates, next_values, which, values)
            val = -arg_val
        case DAGMaxN(args=args_ids):
            vals = [get_values(dag_nodes, arg_id, predicates, next_values, which, values) for arg_id in args_ids]
            val = which.max(which.stack(vals, axis=0), axis=0)
        case DAGMinN(args=args_ids):
            vals = [get_values(dag_nodes, arg_id, predicates, next_values, which, values) for arg_id in args_ids]
            val = which.min(which.stack(vals, axis=0), axis=0)
        case DAGReach(reach=reach_id):
            reach_val = get_values(dag_nodes, reach_id, predicates, next_values, which, values)
            if is_terminal:
                # At the terminal step, Reach is just the value of the reach node.
                val = reach_val
            else:
                # reach OR next_values
                val = which.maximum(reach_val, next_values[dag_root])
        case DAGAvoid(avoid=stay_id):
            stay_val = get_values(dag_nodes, stay_id, predicates, next_values, which, values)
            if is_terminal:
                # At the terminal step, Avoid is just the value of the avoid node.
                val = stay_val
            else:
                # stay AND next_values
                val = which.minimum(stay_val, next_values[dag_root])
        case DAGReachAvoid(reach=reach_id, avoid=stay_id):
            reach_val = get_values(dag_nodes, reach_id, predicates, next_values, which, values)
            stay_val = get_values(dag_nodes, stay_id, predicates, next_values, which, values)
            if is_terminal:
                # At the terminal step, ReachAvoid is just the value of the reach node.
                val = reach_val
            else:
                # reach OR (stay AND next_values)
                val = which.maximum(reach_val, which.minimum(stay_val, next_values[dag_root]))
        case DAGGU(args=args_ids):
            raise ValueError("How to handle GU... ?")
        case _:
            raise NotImplementedError(f"Node type {type(node)} not implemented.")

    values[dag_root] = val
    return val
