import ipdb
import numpy as np

from rraa_rl.collector import RolloutOutput
from rraa_rl.src.env.general_task.herd_os import DAGTransition, HerdOs


def evaluate_triggers(env: HerdOs, trajs: list[RolloutOutput]) -> dict:
    triggers_dict = {}
    for ii, traj in enumerate(trajs):
        triggers = env.get_rules(traj.predicates)
        # triggers = [DAGTransition(trigger.parent, trigger.child, np.any(trigger.condition)) for trigger in triggers]

        for trigger in triggers:
            key = (trigger.parent, trigger.child)
            triggers_dict.get(key, [])
            triggers_dict[key].append(np.any(trigger.condition))

    triggers_dict = {k: np.array(v) for k, v in triggers_dict.items()}
    return triggers_dict
