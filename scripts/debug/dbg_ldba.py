import ipdb
import jax.numpy as jnp
import jax.random as jr
import jax_dataclasses as jdc
import numpy as np
from loguru import logger

from rraa_rl.lcrl.lcrl_wrapper import LCRLWrapper
from rraa_rl.env.general_task.env import EnvStep
from rraa_rl.env.general_task.get_env import get_env_and_cbs
from rraa_rl.env.general_task.gridworld import GridworldMAState


def main():
    env, _, _ = get_env_and_cbs("gridworld_map1", "lcrl")

    pos0 = np.array([0, 0])

    # Go to A, then hit wall.
    # actions = "UURD"
    actions = "UURLDDRRRUURRRDDLRUU"

    action_dict = dict(S=0, U=1, D=2, R=3, L=4)

    actions = [action_dict[a] for a in actions]

    state: LCRLWrapper.State[GridworldMAState] = env.reset(jr.PRNGKey(0))
    with jdc.copy_and_mutate(state) as state:
        state.base.pos = jnp.array(pos0)[None, :]

    for kk, action in enumerate(actions):
        logger.info("automata: {} | pos: {}".format(state.ldba_state.state, state.base.pos))
        action = [jnp.array([action]), jnp.array([0])]
        step: EnvStep[LCRLWrapper.State[GridworldMAState]] = env.step(state, action)
        state_new = step.envstate
        predicates = [k for k, v in step.predicates.items() if jnp.all(v > 0)]
        logger.info(
            "-> automata: {} | pos: {} | term={}, trunc={} | predicates: {}".format(
                state_new.ldba_state.state, state_new.base.pos, step.term, step.trunc, predicates
            )
        )
        state = state_new
        logger.info("---")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
