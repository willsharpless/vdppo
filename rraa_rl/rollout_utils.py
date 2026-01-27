import jax.tree_util as jtu
import jax.numpy as jnp
import jax
import numpy as np

from rraa_rl.collector import RolloutOutput


def extract_rollouts_eval(bT_rollout: RolloutOutput) -> list[RolloutOutput]:
    """Extract eval rollouts into a python list of episodic rollouts."""
    extracted_rollouts = []
    b, T = bT_rollout.shape
    for traj_idx in range(b):
        T_rollout: RolloutOutput = jtu.tree_map(lambda x: x[traj_idx], bT_rollout)

        # Save only up to the index of the first term or trunc
        T_done = T_rollout.term | T_rollout.trunc

        if np.any(T_done):
            # Get the index of the first done
            first_done_idx = np.argmax(T_done)
        else:
            first_done_idx = T - 1

        T_rollout = jtu.tree_map(lambda x: x[: first_done_idx + 1], T_rollout)
        extracted_rollouts.append(T_rollout)
    return extracted_rollouts
