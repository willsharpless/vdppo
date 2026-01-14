import glob
import os
import pickle
import socket
from typing import List, Tuple, TypeVar

import flax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax_dataclasses as jdc
import numpy as np


def internet(host="8.8.8.8", port=53, timeout=3) -> bool:
    """
    Host: 8.8.8.8 (google-public-dns-a.google.com)
    OpenPort: 53/tcp
    Service: domain (DNS/TCP)
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error as ex:
        print(ex)
        return False


def is_connected() -> bool:
    return internet()


def has_nan(x) -> jnp.ndarray:
    return jtu.tree_map(lambda y: jnp.isnan(y).any(), x)


def has_any_nan(x) -> jnp.ndarray:
    return jnp.array(jtu.tree_flatten(has_nan(x))[0]).any()


def has_inf(x) -> jnp.ndarray:
    return jtu.tree_map(lambda y: jnp.isinf(y).any(), x)


def has_any_inf(x) -> jnp.ndarray:
    return jnp.array(jtu.tree_flatten(has_inf(x))[0]).any()


def has_any_nan_or_inf(x) -> jnp.ndarray:
    return has_any_nan(x) | has_any_inf(x)


def compute_norm(grad) -> jnp.ndarray:
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jtu.tree_leaves(grad)))


def compute_norm_and_clip(grad, max_norm: float) -> Tuple[jtu.PyTreeDef, jnp.ndarray, jnp.ndarray]:
    g_norm = compute_norm(grad)
    clipped_g_norm = jnp.maximum(max_norm, g_norm)
    clipped_grad = jtu.tree_map(lambda t: (t / clipped_g_norm) * max_norm, grad)
    clipped_g_norm = compute_norm(clipped_grad)

    return clipped_grad, g_norm, clipped_g_norm


def save_agent(agent, save_dir, epoch) -> None:
    """Save the agent to a file.

    Args:
        agent: Agent.
        save_dir: Directory to save the agent.
        epoch: Epoch number.
    """

    save_dict = dict(
        agent=agent.to_state_dict(),
    )
    save_path = os.path.join(save_dir, f"params_{epoch}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(save_dict, f)


def restore_agent(agent, restore_path, restore_epoch) -> jtu.PyTreeDef:
    """Restore the agent from a file.

    Args:
        agent: Agent.
        restore_path: Path to the directory containing the saved agent.
        restore_epoch: Epoch number.
    """
    candidates = glob.glob(restore_path)

    assert len(candidates) == 1, f"Found {len(candidates)} candidates: {candidates}"

    restore_path = candidates[0] + f"/params_{restore_epoch}.pkl"

    with open(restore_path, "rb") as f:
        load_dict = pickle.load(f)

    agent = flax.serialization.from_state_dict(agent, load_dict["agent"])

    print(f"Restored from {restore_path}")

    return agent


def fmt_desc(s, desc_width=30) -> str:
    """Format description for tqdm progress bar"""
    return f"{s:<{desc_width}}"[:desc_width]


# def extract_rollouts_eval(b_rollout: RolloutOutput) -> List[RolloutOutput]:
#     """Extract eval rollouts into a python list of episodic rollouts."""
#     extracted_rollouts = []
#     b, _ = b_rollout.shape
#     for traj_idx in range(b):
#         rollout = jtu.tree_map(lambda x: x[traj_idx], b_rollout)
#
#         # Save only up to the index of the first term or trunc
#         dones = rollout.T_term | rollout.T_trunc
#         assert jnp.any(dones), "No done signal found in the rollout!"
#
#         # Get the index of the first done
#         first_done_idx = jnp.argmax(dones.astype(jnp.int32))
#         rollout = jtu.tree_map(lambda x: x[: first_done_idx + 1], rollout)
#         extracted_rollouts.append(rollout)
#     return extracted_rollouts


_pytree = TypeVar("_pytree")


def jax2np(pytree: _pytree) -> _pytree:
    return jtu.tree_map(np.array, pytree)


def np2jax(pytree: _pytree) -> _pytree:
    return jtu.tree_map(jnp.array, pytree)


# def convert_rollout_states_to_minstates(rollout: RolloutOutput, env: SingleAgentEnv) -> RolloutOutput:
#     """Convert rollout states to minstates using env.get_minstate()."""
#     return jdc.replace(
#         rollout,
#         T_state_now=env.convert_state_to_minstate(rollout.T_state_now),
#         T_state_next=env.convert_state_to_minstate(rollout.T_state_next),
#     )


def hash_key(key: jnp.ndarray) -> jnp.uint32:
    k = jr.key_data(key)  # shape (2,), dtype=uint32
    return k[0] * jnp.uint32(0x9E3779B1) + k[1]


def tree_where_dim0(mask, a, b):
    """
    mask: (b,) boolean
    a, b: PyTrees with leading dim b
    """
    mask = np.asarray(mask)
    return jtu.tree_map(lambda x, y: np.where(mask[:, None], x, y), a, b)
