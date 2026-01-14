import functools as ft
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from attrs import define
from flax import struct
from jaxtyping import PRNGKeyArray
from valtr.reachability import collect_predicate_info, extract_trigger_predicate_map
from valtr.valtr import to_dag

from rraa_rl.jax_utils import softminimum
from rraa_rl.src.env.general_task.env import Env, EnvStep


class HerdBaseState(NamedTuple):
    # (n_herd, 2) [px, py]
    herd_state: jnp.ndarray
    # (n_herd, 4) [px, py, vx, vy]
    herder_state: jnp.ndarray

    steps: int = 0


@define
class HerdBaseCfg:
    n_herders: int = 2
    n_herd: int = 2

    herd_vel: float = 0.2

    dt: float = 0.2
    acc_maxs: list[float] = [1.0, 2.0]
    vel_maxs: list[float] = [0.5, 1.0]

    agent_radius: float = 0.2
    # Half size.
    halfsize: tuple[float, float] = (5.0, 5.0)

    trunc_steps: int = 500

    herded_radius: float = 1.0  # Radius within which herd agents are considered herded.


class HerdBase(Env):
    """Herding environment with one or more herders and a herd of agents. The herd moves according to some fixed policy.
    The herders can influence the herd by moving around them.

    Each herd agent is a single-integrator that minimizes the soft minimum distance to the herders, the obstacles,
    and other herd agents, where the distances are scaled such that herders have larger influence.
    If the distance is large enough, the herd agents stay still.

    In the discrete action setup, each herder is a double-integrator that can accelerate / decelerate in either axis.

    action: (n_herders, 2): int, {0, 1, 2} for each axis, where 0 = -accel, 1 = no accel, 2 = accel
    """

    def __init__(self, cfg: HerdBaseCfg = HerdBaseCfg()):
        self.cfg = cfg
        assert len(cfg.acc_maxs) == len(cfg.vel_maxs) == cfg.n_herders

    @property
    def n_agents(self) -> int:
        return self.cfg.n_herd

    @property
    def n_actions_per_agent(self) -> list[list[int]]:
        # Each herder has 3 actions per axis.
        n_actions_per_agent = []
        for _ in range(self.cfg.n_herders):
            n_actions_per_agent.append([3, 3])
        return n_actions_per_agent

    def _action_to_controls(self, action: jnp.ndarray):
        """Convert discrete action to continuous accelerations."""
        n_herders = self.cfg.n_herders
        accs = []
        for i in range(n_herders):
            acc_max = self.cfg.acc_maxs[i]
            acc = jnp.where(action[i] == 0, -acc_max, jnp.where(action[i] == 2, acc_max, 0.0))
            accs.append(acc)
        controls = jnp.stack(accs, axis=0)
        return controls

    def dist_to_wall(self, pos: jnp.ndarray):
        """Compute distance to walls given positions."""
        halfsize = self.cfg.halfsize
        px, py = pos[..., 0], pos[..., 1]
        left_dists = px + halfsize[0]
        right_dists = halfsize[0] - px
        bottom_dists = py + halfsize[1]
        top_dists = halfsize[1] - py
        dists = jnp.stack([left_dists, right_dists, bottom_dists, top_dists], axis=-1)  # (n_agents, 4)
        return dists

    def compute_herd_vel(self, n_herd_pos: jnp.ndarray, m_herder_pos: jnp.ndarray):
        def get_weighted_dist(ii: int, herd_pos_new: jnp.ndarray):
            # Compute the minimum distance to the other herd agents.

            n_herd_dist = jnp.linalg.norm(n_herd_pos - herd_pos_new, axis=-1)
            # Ignore self-distance
            n_herd_dist = n_herd_dist.at[ii].set(jnp.inf)
            herd_softmin = softminimum(n_herd_dist)
            herd_min = jnp.min(n_herd_dist)

            # Compute the minimum distance to the herders.
            # (n_herd, 1, 2) - (1, n_herders, 2) -> (n_herd, n_herders, 2) -> (n_herd, n_herders)
            m_herder_dist = jnp.linalg.norm(m_herder_pos - herd_pos_new, axis=-1)
            herder_softmin = softminimum(m_herder_dist)
            herder_min = jnp.min(m_herder_dist)

            # Compute the minimum distance to the walls.
            herd_wall_dists = self.dist_to_wall(herd_pos_new)
            herd_wall_softmin = softminimum(herd_wall_dists, axis=-1)
            herd_wall_min = jnp.min(herd_wall_dists)

            herd_max_dist = 15 * self.cfg.agent_radius
            herder_max_dist = 15 * self.cfg.agent_radius
            wall_max_dist = 15 * self.cfg.agent_radius
            dist_thresh = jnp.array([herd_max_dist, herder_max_dist, wall_max_dist])
            apply_action_herd = jnp.any(jnp.array([herd_min, herder_min, herd_wall_min]) <= dist_thresh)

            w_herd = 0.5
            w_herder = 2.0
            w_wall = 2.0
            vals = jnp.array([herd_softmin, herder_softmin, herd_wall_softmin])
            weights = jnp.array([w_herd, w_herder, w_wall])
            weighted_dist = softminimum(vals * weights)
            return weighted_dist, apply_action_herd

        def get_vel_single(ii: int):
            herd_pos = n_herd_pos[ii]

            # Generate candidate actions uniformly in a circle.
            angles = jnp.linspace(0, 2 * jnp.pi, num=16, endpoint=False)
            deltas = self.cfg.herd_vel * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)  # (num_actions, 2)
            herd_pos_new = herd_pos + deltas  # (num_actions, 2)

            _, apply_action = get_weighted_dist(ii, herd_pos)
            weighted_dists, _ = jax.vmap(ft.partial(get_weighted_dist, ii))(herd_pos_new)  # (num_actions,)
            # Select the action that maximizes the weighted distance.
            best_idx = jnp.argmax(weighted_dists)
            best_delta = deltas[best_idx]
            best_delta = best_delta * jnp.where(apply_action, 1.0, 0.0)
            return best_delta

        n_idxs = jnp.arange(self.cfg.n_herd)
        n_herd_vel = jax.vmap(get_vel_single)(n_idxs)

        return n_herd_vel

    def next_state(self, state: HerdBaseState, control: jnp.ndarray):
        """Compute next state given current state and control inputs."""
        dt = self.cfg.dt

        # Update herder states
        herder_pos = state.herder_state[:, 0:2]
        herder_vel = state.herder_state[:, 2:4]
        herder_acc = control

        # Take velocity limit into account.
        time_till_vmax = jnp.where(
            herder_acc > 0,
            (jnp.array(self.cfg.vel_maxs) - herder_vel) / herder_acc,
            jnp.where(herder_acc < 0, -herder_vel / herder_acc, jnp.inf),
        )
        time_till_vmax = jnp.maximum(time_till_vmax, 0.0)
        acc_dt = jnp.minimum(dt, time_till_vmax)
        noaccel_dt = dt - acc_dt
        # Accelerate for effective_dt, then zero acceleration for the rest of dt.
        herder_vel_new = herder_vel + herder_acc * acc_dt
        herder_pos_mid = herder_pos + herder_vel * acc_dt + 0.5 * herder_acc * acc_dt**2
        herder_pos_new = herder_pos_mid + herder_vel_new * noaccel_dt
        herder_state_new = jnp.concatenate([herder_pos_new, herder_vel_new], axis=-1)

        # Update herd states (simple dynamics: herd agents move towards the average position of the herders)
        herd_pos = state.herd_state
        herd_vel = self.compute_herd_vel(herd_pos, herder_pos)
        herd_state_new = herd_pos + herd_vel * dt

        return HerdBaseState(herd_state=herd_state_new, herder_state=herder_state_new, steps=state.steps + 1)

    def is_herder_collide(self, state: HerdBaseState):
        herder_pos = state.herder_state[:, 0:2]
        n_herders = herder_pos.shape[0]

        def check_pair(i: int, j: int):
            dist = jnp.linalg.norm(herder_pos[i] - herder_pos[j])
            collide = dist < 2 * self.cfg.agent_radius
            return collide

        collide = False
        for i in range(n_herders):
            for j in range(i + 1, n_herders):
                collide = collide | check_pair(i, j)
        return collide

    def is_herder_oob(self, state: HerdBaseState):
        herder_pos = state.herder_state[:, 0:2]
        dists = self.dist_to_wall(herder_pos)
        min_dists = jnp.min(dists, axis=-1)
        oob = jnp.any(min_dists < self.cfg.agent_radius)
        return oob

    def is_herd_herded(self, state: HerdBaseState):
        """All herd agents are fully within a circle in the center."""
        herd_pos = state.herd_state
        dists = jnp.linalg.norm(herd_pos, axis=-1)
        herded = jnp.all((dists + self.cfg.agent_radius) < self.cfg.herded_radius)
        return herded

    def get_predicates_bool(self, state: HerdBaseState):
        predicates = {
            "herder_collide": self.is_herder_collide(state),
            "herder_oob": self.is_herder_oob(state),
            "herd_herded": self.is_herd_herded(state),
        }
        return predicates

    def step(self, state: HerdBaseState, action: jnp.ndarray):
        controls = self._action_to_controls(action)
        state_new = self.next_state(state, controls)
        obs_new = self.get_obs(state_new)

        predicates = self.get_predicates_bool(state_new)
        term = predicates["herder_collide"] | predicates["herder_oob"]
        trunc = state_new.steps >= self.cfg.trunc_steps

        # Make it +1 if true, -1 if false.
        predicates_float = {k: jnp.where(v, 1, -1) for k, v in predicates.items()}

        info = {}
        return EnvStep(state_new, obs_new, predicates_float, term, trunc, info)

    def get_obs(self, state: HerdBaseState):
        # ---------------------------------------------------------------------
        # 1: State
        herd_pos = state.herd_state  # (n_herd, 2)
        obs_herd_pos = herd_pos / jnp.array(self.cfg.halfsize)

        herder_pos = state.herder_state[:, 0:2]  # (n_herders, 2)
        obs_herder_pos = herder_pos / jnp.array(self.cfg.halfsize)

        herder_vel = state.herder_state[:, 2:4]  # (n_herders, 2)
        obs_herder_vel = herder_vel / jnp.array(self.cfg.vel_maxs)[:, None]

        # ---------------------------------------------------------------------
        # 2: Relative positions, break it down into unit vectors and distances.
        n_agents = self.cfg.n_herd + self.cfg.n_herders

        all_pos = jnp.concatenate([herd_pos, herder_pos], axis=0)  # (n_herd + n_herders, 2)
        # Distance from each herd agent to each other agent.
        # (n_agents, n_agents, 2)
        rel_pos = all_pos[None, :, :] - all_pos[:, None, :]
        # Take the upper triangle only to avoid duplicates and self-distances.
        triu_indices = jnp.triu_indices(n_agents, k=1)
        # (n_edges, 2)
        rel_pos_triu = rel_pos[triu_indices]
        # Compute unit vectors and distances.
        rel_dists = jnp.linalg.norm(rel_pos_triu, axis=-1, keepdims=True) + 1e-6
        rel_unit_vecs = rel_pos_triu / rel_dists
        # (n_edges, 2)
        obs_rel_unit_vecs = rel_unit_vecs

        # Normalize distances. Mean ~ half the env size, Std ~ quarter the env size.
        halfsize = 0.5 * sum(self.cfg.halfsize)
        obs_dists = (rel_dists - halfsize) / (0.25 * halfsize)
        # ---------------------------------------------------------------------
        obs = jnp.concatenate(
            [
                obs_herd_pos.flatten(),
                obs_herder_pos.flatten(),
                obs_herder_vel.flatten(),
                obs_rel_unit_vecs.flatten(),
                obs_dists.flatten(),
            ]
        )
        return obs

    def reset(self, key: PRNGKeyArray):
        n_herd = self.cfg.n_herd
        n_herders = self.cfg.n_herders
        key_herd, key_herders = jr.split(key)

        # Uniformly sample herd positions.
        halfsize_x, halfsize_y = self.cfg.halfsize
        maxpos = np.array([halfsize_x, halfsize_y]) - self.cfg.agent_radius
        minpos = -maxpos
        herd_pos = jr.uniform(key_herd, shape=(n_herd, 2), minval=minpos, maxval=maxpos)

        # Uniformly sample herder positions and velocities.
        # (n_herders, 4)
        maxstate = np.zeros((n_herders, 4))
        maxstate[:, 0] = halfsize_x - self.cfg.agent_radius
        maxstate[:, 1] = halfsize_y - self.cfg.agent_radius
        maxstate[:, 2] = np.array(self.cfg.vel_maxs)
        maxstate[:, 3] = np.array(self.cfg.vel_maxs)
        minstate = -maxstate

        herder_state = jr.uniform(key_herders, shape=(n_herders, 4), minval=minstate, maxval=maxstate)

        return HerdBaseState(herd_state=herd_pos, herder_state=herder_state, steps=0)
