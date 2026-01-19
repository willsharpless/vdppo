import functools as ft
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import jax_dataclasses as jdc
import matplotlib.pyplot as plt
import numpy as np
from attrs import define
from jaxtyping import PRNGKeyArray

from rraa_rl.jax_types import BoolScalar
from rraa_rl.jax_utils import softminimum
from rraa_rl.src.env.general_task.env import Env, EnvStep

VEL_ZERO = False
HERD_ZERO = True

# If True, one of the circles is not control invariant.
TEST_INVARIANT = False


class ShouldTermFn:
    def __call__(self, predicates: dict[str, BoolScalar]) -> BoolScalar: ...


@jdc.pytree_dataclass
class HerdBaseState:
    # (n_herd, 2) [px, py]
    herd_state: jnp.ndarray
    # (n_herd, 4) [px, py, vx, vy]
    herder_state: jnp.ndarray

    steps: int = 0


@define
class HerdBaseCfg:
    herd_vel: float = 0.2
    dt: float = 0.2

    # n_herders: int = 2
    # n_herd: int = 2
    # acc_maxs: list[float] = [1.0, 2.0]
    # vel_maxs: list[float] = [0.5, 1.0]
    n_herders: int = 1
    n_herd: int = 1
    acc_maxs: list[float] = [1.0]
    vel_maxs: list[float] = [0.5]

    agent_radius: float = 0.2
    # Half size.
    halfsize: tuple[float, float] = (5.0, 5.0)

    trunc_steps: int = 100

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

    def __init__(self, cfg: HerdBaseCfg = HerdBaseCfg(), should_term_fn: ShouldTermFn = None):
        self.cfg = cfg
        assert len(cfg.acc_maxs) == len(cfg.vel_maxs) == cfg.n_herders

        self.centers = np.array([[-3.0, -3.0], [3.0, 3.0]])
        self.radiuses = np.array([1.0, 1.0])

        self.centers_perturb = self.centers[0:1]
        self.radiuses_perturb = self.radiuses[0:1] + 3 * cfg.agent_radius
        # self.centers = np.array([[-3.0, -3.0]])
        # self.radiuses = np.array([1.0])

        if should_term_fn is None:
            should_term_fn = self._should_term
        self.should_term_fn = should_term_fn

        self._obs_names = None

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

        # Desired velocity.
        herder_vel_cmd = control

        if VEL_ZERO:
            vel_inp = control
            herder_pos_new = herder_pos + vel_inp * dt
            herder_vel_new = herder_vel
        else:
            # Take velocity limit into account.
            #     Max acceleration when cmd=vel_max and current_vel = 0.
            #     =>  acc_max = kp_vel * vel_max   => kp_vel = acc_max / vel_max
            kp_vel = 0.5 * jnp.array(self.cfg.acc_maxs) / jnp.array(self.cfg.vel_maxs)
            herder_acc = kp_vel * (herder_vel_cmd - herder_vel)
            acc_max = jnp.array(self.cfg.acc_maxs)
            herder_acc = jnp.clip(herder_acc, -acc_max[:, None], acc_max[:, None])

            if TEST_INVARIANT:
                # Make the first circle not control invariant by adding a constant upwards acceleration
                # larger than the max acceleration.
                in_circ_perturb = jnp.any(self.is_herder_circ_perturb(state, which=jnp))

                max_acc_max = max(self.cfg.acc_maxs)
                circ1_acc = jnp.array([0.0, 1.1 * max_acc_max])
                herder_acc = herder_acc + jnp.where(in_circ_perturb, circ1_acc, 0.0)

                herder_vel_new = herder_vel + herder_acc * dt

                # Prevent negative velocities in y-axis.
                herder_vel_new = herder_vel_new.at[:, 1].set(jnp.maximum(herder_vel_new[:, 1], 0.0))
            else:
                herder_vel_new = herder_vel + herder_acc * dt

            herder_pos_new = herder_pos + herder_vel * dt + 0.5 * herder_acc * dt**2

            vel_max = jnp.array(self.cfg.vel_maxs)
            herder_vel_new = jnp.clip(herder_vel_new, -vel_max[:, None], vel_max[:, None])

        herder_state_new = jnp.concatenate([herder_pos_new, herder_vel_new], axis=-1)

        # Update herd states (simple dynamics: herd agents move towards the average position of the herders)
        herd_pos = state.herd_state
        herd_vel = self.compute_herd_vel(herd_pos, herder_pos)
        if HERD_ZERO:
            herd_vel = 0
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

    def is_herder_circs(self, state: HerdBaseState, which=jnp):
        h_pos = state.herder_state[..., :, 0:2]
        c_pos = which.array(self.centers)
        # (n_circs, n_herders, 2) -> (n_circs, n_herders)
        ch_dists = which.linalg.norm(h_pos[..., None, :, :] - c_pos[..., :, None, :], axis=-1)
        # (n_circs, )
        c_dists = which.min(ch_dists, axis=-1)
        c_is_herder_inside = c_dists < (which.array(self.radiuses) - self.cfg.agent_radius)
        return c_is_herder_inside

    def is_herder_circ_perturb(self, state: HerdBaseState, which=jnp):
        h_pos = state.herder_state[..., :, 0:2]
        c_pos = which.array(self.centers_perturb)
        # (n_circs, n_herders, 2) -> (n_circs, n_herders)
        ch_dists = which.linalg.norm(h_pos[..., None, :, :] - c_pos[..., :, None, :], axis=-1)
        # (n_circs, )
        c_dists = which.min(ch_dists, axis=-1)
        c_is_herder_inside = c_dists < (which.array(self.radiuses_perturb) - self.cfg.agent_radius)
        return c_is_herder_inside

    def pred_herder_circs(self, state: HerdBaseState, which=jnp):
        """
        Inside the circle is +1.
        Outside the circle is negative.
        - Linearly scale from -1 when distance=edge to -eps when distance=0
        """

        h_pos = state.herder_state[..., :, 0:2]
        c_pos = which.array(self.centers)
        # (n_circs, n_herders, 2) -> (n_circs, n_herders)
        ch_dists = which.linalg.norm(h_pos[..., None, :, :] - c_pos[..., :, None, :], axis=-1)
        # (n_circs, )
        c_dists = which.min(ch_dists, axis=-1)
        c_radiuses = which.array(self.radiuses)
        c_dist_to_circ = c_dists - c_radiuses + self.cfg.agent_radius
        eps = 0.1

        val_at_edge = -1.0
        edge = 2 * which.array(self.cfg.halfsize).max() - c_radiuses.max()
        coef = (val_at_edge + eps) / edge
        pred = jnp.where(c_dist_to_circ <= 0, 1.0, -eps + coef * c_dist_to_circ)
        pred = jnp.clip(pred, -1.0, 1.0)
        return pred

    def get_predicates_bool(self, state: HerdBaseState):
        is_herder_circs = self.is_herder_circs(state)
        predicates = {
            "herder_collide": self.is_herder_collide(state),
            "herder_oob": self.is_herder_oob(state),
            "herd_herded": self.is_herd_herded(state),
            # "herder_c1": is_herder_circs[0],
            # "herder_c2": is_herder_circs[1],
        }
        return predicates

    def get_predicates_float(self, state: HerdBaseState):
        pred_herder_circs = self.pred_herder_circs(state)
        return {"herder_c1": pred_herder_circs[0], "herder_c2": pred_herder_circs[1]}

    def get_predicates(self, state: HerdBaseState):
        predicates_bool = self.get_predicates_bool(state)
        predicates = {k: jnp.where(v, 1.0, -1.0) for k, v in predicates_bool.items()}

        predicates_float = self.get_predicates_float(state)
        predicates = predicates | predicates_float

        return predicates

    def step(self, state: HerdBaseState, action: jnp.ndarray):
        controls = self._action_to_controls(action)
        state_new = self.next_state(state, controls)
        obs_new = self.get_obs(state_new)

        predicates = self.get_predicates(state_new)
        term = self.should_term_fn(predicates)
        trunc = state_new.steps >= self.cfg.trunc_steps

        info = {"age": state_new.steps}
        return EnvStep(state_new, obs_new, predicates, term, trunc, info)

    def get_obs_names(self):
        if self._obs_names is None:
            dummy_state = self.reset(jr.PRNGKey(0))
            _, obs_names = self.get_obs_and_names(dummy_state)
            self._obs_names = obs_names
        return self._obs_names

    def get_obs_and_names(self, state: HerdBaseState):
        def fl(lst: list[list[str]]) -> list[str]:
            return [item for sublist in lst for item in sublist]

        # ---------------------------------------------------------------------
        # 1: State
        herd_pos = state.herd_state  # (n_herd, 2)
        obs_herd_pos = herd_pos / jnp.array(self.cfg.halfsize)
        obs_herd_pos_names = [[f"herd{ii}_px", f"herd{ii}_py"] for ii in range(self.cfg.n_herd)]

        herder_pos = state.herder_state[:, 0:2]  # (n_herders, 2)
        obs_herder_pos = herder_pos / jnp.array(self.cfg.halfsize)
        obs_herder_pos_names = [[f"herder{ii}_px", f"herder{ii}_py"] for ii in range(self.cfg.n_herders)]

        herder_vel = state.herder_state[:, 2:4]  # (n_herders, 2)
        assert herder_vel.shape == (self.cfg.n_herders, 2)
        obs_herder_vel = herder_vel / jnp.array(self.cfg.vel_maxs)[:, None]
        obs_herder_vel_names = [[f"herder{ii}_vx", f"herder{ii}_vy"] for ii in range(self.cfg.n_herders)]

        # ---------------------------------------------------------------------
        # 2: Relative positions, break it down into unit vectors and distances.

        if HERD_ZERO:
            n_pos = self.cfg.n_herders + len(self.centers)
            all_pos = jnp.concatenate([herder_pos, self.centers], axis=0)  # (n_herd + n_herders, 2)
        else:
            n_pos = self.cfg.n_herd + self.cfg.n_herders + len(self.centers)
            all_pos = jnp.concatenate([herd_pos, herder_pos, self.centers], axis=0)  # (n_herd + n_herders, 2)

        # Distance from each herd agent to each other agent.
        # (n_agents, n_agents, 2)
        rel_pos = all_pos[None, :, :] - all_pos[:, None, :]
        # Take the upper triangle only to avoid duplicates and self-distances.
        triu_indices = jnp.triu_indices(n_pos, k=1)
        # (n_edges, 2)
        rel_pos_triu = rel_pos[triu_indices]
        assert rel_pos_triu.shape == (n_pos * (n_pos - 1) // 2, 2)
        # Compute unit vectors and distances.
        rel_dists = jnp.linalg.norm(rel_pos_triu, axis=-1, keepdims=True) + 1e-6
        rel_unit_vecs = rel_pos_triu / rel_dists
        # (n_edges, 2)
        obs_rel_unit_vecs = rel_unit_vecs
        obs_rel_unit_vecs_names = [[f"rel_unitvec{ii}_x", f"rel_unitvec{ii}_y"] for ii, _ in enumerate(rel_unit_vecs)]

        # Normalize distances. Mean ~ half the env size, Std ~ quarter the env size.
        halfsize = 0.5 * sum(self.cfg.halfsize)
        obs_dists = (rel_dists - halfsize) / (0.5 * halfsize)
        obs_dist_names = [f"rel_dist{ii}" for ii, _ in enumerate(rel_dists)]

        # ---------------------------------------------------------------------
        if HERD_ZERO:
            obs = jnp.concatenate(
                [
                    obs_herder_pos.flatten(),
                    obs_herder_vel.flatten(),
                    obs_rel_unit_vecs.flatten(),
                    obs_dists.flatten(),
                ]
            )
            obs_names = [
                *fl(obs_herder_pos_names),
                *fl(obs_herder_vel_names),
                *fl(obs_rel_unit_vecs_names),
                *obs_dist_names,
            ]
        else:
            obs = jnp.concatenate(
                [
                    obs_herd_pos.flatten(),
                    obs_herder_pos.flatten(),
                    obs_herder_vel.flatten(),
                    obs_rel_unit_vecs.flatten(),
                    obs_dists.flatten(),
                ]
            )
            obs_names = [
                *fl(obs_herd_pos_names),
                *fl(obs_herder_pos_names),
                *fl(obs_herder_vel_names),
                *fl(obs_rel_unit_vecs_names),
                *obs_dist_names,
            ]
        return obs, obs_names

    def get_obs(self, state: HerdBaseState):
        obs, _ = self.get_obs_and_names(state)
        return obs

    def reset(self, key: PRNGKeyArray):
        n_herd = self.cfg.n_herd
        n_herders = self.cfg.n_herders
        key_herd, key_herders = jr.split(key)

        # Uniformly sample herd positions.
        halfsize_x, halfsize_y = self.cfg.halfsize
        maxpos = np.array([halfsize_x, halfsize_y]) - self.cfg.agent_radius
        if HERD_ZERO:
            maxpos = np.zeros(2)
        minpos = -maxpos
        herd_pos = jr.uniform(key_herd, shape=(n_herd, 2), minval=minpos, maxval=maxpos)

        # Uniformly sample herder positions and velocities.
        # (n_herders, 4)
        maxstate = np.zeros((n_herders, 4))
        maxstate[:, 0] = halfsize_x - self.cfg.agent_radius
        maxstate[:, 1] = halfsize_y - self.cfg.agent_radius
        maxstate[:, 2] = np.array(self.cfg.vel_maxs)
        maxstate[:, 3] = np.array(self.cfg.vel_maxs)

        if VEL_ZERO:
            maxstate[:, 2] = 0.0
            maxstate[:, 3] = 0.0

        minstate = -maxstate

        herder_state = jr.uniform(key_herders, shape=(n_herders, 4), minval=minstate, maxval=maxstate)

        return HerdBaseState(herd_state=herd_pos, herder_state=herder_state, steps=0)

    @property
    def eval_T(self) -> int:
        return self.cfg.trunc_steps

    def setup_ax(self, ax: plt.Axes):
        cfg = self.cfg
        ax.set_xlim(-1.05 * cfg.halfsize[0], 1.05 * cfg.halfsize[0])
        ax.set_ylim(-1.05 * cfg.halfsize[1], 1.05 * cfg.halfsize[1])
        ax.set_aspect("equal")

        # axvspan and axhspan to mark the boundaries.
        opts = dict(color="black", alpha=0.9)
        ax.axvspan(cfg.halfsize[0], cfg.halfsize[0] + 1.0, **opts)
        ax.axvspan(-cfg.halfsize[0] - 1.0, -cfg.halfsize[0], **opts)
        ax.axhspan(cfg.halfsize[1], cfg.halfsize[1] + 1.0, **opts)
        ax.axhspan(-cfg.halfsize[1] - 1.0, -cfg.halfsize[1], **opts)

        if not HERD_ZERO:
            # Plot the herd circle.
            herd_circle = plt.Circle((0, 0), cfg.herded_radius, color="lightgray", alpha=0.5)
            ax.add_patch(herd_circle)

        if TEST_INVARIANT:
            # Plot the perturbation circles.
            for ii, center in enumerate(self.centers_perturb):
                radius = self.radiuses_perturb[ii]
                circ = plt.Circle((center[0], center[1]), radius, color="C0", alpha=0.2)
                ax.add_patch(circ)

        # Plot the circles.
        for ii, center in enumerate(self.centers):
            radius = self.radiuses[ii]
            circ = plt.Circle((center[0], center[1]), radius, color="C5", alpha=0.3)
            ax.add_patch(circ)
