import functools as ft
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax_dataclasses as jdc
import matplotlib.pyplot as plt
import numpy as np
from attrs import define
from jaxtyping import PRNGKeyArray

from rraa_rl.jax_types import BoolScalar
from rraa_rl.jax_utils import softmaximum, softminimum, tree_stack
from rraa_rl.src.env.general_task.env import BaseEnv, Env, EnvStep
from rraa_rl.train_utils import tree_where

VEL_ZERO = False
# HERD_ZERO = True

# # If True, one of the circles is not control invariant.
# TEST_INVARIANT = False


class ShouldTermFn:
    def __call__(self, predicates: dict[str, BoolScalar]) -> BoolScalar: ...


@jdc.pytree_dataclass
class HerdBaseState:
    # (n_herd, 2) [px, py]
    herd_state: jnp.ndarray
    # (n_herd, 4) [px, py, vx, vy]
    herder_state: jnp.ndarray

    steps: int = 0


@define(slots=False)
class HerdBaseCfg:
    herd_vel: float = 0.25
    herd_vel_self: float = 0.1
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

    herd_zero: bool = True
    """If True, pretend the herd agents don't exist."""

    trunc_steps: int = 100

    herded_radius: float = 1.0  # Radius within which herd agents are considered herded.


@define(slots=False)
class HerdingHerdCfg(HerdBaseCfg):
    p_reset_center: float = 0.1


class HerdBase(BaseEnv):
    """Herding environment with one or more herders and a herd of agents. The herd moves according to some fixed policy.
    The herders can influence the herd by moving around them.

    Each herd agent is a single-integrator that minimizes the soft minimum distance to the herders, the obstacles,
    and other herd agents, where the distances are scaled such that herders have larger influence.
    If the distance is large enough, the herd agents stay still.

    In the discrete action setup, each herder is a double-integrator that can accelerate / decelerate in either axis.

    action: (n_herders, 2): int, {0, 1, 2} for each axis, where 0 = -accel, 1 = no accel, 2 = accel
    """

    Cfg = HerdBaseCfg

    def __init__(self, cfg: HerdBaseCfg = HerdBaseCfg(), should_term_fn: ShouldTermFn = None):
        super().__init__()

        self.cfg = cfg
        assert len(cfg.acc_maxs) == len(cfg.vel_maxs) == cfg.n_herders

        if should_term_fn is None:
            should_term_fn = self._should_term
        self.should_term_fn = should_term_fn

    @property
    def n_agents(self) -> int:
        return self.cfg.n_herders

    @property
    def value_lims(self):
        return -1, 1

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

            # Keep the softmin error <= 0.05 * halfwidth. error <= temperature * log(n)  =>  temperature = error / log(n)
            temperature = 0.05 * min(self.cfg.halfsize) / jnp.log(max(self.cfg.n_herd, self.cfg.n_herders, 4))

            # Compute the minimum distance to the herders.
            # (n_herd, 1, 2) - (1, n_herders, 2) -> (n_herd, n_herders, 2) -> (n_herd, n_herders)
            m_herder_dist = jnp.linalg.norm(m_herder_pos - herd_pos_new, axis=-1)
            herder_softmin = softminimum(m_herder_dist, temperature=temperature)
            herder_min = jnp.min(m_herder_dist)

            # Compute the minimum distance to the walls.
            herd_wall_dists = self.dist_to_wall(herd_pos_new)
            herd_wall_softmin = softminimum(herd_wall_dists, temperature=temperature, axis=-1)
            herd_wall_min = jnp.min(herd_wall_dists)

            # If the distance to the wall is larger than a threshold, then treat the distance as very big.
            # Smoothly increase the effect of this
            wall_dist_thresh = 10 * self.cfg.agent_radius
            coef = 1 + 2 * jnp.tanh(herd_wall_min / wall_dist_thresh * 2)
            herd_wall_softmin = coef * herd_wall_softmin

            herd_max_dist = 15 * self.cfg.agent_radius
            herder_max_dist = 15 * self.cfg.agent_radius
            wall_max_dist = 15 * self.cfg.agent_radius
            dist_thresh = jnp.array([herd_max_dist, herder_max_dist, wall_max_dist])
            apply_action_herd = jnp.any(jnp.array([herd_min, herder_min, herd_wall_min]) <= dist_thresh)

            w_herd = 0.1
            w_herder = 2.0
            w_wall = 1.5
            vals = jnp.array([herd_softmin, herder_softmin, herd_wall_softmin])
            weights = jnp.array([w_herd, w_herder, w_wall])
            # Higher weight => divide by larger number => is minimum more often.
            weighted_dist = softminimum(vals / weights, temperature=temperature)
            closest = jnp.argmin(vals / weights)
            return weighted_dist, apply_action_herd, closest

        def get_vel_single(ii: int):
            herd_pos = n_herd_pos[ii]

            # Generate candidate actions uniformly in a circle.
            angles = jnp.linspace(0, 2 * jnp.pi, num=16, endpoint=False)
            vel_test = self.cfg.herd_vel * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)  # (num_actions, 2)
            herd_pos_new = herd_pos + vel_test * self.cfg.dt  # (num_actions, 2)

            _, apply_action, closest_idx = get_weighted_dist(ii, herd_pos)
            weighted_dists, _, _ = jax.vmap(ft.partial(get_weighted_dist, ii))(herd_pos_new)  # (num_actions,)
            # Select the action that maximizes the weighted distance.
            best_idx = jnp.argmax(weighted_dists)
            best_vel = vel_test[best_idx]
            best_vel = best_vel * jnp.where(apply_action, 1.0, 0.0)

            # If the closest thing is the herd, then move slower than if the closest is a herder.
            closest_is_herd = closest_idx == 0
            best_vel = jnp.where(closest_is_herd, self.cfg.herd_vel_self / self.cfg.herd_vel, 1.0) * best_vel

            return best_vel

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
            herder_vel_new = herder_vel + herder_acc * dt

            herder_pos_new = herder_pos + herder_vel * dt + 0.5 * herder_acc * dt**2

            vel_max = jnp.array(self.cfg.vel_maxs)
            herder_vel_new = jnp.clip(herder_vel_new, -vel_max[:, None], vel_max[:, None])

        herder_state_new = jnp.concatenate([herder_pos_new, herder_vel_new], axis=-1)

        # Update herd states (simple dynamics: herd agents move towards the average position of the herders)
        herd_pos = state.herd_state
        herd_vel = self.compute_herd_vel(herd_pos, herder_pos)
        if self.cfg.herd_zero:
            herd_vel = 0
        herd_state_new = herd_pos + herd_vel * dt

        info_dyn = {"dyn/herd_vel": herd_vel}

        return HerdBaseState(herd_state=herd_state_new, herder_state=herder_state_new, steps=state.steps + 1), info_dyn

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
        herded = jnp.all((dists + self.cfg.agent_radius) <= self.cfg.herded_radius)
        return herded

    def get_predicates_bool(self, state: HerdBaseState):
        predicates = {
            "herder_collide": self.is_herder_collide(state),
            "herder_oob": self.is_herder_oob(state),
            "herd_herded": self.is_herd_herded(state),
        }
        return predicates

    def get_predicates_float(self, state: HerdBaseState):
        return {}

    def get_predicates(self, state: HerdBaseState):
        predicates_bool = self.get_predicates_bool(state)
        predicates = {k: jnp.where(v, 1.0, -1.0) for k, v in predicates_bool.items()}

        predicates_float = self.get_predicates_float(state)
        predicates = predicates | predicates_float

        return predicates

    def step(self, state: HerdBaseState, action: jnp.ndarray):
        controls = self._action_to_controls(action)
        state_new, info_dyn = self.next_state(state, controls)
        obs_new = self.get_obs(state_new)

        predicates = self.get_predicates(state_new)
        term = self.should_term_fn(predicates)
        trunc = state_new.steps >= self.cfg.trunc_steps

        info = {"age": state_new.steps} | info_dyn
        return EnvStep(state_new, obs_new, predicates, term, trunc, info)

    def get_objects_pos(self, state: HerdBaseState):
        herd_pos = state.herd_state  # (n_herd, 2)
        herder_pos = state.herder_state[:, 0:2]  # (n_herders, 2)

        if self.cfg.herd_zero:
            all_pos = herder_pos
        else:
            all_pos = jnp.concatenate([herd_pos, herder_pos], axis=0)  # (n_herd + n_herders, 2)

        return all_pos

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
        all_pos = self.get_objects_pos(state)
        n_pos = len(all_pos)

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
        if self.cfg.herd_zero:
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
        if self.cfg.herd_zero:
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

        if not self.cfg.herd_zero:
            # Plot the herd circle.
            herd_circle = plt.Circle((0, 0), cfg.herded_radius, color="lightgray", alpha=0.5)
            ax.add_patch(herd_circle)


@define(slots=False)
class HerdBasePlayCfg(HerdBaseCfg):
    test_invariant: bool = False


class HerdBasePlay(HerdBase):
    """For testing."""

    Cfg = HerdBasePlayCfg

    def __init__(self, cfg: HerdBasePlayCfg, should_term_fn: ShouldTermFn = None):
        super().__init__(cfg, should_term_fn=should_term_fn)
        self.cfg = cfg

        self.centers = np.array([[-3.0, -3.0], [3.0, 3.0]])
        self.radiuses = np.array([1.0, 1.0])

        self.centers_perturb = self.centers[0:1]
        self.radiuses_perturb = self.radiuses[0:1] + 3 * cfg.agent_radius

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

            if self.cfg.test_invariant:
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
        if self.cfg.herd_zero:
            herd_vel = 0
        herd_state_new = herd_pos + herd_vel * dt

        next_state = HerdBaseState(herd_state=herd_state_new, herder_state=herder_state_new, steps=state.steps + 1)
        info_dyn = {}
        return next_state, info_dyn

    def get_predicates_bool(self, state: HerdBaseState):
        predicates = super().get_predicates_bool(state)
        return predicates

    def get_predicates_float(self, state: HerdBaseState):
        predicates = super().get_predicates_float(state)
        pred_herder_circs = self.pred_herder_circs(state)
        return {"herder_c1": pred_herder_circs[0], "herder_c2": pred_herder_circs[1]} | predicates

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

    def setup_ax(self, ax: plt.Axes):
        super().setup_ax(ax)

        if self.cfg.test_invariant:
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


class HerdingHerd(HerdBase):
    Cfg = HerdingHerdCfg

    def __init__(self, cfg: HerdingHerdCfg = HerdingHerdCfg(), should_term_fn: ShouldTermFn = None):

        super().__init__(cfg, should_term_fn=should_term_fn)
        self.cfg = cfg

    def reset(self, key: PRNGKeyArray):
        # With some prob, reset the herd in the center.
        p_reset_center = self.cfg.p_reset_center
        # p_reset_center = 0.5
        # With some prob, reset the herd within small circle, and herder agents on outside pointing inwards.
        p_reset_herd = 0.25
        p_reset_orig = 1.0 - p_reset_center - p_reset_herd

        key_orig, key_center, key_herding, key_which = jr.split(key, 4)

        herd_state_orig = super().reset(key_orig)
        herd_state_center = self.reset_center(key_center)
        herd_state_herding, _ = self.reset_herding(key_herding)

        # reset_center = jr.bernoulli(key_do_center, p=p_reset_center)
        which_reset = jr.categorical(key_which, jnp.array([p_reset_orig, p_reset_center, p_reset_herd]))

        herd_state_stack = tree_stack([herd_state_orig, herd_state_center, herd_state_herding])
        herd_state = jtu.tree_map(lambda x: x[which_reset], herd_state_stack)

        return herd_state

    def reset_center(self, key: PRNGKeyArray):
        # All three herd agents in the center, as close as possible without overlapping.
        # Uniformly spread out in a circle, randomize the rotation.
        # Herders are randomly placed on the outside.
        cfg = self.cfg

        key_herd_angle0, key_herd_radius, key_herder_angle, key_herder_radius, key_herder_vel = jr.split(key, 5)

        # -------------------------------
        angle0 = jr.uniform(key_herd_angle0, minval=0.0, maxval=2 * jnp.pi)
        herd_angles = angle0 + jnp.linspace(0, 2 * jnp.pi, num=self.cfg.n_herd, endpoint=False)
        # Should be >= r/sin(pi/n) to not be overlapping.
        min_radius = cfg.agent_radius / jnp.sin(jnp.pi / cfg.n_herd)
        max_radius = cfg.herded_radius - cfg.agent_radius
        herd_radius = jr.uniform(key=key_herd_radius, minval=1.01 * min_radius, maxval=1.01 * max_radius)

        herd_pos_x = herd_radius * jnp.cos(herd_angles)
        herd_pos_y = herd_radius * jnp.sin(herd_angles)
        her_pos = jnp.stack([herd_pos_x, herd_pos_y], axis=-1)
        # -------------------------------

        herder_angles = jr.uniform(key_herder_angle, shape=(cfg.n_herders,), minval=0.0, maxval=2 * jnp.pi)
        min_radius = herd_radius + 2.1 * cfg.agent_radius
        max_radius = herd_radius + 5.0 * cfg.agent_radius
        herder_radius = jr.uniform(
            key=key_herder_radius,
            shape=(cfg.n_herders,),
            minval=min_radius,
            maxval=max_radius,
        )
        herder_pos_x = herder_radius * jnp.cos(herder_angles)
        herder_pos_y = herder_radius * jnp.sin(herder_angles)
        herder_pos = jnp.stack([herder_pos_x, herder_pos_y], axis=-1)

        herder_vel = jr.uniform(
            key=key_herder_vel,
            shape=(cfg.n_herders, 2),
            minval=-jnp.array(cfg.vel_maxs) * 0.5,
            maxval=jnp.array(cfg.vel_maxs) * 0.5,
        )
        herder_state = jnp.concatenate([herder_pos, herder_vel], axis=-1)

        return HerdBaseState(herd_state=her_pos, herder_state=herder_state, steps=0)

    def reset_herding(self, key: PRNGKeyArray):
        # Two herd agents initialized on opposite sides of a circle of varying radius.
        # All other herd agents initialized randomly inside the circle.
        # The center of the circle is close to the herding center.
        cfg = self.cfg
        if cfg.n_herd == 1:
            raise NotImplementedError("TODO")

        key_herd, key_herders = jr.split(key, 2)

        # --------------------------------------------
        key_circle_radius, key_circle_center, key_angles, key_radius_frac = jr.split(key_herd, 4)

        min_radius = cfg.agent_radius / jnp.sin(jnp.pi / cfg.n_herd)
        min_radius = min_radius + cfg.agent_radius
        max_radius = cfg.herded_radius
        radius = jr.uniform(key_circle_radius, minval=min_radius, maxval=max_radius)

        max_pos = 0.5 * cfg.halfsize[0]
        min_pos = -max_pos
        circle_center = jr.uniform(key_circle_center, shape=(2,), minval=min_pos, maxval=max_pos)

        angles = jr.uniform(key_angles, shape=(cfg.n_herd,), minval=0.0, maxval=2 * jnp.pi)
        angles = angles.at[1].set(angles[0] + jnp.pi)

        radius_fracs = jr.uniform(key_radius_frac, shape=(cfg.n_herd,), minval=0.0, maxval=1.0)
        radius_fracs = radius_fracs.at[:2].set(1.0)

        herd_pos_x = circle_center[0] + radius * radius_fracs * jnp.cos(angles)
        herd_pos_y = circle_center[1] + radius * radius_fracs * jnp.sin(angles)
        herd_pos = jnp.stack([herd_pos_x, herd_pos_y], axis=-1)

        herd_pos = jnp.clip(herd_pos, -cfg.halfsize[0] + cfg.agent_radius, cfg.halfsize[0] - cfg.agent_radius)

        # --------------------------------------------
        key_angles, key_radius, key_vel = jr.split(key_herders, 3)

        # Herders are placed such that the herd is between them and the center.
        herd_circle_angle = jnp.arctan2(circle_center[1], circle_center[0])
        herd_circle_radius = jnp.linalg.norm(circle_center) + radius

        # Compute the max angle such that the line from the herder to the center is tangential to the herd circle.
        inside = radius / herd_circle_radius
        inside_safe = jnp.clip(inside, -0.999, 0.999)
        max_angle_offset = jnp.arcsin(inside_safe)

        herder_angles = herd_circle_angle + jr.uniform(
            key_angles,
            shape=(cfg.n_herders,),
            minval=-max_angle_offset,
            maxval=max_angle_offset,
        )

        herder_radius = jr.uniform(
            key_radius,
            shape=(cfg.n_herders,),
            # minval=herd_circle_radius - 1.0 * cfg.agent_radius,
            minval=herd_circle_radius + 1.0 * cfg.agent_radius,
            maxval=herd_circle_radius + 7.0 * cfg.agent_radius,
        )
        herder_pos_x = herder_radius * jnp.cos(herder_angles)
        herder_pos_y = herder_radius * jnp.sin(herder_angles)
        herder_pos = jnp.stack([herder_pos_x, herder_pos_y], axis=-1)

        herder_pos = jnp.clip(herder_pos, -cfg.halfsize[0] + cfg.agent_radius, cfg.halfsize[0] - cfg.agent_radius)

        herder_vel = jr.uniform(
            key=key_vel,
            shape=(cfg.n_herders, 2),
            minval=-jnp.array(cfg.vel_maxs) * 0.5,
            maxval=jnp.array(cfg.vel_maxs) * 0.5,
        )
        herder_state = jnp.concatenate([herder_pos, herder_vel], axis=-1)

        info = {
            "herd/circle_center": circle_center,
            "herd/radius": radius,
            "herder_radius": herder_radius,
            "hred_circle_angle": herd_circle_angle,
        }
        return HerdBaseState(herd_state=herd_pos, herder_state=herder_state, steps=0), info

    def get_predicates_float(self, state: HerdBaseState):
        return {
            "herd_herded": self.is_herd_herded_float(state),
        }

    def is_herd_herded_float(self, state: HerdBaseState):
        """All herd agents are fully within a circle in the center.
        1 when herded, -eps on boundary, scales linearly with soft maximum of furthest herd agent."""
        herd_pos = state.herd_state
        n_dists = jnp.linalg.norm(herd_pos, axis=-1)

        # In boundary if dists + agent_radius <= herded_radius
        n_dist_to_boundary = n_dists + self.cfg.agent_radius - self.cfg.herded_radius

        max_dist = jnp.max(n_dist_to_boundary)

        if self.cfg.n_herd == 1:
            # No need for softmax if only one herd agent.
            max_dist_soft = max_dist
        else:
            # error <= temperature * log(n). I want error <= 0.1 * halfwidth, so temperature = 0.1 * halfwidth / log(n)
            temperature = 0.1 * min(self.cfg.halfsize) / jnp.log(self.cfg.n_herd)
            max_dist_soft = softmaximum(n_dist_to_boundary, temperature=temperature)

        eps = 0.1

        # Linear scaling from -eps to -1. Reach -1 at distance = halfsize.
        t = max_dist_soft / min(self.cfg.halfsize)
        t = jnp.clip(t, 0.0, 1.0)
        outside_val = -(eps + (1.0 - eps) * t)

        val = jnp.where(max_dist <= 0, 1.0, outside_val)
        val = jnp.clip(val, -1.0, 1.0)
        return val


# def all_in_circle(pos, radius, circle_radius):
#     assert pos.shape == (3, 2)
#     dists = np.linalg.norm(pos, axis=-1)
#     assert dists.shape == (3,)
#     all_in_circle = np.all((dists + radius) <= circle_radius)
#     return np.where(all_in_circle, 1, -1)
