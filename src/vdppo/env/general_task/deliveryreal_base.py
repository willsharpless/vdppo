import functools as ft
from typing import Any

import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax_dataclasses as jdc
import matplotlib.pyplot as plt
import numpy as np
from attrs import define
from jaxtyping import PRNGKeyArray
from loguru import logger

from vdppo.common.geometry import RectCenterExtent, dist_pt_to_rect
from vdppo.common.jax_types import BoolScalar
from vdppo.common.jax_utils import softminimum, tree_stack
from vdppo.env.general_task.env import BaseEnv, Env, EnvStep
from vdppo.common.train_utils import tree_where

VEL_ZERO = False
# HERD_ZERO = True

# # If True, one of the circles is not control invariant.
# TEST_INVARIANT = False


class ShouldTermFn:
    def __call__(self, predicates: dict[str, BoolScalar]) -> BoolScalar: ...


@jdc.pytree_dataclass
class DeliveryRealBaseState:
    # (n_herd, 2) [px, py]
    herd_state: jnp.ndarray
    # (n_herd, 4) [px, py, vx, vy]
    herder_state: jnp.ndarray

    steps: int

    centers: jnp.ndarray


@define(slots=False)
class DeliveryRealBaseCfg:
    herd_vel: float = 0.2
    dt: float = 0.2

    n_herders: int = 3
    n_herd: int = 3
    acc_maxs: list[float] = [1.5, 1.5, 1.0]
    vel_maxs: list[float] = [0.8, 0.8, 0.4]

    agent_radius: float = 0.4
    # base_agent_radius: float = 0.75
    base_agent_radius: float = 1.32

    # Half size.
    halfsize: tuple[float, float] = (5.0, 5.0)

    herd_zero: bool = True
    """If True, pretend the herd agents don't exist."""

    trunc_steps: int = 100
    # eval_steps: int = 500
    eval_steps: int = 200

    herded_radius: float = 1.0  # Radius within which herd agents are considered herded.

    centers: list[list[float]] = [
        [-2.0, 0.0],
        [3.0, 1.0],
    ]
    radiuses: list[float] = [0.5] * len(centers)

    obstacle_centers: list[list[float]] = [
        [-2.8, 3.2],  # upper left
        [-1.0 - 0.5, -2.5 - 0.5],  # lower left
        [0.1 - 0.5, -0.3 - 0.2],  # lower middle
        [0.3 - 0.5, 0.1 - 0.4],  # lower middle addendum
        [2.75, 3.7],  # upper right
    ]
    obstacle_radiuses: list[float] = [0.4, 0.3, 0.25, 0.4, 0.4]
    # obstacle_radiuses: list[float] = [0.8, 0.8, 0.8, 0.8, 0.9]
    obstacle_lw_ratios: list[float] = [1.0, 5.0, 5.0, 0.9, 0.7]
    obstacle_shape_norm: float = float("inf")

    air_obstacle_centers: list[list[float]] = [[0.0, 0.0]]
    air_obstacle_radiuses: list[float] = [0.4]
    air_obstacle_lw_ratios: list[float] = [1.0]
    air_obstacles: bool = True

    base_agent: bool = True

    dynamic_targets: bool = True
    update_targets: bool = True

    p_reset_task: float = 0.2
    p_reset_heuristic: float = 0.5

    p_reset_atgoal: float = 0.01

    def obstacles_to_aabbs(self, which=jnp, add_radius: bool = False):
        centers = which.array(self.obstacle_centers)  # (n_obst, 2)
        radii = which.array(self.obstacle_radiuses)  # (n_obst,)
        ratios = which.array(self.obstacle_lw_ratios)  # (n_obst,)

        effective_radii = radii

        half_x = effective_radii * ratios
        half_y = effective_radii

        if add_radius:
            half_x = half_x + self.agent_radius
            half_y = half_y + self.agent_radius

        xmin = centers[:, 0] - half_x
        xmax = centers[:, 0] + half_x
        ymin = centers[:, 1] - half_y
        ymax = centers[:, 1] + half_y

        return which.stack([xmin, ymin, xmax, ymax], axis=-1)

    def air_obstacles_to_aabbs(self, which=jnp, add_radius: bool = False):
        centers = which.array(self.air_obstacle_centers)  # (n_obst, 2)
        radii = which.array(self.air_obstacle_radiuses)  # (n_obst,)
        ratios = which.array(self.air_obstacle_lw_ratios)  # (n_obst,)

        effective_radii = radii

        half_x = effective_radii * ratios
        half_y = effective_radii

        if add_radius:
            half_x = half_x + self.agent_radius
            half_y = half_y + self.agent_radius

        xmin = centers[:, 0] - half_x
        xmax = centers[:, 0] + half_x
        ymin = centers[:, 1] - half_y
        ymax = centers[:, 1] + half_y

        return which.stack([xmin, ymin, xmax, ymax], axis=-1)

    def sample_center_outside_obst(self, key: PRNGKeyArray):
        n_targets = len(self.centers)
        valid_centers = jnp.zeros((n_targets, 2))
        halfsize_x, halfsize_y = self.halfsize
        maxpos_per_ag = np.zeros((1, 2))
        maxpos_per_ag[:, 0] = halfsize_x - self.agent_radius
        maxpos_per_ag[:, 1] = halfsize_y - self.agent_radius

        def sample_valid_for_ag(key, c_ix):

            def sample_valid_position(key):
                pos_try = jr.uniform(key, shape=(1, 2), minval=-maxpos_per_ag, maxval=maxpos_per_ag)

                obst_centers = jnp.array(self.obstacle_centers)  # (n_obst, 2)
                obst_radii = jnp.array(self.obstacle_radiuses)  # (n_obst,)
                obstacle_lw_ratios = jnp.array(self.obstacle_lw_ratios)  # (n_obst,)

                # Box half-extents: (n_obst, 2)
                half_extents = jnp.stack(
                    [obst_radii * obstacle_lw_ratios, obst_radii], axis=-1  # half-width (x)  # half-height (y)
                )

                # Relative position: (n_obst, 2)
                rel_pos = pos_try - obst_centers

                # Box SDF
                q = jnp.abs(rel_pos) - half_extents  # (n_obst, 2)
                outside_dist = jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1)  # (n_obst,)
                inside_dist = jnp.minimum(jnp.max(q, axis=-1), 0.0)  # (n_obst,)
                sdf = outside_dist + inside_dist  # (n_obst,)

                ## AERIAL
                air_obst_centers = jnp.array(self.air_obstacle_centers)  # (n_obst, 2)
                air_obst_radii = jnp.array(self.air_obstacle_radiuses)  # (n_obst,)
                air_obst_lw_ratios = jnp.array(self.air_obstacle_lw_ratios)  # (n_obst,)

                # Box half-extents: (n_obst, 2)
                air_half_extents = jnp.stack(
                    [air_obst_radii * air_obst_lw_ratios, air_obst_radii], axis=-1  # half-width (x)  # half-height (y)
                )

                # Relative position: (n_obst, 2)
                air_rel_pos = pos_try - air_obst_centers

                # Box SDF
                q = jnp.abs(air_rel_pos) - air_half_extents  # (n_obst, 2)
                air_outside_dist = jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1)  # (n_obst,)
                air_inside_dist = jnp.minimum(jnp.max(q, axis=-1), 0.0)  # (n_obst,)
                air_sdf = air_outside_dist + air_inside_dist  # (n_obst,)

                # jnp conditional if air_obstacles
                sdf = jnp.where(self.air_obstacles, jnp.minimum(sdf, air_sdf), sdf)

                # Valid if SDF >= agent_radius for all obstacles
                is_valid = jnp.all(sdf >= self.agent_radius)
                return is_valid, pos_try

            def sample_until_valid(carry):
                key, is_valid, pos = carry
                key, key_new = jax.random.split(key)
                is_valid_new, pos_new = sample_valid_position(key_new)
                pos = jnp.where(is_valid, pos, pos_new)
                is_valid = is_valid | is_valid_new
                return (key, is_valid, pos)

            init_carry = (key, False, jnp.zeros((1, 2)))

            key, _, valid_center = jax.lax.while_loop(lambda carry: ~carry[1], sample_until_valid, init_carry)

            return valid_center

        valid_centers = jax.vmap(sample_valid_for_ag)(jr.split(key, n_targets), jnp.arange(n_targets))
        # assert valid_centers.shape == (2, 1, 2)
        valid_centers = valid_centers.squeeze(1)
        # assert valid_centers.shape == (2, 2)

        return valid_centers

    reset_targets_fn = sample_center_outside_obst
    update_targets_fn = sample_center_outside_obst  # random jump
    update_cond_fn = "agent_in_respective_target"


class DeliveryRealBase(BaseEnv):
    """
    DeliveryReal env -- made from herd env (eg. num agents = n_herders) to use same callbacks/plotting/utils. Agents move

    Also, in case "dummy" agents (herded) are desired (moving obstacles). Otherwise, just a multi-agent env designed for multi-reach-avoiding.

    Predicates include:
        - reaching targets (DeliveryReal locs)
        - avoiding obstacles (city)

    Additionally, one may instantiate a 'base' agent, which is slower agent which the other agents may need to revisit.

    In the discrete action setup, each agent is a double-integrator that can accelerate / decelerate in either axis. Herd agents are single-integrators with built in policies.
    action: (n_herders, 2): int, {0, 1, 2} for each axis, where 0 = -accel, 1 = no accel, 2 = accel
    """

    Cfg = DeliveryRealBaseCfg
    State = DeliveryRealBaseState

    def __init__(self, cfg: DeliveryRealBaseCfg = DeliveryRealBaseCfg(), should_term_fn: ShouldTermFn = None):
        super().__init__()

        self.cfg = cfg
        assert len(cfg.acc_maxs) == len(cfg.vel_maxs) == cfg.n_herders

        self.cfg.update_cond_fn = getattr(self, self.cfg.update_cond_fn)

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
    def action_dim(self) -> int:
        return 2

    @property
    def control_lim_lo(self) -> list[list[float]]:
        return [[-self.cfg.acc_maxs[i]] * self.action_dim for i in range(self.cfg.n_herders)]

    @property
    def control_lim_hi(self) -> list[list[float]]:
        return [[self.cfg.acc_maxs[i]] * self.action_dim for i in range(self.cfg.n_herders)]

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

    def any_agent_in_target(self, state):
        def check_target_ix(center_ix):
            return self.is_herder_in_dyn_target(state, which=jnp, center_ix=center_ix)

        return jax.vmap(check_target_ix)(jnp.arange(len(self.cfg.centers)))

    def agent_in_respective_target(self, state):
        def check_target_ix(center_ix):
            return self.is_herderX_circs(state, herder_ix=center_ix, center_ix=center_ix)

        return jax.vmap(check_target_ix)(jnp.arange(len(self.cfg.centers)))

    def next_state(self, state: DeliveryRealBaseState, control: jnp.ndarray):
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
            herder_acc = kp_vel[:, None] * (herder_vel_cmd - herder_vel)
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

        # Update targets
        if self.cfg.dynamic_targets and self.cfg.update_targets:
            centers_new = self.cfg.update_targets_fn(jr.PRNGKey(state.steps))
            update_cond = self.cfg.update_cond_fn(state)
            centers = jnp.where(update_cond[:, None], centers_new, state.centers)
        else:
            centers = state.centers

        return DeliveryRealBaseState(
            herd_state=herd_state_new, herder_state=herder_state_new, steps=state.steps + 1, centers=centers
        )

    ## BOOL PREDICATES (SPARSE)

    def _has_collide(self, n_pos: jnp.ndarray, radius: float | jnp.ndarray):
        n = len(n_pos)

        def check_pair(i: int, j: int):
            dist = jnp.linalg.norm(n_pos[i] - n_pos[j])
            collide = dist < 2 * radius
            return collide

        collide = False
        for i in range(n):
            for j in range(i + 1, n):
                collide = collide | check_pair(i, j)

        return collide

    def is_just_herders_collide(self, state: DeliveryRealBaseState):
        herder_pos = state.herder_state[:-1, 0:2]
        return self._has_collide(herder_pos, self.cfg.agent_radius)

    def is_herder_oob(self, state: DeliveryRealBaseState):
        herder_pos = state.herder_state[:, 0:2]
        dists = self.dist_to_wall(herder_pos)
        min_dists = jnp.min(dists, axis=-1)
        oob = jnp.any(min_dists < self.cfg.agent_radius)
        return oob

    def is_herder_in_obstacles(self, state: DeliveryRealBaseState, which=jnp):
        herder_pos = state.herder_state[..., :, 0:2]  # (n_herders, 2)

        obst_centers = which.array(self.cfg.obstacle_centers)  # (n_obst, 2)
        obst_radii = which.array(self.cfg.obstacle_radiuses)  # (n_obst,)
        obstacle_lw_ratios = which.array(self.cfg.obstacle_lw_ratios)  # (n_obst,)

        agent_radii = which.where(
            self.cfg.base_agent * (which.arange(self.cfg.n_herders) == self.cfg.n_herders - 1),
            self.cfg.base_agent_radius,
            self.cfg.agent_radius,
        )  # (n_herders,)

        # Box half-extents: (n_obst, 2)
        half_extents = which.stack(
            [obst_radii * obstacle_lw_ratios, obst_radii], axis=-1  # half-width (x)  # half-height (y)
        )

        # Relative position: (n_obst, n_herders, 2)
        rel_pos = herder_pos[None, :, :] - obst_centers[:, None, :]

        # Box SDF: distance to nearest point on box surface
        # q = |rel_pos| - half_extents
        q = which.abs(rel_pos) - half_extents[:, None, :]  # (n_obst, n_herders, 2)

        # SDF = length(max(q, 0)) + min(max(q.x, q.y), 0)
        outside_dist = which.linalg.norm(which.maximum(q, 0.0), axis=-1)  # (n_obst, n_herders)
        inside_dist = which.minimum(which.max(q, axis=-1), 0.0)  # (n_obst, n_herders)
        sdf = outside_dist + inside_dist  # (n_obst, n_herders)

        # Collision when SDF < agent_radius
        c_is_herder_inside = which.any(sdf < agent_radii[None, :])

        return c_is_herder_inside

    def is_herder_in_air_obstacles(self, state: DeliveryRealBaseState, which=jnp):
        n_move = self.cfg.n_herders - 1
        n_pos_move = state.herder_state[:n_move, :2]
        assert n_pos_move.shape == (n_move, 2)

        n_in_aerial_obs = jax.vmap(ft.partial(self._in_aerial_obs, radius=self.cfg.agent_radius))(n_pos_move)
        return which.any(n_in_aerial_obs)

    def is_herder_in_target(self, state: DeliveryRealBaseState, which=jnp, center=[0.0, 0.0], radius=0.5):
        h_pos = state.herder_state[..., :, 0:2]
        ch_dists = which.linalg.norm(h_pos - which.array(center), axis=-1)
        c_is_herder_inside = which.any(ch_dists < (which.array(radius)))
        return c_is_herder_inside

    def is_herder_in_dyn_target(self, state: DeliveryRealBaseState, which=jnp, center_ix=0, radius=0.5):
        h_pos = state.herder_state[..., :, 0:2]
        ch_dists = which.linalg.norm(h_pos - state.centers[center_ix], axis=-1)
        c_is_herder_inside = which.any(ch_dists < (which.array(radius) - self.cfg.agent_radius))
        return c_is_herder_inside

    def is_herder_circs(self, state: DeliveryRealBaseState, which=jnp):
        h_pos = state.herder_state[..., :, 0:2]
        c_pos = which.array(self.cfg.centers)
        # (n_circs, n_herders, 2) -> (n_circs, n_herders)
        ch_dists = which.linalg.norm(h_pos[..., None, :, :] - c_pos[..., :, None, :], axis=-1)
        # (n_circs, )
        c_dists = which.min(ch_dists, axis=-1)
        c_is_herder_inside = c_dists < (which.array(self.radiuses) - self.cfg.agent_radius)
        return c_is_herder_inside

    def is_herder_at_base_ag(self, state: DeliveryRealBaseState, which=jnp):
        h_pos = state.herder_state[..., :, 0:2]
        base_ag_pos = state.herder_state[-1, 0:2]
        ch_dists = which.linalg.norm(h_pos - base_ag_pos, axis=-1)
        c_all_herder_inside = which.all(ch_dists < self.cfg.agent_radius)
        return c_all_herder_inside

    def is_herderX_at_base_ag(self, state: DeliveryRealBaseState, herder_ix: int, which=jnp):
        h_pos = state.herder_state[..., herder_ix, 0:2]
        base_ag_pos = state.herder_state[-1, 0:2]
        ch_dists = which.linalg.norm(h_pos - base_ag_pos, axis=-1)
        c_all_herder_inside = which.all(ch_dists < self.cfg.agent_radius)
        return c_all_herder_inside

    def is_herderX_circs(self, state: DeliveryRealBaseState, herder_ix: int, center_ix: int, which=jnp, radius=0.5):
        # assert self.cfg.dynamic_targets == True
        h_pos = state.herder_state[..., herder_ix, 0:2]
        ch_dists = which.linalg.norm(h_pos - state.centers[center_ix], axis=-1)
        c_is_herder_inside = which.any(ch_dists < (which.array(radius) - self.cfg.agent_radius))
        return c_is_herder_inside

    def get_predicates_bool(self, state: DeliveryRealBaseState):
        predicates = {
            "aerial_collide": self.is_just_herders_collide(state),
            "oob": self.is_herder_oob(state),
            "obstacles": self.is_herder_in_obstacles(state),
            "target0": self.is_herder_in_target(state, center=self.cfg.centers[0], radius=self.cfg.radiuses[0]),
            "target1": self.is_herder_in_target(state, center=self.cfg.centers[1], radius=self.cfg.radiuses[1]),
            # "target2": self.is_herder_in_target(state, center=self.cfg.centers[2], radius=self.cfg.radiuses[2]),
            # "target3": self.is_herder_in_target(state, center=self.cfg.centers[3], radius=self.cfg.radiuses[3]),
            # "target4": self.is_herder_in_target(state, center=self.cfg.centers[4], radius=self.cfg.radiuses[4]),
            "ags_to_base_agent": self.is_herder_at_base_ag(state),
            "ag0_target0": self.is_herderX_circs(state, herder_ix=0, center_ix=0),
            "ag1_target0": self.is_herderX_circs(state, herder_ix=1, center_ix=0),
            "ag0_target1": self.is_herderX_circs(state, herder_ix=0, center_ix=1),
            "ag1_target1": self.is_herderX_circs(state, herder_ix=1, center_ix=1),
            "ag0_base": self.is_herderX_at_base_ag(state, herder_ix=0),
            "ag1_base": self.is_herderX_at_base_ag(state, herder_ix=1),
            "air_obstacles": self.is_herder_in_air_obstacles(state),
        }
        if self.cfg.dynamic_targets:
            predicates["target0"] = self.is_herder_in_dyn_target(state, center_ix=0)
            predicates["target1"] = self.is_herder_in_dyn_target(state, center_ix=1)
        return predicates

    ## FLOAT PREDICATES (DENSE)

    def pred_herder_circs(self, state: DeliveryRealBaseState, which=jnp):
        """
        Inside the circle is +1.
        Outside the circle is negative.
        - Linearly scale from -1 when distance=edge to -eps when distance=0
        """

        h_pos = state.herder_state[..., :, 0:2]
        c_pos = which.array(self.cfg.centers)
        # (n_circs, n_herders, 2) -> (n_circs, n_herders)
        ch_dists = which.linalg.norm(h_pos[..., None, :, :] - c_pos[..., :, None, :], axis=-1)
        # (n_circs, )
        c_dists = which.min(ch_dists, axis=-1)
        c_radiuses = which.array(self.cfg.radiuses)
        c_dist_to_circ = c_dists - c_radiuses + self.cfg.agent_radius
        eps = 0.1

        val_at_edge = -1.0
        edge = 2 * which.array(self.cfg.halfsize).max() - c_radiuses.max()
        coef = (val_at_edge + eps) / edge
        pred = jnp.where(c_dist_to_circ <= 0, 1.0, -eps + coef * c_dist_to_circ)
        pred = jnp.clip(pred, -1.0, 1.0)
        return pred

    def pred_herder_circs_dyn(self, state: DeliveryRealBaseState, which=jnp, center_ix=0, radius=0.5):
        """
        Inside the circle is +1.
        Outside the circle is negative.
        - Linearly scale from -1 when distance=edge to -eps when distance=0
        """
        h_pos = state.herder_state[..., :, 0:2]
        ch_dists = which.linalg.norm(h_pos - state.centers[center_ix], axis=-1).min(axis=-1)
        c_dist_to_circ = ch_dists - (which.array(radius) - self.cfg.agent_radius)
        eps = 0.1
        val_at_edge = -1.0
        edge = 2 * which.array(self.cfg.halfsize).max() - which.array(radius)
        coef = (val_at_edge + eps) / edge
        pred = jnp.where(c_dist_to_circ <= 0, 1.0, -eps + coef * c_dist_to_circ)
        pred = jnp.clip(pred, -1.0, 1.0)
        return pred

    def get_predicates_float(self, state: DeliveryRealBaseState):
        pred_herder_circs = self.pred_herder_circs(state)
        predicates = {
            "target0_dense": pred_herder_circs[0],
            "target1_dense": pred_herder_circs[1],
            "target2_dense": pred_herder_circs[2],
            "target3_dense": pred_herder_circs[3],
            "target4_dense": pred_herder_circs[4],
        }
        if self.cfg.dynamic_targets:
            predicates["target0_dense"] = self.pred_herder_circs_dyn(state, center_ix=0)
            predicates["target1_dense"] = self.pred_herder_circs_dyn(state, center_ix=1)
        return predicates

    def get_predicates(self, state: DeliveryRealBaseState):
        predicates_bool = self.get_predicates_bool(state)
        predicates = {k: jnp.where(v, 1.0, -1.0) for k, v in predicates_bool.items()}

        predicates_float = self.get_predicates_float(state)
        predicates = predicates | predicates_float

        return predicates

    def step(self, state: DeliveryRealBaseState, action: jnp.ndarray):
        controls = self._action_to_controls(action)
        state_new = self.next_state(state, controls)
        obs_new = self.get_obs(state_new)

        predicates = self.get_predicates(state_new)
        term = self.should_term_fn(predicates)
        trunc = state_new.steps >= self.cfg.trunc_steps

        info = {"age": state_new.steps}
        return EnvStep(state_new, obs_new, predicates, term, trunc, info)

    def step_control(self, state: DeliveryRealBaseState, controls: jnp.ndarray):
        state_new = self.next_state(state, controls)
        obs_new = self.get_obs(state_new)

        predicates = self.get_predicates(state_new)
        term = self.should_term_fn(predicates)
        trunc = state_new.steps >= self.cfg.trunc_steps

        info = {"age": state_new.steps}
        return EnvStep(state_new, obs_new, predicates, term, trunc, info)

    def get_objects_pos(self, state: DeliveryRealBaseState):
        assert self.cfg.herd_zero
        assert self.cfg.dynamic_targets

        herder_pos = state.herder_state[:, 0:2]  # (n_herders, 2)

        target_pos = state.centers  # (n_targets, 2)

        air_obs_pos = np.array(self.cfg.air_obstacle_centers)
        assert air_obs_pos.shape == (1, 2)

        obstacle_pos = np.array(self.cfg.obstacle_centers)
        assert obstacle_pos.shape == (4, 2)

        all_pos = jnp.concatenate([herder_pos, target_pos, air_obs_pos, obstacle_pos], axis=0)
        return all_pos

    def get_obs_and_names(self, state: DeliveryRealBaseState):
        def fl(lst: list[list[str]]) -> list[str]:
            return [item for sublist in lst for item in sublist]

        # ---------------------------------------------------------------------
        # 1: State
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

        # # ---------------------------------------------------------------------
        # # 3: Dynamic target centers (if enabled)
        # assert self.cfg.dynamic_targets
        # if self.cfg.dynamic_targets:
        #     obs_centers = state.centers / jnp.array(self.cfg.halfsize)  # (n_targets, 2)
        #     obs_centers_names = [[f"target{ii}_cx", f"target{ii}_cy"] for ii in range(len(state.centers))]

        # ---------------------------------------------------------------------
        assert self.cfg.herd_zero
        if self.cfg.dynamic_targets:
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

        return obs, obs_names

    def get_obs(self, state: DeliveryRealBaseState):
        obs, _ = self.get_obs_and_names(state)
        return obs

    def sample_pos_outside_obst(
        self, key: PRNGKeyArray, herd_pos: jnp.ndarray, maxpos_per_ag: jnp.ndarray, minpos_per_ag: jnp.ndarray
    ):
        n_herd = self.cfg.n_herd
        n_herders = self.cfg.n_herders
        valid_pos = jnp.zeros((n_herders, 2))

        centers = jnp.array(self.cfg.centers)

        def sample_valid_for_ag(key, ag_ix):

            def sample_valid_position(key):
                pos_try = jr.uniform(key, shape=(1, 2), minval=minpos_per_ag, maxval=maxpos_per_ag)
                pos_try_full = jnp.zeros((n_herders, 2)).at[:, :2].set(pos_try)
                valid_vel = jnp.zeros_like(pos_try_full)
                herder_state_try = jnp.concatenate([pos_try_full, valid_vel], axis=-1)
                state_try = DeliveryRealBaseState(
                    herd_state=herd_pos, herder_state=herder_state_try, steps=0, centers=centers
                )
                is_not_valid = self.is_herder_in_obstacles(state_try)
                return ~is_not_valid, pos_try

            def sample_until_valid(carry):
                key, is_valid, pos = carry
                key, key_new = jax.random.split(key)
                is_valid_new, pos_new = sample_valid_position(key_new)
                pos = jnp.where(is_valid, pos, pos_new)
                is_valid = is_valid | is_valid_new
                return (key, is_valid, pos)

            init_carry = (key, False, jnp.zeros((1, 2)))

            key, _, valid_pos_ag = jax.lax.while_loop(lambda carry: ~carry[1], sample_until_valid, init_carry)

            return valid_pos_ag

        keys = jr.split(key, n_herders)
        agent_indices = jnp.arange(n_herders)
        valid_pos = jax.vmap(sample_valid_for_ag)(keys, agent_indices)

        return valid_pos.squeeze(1)

    # # standard reset old
    # def reset_old(self, key: PRNGKeyArray):
    #     n_herd = self.cfg.n_herd
    #     n_herders = self.cfg.n_herders
    #     key_herd, key_herders = jr.split(key)

    #     # Uniformly sample herd positions.
    #     halfsize_x, halfsize_y = self.cfg.halfsize
    #     maxpos = np.array([halfsize_x, halfsize_y]) - self.cfg.agent_radius
    #     if self.cfg.herd_zero:
    #         maxpos = np.zeros(2)
    #     minpos = -maxpos
    #     herd_pos = jr.uniform(key_herd, shape=(n_herd, 2), minval=minpos, maxval=maxpos)

    #     # Uniformly sample herder positions and velocities.
    #     # (n_herders, 4)
    #     maxpos_per_ag = np.zeros((1, 2))
    #     maxpos_per_ag[:, 0] = halfsize_x - self.cfg.agent_radius
    #     maxpos_per_ag[:, 1] = halfsize_y - self.cfg.agent_radius
    #     maxvel = np.zeros((n_herders, 2))
    #     maxvel[:, 0] = np.array(self.cfg.vel_maxs)
    #     maxvel[:, 1] = np.array(self.cfg.vel_maxs)

    #     if VEL_ZERO:
    #         maxvel[:, 0] = 0.0
    #         maxvel[:, 1] = 0.0

    #     herder_pos_valid = self.sample_pos_outside_obst(key_herders, herd_pos, maxpos_per_ag=maxpos_per_ag, minpos_per_ag=-maxpos_per_ag)
    #     herder_vel = jr.uniform(key_herders, shape=(n_herders, 2), minval=-maxvel, maxval=maxvel)
    #     herder_state = jnp.concatenate([herder_pos_valid, herder_vel], axis=-1)

    #     # Sample dynamic target positions.
    #     if self.cfg.dynamic_targets:
    #         centers = self.cfg.reset_targets_fn(key)
    #     else:
    #         centers = jnp.array(self.cfg.centers)

    #     return DeliveryRealBaseState(herd_state=herd_pos, herder_state=herder_state, steps=0, centers=centers)

    def _reset_single(self, key: PRNGKeyArray):
        key_uniform, key_task, key_goal, key_which, key_center, key_atgoal, key_which_atgoal = jr.split(key, 7)

        herd_pos = jnp.zeros((self.cfg.n_herd, 2))

        assert self.cfg.dynamic_targets
        centers = self.cfg.reset_targets_fn(key_center)

        # ----
        p_reset_task = self.cfg.p_reset_task
        p_reset_heuristic = self.cfg.p_reset_heuristic
        p_reset_orig = 1.0 - p_reset_task - p_reset_heuristic
        assert p_reset_orig >= 0.0

        herder_state_uniform = self._reset_uniform(key_uniform)
        herder_state_task = self._reset_task(key_task)
        herder_state_heuristic = self._reset_heuristic(key_task, centers)

        probs = np.array([p_reset_orig, p_reset_task, p_reset_heuristic])
        assert np.isclose(probs.sum(), 1.0)
        which_reset = jr.categorical(key_which, probs)

        stack_list = [herder_state_uniform, herder_state_task, herder_state_heuristic]
        assert len(probs) == len(stack_list)
        herder_state_stack = tree_stack(stack_list)
        herder_state = jtu.tree_map(lambda x: x[which_reset], herder_state_stack)
        # ----

        # Small probability of reset the agent near the goal.
        p_reset_atgoal = self.cfg.p_reset_atgoal
        which_atgoal = jr.bernoulli(key_which_atgoal, p_reset_atgoal, shape=(2,))
        pos_atgoal = self._rand_pos_atgoal(key_atgoal, centers)
        assert pos_atgoal.shape == (2, 2)

        pos = jnp.where(which_atgoal[:, None], pos_atgoal, herder_state[:2, :2])
        assert pos.shape == (2, 2)
        herder_state = herder_state.at[:2, :2].set(pos)

        return DeliveryRealBaseState(herd_state=herd_pos, herder_state=herder_state, steps=0, centers=centers)

    def is_invalid_real_eval_state(self, state: DeliveryRealBaseState):
        predicates = self.get_predicates(state)
        predicates = {k: v > 0 for k, v in predicates.items()}
        pred_names = ["aerial_collide", "obstacles", "oob", "air_obstacles"]
        is_unsafe = jnp.any(jnp.array([predicates[name] for name in pred_names]))
        return is_unsafe

    def is_valid_real_eval_state(self, state: DeliveryRealBaseState):
        return ~self.is_invalid_real_eval_state(state)

    def reset(self, key: PRNGKeyArray):
        """Small effort rejection sampling to avoid obstacles."""
        n_samples = 4
        b_key = jr.split(key, n_samples)
        b_state = jax.vmap(self._reset_single)(b_key)
        b_valid = jax.vmap(self.is_valid_real_eval_state)(b_state)
        valid_idx = jnp.argmax(b_valid)

        state = jtu.tree_map(lambda x: x[valid_idx], b_state)
        return state

    def reset_batch(self, key: PRNGKeyArray, batch_size: int, sample_multiplier: int = 4) -> Any:
        """Rejection sampling, but we can be more efficient in batch."""
        n_samples = batch_size * sample_multiplier

        b_key = jr.split(key, n_samples)
        b_state = jax.vmap(self._reset_single)(b_key)
        b_invalid = jax.vmap(self.is_invalid_real_eval_state)(b_state)

        # argsort to get valid samples first.
        sorted_indices = jnp.argsort(b_invalid, axis=0)

        # Get the first batch_size valid samples.
        selected_indices = sorted_indices[:batch_size]

        b_state_valid = jtu.tree_map(lambda x: x[selected_indices], b_state)
        return b_state_valid

    def reset_batch_eval(self, key: PRNGKeyArray, batch_size: int) -> Any:
        # Try much harder to get valid samples during eval.
        logger.debug("reset_batch_eval called")
        return self.reset_batch(key, batch_size, sample_multiplier=32)

    def _rand_pos_atgoal(self, key: PRNGKeyArray, n_centers: jnp.ndarray):
        n_herder_move = 2
        assert n_centers.shape == (2, 2)
        # Goal radius is 0.5 (is_herderX_circs).
        goal_radius = 0.5
        # At goal if dist < radius - agent_radius.
        radius_inner = goal_radius - self.cfg.agent_radius

        # Randomly sample a position within the inner circle, parametrize by angle and radius.
        key_angle, key_radius = jr.split(key)
        n_angle = jr.uniform(key_angle, shape=(2,), minval=0.0, maxval=2 * jnp.pi)
        n_radius = jr.uniform(key_radius, shape=(2,), minval=0.0, maxval=radius_inner + 2 * self.cfg.agent_radius)
        n_pos_offset = n_radius[:, None] * jnp.stack([jnp.cos(n_angle), jnp.sin(n_angle)], axis=-1)
        assert n_pos_offset.shape == (n_herder_move, 2)

        n_herder_pos = n_centers + n_pos_offset
        assert n_herder_pos.shape == (n_herder_move, 2)

        # Clip to within the environment bounds.
        halfsize_x, halfsize_y = self.cfg.halfsize
        minpos = jnp.array([-0.99 * halfsize_x, -0.99 * halfsize_y])
        maxpos = jnp.array([0.99 * halfsize_x, 0.99 * halfsize_y])
        n_herder_pos = jnp.clip(n_herder_pos, minpos, maxpos)

        return n_herder_pos

    def _reset_uniform(self, key: PRNGKeyArray):
        n_herders = self.cfg.n_herders
        halfsize_x, halfsize_y = self.cfg.halfsize

        key_pos, key_vel = jr.split(key)

        minpos = np.array([-halfsize_x, -halfsize_y])
        maxpos = np.array([halfsize_x, halfsize_y])
        n_herder_pos = jr.uniform(key_pos, shape=(n_herders, 2), minval=minpos, maxval=maxpos)

        vel_max = np.array(self.cfg.vel_maxs)[:, None]
        n_herder_vel = jr.uniform(key_vel, shape=(n_herders, 2), minval=-vel_max, maxval=vel_max)

        n_herder_state = jnp.concatenate([n_herder_pos, n_herder_vel], axis=-1)
        assert n_herder_state.shape == (n_herders, 4)

        return n_herder_state

    def _reset_task(self, key: PRNGKeyArray):
        n_herders = self.cfg.n_herders
        halfsize_x, halfsize_y = self.cfg.halfsize

        key1, key2, key_vel = jr.split(key, 3)

        # Agents: Uniform.
        minpos = np.array([-0.95 * halfsize_x, -0.95 * halfsize_y])
        maxpos = np.array([0.95 * halfsize_x, 0.95 * halfsize_y])
        n_herder_pos = jr.uniform(key1, shape=(n_herders - 1, 2), minval=minpos, maxval=maxpos)

        # Base agent: near origin but within [-0.9, 0.9] box
        maxstate_base = np.array([[0.9 - self.cfg.base_agent_radius, 0.9 - self.cfg.base_agent_radius]])
        base_state = jr.uniform(key2, shape=(1, 2), minval=-maxstate_base, maxval=maxstate_base)

        n_herder_pos = jnp.concatenate([n_herder_pos, base_state], axis=0)

        vel_max = np.array(self.cfg.vel_maxs)[:, None]
        n_herder_vel = jr.uniform(key_vel, shape=(n_herders, 2), minval=-vel_max, maxval=vel_max)

        n_herder_state = jnp.concatenate([n_herder_pos, n_herder_vel], axis=-1)
        assert n_herder_state.shape == (n_herders, 4)

        return n_herder_state

    def _in_obs(self, pos: jnp.ndarray, radius: jnp.ndarray, which=jnp):
        obst_centers = which.array(self.cfg.obstacle_centers)  # (n_obst, 2)
        obst_radii = which.array(self.cfg.obstacle_radiuses)  # (n_obst,)
        obst_lw_ratios = which.array(self.cfg.obstacle_lw_ratios)  # (n_obst,)
        n_obst = len(obst_centers)

        half_extents = which.stack([obst_radii * obst_lw_ratios, obst_radii], axis=-1)
        o_rects = RectCenterExtent(center=obst_centers, extent=half_extents)  # (n_obst, ...)

        o_dist_obst = jax.vmap(ft.partial(dist_pt_to_rect, pos))(o_rects)
        assert o_dist_obst.shape == (n_obst,)

        return which.any(o_dist_obst < radius)

    def _in_aerial_obs(self, pos: jnp.ndarray, radius: jnp.ndarray, which=jnp):
        air_obst_centers = which.array(self.cfg.air_obstacle_centers)
        air_obst_radii = which.array(self.cfg.air_obstacle_radiuses)
        air_obst_lw_ratios = which.array(self.cfg.air_obstacle_lw_ratios)  # (n_obst,)
        n_obst = len(air_obst_centers)

        half_extents = which.stack([air_obst_radii * air_obst_lw_ratios, air_obst_radii], axis=-1)
        o_rects = RectCenterExtent(center=air_obst_centers, extent=half_extents)

        o_dist_obst = jax.vmap(ft.partial(dist_pt_to_rect, pos))(o_rects)
        assert o_dist_obst.shape == (n_obst,)

        return which.any(o_dist_obst < radius)

    def _is_oob(self, pos: jnp.ndarray, which=jnp):
        dists = self.dist_to_wall(pos)
        min_dists = jnp.min(dists, axis=-1)
        return jnp.any(min_dists < self.cfg.agent_radius)

    def _is_valid_base(self, pos_base: jnp.ndarray, which=jnp):
        # Shouldn't collide with obstacles, shouldn't be out of bounds.
        radius_base = self.cfg.base_agent_radius
        base_in_obs = self._in_obs(pos_base, radius_base, which=which)
        # -------------
        base_oob = self._is_oob(pos_base, which=which)
        # -------------
        base_unsafe = base_in_obs | base_oob
        return ~base_unsafe

    def _is_valid_move(self, n_pos_move: jnp.ndarray):
        n_move = self.cfg.n_herders - 1
        assert n_pos_move.shape == (n_move, 2)

        # Shouldn't collide with obstacles,
        n_in_obs = jax.vmap(ft.partial(self._in_obs, radius=self.cfg.agent_radius))(n_pos_move)
        in_obs = jnp.any(n_in_obs)

        # Aerial obstacles.
        n_in_aerial_obs = jax.vmap(ft.partial(self._in_aerial_obs, radius=self.cfg.agent_radius))(n_pos_move)
        in_aerial_obs = jnp.any(n_in_aerial_obs)

        # Aerial collide
        aerial_collide = self._has_collide(n_pos_move, self.cfg.agent_radius)

        # Not out of bounds.
        n_oob = jax.vmap(self._is_oob)(n_pos_move)
        oob = jnp.any(n_oob)

        is_unsafe = in_obs | in_aerial_obs | aerial_collide | oob
        return ~is_unsafe

    def _reset_heuristic(self, key: PRNGKeyArray, n_centers: jnp.ndarray):
        """
        1. Rejection sample a position for the base that minimizes the distance to the closest target and is valid.
        2. Rejection sample positions for the other herders that is close to their target and is valid.
        """
        n_herders = self.cfg.n_herders
        n_move = n_herders - 1

        assert n_centers.shape == (n_move, 2)

        agent_radius = self.cfg.agent_radius

        key_base, key_herders, key_vel = jr.split(key, 3)

        # 1. Sample for the base.
        n_samples_base = 16
        b_key_base = jr.split(key_base, n_samples_base)

        # To make the sampling more focused, consider the midpoint between the two targets.
        # Sample within a circle with radius equal to half the distance between the two targets.
        sample_circ_center = 0.5 * (n_centers[0] + n_centers[1])
        sample_circ_radius = 0.5 * jnp.linalg.norm(n_centers[0] - n_centers[1])

        def sample_base_pos(key_):
            key_0, key_1 = jr.split(key_)
            angle_ = jr.uniform(key_0, minval=0.0, maxval=2 * jnp.pi)
            radius_ = jr.uniform(key_1, minval=0.0, maxval=sample_circ_radius)
            pos_offset_ = radius_ * jnp.array([jnp.cos(angle_), jnp.sin(angle_)])
            pos_ = sample_circ_center + pos_offset_
            return pos_

        b_pos_base = jax.vmap(sample_base_pos)(b_key_base)
        b_base_valid = jax.vmap(self._is_valid_base)(b_pos_base)

        # Of the valid ones, find the one that is closest to any target.
        bn_dist = jnp.linalg.norm(b_pos_base[:, None] - n_centers[None, :], axis=-1)
        assert bn_dist.shape == (n_samples_base, n_move)

        b_dist_closest = jnp.min(bn_dist, axis=1)
        assert b_dist_closest.shape == (n_samples_base,)

        b_cost = jnp.where(b_base_valid, b_dist_closest, 1e6)
        idx = jnp.argmin(b_cost)
        pos_base = b_pos_base[idx, :]
        assert pos_base.shape == (2,)

        # 2. Sample for the other herders. Sample around their targets, try to ensure validity.
        n_samples_herders = 32
        b_key_herders = jr.split(key_herders, n_samples_herders)

        def sample_herder_pos(key_):
            n_angle_ = jr.uniform(key_, minval=0.0, maxval=2 * jnp.pi, shape=n_move)
            n_radius_ = jr.uniform(key_, minval=0.0, maxval=5 * agent_radius, shape=n_move)
            n_pos_offset_ = n_radius_[:, None] * jnp.stack([jnp.cos(n_angle_), jnp.sin(n_angle_)], axis=-1)
            assert n_pos_offset_.shape == (n_move, 2)

            n_pos_ = n_centers + n_pos_offset_
            assert n_pos_.shape == (n_move, 2)

            # Clip to within the environment bounds.
            minpos = -0.99 * np.array(self.cfg.halfsize)
            maxpos = 0.99 * np.array(self.cfg.halfsize)
            n_pos_ = jnp.clip(n_pos_, minpos, maxpos)

            return n_pos_

        bn_move_pos = jax.vmap(sample_herder_pos)(b_key_herders)
        assert bn_move_pos.shape == (n_samples_herders, n_move, 2)

        b_move_valid = jax.vmap(self._is_valid_move)(bn_move_pos)
        idx = jnp.argmax(b_move_valid)
        n_move_pos = bn_move_pos[idx, :, :]
        assert n_move_pos.shape == (n_move, 2)

        n_herder_pos = jnp.concatenate([n_move_pos, pos_base[None, :]], axis=0)

        vel_max = np.array(self.cfg.vel_maxs)[:, None]
        n_herder_vel = jr.uniform(key_vel, shape=(n_herders, 2), minval=-vel_max, maxval=vel_max)

        n_herder_state = jnp.concatenate([n_herder_pos, n_herder_vel], axis=-1)
        assert n_herder_state.shape == (n_herders, 4)

        return n_herder_state

    @property
    def eval_T(self) -> int:
        # return self.cfg.trunc_steps
        return self.cfg.base.eval_steps

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

        aabbs = cfg.obstacles_to_aabbs(which=np, add_radius=False)
        for aabb in aabbs:
            xmin, ymin, xmax, ymax = aabb
            width = xmax - xmin
            height = ymax - ymin
            rect = plt.Rectangle((xmin, ymin), width, height, facecolor="black", edgecolor="none", alpha=0.3)
            ax.add_patch(rect)

        air_aabbs = cfg.air_obstacles_to_aabbs(which=np, add_radius=False)
        for aabb in air_aabbs:
            xmin, ymin, xmax, ymax = aabb
            width = xmax - xmin
            height = ymax - ymin
            rect = plt.Rectangle((xmin, ymin), width, height, facecolor="yellow", edgecolor="none", alpha=0.2)
            ax.add_patch(rect)


# class DeliveryRealBasePlay(DeliveryRealBase):
#     """For testing."""
#
#     Cfg = DeliveryRealBasePlayCfg
#
#     def __init__(self, cfg: DeliveryRealBasePlayCfg, should_term_fn: ShouldTermFn = None):
#         super().__init__(cfg, should_term_fn=should_term_fn)
#         self.cfg = cfg
#
#         self.centers = np.array([[-3.0, -3.0], [3.0, 3.0]])
#         self.radiuses = np.array([1.0, 1.0])
#
#         self.centers_perturb = self.centers[0:1]
#         self.radiuses_perturb = self.radiuses[0:1] + 3 * cfg.agent_radius
