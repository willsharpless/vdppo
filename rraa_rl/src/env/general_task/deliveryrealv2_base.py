import functools as ft
from typing import Any
import ipdb

import jax
import jax.numpy as jnp
import jax.random as jr
import jax_dataclasses as jdc
import matplotlib.pyplot as plt
import numpy as np
from attrs import define
from jaxtyping import PRNGKeyArray

from rraa_rl.jax_types import BoolScalar
from rraa_rl.jax_utils import softminimum, tree_stack
from rraa_rl.train_utils import tree_where
import jax.tree_util as jtu
from rraa_rl.src.env.general_task.env import BaseEnv, Env, EnvStep

VEL_ZERO = False
# HERD_ZERO = True

# # If True, one of the circles is not control invariant.
# TEST_INVARIANT = False


class ShouldTermFn:
    def __call__(self, predicates: dict[str, BoolScalar]) -> BoolScalar: ...


@jdc.pytree_dataclass
class DeliveryRealv2BaseState:
    # (n_herd, 2) [px, py]
    herd_state: jnp.ndarray
    # (n_herd, 4) [px, py, vx, vy]
    herder_state: jnp.ndarray

    steps: int

    centers: jnp.ndarray

@define(slots=False)
class DeliveryRealv2BaseCfg:
    herd_vel: float = 0.2
    dt: float = 0.2

    n_herders: int = 3
    n_herd: int = 3
    acc_maxs: list[float] = [2.0, 2.0, 1.0]
    vel_maxs: list[float] = [1.0, 1.0, 0.1]

    agent_radius: float = 0.4
    base_agent_radius: float = 0.7

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
        [-2.8, 3.2], # upper left
        [-1.0-0.5, -2.5-0.5], # lower left
        [0.1-0.5, -0.3-0.2], # lower middle
        [0.3-0.5, 0.1-0.4], # lower middle addendum
        [2.75, 3.7], # upper right
    ]
    obstacle_radiuses: list[float] = [0.4, 0.3, 0.25, 0.4, 0.4]
    # obstacle_radiuses: list[float] = [0.8, 0.8, 0.8, 0.8, 0.9]
    obstacle_lw_ratios: list[float] = [1.0, 5., 5., 0.9, 0.7]
    obstacle_shape_norm: float = float("inf")

    air_obstacle_centers: list[list[float]] = [
        [0., 0.]
    ]
    air_obstacle_radiuses: list[float] = [0.4]
    air_obstacle_lw_ratios: list[float] = [1.0]
    air_obstacles: bool = True

    base_agent: bool = True

    dynamic_targets: bool = True
    update_targets: bool = True

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
                half_extents = jnp.stack([
                    obst_radii * obstacle_lw_ratios,  # half-width (x)
                    obst_radii                         # half-height (y)
                ], axis=-1)

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
                air_half_extents = jnp.stack([
                    air_obst_radii * air_obst_lw_ratios,  # half-width (x)
                    air_obst_radii                         # half-height (y)
                ], axis=-1)

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

            key, _, valid_center = jax.lax.while_loop(
                lambda carry: ~carry[1],
                sample_until_valid,
                init_carry
            )

            return valid_center

        valid_centers = jax.vmap(sample_valid_for_ag)(jr.split(key, n_targets), jnp.arange(n_targets))
        # assert valid_centers.shape == (2, 1, 2)
        valid_centers = valid_centers.squeeze(1)
        # assert valid_centers.shape == (2, 2)

        return valid_centers

    reset_targets_fn = sample_center_outside_obst
    update_targets_fn = sample_center_outside_obst # random jump
    update_cond_fn = 'agent_in_respective_target'

class DeliveryRealv2Base(BaseEnv):
    """
    DeliveryRealv2 env -- made from herd env (eg. num agents = n_herders) to use same callbacks/plotting/utils. Agents move

    Also, in case "dummy" agents (herded) are desired (moving obstacles). Otherwise, just a multi-agent env designed for multi-reach-avoiding. 

    Predicates include:
        - reaching targets (DeliveryRealv2 locs)
        - avoiding obstacles (city)

    Additionally, one may instantiate a 'base' agent, which is slower agent which the other agents may need to revisit.

    In the discrete action setup, each agent is a double-integrator that can accelerate / decelerate in either axis. Herd agents are single-integrators with built in policies.
    action: (n_herders, 2): int, {0, 1, 2} for each axis, where 0 = -accel, 1 = no accel, 2 = accel
    """

    Cfg = DeliveryRealv2BaseCfg
    State = DeliveryRealv2BaseState

    def __init__(self, cfg: DeliveryRealv2BaseCfg = DeliveryRealv2BaseCfg(), should_term_fn: ShouldTermFn = None):
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

    def next_state(self, state: DeliveryRealv2BaseState, control: jnp.ndarray):
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

        return DeliveryRealv2BaseState(herd_state=herd_state_new, herder_state=herder_state_new, steps=state.steps + 1, centers=centers)

    ## BOOL PREDICATES (SPARSE)

    def is_herder_collide(self, state: DeliveryRealv2BaseState):
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
    
    def is_just_herders_collide(self, state: DeliveryRealv2BaseState):
        herder_pos = state.herder_state[:-1, 0:2]
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

    def is_herder_oob(self, state: DeliveryRealv2BaseState):
        herder_pos = state.herder_state[:, 0:2]
        dists = self.dist_to_wall(herder_pos)
        min_dists = jnp.min(dists, axis=-1)
        oob = jnp.any(min_dists < self.cfg.agent_radius)
        return oob

    def is_herd_herded(self, state: DeliveryRealv2BaseState):
        """All herd agents are fully within a circle in the center."""
        herd_pos = state.herd_state
        dists = jnp.linalg.norm(herd_pos, axis=-1)
        herded = jnp.all((dists + self.cfg.agent_radius) < self.cfg.herded_radius)
        return herded

    # def is_herder_in_obstacles(self, state: DeliveryRealv2BaseState, which=jnp):
    #     herder_pos = state.herder_state[..., :, 0:2] 
        
    #     obst_centers = which.array(self.cfg.obstacle_centers)
    #     obst_radii = which.array(self.cfg.obstacle_radiuses)
    #     obstacle_lw_ratios = which.array(self.cfg.obstacle_lw_ratios)

    #     rel_pos = herder_pos[..., None, :, :] - obst_centers[..., :, None, :]
    #     semi_axes = which.stack([
    #         obst_radii * obstacle_lw_ratios,  # x semi-axis (length)
    #         obst_radii                          # y semi-axis (width)
    #     ], axis=-1)  # (n_obst, 2)
    #     normalized_pos = rel_pos / semi_axes[..., :, None, :]
    #     ch_dists = which.linalg.norm(normalized_pos, axis=-1, 
    #                                  ord=self.cfg.obstacle_shape_norm)
    #     # (n_obst, )
    #     c_dists = which.min(ch_dists, axis=-1)
    #     c_is_herder_inside = jnp.any(c_dists < (obst_radii - self.cfg.agent_radius))
    #     return c_is_herder_inside
    
    # def is_herder_in_obstacles(self, state: DeliveryRealv2BaseState, which=jnp):
    #     herder_pos = state.herder_state[..., :, 0:2] 
        
    #     obst_centers = which.array(self.cfg.obstacle_centers)
    #     obst_radii = which.array(self.cfg.obstacle_radiuses)
    #     obstacle_lw_ratios = which.array(self.cfg.obstacle_lw_ratios)
    #     agent_radii = which.where(
    #         self.cfg.base_agent * (which.arange(self.cfg.n_herders) == self.cfg.n_herders-1), 
    #         self.cfg.base_agent_radius, self.cfg.agent_radius
    #     )

    #     rel_pos = herder_pos[..., None, :, :] - obst_centers[..., :, None, :]
    #     semi_axes = which.stack([
    #         obst_radii * obstacle_lw_ratios,  # x semi-axis (length)
    #         obst_radii                          # y semi-axis (width)
    #     ], axis=-1)  # (n_obst, 2)
    #     normalized_pos = rel_pos / semi_axes[..., :, None, :]
    #     ch_dists = which.linalg.norm(normalized_pos, axis=-1, 
    #                                  ord=self.cfg.obstacle_shape_norm)
    #     # (n_obst, )
    #     min_semi_axes = which.min(semi_axes, axis=-1)  # (5,)
    #     normalized_agent_radii = agent_radii[None, :] / min_semi_axes[:, None]  # (5, 3)
    #     # normalized_agent_radii = agent_radii[None, :] / obst_radii[:, None]
    #     c_is_herder_inside = which.any(ch_dists < 1. - normalized_agent_radii)
    #     return c_is_herder_inside

    def is_herder_in_obstacles(self, state: DeliveryRealv2BaseState, which=jnp):
        herder_pos = state.herder_state[..., :, 0:2]  # (n_herders, 2)
        
        obst_centers = which.array(self.cfg.obstacle_centers)  # (n_obst, 2)
        obst_radii = which.array(self.cfg.obstacle_radiuses)  # (n_obst,)
        obstacle_lw_ratios = which.array(self.cfg.obstacle_lw_ratios)  # (n_obst,)
        
        agent_radii = which.where(
            self.cfg.base_agent * (which.arange(self.cfg.n_herders) == self.cfg.n_herders - 1),
            self.cfg.base_agent_radius,
            self.cfg.agent_radius
        )  # (n_herders,)
        
        # Box half-extents: (n_obst, 2)
        half_extents = which.stack([
            obst_radii * obstacle_lw_ratios,  # half-width (x)
            obst_radii                         # half-height (y)
        ], axis=-1)
        
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

    # def is_herder_in_air_obstacles(self, state: DeliveryRealv2BaseState, which=jnp):
    #     # herder_pos = state.herder_state[:-1, 0:2]  # (n_herders, 2) # NOTE just applied to non-base agent
    #     herder_pos = state.herder_state[:-1, 0:2]  # (n_herders, 2) # NOTE just applied to non-base agent

    #     obst_centers = which.array(self.cfg.air_obstacle_centers)  # (n_obst, 2)
    #     obst_radii = which.array(self.cfg.air_obstacle_radiuses)  # (n_obst,)
    #     obstacle_lw_ratios = which.array(self.cfg.air_obstacle_lw_ratios)  # (n_obst,)
        
    #     # Box half-extents: (n_obst, 2)
    #     half_extents = which.stack([
    #         obst_radii * obstacle_lw_ratios,  # half-width (x)
    #         obst_radii                         # half-height (y)
    #     ], axis=-1)
        
    #     # Relative position: (n_obst, n_herders, 2)
    #     rel_pos = herder_pos[None, :, :] - obst_centers[:, None, :]
        
    #     # Box SDF: distance to nearest point on box surface
    #     # q = |rel_pos| - half_extents
    #     q = which.abs(rel_pos) - half_extents[:, None, :]  # (n_obst, n_herders, 2)
        
    #     # SDF = length(max(q, 0)) + min(max(q.x, q.y), 0)
    #     outside_dist = which.linalg.norm(which.maximum(q, 0.0), axis=-1)  # (n_obst, n_herders)
    #     inside_dist = which.minimum(which.max(q, axis=-1), 0.0)  # (n_obst, n_herders)
    #     sdf = outside_dist + inside_dist  # (n_obst, n_herders)
        
    #     # Collision when SDF < agent_radius
    #     c_is_herder_inside = which.any(sdf < self.cfg.agent_radius)
        
    #     return c_is_herder_inside

    def is_herder_in_air_obstacles(self, state: DeliveryRealv2BaseState, which=jnp):
        c_is_herder_inside = which.any(jnp.abs(state.herder_state[..., :-1, 0:2]).max(axis=-1) - self.cfg.agent_radius < 1.)
        return c_is_herder_inside

    def is_herder_in_target(self, state: DeliveryRealv2BaseState, which=jnp, center=[0., 0.], radius=0.5):
        h_pos = state.herder_state[..., :, 0:2]
        ch_dists = which.linalg.norm(h_pos - which.array(center), axis=-1)
        c_is_herder_inside = which.any(ch_dists < (which.array(radius) ))
        return c_is_herder_inside

    def is_herder_in_dyn_target(self, state: DeliveryRealv2BaseState, which=jnp, center_ix=0, radius=0.5):
        h_pos = state.herder_state[..., :, 0:2]
        ch_dists = which.linalg.norm(h_pos - state.centers[center_ix], axis=-1)
        c_is_herder_inside = which.any(ch_dists < (which.array(radius) - self.cfg.agent_radius))
        return c_is_herder_inside

    def is_herder_circs(self, state: DeliveryRealv2BaseState, which=jnp):
        h_pos = state.herder_state[..., :, 0:2]
        c_pos = which.array(self.cfg.centers)
        # (n_circs, n_herders, 2) -> (n_circs, n_herders)
        ch_dists = which.linalg.norm(h_pos[..., None, :, :] - c_pos[..., :, None, :], axis=-1)
        # (n_circs, )
        c_dists = which.min(ch_dists, axis=-1)
        c_is_herder_inside = c_dists < (which.array(self.radiuses) - self.cfg.agent_radius)
        return c_is_herder_inside

    def is_herder_at_base_ag(self, state: DeliveryRealv2BaseState, which=jnp):
        h_pos = state.herder_state[..., :, 0:2]
        base_ag_pos = state.herder_state[-1, 0:2]
        ch_dists = which.linalg.norm(h_pos - base_ag_pos, axis=-1)
        c_all_herder_inside = which.all(ch_dists < self.cfg.agent_radius)
        return c_all_herder_inside
    
    def is_herderX_at_base_ag(self, state: DeliveryRealv2BaseState, herder_ix:int, which=jnp):
        h_pos = state.herder_state[..., herder_ix, 0:2]
        base_ag_pos = state.herder_state[-1, 0:2]
        ch_dists = which.linalg.norm(h_pos - base_ag_pos, axis=-1)
        c_all_herder_inside = which.all(ch_dists < self.cfg.agent_radius)
        return c_all_herder_inside
    
    def is_herderX_circs(self, state: DeliveryRealv2BaseState, herder_ix:int, center_ix:int, which=jnp, radius=0.5):
        # assert self.cfg.dynamic_targets == True
        h_pos = state.herder_state[..., herder_ix, 0:2]
        ch_dists = which.linalg.norm(h_pos - state.centers[center_ix], axis=-1)
        c_is_herder_inside = which.any(ch_dists < (which.array(radius) - self.cfg.agent_radius))
        return c_is_herder_inside

    def get_predicates_bool(self, state: DeliveryRealv2BaseState):
        predicates = {
            "collide": self.is_herder_collide(state),
            "aerial_collide": self.is_just_herders_collide(state),
            "oob": self.is_herder_oob(state),
            # "herd_herded": self.is_herd_herded(state),
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
            "air_obstacles": self.is_herder_in_air_obstacles(state)
        }
        if self.cfg.dynamic_targets:
            predicates["target0"] = self.is_herder_in_dyn_target(state, center_ix=0)
            predicates["target1"] = self.is_herder_in_dyn_target(state, center_ix=1)
        return predicates

    ## FLOAT PREDICATES (DENSE)
    
    def pred_herder_circs(self, state: DeliveryRealv2BaseState, which=jnp):
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
    
    def pred_herder_circs_dyn(self, state: DeliveryRealv2BaseState, which=jnp, center_ix=0, radius=0.5):
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

    def get_predicates_float(self, state: DeliveryRealv2BaseState):
        pred_herder_circs = self.pred_herder_circs(state)
        predicates = {
            "target0_dense": pred_herder_circs[0],
            "target1_dense": pred_herder_circs[1],
            "target2_dense": pred_herder_circs[2],
            "target3_dense": pred_herder_circs[3],
            "target4_dense": pred_herder_circs[4]
        }
        if self.cfg.dynamic_targets:
            predicates["target0_dense"] = self.pred_herder_circs_dyn(state, center_ix=0)
            predicates["target1_dense"] = self.pred_herder_circs_dyn(state, center_ix=1)
        return predicates

    def get_predicates(self, state: DeliveryRealv2BaseState):
        predicates_bool = self.get_predicates_bool(state)
        predicates = {k: jnp.where(v, 1.0, -1.0) for k, v in predicates_bool.items()}

        predicates_float = self.get_predicates_float(state)
        predicates = predicates | predicates_float

        return predicates

    def step(self, state: DeliveryRealv2BaseState, action: jnp.ndarray):
        controls = self._action_to_controls(action)
        state_new = self.next_state(state, controls)
        obs_new = self.get_obs(state_new)

        predicates = self.get_predicates(state_new)
        term = self.should_term_fn(predicates)
        trunc = state_new.steps >= self.cfg.trunc_steps

        info = {"age": state_new.steps}
        return EnvStep(state_new, obs_new, predicates, term, trunc, info)
    
    def step_control(self, state: DeliveryRealv2BaseState, controls: jnp.ndarray):
        state_new = self.next_state(state, controls)
        obs_new = self.get_obs(state_new)

        predicates = self.get_predicates(state_new)
        term = self.should_term_fn(predicates)
        trunc = state_new.steps >= self.cfg.trunc_steps

        info = {"age": state_new.steps}
        return EnvStep(state_new, obs_new, predicates, term, trunc, info)

    def get_objects_pos(self, state: DeliveryRealv2BaseState):
        herd_pos = state.herd_state  # (n_herd, 2)
        herder_pos = state.herder_state[:, 0:2]  # (n_herders, 2)

        if self.cfg.herd_zero:
            all_pos = herder_pos
        else:
            all_pos = jnp.concatenate([herd_pos, herder_pos], axis=0)  # (n_herd + n_herders, 2)

        return all_pos

    def get_obs_and_names(self, state: DeliveryRealv2BaseState):
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
        # 3: Dynamic target centers (if enabled)
        if self.cfg.dynamic_targets:
            obs_centers = state.centers / jnp.array(self.cfg.halfsize)  # (n_targets, 2)
            obs_centers_names = [[f"target{ii}_cx", f"target{ii}_cy"] for ii in range(len(state.centers))]
        
        # ---------------------------------------------------------------------
        if self.cfg.herd_zero:
            if self.cfg.dynamic_targets:
                obs = jnp.concatenate(
                    [
                        obs_herder_pos.flatten(),
                        obs_herder_vel.flatten(),
                        obs_rel_unit_vecs.flatten(),
                        obs_dists.flatten(),
                        obs_centers.flatten(),
                    ]
                )
                obs_names = [
                    *fl(obs_herder_pos_names),
                    *fl(obs_herder_vel_names),
                    *fl(obs_rel_unit_vecs_names),
                    *obs_dist_names,
                    *fl(obs_centers_names),
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
        else:
            if self.cfg.dynamic_targets:
                obs = jnp.concatenate(
                    [
                        obs_herd_pos.flatten(),
                        obs_herder_pos.flatten(),
                        obs_herder_vel.flatten(),
                        obs_rel_unit_vecs.flatten(),
                        obs_dists.flatten(),
                        obs_centers.flatten(),
                    ]
                )
                obs_names = [
                    *fl(obs_herd_pos_names),
                    *fl(obs_herder_pos_names),
                    *fl(obs_herder_vel_names),
                    *fl(obs_rel_unit_vecs_names),
                    *obs_dist_names,
                    *fl(obs_centers_names),
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

    def get_obs(self, state: DeliveryRealv2BaseState):
        obs, _ = self.get_obs_and_names(state)
        return obs

    def sample_pos_outside_obst(self, key: PRNGKeyArray, herd_pos: jnp.ndarray, maxpos_per_ag: jnp.ndarray, minpos_per_ag: jnp.ndarray):
        n_herd = self.cfg.n_herd
        n_herders = self.cfg.n_herders
        valid_pos = jnp.zeros((n_herders, 2))

        centers = jnp.array(self.cfg.centers)

        def sample_valid_for_ag(key, ag_ix):

            def sample_valid_position(key):
                pos_try = jr.uniform(key, shape=(1, 2), minval=minpos_per_ag, maxval=maxpos_per_ag)
                pos_try_full = jnp.zeros((n_herders, 2)).at[:,:2].set(pos_try)
                valid_vel = jnp.zeros_like(pos_try_full)
                herder_state_try = jnp.concatenate([pos_try_full, valid_vel], axis=-1)
                state_try = DeliveryRealv2BaseState(herd_state=herd_pos, herder_state=herder_state_try, steps=0, centers=centers)
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

            key, _, valid_pos_ag = jax.lax.while_loop(
                lambda carry: ~carry[1],
                sample_until_valid,
                init_carry
            )

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

    #     return DeliveryRealv2BaseState(herd_state=herd_pos, herder_state=herder_state, steps=0, centers=centers)
    
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
        key1, key2, key3, key4 = jr.split(key_herders, 4)

        # Agent 1: left box (x < -1)
        minstate_1 = np.array([[-halfsize_x + self.cfg.agent_radius, -1.0 + self.cfg.agent_radius]])
        maxstate_1 = np.array([[-1.0 - self.cfg.agent_radius, 1.0 - self.cfg.agent_radius]])
        non_base_state_1 = jr.uniform(key1, shape=(1, 2), minval=minstate_1, maxval=maxstate_1)

        # Agent 2: right box (x > 1)
        minstate_2 = np.array([[1.0 + self.cfg.agent_radius, -1.0 + self.cfg.agent_radius]])
        maxstate_2 = np.array([[halfsize_x - self.cfg.agent_radius, 1.0 - self.cfg.agent_radius]])
        non_base_state_2 = jr.uniform(key2, shape=(1, 2), minval=minstate_2, maxval=maxstate_2)

        # Base agent: near origin but within [-0.9, 0.9] box
        maxstate_base = np.array([[0.9 - self.cfg.base_agent_radius, 0.9 - self.cfg.base_agent_radius]])
        base_state = jr.uniform(key3, shape=(1, 2), minval=-maxstate_base, maxval=maxstate_base)

        herder_pos = jnp.concatenate([non_base_state_1, non_base_state_2, base_state], axis=0)

        maxvel = np.zeros((n_herders, 2))
        maxvel[:, 0] = np.array(self.cfg.vel_maxs)
        maxvel[:, 1] = np.array(self.cfg.vel_maxs)
        herder_vel = jr.uniform(key4, shape=(n_herders, 2), minval=-maxvel, maxval=maxvel)
        herder_state = jnp.concatenate([herder_pos, herder_vel], axis=-1)

        # Sample dynamic target positions.
        if self.cfg.dynamic_targets:
            centers = self.cfg.reset_targets_fn(key)
        else:
            centers = jnp.array(self.cfg.centers)

        return DeliveryRealv2BaseState(herd_state=herd_pos, herder_state=herder_state, steps=0, centers=centers)

    # def reset_orig(self, key: PRNGKeyArray, centers: jnp.ndarray):
    #     n_herd = self.cfg.n_herd
    #     n_herders = self.cfg.n_herders
    #     key_herd, key_herders, key_herders_vel = jr.split(key, 3)

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
    #     herder_vel = jr.uniform(key_herders_vel, shape=(n_herders, 2), minval=-maxvel, maxval=maxvel)
    #     herder_state = jnp.concatenate([herder_pos_valid, herder_vel], axis=-1)

    #     return DeliveryRealv2BaseState(herd_state=herd_pos, herder_state=herder_state, steps=0, centers=centers)

    # # mixed reset: including some ``good'' samples
    # def reset(self, key: PRNGKeyArray):
    #     # With some prob, reset the herd within small circle, and herder agents on outside pointing inwards.
    #     p_reset_close_center = 0.2
    #     p_reset_med_center = 0.2
    #     p_reset_together = 0.2
    #     p_reset_orig = 1.0 - p_reset_close_center - p_reset_med_center - p_reset_together

    #     key_close, key_med, key_together, key_orig, key_centers, key_which = jr.split(key, 6)

    #     # Sample dynamic target positions.
    #     if self.cfg.dynamic_targets:
    #         centers = self.cfg.reset_targets_fn(key_centers)
    #     else:
    #         centers = self.cfg.centers

    #     herd_state_orig = self.reset_orig(key_orig, centers)
    #     herd_state_close = self.reset_center(key_close, centers, radius=1.)
    #     herd_state_med = self.reset_center(key_med, centers, radius=2.)
    #     herd_state_together = self.reset_together(key_together, centers, radius=1.)

    #     probs = jnp.array([p_reset_close_center, p_reset_med_center, p_reset_together, p_reset_orig])
    #     which_reset = jr.categorical(key_which, probs)

    #     stack_list = [
    #         herd_state_orig,
    #         herd_state_close,
    #         herd_state_med,
    #         herd_state_together,
    #     ]
    #     assert len(probs) == len(stack_list)

    #     herd_state_stack = tree_stack(stack_list)
    #     herd_state = jtu.tree_map(lambda x: x[which_reset], herd_state_stack)

    #     return herd_state

    # def reset_center(self, key:PRNGKeyArray, centers: jnp.ndarray, radius: float):
    #     n_herd = self.cfg.n_herd
    #     key_herd, key_herders, key_herders_vel = jr.split(key, 3)

    #     # Uniformly sample herd positions.
    #     halfsize_x, halfsize_y = self.cfg.halfsize
    #     maxpos = np.array([halfsize_x, halfsize_y]) - self.cfg.agent_radius
    #     if self.cfg.herd_zero:
    #         maxpos = np.zeros(2)
    #     minpos = -maxpos
    #     herd_pos = jr.uniform(key_herd, shape=(n_herd, 2), minval=minpos, maxval=maxpos)

    #     # Sample valid herders positions for min(num_centers, num_agents) agents near each center
    #     n_agents_to_sample = min(centers.shape[0], self.cfg.n_herders)
    #     keys_herders = jr.split(key_herders, n_agents_to_sample)
        
    #     # Sample position for each agent near its corresponding center
    #     herder_positions_near_centers = []
    #     for i in range(n_agents_to_sample):
    #         center = centers[i]
    #         key_agent = keys_herders[i]
    #         herder_pos_all_near_centeri = self.sample_pos_outside_obst(key_agent, herd_pos, maxpos_per_ag=center + radius, minpos_per_ag=center - radius)
    #         herder_pos_i_near_centeri = herder_pos_all_near_centeri[0:1]  # Take only the first agent's position from this sample
    #         herder_positions_near_centers.append(herder_pos_i_near_centeri)

    #     # Stack positions
    #     herder_pos_valid_near = jnp.concatenate(herder_positions_near_centers, axis=0)
        
    #     # If we have more agents than centers, sample remaining agents randomly
    #     if self.cfg.n_herders > n_agents_to_sample:
    #         n_remaining = self.cfg.n_herders - n_agents_to_sample
    #         maxpos_per_ag = np.zeros((1, 2))
    #         maxpos_per_ag[:, 0] = halfsize_x - self.cfg.agent_radius
    #         maxpos_per_ag[:, 1] = halfsize_y - self.cfg.agent_radius
            
    #         key_remaining = jr.fold_in(key_herders, n_agents_to_sample)
    #         herder_pos_remaining = self.sample_pos_outside_obst(key_remaining, herd_pos, maxpos_per_ag=maxpos_per_ag, minpos_per_ag=-maxpos_per_ag)
    #         herder_pos_remaining_n_agents_to_sample = herder_pos_remaining[0:n_remaining]
    #         herder_pos_valid_near = jnp.concatenate([herder_pos_valid_near, herder_pos_remaining_n_agents_to_sample], axis=0)

    #     # Sample vels
    #     maxvel = np.zeros((self.cfg.n_herders, 2))
    #     maxvel[:, 0] = np.array(self.cfg.vel_maxs)
    #     maxvel[:, 1] = np.array(self.cfg.vel_maxs)
    #     if VEL_ZERO:
    #         maxvel[:, 0] = 0.0
    #         maxvel[:, 1] = 0.0
    #     herder_vel = jr.uniform(key_herders_vel, shape=(self.cfg.n_herders, 2), minval=-maxvel, maxval=maxvel)

    #     herder_state = jnp.concatenate([herder_pos_valid_near, herder_vel], axis=-1)

    #     return DeliveryRealv2BaseState(herd_state=herd_pos, herder_state=herder_state, steps=0, centers=centers)

    # def reset_together(self, key:PRNGKeyArray, centers: jnp.ndarray, radius: float):
    #     n_herd = self.cfg.n_herd
    #     key_herd, key_herders, key_herders_vel, key_herders_together = jr.split(key, 4)

    #     # Uniformly sample herd positions.
    #     halfsize_x, halfsize_y = self.cfg.halfsize
    #     maxpos = np.array([halfsize_x, halfsize_y]) - self.cfg.agent_radius
    #     if self.cfg.herd_zero:
    #         maxpos = np.zeros(2)
    #     minpos = -maxpos
    #     herd_pos = jr.uniform(key_herd, shape=(n_herd, 2), minval=minpos, maxval=maxpos)

    #     maxpos_per_ag = np.zeros((1, 2))
    #     maxpos_per_ag[:, 0] = halfsize_x - self.cfg.agent_radius
    #     maxpos_per_ag[:, 1] = halfsize_y - self.cfg.agent_radius

    #     # Sample herder positions and velocities.
    #     herder_pos_valid = self.sample_pos_outside_obst(key_herders, herd_pos, maxpos_per_ag=maxpos_per_ag, minpos_per_ag=-maxpos_per_ag)
    #     herder_pos_valid_together = self.sample_pos_outside_obst(key_herders_together, herd_pos, maxpos_per_ag=herder_pos_valid[0:1] + radius, minpos_per_ag=-herder_pos_valid[0:1] - radius)

    #     # Sample vel
    #     maxvel = np.zeros((self.cfg.n_herders, 2))
    #     maxvel[:, 0] = np.array(self.cfg.vel_maxs)
    #     maxvel[:, 1] = np.array(self.cfg.vel_maxs)
    #     if VEL_ZERO:
    #         maxvel[:, 0] = 0.0
    #         maxvel[:, 1] = 0.0
    #     herder_vel = jr.uniform(key_herders_vel, shape=(self.cfg.n_herders, 2), minval=-maxvel, maxval=maxvel)

    #     herder_state = jnp.concatenate([herder_pos_valid_together, herder_vel], axis=-1)

    #     return DeliveryRealv2BaseState(herd_state=herd_pos, herder_state=herder_state, steps=0, centers=centers)

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


# class DeliveryRealv2BasePlay(DeliveryRealv2Base):
#     """For testing."""
#
#     Cfg = DeliveryRealv2BasePlayCfg
#
#     def __init__(self, cfg: DeliveryRealv2BasePlayCfg, should_term_fn: ShouldTermFn = None):
#         super().__init__(cfg, should_term_fn=should_term_fn)
#         self.cfg = cfg
#
#         self.centers = np.array([[-3.0, -3.0], [3.0, 3.0]])
#         self.radiuses = np.array([1.0, 1.0])
#
#         self.centers_perturb = self.centers[0:1]
#         self.radiuses_perturb = self.radiuses[0:1] + 3 * cfg.agent_radius
