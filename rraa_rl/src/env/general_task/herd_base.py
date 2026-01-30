import functools as ft

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax_dataclasses as jdc
import matplotlib.pyplot as plt
import numpy as np
from attrs import define
from jaxtyping import PRNGKeyArray
from rraa_rl.geometry import AABB, LineSegment, dist_pt_to_aabb, segment_intersects_aabb
from rraa_rl.jax_types import BoolScalar
from rraa_rl.jax_utils import softmaximum, softminimum, tree_stack
from rraa_rl.src.env.general_task.env import BaseEnv, EnvStep

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

    # Multiplier, ONLY USED FOR VIZ.
    pos_multiplier: float = 1.0


@define(slots=False)
class HerdingHerdCfg(HerdBaseCfg):
    p_reset_center: float = 0.1
    p_reset_task: float = 0.2
    p_reset_herd: float = 0.3
    p_reset_gate: float = 0.3
    p_reset_gap: float = 0.01

    wall_thick_x: float = 0.05


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
            should_term_fn = lambda predicates: False
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

    def _action_to_controls(self, action: list[jnp.ndarray]):
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
        return jnp.zeros_like(n_herd_pos)

    def next_state(self, state: HerdBaseState, control: jnp.ndarray):
        """Compute next state given current state and control inputs."""
        dt = self.cfg.dt

        # Update herder states
        herder_pos = state.herder_state[:, 0:2]
        herder_vel = state.herder_state[:, 2:4]

        # Desired velocity.
        herder_vel_cmd = control
        assert herder_vel_cmd.shape == herder_vel.shape == (self.n_agents, 2)

        if VEL_ZERO:
            vel_inp = control
            herder_pos_new = herder_pos + vel_inp * dt
            herder_vel_new = herder_vel
        else:
            # Take velocity limit into account.
            #     Max acceleration when cmd=vel_max and current_vel = 0.
            #     =>  acc_max = kp_vel * vel_max   => kp_vel = acc_max / vel_max
            acc_maxs = jnp.array(self.cfg.acc_maxs)
            vel_maxs = jnp.array(self.cfg.vel_maxs)
            assert acc_maxs.shape == vel_maxs.shape == (self.n_agents,)

            kp_vel = 0.5 * acc_maxs / vel_maxs
            assert kp_vel.shape == (self.n_agents,)

            # (n_agents, 1) * (n_agents, 2)
            herder_acc = kp_vel[:, None] * (herder_vel_cmd - herder_vel)
            assert herder_acc.shape == (self.n_agents, 2)

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
        # assert dists.shape == (self.cfg.n_herders, 6)
        # dists = dists[..., 5:]
        min_dists = jnp.min(dists, axis=-1)
        oob = jnp.any(min_dists < self.cfg.agent_radius)
        return oob

    def get_predicates_bool(self, state: HerdBaseState):
        predicates = {
            "herder_collide": self.is_herder_collide(state),
            "herder_oob": self.is_herder_oob(state),
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

    def step(self, state: HerdBaseState, action: list[jnp.ndarray]):
        controls = self._action_to_controls(action)
        state_new, info_dyn = self.next_state(state, controls)
        obs_new = self.get_obs(state_new)

        predicates = self.get_predicates(state_new)
        term = self.should_term_fn(predicates)
        trunc = state_new.steps >= self.cfg.trunc_steps

        info = {"age": state_new.steps} | info_dyn
        return EnvStep(state_new, obs_new, predicates, term, trunc, info)

    def step_control(self, state: HerdBaseState, controls: jnp.ndarray):
        state_new , info_dyn = self.next_state(state, controls)
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

    def reset_eval(self, key: PRNGKeyArray):
        n_herd = self.cfg.n_herd
        n_herders = self.cfg.n_herders
        key_herd, key_herders = jr.split(key)

        # Uniformly sample herd positions.
        halfsize_x, halfsize_y = self.cfg.halfsize
        maxpos = np.array([-0.5, halfsize_y]) - self.cfg.agent_radius
        if self.cfg.herd_zero:
            maxpos = np.zeros(2)
        minpos = np.array([-halfsize_x, -halfsize_y]) + self.cfg.agent_radius
        herd_pos = jr.uniform(key_herd, shape=(n_herd, 2), minval=minpos, maxval=maxpos)

        # Uniformly sample herder positions in right half-plane
        maxstate = np.zeros((n_herders, 4))
        minstate = np.zeros((n_herders, 4))
        upper_x = halfsize_x - self.cfg.agent_radius
        upper_y = halfsize_y - self.cfg.agent_radius
        lower_x = 0.5 + self.cfg.agent_radius
        lower_y = -halfsize_y + self.cfg.agent_radius

        maxstate[:, 0] = upper_x - self.cfg.agent_radius
        maxstate[:, 1] = upper_y - self.cfg.agent_radius
        minstate[:, 0] = lower_x - self.cfg.agent_radius
        minstate[:, 1] = lower_y - self.cfg.agent_radius

        # maxstate[:, 2] = np.array(self.cfg.vel_maxs)
        # maxstate[:, 3] = np.array(self.cfg.vel_maxs)
        maxstate[:, 2] = 0.0
        maxstate[:, 3] = 0.0

        herder_state = jr.uniform(key_herders, shape=(n_herders, 4), minval=minstate, maxval=maxstate)

        return HerdBaseState(herd_state=herd_pos, herder_state=herder_state, steps=0)

    @property
    def eval_T(self) -> int:
        return self.cfg.trunc_steps

    def setup_ax(self, ax: plt.Axes):
        cfg = self.cfg
        mul = cfg.pos_multiplier
        ax.set_xlim(-1.05 * cfg.halfsize[0] * mul, 1.05 * cfg.halfsize[0] * mul)
        ax.set_ylim(-1.05 * cfg.halfsize[1] * mul, 1.05 * cfg.halfsize[1] * mul)
        ax.set_aspect("equal")

        # axvspan and axhspan to mark the boundaries.
        opts = dict(color="black", alpha=0.9)
        ax.axvspan(cfg.halfsize[0] * mul, (cfg.halfsize[0] + 1.0) * mul, **opts)
        ax.axvspan((-cfg.halfsize[0] - 1.0) * mul, -cfg.halfsize[0], **opts)
        ax.axhspan(cfg.halfsize[1] * mul, (cfg.halfsize[1] + 1.0) * mul, **opts)
        ax.axhspan((-cfg.halfsize[1] - 1.0) * mul, -cfg.halfsize[1] * mul, **opts)

    def is_valid_real_eval_state(self, state):
        predicates = self.get_predicates(state)
        is_unsafe = predicates["herder_unsafe"] > 0
        return ~is_unsafe

class HerdingHerd(HerdBase):
    Cfg = HerdingHerdCfg
    State = HerdBaseState

    def __init__(self, cfg: HerdingHerdCfg = HerdingHerdCfg(), should_term_fn: ShouldTermFn = None):

        super().__init__(cfg, should_term_fn=should_term_fn)
        self.cfg = cfg

        halfsize = min(cfg.halfsize)
        self.herded_center = np.array([0.56 * halfsize, -0.56 * halfsize])
        self.gates = np.array([[-0.3 * halfsize, 0.5 * halfsize], [0.3 * halfsize, 0.5 * halfsize]])

        # Have a vertical wall, with a gap.
        self.wall_x = 0.0
        self.wall_thick_x = cfg.wall_thick_x
        self.gap_y = self.gates[0, 1]
        self.gap_halfheight = 3 * cfg.agent_radius

        self.wall_lower_aabb, self.wall_upper_aabb = self._get_wall_gap_aabb(
            self.wall_x, self.wall_thick_x, self.gap_y, self.gap_halfheight, cfg.halfsize[1]
        )

    @staticmethod
    def _get_wall_gap_aabb(
        wall_x: float, wall_thick_x: float, gap_y: float, gap_halfheight: float, halfsize_y: float
    ) -> tuple[AABB, AABB]:
        minpos = np.array([wall_x - wall_thick_x / 2, -halfsize_y])
        maxpos = np.array([wall_x + wall_thick_x / 2, gap_y - gap_halfheight])
        aabb_lower = AABB(minpos=minpos, maxpos=maxpos)

        minpos = np.array([wall_x - wall_thick_x / 2, gap_y + gap_halfheight])
        maxpos = np.array([wall_x + wall_thick_x / 2, halfsize_y])
        aabb_upper = AABB(minpos=minpos, maxpos=maxpos)
        return aabb_lower, aabb_upper

    @property
    def n_gates(self) -> int:
        return len(self.gates)

    @ft.partial(jax.jit, static_argnames=("self",))
    def reset(self, key: PRNGKeyArray):
        # With some prob, reset the herd in the center.
        p_reset_center = self.cfg.p_reset_center
        # p_reset_task = 0.2
        # p_reset_herd = 0.3
        # p_reset_gate = 0.3
        # p_reset_gap = 0.01
        # p_reset_orig = 1.0 - p_reset_center - p_reset_herd - p_reset_gate

        p_reset_task = self.cfg.p_reset_task
        p_reset_herd = self.cfg.p_reset_herd
        p_reset_gate = self.cfg.p_reset_gate
        p_reset_gap = self.cfg.p_reset_gap
        p_reset_orig = 1.0 - p_reset_task - p_reset_center - p_reset_herd - p_reset_gate - p_reset_gap
        assert p_reset_orig >= 0.0

        key_orig, key_task, key_center, key_gap, key_herding, key_gate, key_which, key_which_gate = jr.split(key, 8)
        key_gates = jr.split(key_gate, self.n_gates)

        herd_state_orig = super().reset(key_orig)
        herd_state_task = self.reset_task(key_task)
        herd_state_center = self.reset_center(key_center)
        herd_state_gap = self.reset_gap(key_gap)
        herd_state_herding, _ = self.reset_herding(key_herding, center=self.herded_center)

        herd_state_gates, _ = jax.vmap(self.reset_herding)(key_gates, self.gates)
        which_gate = jr.randint(key_which_gate, shape=(), minval=0, maxval=self.n_gates)
        herd_state_gate = jtu.tree_map(lambda x: x[which_gate], herd_state_gates)

        # reset_center = jr.bernoulli(key_do_center, p=p_reset_center)
        probs = np.array([p_reset_orig, p_reset_task, p_reset_center, p_reset_gap, p_reset_herd, p_reset_gate])
        assert np.isclose(probs.sum(), 1.0)
        which_reset = jr.categorical(key_which, probs)

        stack_list = [
            herd_state_orig,
            herd_state_task,
            herd_state_center,
            herd_state_gap,
            herd_state_herding,
            herd_state_gate,
        ]
        assert len(probs) == len(stack_list)

        herd_state_stack = tree_stack(stack_list)
        herd_state = jtu.tree_map(lambda x: x[which_reset], herd_state_stack)

        return herd_state

    def reset_task(self, key: PRNGKeyArray):
        # Randomly reset the herd agents on the left, and the herders on the right.
        key_herd, key_herders = jr.split(key)

        halfsize = np.array(self.cfg.halfsize)

        pos_lo = np.array([-0.9 * halfsize[0], -0.9 * halfsize[1]])
        pos_hi = np.array([-0.1 * halfsize[0], 0.9 * halfsize[1]])
        herd_pos = jr.uniform(key_herd, shape=(self.cfg.n_herd, 2), minval=pos_lo, maxval=pos_hi)

        pos_lo = np.array([0.1 * halfsize[0], -0.9 * halfsize[1]])
        pos_hi = np.array([0.9 * halfsize[0], 0.9 * halfsize[1]])
        herder_pos = jr.uniform(key_herders, shape=(self.cfg.n_herders, 2), minval=pos_lo, maxval=pos_hi)

        herder_vel = jr.uniform(
            key_herders,
            shape=(self.cfg.n_herders, 2),
            minval=-jnp.array(self.cfg.vel_maxs) * 0.5,
            maxval=jnp.array(self.cfg.vel_maxs) * 0.5,
        )
        herder_state = jnp.concatenate([herder_pos, herder_vel], axis=-1)

        return HerdBaseState(herd_state=herd_pos, herder_state=herder_state, steps=0)

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
        herd_pos = self.herded_center + jnp.stack([herd_pos_x, herd_pos_y], axis=-1)
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
        herder_pos = self.herded_center + jnp.stack([herder_pos_x, herder_pos_y], axis=-1)

        herder_vel = jr.uniform(
            key=key_herder_vel,
            shape=(cfg.n_herders, 2),
            minval=-jnp.array(cfg.vel_maxs) * 0.5,
            maxval=jnp.array(cfg.vel_maxs) * 0.5,
        )
        herder_state = jnp.concatenate([herder_pos, herder_vel], axis=-1)

        return HerdBaseState(herd_state=herd_pos, herder_state=herder_state, steps=0)

    def reset_gap(self, key: PRNGKeyArray):
        cfg = self.cfg
        key_base, key_which, key_pos = jr.split(key, 3)

        herd_state = self.reset_task(key)

        # Choose a random herder agent and position it within the gap.
        agent_idx = jr.randint(key_which, shape=(), minval=0, maxval=self.cfg.n_herders)

        # Position within the gap.
        wall_x = self.wall_x
        wall_thick_x = self.wall_thick_x
        agent_radius = cfg.agent_radius
        gap_y = self.gap_y
        gap_halfheight = self.gap_halfheight

        minpos = jnp.array([wall_x - wall_thick_x - agent_radius, gap_y - gap_halfheight + agent_radius])
        maxpos = jnp.array([wall_x + wall_thick_x + agent_radius, gap_y + gap_halfheight - agent_radius])
        herder_pos_in_gap = jr.uniform(key_pos, shape=(2,), minval=minpos, maxval=maxpos)

        with jdc.copy_and_mutate(herd_state) as herd_state:
            herd_state.herder_state = herd_state.herder_state.at[agent_idx, 0:2].set(herder_pos_in_gap)

        return herd_state

    def reset_herding(self, key: PRNGKeyArray, center):
        # Two herd agents initialized on opposite sides of a circle of varying radius.
        # All other herd agents initialized randomly inside the circle.
        # The center of the circle is close to the herding center.
        cfg = self.cfg
        if cfg.n_herd == 1:
            return super().reset(key), {}

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
        herd_pos = center + jnp.stack([herd_pos_x, herd_pos_y], axis=-1)

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
        herder_pos = center + jnp.stack([herder_pos_x, herder_pos_y], axis=-1)

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

    def is_herd_herded(self, state: HerdBaseState):
        """All herd agents are fully within a circle in the center."""
        herd_pos = state.herd_state
        dists = jnp.linalg.norm(herd_pos - self.herded_center, axis=-1)
        herded = jnp.all((dists + self.cfg.agent_radius) <= self.cfg.herded_radius)
        return herded

    def is_herd_in_gates(self, state: HerdBaseState):
        """For each gate, check if all herd agents are fully within the gate circle."""
        n_pos = state.herd_state
        n_gates = self.n_gates
        m_pos_gates = self.gates

        mn_dists = jnp.linalg.norm(n_pos[None, :, :] - m_pos_gates[:, None, :], axis=-1)
        assert mn_dists.shape == (n_gates, self.cfg.n_herd)
        m_is_herd_in_gates = jnp.all((mn_dists + self.cfg.agent_radius) <= self.cfg.herded_radius, axis=-1)

        return m_is_herd_in_gates

    def is_herder_collide_wall(self, state: HerdBaseState):
        # Check if any herder collides with either the bottom or top sections of the wall.
        herder_pos = state.herder_state[:, 0:2]
        herder_x = herder_pos[:, 0]
        herder_y = herder_pos[:, 1]
        collide_bottom = jnp.logical_and(
            jnp.abs(herder_x - self.wall_x) < (self.cfg.agent_radius - 0.5 * self.wall_thick_x),
            herder_y < (self.gap_y - self.gap_halfheight + self.cfg.agent_radius),
        )
        collide_top = jnp.logical_and(
            jnp.abs(herder_x - self.wall_x) < (self.cfg.agent_radius - 0.5 * self.wall_thick_x),
            herder_y > (self.gap_y + self.gap_halfheight - self.cfg.agent_radius),
        )
        collide = jnp.any(collide_bottom | collide_top)
        return collide

    def get_predicates_bool(self, state: HerdBaseState):
        predicates = super().get_predicates_bool(state)
        predicates = {
            "herd_herded": self.is_herd_herded(state),
            "herder_collide_wall": self.is_herder_collide_wall(state),
        } | predicates

        is_herd_in_gates = self.is_herd_in_gates(state)
        for ii in range(self.n_gates):
            predicates[f"herd_gate_{ii}"] = is_herd_in_gates[ii]

        return predicates

    def get_predicates_float(self, state: HerdBaseState):
        predicates_bool = self.get_predicates_bool(state)
        predicates = {k: jnp.where(v, 1.0, -1.0) for k, v in predicates_bool.items()}

        predicates = predicates | super().get_predicates_float(state)
        predicates = {
            "herd_herded": self.is_herd_herded_float(state),
            "herd_herder_collide": self.herd_herder_collide(state),
        } | predicates

        is_herder_unsafe = jnp.stack(
            [predicates["herder_oob"], predicates["herder_collide"], predicates["herd_herder_collide"]], axis=-1
        ).max(axis=-1)
        predicates = predicates | {"herder_unsafe": is_herder_unsafe}

        return predicates

    def is_herd_herded_float(self, state: HerdBaseState):
        """All herd agents are fully within a circle in the center.
        1 when herded, -eps on boundary, scales linearly with soft maximum of furthest herd agent."""
        herd_pos = state.herd_state
        n_dists = jnp.linalg.norm(herd_pos - self.herded_center, axis=-1)

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

    def herd_herder_collide(self, state: HerdBaseState):
        """+1 if herd-herder collisions. -eps at boundary, scales linearly with soft minimum of distances."""
        n_herd_pos = state.herd_state[:, :2]
        m_herder_pos = state.herder_state[:, :2]
        nm_dists = jnp.linalg.norm(
            n_herd_pos[:, None, :] - m_herder_pos[None, :, :],
            axis=-1,
        )
        b_dists = nm_dists.flatten()

        # In boundary if dists <= 2 * agent_radius
        b_dist_to_boundary = b_dists - 2 * self.cfg.agent_radius

        min_dist = b_dist_to_boundary.min()

        if self.cfg.n_herd * self.cfg.n_herders == 1:
            # No need for softmin if only one pair.
            min_dist_soft = b_dist_to_boundary[0]
        else:
            # error <= temperature * log(n). I want error <= 0.05 * halfwidth, so temperature = 0.1 * halfwidth / log(n)
            temperature = 0.05 * min(self.cfg.halfsize) / jnp.log(self.cfg.n_herd)
            min_dist_soft = softminimum(b_dist_to_boundary, temperature=temperature)

        eps = 0.1

        # Linear scaling from -eps to -1. -eps at mindist=0, -1 at mindist = 1 * agent_radius.
        t = min_dist_soft / (1.0 * self.cfg.agent_radius)
        t = jnp.clip(t, 0.0, 1.0)
        outside_val = -(eps + (1.0 - eps) * t)

        val = jnp.where(min_dist <= 0.0, 1.0, outside_val)
        val = jnp.clip(val, -1.0, 1.0)
        return val

    def get_objects_pos(self, state: HerdBaseState):
        herd_pos = state.herd_state  # (n_herd, 2)
        herder_pos = state.herder_state[:, 0:2]  # (n_herders, 2)
        herded_center_pos = self.herded_center
        gates_pos = self.gates
        all_pos = jnp.concatenate([herd_pos, herder_pos, herded_center_pos[None, :], gates_pos], axis=0)
        return all_pos

    def setup_ax(self, ax: plt.Axes):
        cfg = self.cfg
        super().setup_ax(ax)
        mul = self.cfg.pos_multiplier

        assert not self.cfg.herd_zero

        # Plot the herd circle.
        herd_circle = plt.Circle(self.herded_center * mul, cfg.herded_radius * mul, color="C1", alpha=0.2)
        ax.add_patch(herd_circle)
        ax.text(
            self.herded_center[0] * mul,
            self.herded_center[1] * mul,
            "2",
            color="black",
            fontsize=10,
            ha="center",
            va="center",
            alpha=0.5
        )

        # Plot the gates if they are active.
        for ii in range(self.n_gates):
            if self.is_predicate_active(f"herd_gate_{ii}"):
                gate_circle = plt.Circle(self.gates[ii] * mul, cfg.herded_radius * mul, color="lightgray", alpha=0.4)
                ax.add_patch(gate_circle)

                # Draw the number of the gate.
                ax.text(
                    self.gates[ii][0] * mul,
                    self.gates[ii][1] * mul,
                    f"{ii}",
                    color="black",
                    fontsize=12,
                    ha="center",
                    va="center",
                    alpha=0.5,
                )

        # Visualize the bottom and top parts of the wall with rectangles.
        wall_thick_x_vis = self.wall_thick_x
        wall_bottom = plt.Rectangle(
            (-wall_thick_x_vis / 2 * mul, -cfg.halfsize[1] * mul),
            wall_thick_x_vis * mul,
            (self.gap_y - self.gap_halfheight + cfg.halfsize[1]) * mul,
            color="black",
            alpha=0.8,
        )
        wall_top = plt.Rectangle(
            (-wall_thick_x_vis / 2 * mul, (self.gap_y + self.gap_halfheight) * mul),
            wall_thick_x_vis * mul,
            (cfg.halfsize[1] - (self.gap_y + self.gap_halfheight)) * mul,
            color="black",
            alpha=0.8,
        )
        ax.add_patch(wall_bottom)
        ax.add_patch(wall_top)

    def dist_to_wall(self, pos: jnp.ndarray):
        """Compute distance to walls given positions."""
        halfsize = self.cfg.halfsize
        px, py = pos[..., 0], pos[..., 1]

        # Four walls of the room.
        left_dists = px + halfsize[0]
        right_dists = halfsize[0] - px
        bottom_dists = py + halfsize[1]
        top_dists = halfsize[1] - py

        no_batch = pos.shape == (2,)
        if no_batch:
            # Wall with gap.
            wall_lower_dist = dist_pt_to_aabb(pos, aabb=self.wall_lower_aabb)
            wall_upper_dist = dist_pt_to_aabb(pos, aabb=self.wall_upper_aabb)
        else:
            # Wall with gap.
            wall_lower_dist = jax.vmap(ft.partial(dist_pt_to_aabb, aabb=self.wall_lower_aabb))(pos)
            wall_upper_dist = jax.vmap(ft.partial(dist_pt_to_aabb, aabb=self.wall_upper_aabb))(pos)

        dists = jnp.stack(
            [left_dists, right_dists, bottom_dists, top_dists, wall_lower_dist, wall_upper_dist], axis=-1
        )  # (n_agents, 4)
        return dists

    def compute_herd_vel(self, n_herd_pos: jnp.ndarray, m_herder_pos: jnp.ndarray):
        def get_weighted_dist(ii: int, herd_pos_new: jnp.ndarray):
            # Compute the minimum distance to the other herd agents.

            # Keep the softmin error <= 0.05 * halfwidth. error <= temperature * log(n)  =>  temperature = error / log(n)
            temperature = 0.05 * min(self.cfg.halfsize) / jnp.log(max(self.cfg.n_herd, self.cfg.n_herders, 4))

            n_herd_dist = jnp.linalg.norm(n_herd_pos - herd_pos_new, axis=-1)
            # Ignore self-distance
            n_herd_dist = n_herd_dist.at[ii].set(jnp.inf)
            # Take the geometry into account.
            n_herd_dist = n_herd_dist - 2 * self.cfg.agent_radius

            herd_softmin = softminimum(n_herd_dist, temperature=temperature)
            herd_min = jnp.min(n_herd_dist)

            # Compute the minimum distance to the herders.
            # (n_herd, 1, 2) - (1, n_herders, 2) -> (n_herd, n_herders, 2) -> (n_herd, n_herders)
            m_herder_dist = jnp.linalg.norm(m_herder_pos - herd_pos_new, axis=-1)
            # Take the geometry into account.
            m_herder_dist = m_herder_dist - 2 * self.cfg.agent_radius

            # Set the distance to infinity if the herder has no line of sight to the herd agent.
            def intersects_any(herder_pos_):
                segment = LineSegment(herd_pos_new, herder_pos_)
                intersects_upper = segment_intersects_aabb(segment, aabb=self.wall_upper_aabb)
                intersects_lower = segment_intersects_aabb(segment, aabb=self.wall_lower_aabb)
                return intersects_upper | intersects_lower

            m_intersect_any = jax.vmap(intersects_any)(m_herder_pos)
            m_herder_dist = jnp.where(m_intersect_any, jnp.inf, m_herder_dist)

            herder_softmin = softminimum(m_herder_dist, temperature=temperature)
            herder_min = jnp.min(m_herder_dist)

            # Compute the minimum distance to the walls.
            herd_wall_dists_all = self.dist_to_wall(herd_pos_new)
            assert herd_wall_dists_all.shape == (6,)
            # Take geometry into account.
            herd_wall_dists_all = herd_wall_dists_all - self.cfg.agent_radius

            herd_wall_dists = herd_wall_dists_all[:4]
            herd_wall_gap_dists = herd_wall_dists_all[4:]

            herd_wall_softmin = softminimum(herd_wall_dists, temperature=temperature, axis=-1)
            herd_wall_min = jnp.min(herd_wall_dists)

            herd_wall_gap_softmin = softminimum(herd_wall_gap_dists, temperature=temperature, axis=-1)
            herd_wall_gap_min = jnp.min(herd_wall_gap_dists)

            # If the distance to the wall is larger than a threshold, then treat the distance as very big.
            # Smoothly increase the effect of this
            wall_dist_thresh = 10 * self.cfg.agent_radius
            coef = 1 + 2 * jnp.tanh(herd_wall_min / wall_dist_thresh * 2)
            herd_wall_softmin = coef * herd_wall_softmin

            w_herd = 0.1
            w_herder = 2.0
            w_wall = 1.5
            w_wall_gap = 0.05
            vals = jnp.array([herd_softmin, herder_softmin, herd_wall_softmin, herd_wall_gap_softmin])
            weights = jnp.array([w_herd, w_herder, w_wall, w_wall_gap])
            # Higher weight => divide by larger number => is minimum more often.
            weighted_dist = softminimum(vals / weights, temperature=temperature)
            closest = jnp.argmin(vals / weights)
            return weighted_dist, closest, herder_min

        def get_vel_single(ii: int):
            herd_pos = n_herd_pos[ii]

            # Generate candidate actions uniformly in a circle.
            angles = jnp.linspace(0, 2 * jnp.pi, num=16, endpoint=False)
            vel_test = self.cfg.herd_vel * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)  # (num_actions, 2)
            herd_pos_new = herd_pos + vel_test * self.cfg.dt  # (num_actions, 2)

            _, closest_idx, herder_min_dist = get_weighted_dist(ii, herd_pos)
            weighted_dists, _, _ = jax.vmap(ft.partial(get_weighted_dist, ii))(herd_pos_new)  # (num_actions,)
            # Select the action that maximizes the weighted distance.
            best_idx = jnp.argmax(weighted_dists)
            best_vel = vel_test[best_idx]

            # As the herder moves away, the herd slows down.
            eff_range = 4.0
            sigma = eff_range / 2
            free_range = 4 * self.cfg.agent_radius
            tmp = jnp.maximum(herder_min_dist - free_range, 0.0)
            # Use Gaussian kernel.
            herder_vel_coef = jnp.exp(-0.5 * (tmp / sigma) ** 2)

            # If the closest thing is the herd, then move slower than if the closest is a herder.
            closest_is_herd = closest_idx == 0
            vel_coef = jnp.where(closest_is_herd, self.cfg.herd_vel_self / self.cfg.herd_vel, herder_vel_coef)
            best_vel = vel_coef * best_vel

            return best_vel

        n_idxs = jnp.arange(self.cfg.n_herd)
        n_herd_vel = jax.vmap(get_vel_single)(n_idxs)

        return n_herd_vel
