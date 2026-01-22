import functools as ft
from typing import Any

import einops as ei
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

from rraa_rl.geometry import AABB, LineSegment, dist_pt_to_aabb, segment_intersects_aabb
from rraa_rl.jax_types import BoolScalar
from rraa_rl.jax_utils import softmaximum, softminimum, tree_stack
from rraa_rl.src.env.general_task.env import (BaseEnv, Env, EnvCfg, EnvStep, EnvUsingBase, StateWithTemporalNode,
                                              StaticTemporalNodeMixin, StaticTemporalNodeMixinCfg)
from rraa_rl.src.env.general_task.herd_base import (HerdBase, HerdBaseCfg, HerdBasePlay, HerdBasePlayCfg, HerdingHerd,
                                                    HerdingHerdCfg)
from rraa_rl.train_utils import tree_where


class GridworldMap:
    """In the valtr codebase, we ended up with the following convention:

     len_y, len_x
    height, width = map.shape

    positions go from 0, 1, ..., len_x - 1 (or len_y - 1)
    """

    def __init__(self, len_x: int, len_y: int, predicates: dict[str, np.ndarray]):
        self._len_x = len_x
        self._len_y = len_y
        self.predicates = predicates

    @property
    def len_x(self) -> int:
        return self._len_x

    @property
    def len_y(self) -> int:
        return self._len_y

    @staticmethod
    def Map5() -> "GridworldMap":
        map_str = """
            |   #    |
            |   #    |
            | K ###  |
            |     #  |
            |#### # B|
            |   #    |
            | A #    |
            |   D    |
        """
        d_raw, len_x, len_y = GridworldMap.parse_room_str(map_str, boundary="|")

        predicates = {
            "A": np.where(d_raw["A"], 1, -1),
            "B": np.where(d_raw["B"], 1, -1),
            "D": np.where(d_raw["D"], 1, -1),
            "K": np.where(d_raw["K"], 1, -1),
            "w": np.where(d_raw["#"], 1, -1),
        }

        return GridworldMap(len_x, len_y, predicates)

    def get_predicates(self, pos: jnp.ndarray, which=jnp):
        px, py = pos[..., 0], pos[..., 1]

        d_predicates = {}
        for pred_name, pred_map in self.predicates.items():
            # pred_map: (len_x, len_y)
            pred_map_jnp = which.asarray(pred_map)
            pred_values = pred_map_jnp[px, py]  # (...,)
            d_predicates[pred_name] = pred_values

        return d_predicates

    @staticmethod
    def parse_room_str(map_str: str, boundary: str = "|") -> tuple[dict[str, np.ndarray], int, int]:
        map_str = map_str.strip("\n").strip()

        # Figure out how many rows and columns.
        lines = map_str.split("\n")
        len_y = len(lines)
        len_x = len(lines[0].split(boundary)[1])

        # For each unique character, create an entry in the dict.
        d = {}
        for ii, l in enumerate(lines):
            l = l.split(boundary)[1]
            assert len(l) == len_x

            for jj, c in enumerate(l):
                if c not in d:
                    d[c] = np.zeros((len_x, len_y), dtype=bool)

                d[c][jj, ii] = True

        # Flip things. When we iterated above, (0, 0) was top-left. We want (0, 0) to be bottom-left.
        for k in d.keys():
            d[k] = d[k][:, ::-1]

        return d, len_x, len_y


@jdc.pytree_dataclass
class GridworldMAState:
    # (n_agents, 2) = [pos_x, pos_y]
    pos: jnp.ndarray

    steps: int = 0


@define(slots=False)
class GridworldMACfg(EnvCfg, StaticTemporalNodeMixinCfg):
    specification: str = None
    map: GridworldMap = None
    n_agents: int = 1

    trunc_steps: int = 100


class GridworldMABase(BaseEnv):
    """Gridworld."""

    Cfg = GridworldMACfg

    def __init__(self, cfg: GridworldMACfg):
        super().__init__()

        self.cfg = cfg

    @property
    def n_agents(self) -> int:
        return self.cfg.n_agents

    @property
    def value_lims(self):
        return -1, 1

    @property
    def map(self) -> GridworldMap:
        return self.cfg.map

    @property
    def n_actions_per_agent(self) -> list[list[int]]:
        # Each agent has 5 actions: stay, up, down, right, left
        n_actions_per_agent = []
        for _ in range(self.cfg.n_agents):
            n_actions_per_agent.append([5])
        return n_actions_per_agent

    @property
    def action_deltas(self) -> jnp.ndarray:
        # [stay, up, down, right, left]. [x, y]. Origin is bottom left, increases to top right.
        deltas = jnp.array([[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]])
        return deltas

    def action_to_deltas(self, action: jnp.ndarray) -> jnp.ndarray:
        assert action.shape == (self.n_agents,)
        deltas = self.action_deltas[action]  # (n_agents, 2)
        assert deltas.shape == (self.n_agents, 2)
        return deltas

    def clip_pos(self, pos: jnp.ndarray) -> jnp.ndarray:
        len_x, len_y = self.map.len_x, self.map.len_y
        assert pos.shape[-1] == (2,)
        x_clip = jnp.clip(pos[..., 0], 0, len_x - 1)
        y_clip = jnp.clip(pos[..., 1], 0, len_y - 1)
        pos_clip = jnp.stack([x_clip, y_clip], axis=-1)
        return pos_clip

    def next_state(self, state: GridworldMAState, action: jnp.ndarray) -> GridworldMAState:
        deltas = self.action_to_deltas(action)
        new_pos = self.clip_pos(state.pos + deltas)
        with jdc.copy_and_mutate(state) as state_new:
            state_new.pos = new_pos
            state_new.steps = state.steps + 1
        return state_new

    def step(self, state: GridworldMAState, action: jnp.ndarray):
        state_new = self.next_state(state, action)
        obs_new = self.get_obs(state_new)
        predicates = self.get_predicates(state_new)
        term = False
        trunc = state_new.steps >= self.cfg.trunc_steps

        info = {"age": state_new.steps}
        return EnvStep(state_new, obs_new, predicates, term, trunc, info)

    def get_obs_and_names(self, state: GridworldMAState) -> tuple[jnp.ndarray, list[str]]:
        if self.n_agents > 1:
            raise NotImplementedError("Multi-agent observations not implemented yet.")

        # For a single agent and a small map, just do a one-hot encoding.
        len_x, len_y = self.map.len_x, self.map.len_y
        # (n_agents, 2) -> (2,)
        agent_pos = state.pos.squeeze(0)

        obs = jnp.zeros((len_x, len_y), dtype=jnp.float32)
        obs = obs.at[agent_pos[0], agent_pos[1]].set(1.0)
        obs = obs.flatten()
        obs_names = [f"cell_{x}_{y}" for x in range(len_x) for y in range(len_y)]
        return obs, obs_names

    def get_predicates(self, state: GridworldMAState) -> dict[str, jnp.ndarray]:
        # vmap over agents.
        return self.map.get_predicates(state.pos)

    def reset(self, key: PRNGKeyArray) -> GridworldMAState:
        len_x, len_y = self.map.len_x, self.map.len_y
        pos_min = jnp.array([0, 0])
        pos_max = jnp.array([len_x - 1, len_y - 1])
        pos = jr.randint(key, (self.n_agents, 2), pos_min, pos_max)
        state = GridworldMAState(pos=pos, steps=0)
        return state

    @property
    def eval_T(self) -> int:
        return self.cfg.trunc_steps

    def setup_ax(self, ax: plt.Axes):
        len_x, len_y = self.map.len_x, self.map.len_y
        ax.set_xlim(-0.5, len_x - 0.5)
        ax.set_ylim(-0.5, len_y - 0.5)

        # Integer ticks at cell centers
        ax.set_xticks(np.arange(len_x))
        ax.set_yticks(np.arange(len_y))

        # Grid lines at half-integers
        ax.set_xticks(np.arange(-0.5, len_x, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len_y, 1), minor=True)

        # No major grid, white minor grid.
        ax.grid(False, which="major")
        ax.grid(which="minor", color="white", linewidth=1)

        ax.tick_params(which="minor", bottom=True, left=True)
        ax.tick_params(which="minor", color="black", labelcolor="black", length=3, width=1)
        # ax.tick_params(which="major", color="black", labelcolor="black", length=3, width=1)


class GridworldMA(StaticTemporalNodeMixin, EnvUsingBase):
    Cfg = GridworldMACfg
    State = StateWithTemporalNode[GridworldMAState]

    def __init__(self, cfg: GridworldMACfg):
        self.cfg = cfg
        base_env = GridworldMABase(cfg)
        EnvUsingBase.__init__(self, cfg, self.specification, base_env)
        StaticTemporalNodeMixin.__init__(self, cfg)
        self.base = base_env

    def reset_batch(self, key: PRNGKeyArray, batch_size: int, init: bool = False) -> StateWithTemporalNode:
        key_reset, key_steps = jr.split(key)
        b_state: StateWithTemporalNode[GridworldMAState] = super().reset_batch(key, batch_size)

        if init:
            # Randomize the initial timestep.
            with jdc.copy_and_mutate(b_state) as b_state_new:
                b_state_new.base.steps = jr.randint(key_steps, (batch_size,), 0, self.base.cfg.trunc_steps)
        else:
            b_state_new = b_state

        return b_state_new

    @property
    def specification(self):
        return self.cfg.specification
