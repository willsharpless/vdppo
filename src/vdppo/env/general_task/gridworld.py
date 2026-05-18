from typing import Any, NamedTuple

import einops as ei
import flax.linen as nn
import jax.numpy as jnp
import jax.random as jr
import jax_dataclasses as jdc
import matplotlib.pyplot as plt
import numpy as np
from attrs import define
from jaxtyping import PRNGKeyArray
from matplotlib.colors import to_rgba

from vdppo.common.emoji_util import plot_emoji
from vdppo.env.general_task.env import (AugObs, BaseEnv, EnvCfg, EnvStep, EnvUsingBase, StateWithTemporalNode,
                                              StaticTemporalNodeMixin, StaticTemporalNodeMixinCfg)

plt.style.use("seaborn-v0_8-darkgrid")


class BoolExpression:
    """So that we can specify either ANY agent or ALL agents"""

    def __call__(self, bools: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError("")


@define
class AnyAgent(BoolExpression):
    """Any agent within the specified set that is True will make the expression True. None means all agents."""

    valid: jnp.ndarray = None  # (n_agents,)

    def __call__(self, bools: jnp.ndarray) -> jnp.ndarray:
        if self.valid is None:
            return jnp.any(bools)
        else:
            bools_valid = bools[self.valid]
            return jnp.any(bools_valid)


@define
class AllAgent(BoolExpression):
    """All agent within the specified set being True will make the expression True. None means all agents."""

    valid: jnp.ndarray = None  # (n_agents,)

    def __call__(self, bools: jnp.ndarray) -> jnp.ndarray:
        if self.valid is None:
            return jnp.all(bools)
        else:
            bools_valid = bools[self.valid]
            return jnp.all(bools_valid)


@define
class GridworldPredicateCfg:
    sparse_predicates: list[str] = None

    eps: float = 0.1
    # If not satisfied, then the one-hop neighbors are -eps.

    delta: float = 0.1
    # The n-hop neighbors are -eps - (n-1) * delta, clipped at -1.


class GridworldObs(NamedTuple):
    # (n_agent, 2)
    n_pos: jnp.ndarray


class GridworldMap:
    """In the valtr codebase, we ended up with the following convention:

     len_y, len_x
    height, width = map.shape

    positions go from 0, 1, ..., len_x - 1 (or len_y - 1)
    """

    def __init__(
        self,
        len_x: int,
        len_y: int,
        predicates_bool: dict[str, np.ndarray],
        predicate_expr: dict[str, BoolExpression],
        d_raw: dict[str, np.ndarray],
        color_dict: dict[str, Any] = None,
        label_dict: dict[str, str] = None,
        pred_cfg: GridworldPredicateCfg = GridworldPredicateCfg(),
    ):
        self._len_x = len_x
        self._len_y = len_y
        self.predicates_bool = predicates_bool
        self.predicate_expr = predicate_expr

        self.d_raw = d_raw
        self.color_dict = color_dict if color_dict is not None else {}
        self.label_dict = label_dict if label_dict is not None else {}

        # Construct the visualization color map.
        empty_map = np.full((len_x, len_y, 4), fill_value=0)
        for k, v in d_raw.items():
            if k in self.color_dict:
                color = to_rgba(self.color_dict[k])
                empty_map = np.where(v[..., None], color, empty_map)
        self.map_viz_color = empty_map
        self.pred_cfg = pred_cfg

    def show_map(self, ax: plt.Axes):
        ax.imshow(np.swapaxes(self.map_viz_color, 0, 1), origin="lower", alpha=0.7)

        annotate_cell(self.d_raw, self.label_dict, ax, offset=np.array([0.28, 0.28]), size_data=0.3, fontsize=10)

    @property
    def len_x(self) -> int:
        return self._len_x

    @property
    def len_y(self) -> int:
        return self._len_y

    @staticmethod
    def Map1() -> "GridworldMap":
        map_str = """
            | A#    |
            | ## ## |
            |    #B |
        """
        d_raw, len_x, len_y = GridworldMap.parse_room_str(map_str, boundary="|")

        predicates = {
            "A": d_raw["A"],
            "B": d_raw["B"],
            "w": d_raw["#"],
        }

        predicate_expr = {
            "A": AnyAgent(),
            "B": AnyAgent(),
            "w": AnyAgent(),
        }

        color_dict = {
            "#": "C3",
        }

        label_dict = {
            "A": "A",
            "B": "B",
        }

        return GridworldMap(len_x, len_y, predicates, predicate_expr, d_raw, color_dict, label_dict)

    @staticmethod
    def Map5() -> "GridworldMap":
        map_str = """
            |   #    |
            |   #    |
            | K ###  |
            |     #  |
            |#### # B|
            |...#    |
            |.A.#    |
            |...D    |
        """
        d_raw, len_x, len_y = GridworldMap.parse_room_str(map_str, boundary="|")

        predicates = {
            "A": d_raw["A"],
            "B": d_raw["B"],
            "D": d_raw["D"],
            "K": d_raw["K"],
            "w": d_raw["#"],
            ".": d_raw["."] | d_raw["#"] | d_raw["A"] | d_raw["D"],
        }
        predicate_expr = {
            "A": AnyAgent(),
            "B": AnyAgent(),
            "D": AnyAgent(),
            "K": AnyAgent(),
            "w": AnyAgent(),
            ".": AnyAgent(),
        }

        color_dict = {
            "#": to_rgba([0.028, 0.62, 0.59], alpha=0.0),
            "K": to_rgba("C1", alpha=0.0),
            "D": to_rgba("C1", alpha=0.0),
        }

        # label_dict = {
        #     "A": "A",
        #     "B": "B",
        #     "K": ":key:",
        #     "D": ":door:",
        # }
        label_dict = {
            "A": "a",
            "B": "b",
            "K": "k",
            "D": "d",
            "#": "w",
            ".": "",
        }

        return GridworldMap(len_x, len_y, predicates, predicate_expr, d_raw, color_dict, label_dict)

    @staticmethod
    def Map6() -> "GridworldMap":
        map_str = """
            |    B   |
            |        |
            |  ..a.  |
            |  ....  |
            |C ....  |
            |  ...b  |
            |        |
            |     A  |
        """
        d_raw, len_x, len_y = GridworldMap.parse_room_str(map_str, boundary="|")

        predicates = {
            "A": d_raw["A"] | d_raw["a"],
            "B": d_raw["B"] | d_raw["b"],
            "C": d_raw["C"],
            "q": d_raw["."] | d_raw["a"] | d_raw["b"],
        }
        predicate_expr = {
            "A": AnyAgent(),
            "B": AnyAgent(),
            "C": AnyAgent(),
            "q": AnyAgent(),
        }

        color_dict = {
            ".": to_rgba("C2", alpha=0.7),
            "a": to_rgba("C2", alpha=0.7),
            "b": to_rgba("C2", alpha=0.7),
        }

        label_dict = {
            "A": "A",
            "a": "A",
            "B": "B",
            "b": "B",
            "C": "C",
        }

        return GridworldMap(len_x, len_y, predicates, predicate_expr, d_raw, color_dict, label_dict)

    @staticmethod
    def Map7() -> "GridworldMap":
        map_str = """
            |        |
            | a.  .. |
            | ..  .. |
            | ..C .. |
            | ..  a.B|
            | ..  .. |
            | .b  .b |
            | A      |
        """
        d_raw, len_x, len_y = GridworldMap.parse_room_str(map_str, boundary="|")

        predicates = {
            "A": d_raw["A"] | d_raw["a"],
            "B": d_raw["B"] | d_raw["b"],
            "C": d_raw["C"],
            "q": d_raw["."] | d_raw["a"] | d_raw["b"],
        }
        predicate_expr = {
            "A": AnyAgent(),
            "B": AnyAgent(),
            "C": AnyAgent(),
            "q": AnyAgent(),
        }

        color_dict = {
            ".": to_rgba("C2", alpha=0.0),
            "a": to_rgba("C2", alpha=0.0),
            "b": to_rgba("C2", alpha=0.0),
        }

        # label_dict = {
        #     "A": "A",
        #     "a": "A",
        #     "B": "B",
        #     "b": "B",
        #     "C": "C",
        # }
        label_dict = {
            "A": "a",
            "a": "a",
            "B": "b",
            "b": "b",
            "C": "g",
        }

        return GridworldMap(len_x, len_y, predicates, predicate_expr, d_raw, color_dict, label_dict)

    def get_predicates(self, pos: jnp.ndarray, which=jnp):
        # (n_agents, 2)
        n_agents, _ = pos.shape
        px, py = pos[..., 0], pos[..., 1]

        predicates_bool = {}
        for pred_name, pred_map in self.predicates_bool.items():
            # pred_map: (len_x, len_y)
            pred_map_jnp = which.asarray(pred_map)
            n_pred_values = pred_map_jnp[px, py]  # (...,)
            assert n_pred_values.shape == (n_agents,)
            pred_value = self.predicate_expr[pred_name](n_pred_values)
            predicates_bool[pred_name] = pred_value

        # Convert from bool to float.
        predicates_float = {k: self.pred_bool_to_float(k, v) for k, v in predicates_bool.items()}
        # predicates = {k: which.where(v, 1.0, -1.0) for k, v in d_predicates_bool.items()}
        return predicates_float

    def pred_bool_to_float(self, name: str, pred_bool: jnp.ndarray) -> jnp.ndarray:
        sparse_predicates = self.pred_cfg.sparse_predicates
        if sparse_predicates is None or name in sparse_predicates:
            return jnp.where(pred_bool, 1.0, -1.0)

        raise NotImplementedError("Dense predicates not implemented yet.")

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

    eval_formulae: dict[str, str] = None

    trunc_steps: int = 100


class GridworldMABase(BaseEnv):
    """Gridworld."""

    Cfg = GridworldMACfg

    def __init__(self, cfg: GridworldMACfg):
        super().__init__()

        self.cfg = cfg

    def get_eval_formulae(self) -> dict[str, str]:
        return self.cfg.eval_formulae or {}

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
    def action_dim(self) -> int:
        return self.n_agents

    @property
    def control_lim_lo(self) -> list[list[float]]:
        return [0] * self.n_agents

    @property
    def control_lim_hi(self) -> list[list[float]]:
        return [1] * self.n_agents

    @property
    def action_deltas(self) -> jnp.ndarray:
        # [stay, up, down, right, left]. [x, y]. Origin is bottom left, increases to top right.
        deltas = jnp.array([[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]])
        return deltas

    def action_to_deltas(self, action: list[jnp.ndarray]) -> jnp.ndarray:
        assert len(action) == self.n_agents

        action_deltas = self.action_deltas

        deltas = []
        for ii in range(self.n_agents):
            agent_action = action[ii].squeeze()
            deltas.append(action_deltas[agent_action])
        deltas = jnp.stack(deltas, axis=0)
        assert deltas.shape == (self.n_agents, 2)
        return deltas

    def clip_pos(self, pos: jnp.ndarray) -> jnp.ndarray:
        len_x, len_y = self.map.len_x, self.map.len_y
        assert pos.shape[-1] == 2
        x_clip = jnp.clip(pos[..., 0], 0, len_x - 1)
        y_clip = jnp.clip(pos[..., 1], 0, len_y - 1)
        pos_clip = jnp.stack([x_clip, y_clip], axis=-1)
        return pos_clip

    def next_state(self, state: GridworldMAState, action: list[jnp.ndarray]) -> GridworldMAState:
        deltas = self.action_to_deltas(action)
        new_pos = self.clip_pos(state.pos + deltas)
        with jdc.copy_and_mutate(state) as state_new:
            state_new.pos = new_pos
            state_new.steps = state.steps + 1
        return state_new

    def step(self, state: GridworldMAState, action: list[jnp.ndarray]):
        state_new = self.next_state(state, action)
        obs_new = self.get_obs(state_new)
        predicates = self.get_predicates(state)
        term = False
        trunc = state_new.steps >= self.cfg.trunc_steps

        info = {"age": state_new.steps}
        return EnvStep(state_new, obs_new, predicates, term, trunc, info)

    def step_control(self, state: GridworldMAState, controls: jnp.ndarray):
        controls = jnp.asarray(controls).reshape((self.n_agents,))
        n_actions = self.action_deltas.shape[0]
        actions = jnp.minimum(jnp.floor(controls * n_actions).astype(jnp.int32), n_actions - 1)
        state_new = self.next_state(state, actions)
        obs_new = self.get_obs(state_new)

        predicates = self.get_predicates(state_new)
        term = False
        trunc = state_new.steps >= self.cfg.trunc_steps

        info = {"age": state_new.steps}
        return EnvStep(state_new, obs_new, predicates, term, trunc, info)

    def add_obs_preprocessor(self, module: nn.Module):
        return GridworldLearnedEmbed(module, self.n_agents, self.map.len_x, self.map.len_y)

    def get_obs_and_names(self, state: GridworldMAState) -> tuple[GridworldObs, list[str]]:
        if self.n_agents > 1:
            raise NotImplementedError("Multi-agent observations not implemented yet.")

        obs = GridworldObs(n_pos=state.pos)
        # # For a single agent and a small map, just do a one-hot encoding.
        # len_x, len_y = self.map.len_x, self.map.len_y
        # # (n_agents, 2) -> (2,)
        # agent_pos = state.pos.squeeze(0)
        #
        # obs = jnp.zeros((len_x, len_y), dtype=jnp.float32)
        # obs = obs.at[agent_pos[0], agent_pos[1]].set(1.0)
        # obs = obs.flatten()
        obs_names = []
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
        plt.style.use("seaborn-v0_8-darkgrid")
        len_x, len_y = self.map.len_x, self.map.len_y
        ax.set_xlim(-0.5, len_x - 0.5)
        ax.set_ylim(-0.5, len_y - 0.5)

        # # Integer ticks at cell centers
        # ax.set_xticks(np.arange(len_x))
        # ax.set_yticks(np.arange(len_y))

        # # Grid lines at half-integers
        # ax.set_xticks(np.arange(-0.5, len_x, 1), minor=True)
        # ax.set_yticks(np.arange(-0.5, len_y, 1), minor=True)

        # # No major grid, white minor grid.
        # ax.grid(False, which="major")
        ax.grid(which="major", color="white", linewidth=1)

        # ax.tick_params(which="minor", bottom=True, left=True)
        # ax.tick_params(which="minor", color="black", labelcolor="black", length=3, width=1)
        # ax.tick_params(which="major", color="black", labelcolor="black", length=3, width=1)

        ax.set_xticks(np.arange(len_x + 1) - 0.5)
        ax.set_yticks(np.arange(len_y + 1) - 0.5)
        ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)

        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(2)

        # Visualize the map.
        self.map.show_map(ax)
        # ax.imshow(self.map.map_viz_color.T, origin="lower", alpha=0.7)

    def is_valid_real_eval_state(self, state):
        predicates = self.get_predicates(state)
        is_unsafe = False
        if "w" in predicates:
            is_unsafe = predicates["w"] > 0
        if "." in predicates:
            is_unsafe = predicates["."] > 0
        if "q" in predicates:
            is_unsafe = predicates["q"] > 0
        return ~is_unsafe

    def get_all_states(self) -> GridworldMAState:
        assert self.n_agents == 1
        # We want to return states such that the pos has shape (len_x, len_y, n_agents=1, 2)
        len_x, len_y = self.map.len_x, self.map.len_y
        xs = jnp.arange(len_x)
        ys = jnp.arange(len_y)
        bb_X, bb_Y = jnp.meshgrid(xs, ys, indexing="ij")
        bb_pos = jnp.stack([bb_X, bb_Y], axis=-1)
        bb_pos = bb_pos[:, :, None, :]  # (len_x, len_y, n_agents=1, 2)
        bb_steps = jnp.zeros((len_x, len_y), dtype=jnp.int32)
        return GridworldMAState(bb_pos, bb_steps)


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


def annotate_cell(
    d_raw: dict[str, np.ndarray],
    label_dict: dict[str, str],
    ax: plt.Axes,
    fontsize: int = 20,
    size_data: float = 0.8,
    offset: np.ndarray = np.array([0.0, 0.0]),
):
    for k, v in d_raw.items():
        if k not in label_dict:
            continue

        label = label_dict[k]
        is_emoji = label.startswith(":") and label.endswith(":")
        xs, ys = np.where(v)
        for x, y in zip(xs, ys):
            if is_emoji:
                plot_emoji(
                    np.array([x, y]) + offset,
                    size_data=size_data,
                    emoji_str=label_dict[k],
                    size=512,
                    ax=ax,
                    extent="lower",
                )
            else:
                ax.text(
                    x + offset[0],
                    y + offset[1],
                    label_dict[k],
                    color="black",
                    fontsize=fontsize,
                    ha="center",
                    va="center",
                )


class GridworldLearnedEmbed(nn.Module):
    nn: nn.Module
    n_agents: int
    len_x: int
    len_y: int
    n_feat: int = 32

    @nn.compact
    def __call__(self, obs: AugObs):
        base_obs = obs.base
        assert isinstance(base_obs, GridworldObs)

        n_pos = base_obs.n_pos
        assert n_pos.shape[-2:] == (self.n_agents, 2)

        # Learn an embedding for each cell in the grid.
        cell_embed = self.param("cell_embed", nn.initializers.xavier_uniform(), (self.len_x, self.len_y, self.n_feat))
        n_feats = cell_embed[n_pos[..., 0], n_pos[..., 1], :]
        assert n_feats.shape[-2:] == (self.n_agents, self.n_feat)

        # Also add in the position normalized to [-1, 1].
        pos_max = jnp.array([self.len_x - 1, self.len_y - 1], dtype=jnp.float32)
        # [0, pos_max] -> [0, 1] -> [-1, 1]
        n_pos_norm = (n_pos / pos_max) * 2.0 - 1.0
        n_feats = jnp.concatenate([n_feats, n_pos_norm], axis=-1)

        # Allow arbitrary batch dimensions
        combined_obs = ei.rearrange(n_feats, "... n_agents n_feat -> ... (n_agents n_feat)")

        obs_new = obs._replace(base=combined_obs)
        return self.nn(obs_new)
