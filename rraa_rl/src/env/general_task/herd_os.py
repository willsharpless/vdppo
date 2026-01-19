import copy
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax_dataclasses as jdc
import numpy as np
from attrs import define
from jaxtyping import PRNGKeyArray

from rraa_rl.jax_types import BoolScalar
from rraa_rl.jax_utils import tree_cat
from rraa_rl.src.env.general_task.env import Env, EnvStep, get_rules
from rraa_rl.src.env.general_task.herd_base import HerdBase, HerdBaseCfg, HerdBaseState


@jdc.pytree_dataclass
class HerdOsState:
    temporal_node_idx: int
    base: HerdBaseState


@define
class HerdOsCfg:
    base: HerdBaseCfg = HerdBaseCfg()
    # What fraction of the batch is which temporal node at reset. This is in reverse topological order.
    # temporal_node_fracs: list[float] = [0.6, 0.4]
    # temporal_node_fracs: list[float] = [0.4, 0.3, 0.3]
    temporal_node_fracs: list[float] | None = None

    do_temporal_transition: bool = False
    """If true (e.g., eval), then change the temporal node according to the DAG transitions."""

    eval_T: int = 200

    @property
    def root_only(self):
        return self.temporal_node_fracs[0] == 1.0


class HerdOs(Env):
    """Herding environment with one or more herders and a herd of agents. The herd moves according to some fixed policy.
    The herders can influence the herd by moving around them.

    Each herd agent is a single-integrator that minimizes the soft minimum distance to the herders, the obstacles,
    and other herd agents, where the distances are scaled such that herders have larger influence.
    If the distance is large enough, the herd agents stay still.

    In the discrete action setup, each herder is a double-integrator that can accelerate / decelerate in either axis.

    action: (n_herders, 2): int, {0, 1, 2} for each axis, where 0 = -accel, 1 = no accel, 2 = accel
    """

    Cfg = HerdOsCfg
    State = HerdOsState

    def __init__(self, cfg: HerdOsCfg = HerdOsCfg()):
        self.cfg = cfg
        self.base = HerdBase(cfg.base, should_term_fn=self.should_terminate)
        super().__init__(self.specification)

        if self.cfg.temporal_node_fracs is None:
            # Split it evenly.
            self.cfg.temporal_node_fracs = np.full(self.n_temporal_nodes, 1.0 / self.n_temporal_nodes).tolist()

        assert len(self.cfg.temporal_node_fracs) == len(self.temporal_nodes)

    @property
    def n_agents(self) -> int:
        return self.base.n_agents

    @property
    def value_lims(self):
        return self.base.value_lims

    @property
    def n_actions_per_agent(self) -> list[list[int]]:
        return self.base.n_actions_per_agent

    @property
    def max_entropy(self) -> float:
        return self.base.max_entropy

    @property
    def n_temporal_nodes(self):
        return len(self.temporal_nodes)

    @property
    def specification(self):
        # return "F(G(herd_herded)) && G( !herder_collide ) && G( !herder_oob )"
        # return "( !herder_collide && ! herder_oob ) U ( G(herd_herded && !herder_collide && ! herder_oob) )"
        return "F herder_c1 && F herder_c2"
        # return "F G (herder_c1 || herder_c2)"
        # return "F herder_c1"
        # return "!herder_collide U herder_c1"
        # return "G F herder_c1 && G F herder_c2"

    def should_terminate(self, predicates: dict[str, jnp.ndarray]) -> BoolScalar:
        eps = 0.1

        # Terminate when leaving the allowed area.
        is_oob = predicates["herder_oob"] > eps
        should_term = is_oob

        # Terminate when reaching the goal, or leaving the allowed area.
        # is_goal = predicates["herder_c1"] > eps
        # should_term = is_goal | is_oob

        return should_term

    def step(self, state: HerdOsState, action: jnp.ndarray):
        base_step: EnvStep = self.base.step(state.base, action)

        temporal_node_idx = state.temporal_node_idx

        # Temporal transitions with G require the value function, so we pass that in as a callback.
        # if self.cfg.do_temporal_transition:
        #     logger.debug("Doing temporal transition")
        #     predicates = base_step.predicates
        #     temporal_node_idx = self.transition_temporal_node(predicates, temporal_node_idx)

        state_new = jdc.replace(state, temporal_node_idx=temporal_node_idx, base=base_step.envstate)
        obs = self._augment_obs(state_new, base_step.obs)
        step = base_step._replace(envstate=state_new, obs=obs)

        return step

    def transition_temporal_node(
        self, predicates: dict[str, jnp.ndarray], t_value: jnp.ndarray, temporal_node_idx: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute the new temporal node idx after applying the DAG transitions."""
        all_triggers = get_rules(self.temporal_nodes, self.dag_nodes, predicates, t_value)
        t_parents = jnp.stack([trigger.parent for trigger in all_triggers])
        t_conditions = jnp.stack([trigger.condition for trigger in all_triggers])
        t_children = jnp.stack([trigger.child for trigger in all_triggers])

        t_is_valid = (t_parents == temporal_node_idx) & (t_conditions > 0.0)

        # If there are multiple valid transitions, pick the one with the highest condition value.
        t_has_valid = jnp.any(t_is_valid)
        t_conditions_masked = jnp.where(t_is_valid, t_conditions, -jnp.inf)
        transition_idx = jnp.argmax(t_conditions_masked)

        temporal_node_idx_new = jnp.where(t_has_valid, t_children[transition_idx], temporal_node_idx)

        # jd.print("-----------", ordered=True)
        # jd.print("t_parents: {}", t_parents, ordered=True)
        # jd.print("t_conditions: {}", t_conditions, ordered=True)
        # jd.print("t_children: {}", t_children, ordered=True)
        # jd.print("{} -> {}", temporal_node_idx, temporal_node_idx_new, ordered=True)

        return temporal_node_idx_new

    def get_obs(self, state: Any) -> Any:
        base_obs = self.base.get_obs(state.base)
        return self._augment_obs(state, base_obs)

    def get_obs_names(self) -> list[str]:
        return self.base.get_obs_names() + self.augment_obs_names()

    def get_predicates(self, state: HerdOsState) -> dict[str, jnp.ndarray]:
        return self.base.get_predicates(state.base)

    def reset(self, key: PRNGKeyArray) -> HerdOsState:
        key_base, key_node = jr.split(key)
        base_state = self.base.reset(key_base)

        # Sample temporal node idx according to configured fractions.
        node_fracs = jnp.array(self.cfg.temporal_node_fracs)
        node_fracs = node_fracs / jnp.sum(node_fracs)
        temporal_node_idx = jr.choice(key_node, a=self.n_temporal_nodes, p=node_fracs)
        state = HerdOsState(temporal_node_idx=temporal_node_idx, base=base_state)
        return state

    def reset_batch(self, key: PRNGKeyArray, batch_size: int) -> Any:
        # Instead of randomly sampling temporal nodes, we assign fixed fractions of the batch to each temporal node.
        base_state = self.base.reset_batch(key, batch_size)

        n_per_temporal_node = np.round(np.array(self.cfg.temporal_node_fracs) * batch_size).astype(int)
        n_per_temporal_node[-1] = batch_size - n_per_temporal_node[:-1].sum()

        temporal_node_idxs = jnp.concatenate([jnp.full((n,), idx) for idx, n in enumerate(n_per_temporal_node)], axis=0)
        state = HerdOsState(
            temporal_node_idx=temporal_node_idxs,
            base=base_state,
        )
        return state

    def get_eval_states(self, n_envs: int) -> HerdOsState:
        # Assign envs evenly to each temporal node.
        n_envs_per_node = np.full((self.n_temporal_nodes,), n_envs // self.n_temporal_nodes)
        n_envs_per_node[0] = n_envs - n_envs_per_node[1:].sum()

        key = jr.PRNGKey(seed=12345)
        max_n_envs_per_node = n_envs_per_node.max()
        m_state_base = self.base.reset_batch(key, max_n_envs_per_node)

        states = []
        for ii, n_envs_this in enumerate(n_envs_per_node):
            state_base = jtu.tree_map(lambda x: x[:n_envs_this], m_state_base)
            state = HerdOsState(
                temporal_node_idx=jnp.full((n_envs_this,), ii),
                base=state_base,
            )
            states.append(state)

        b_state0 = tree_cat(states, axis=0)
        return b_state0

    @property
    def eval_T(self) -> int:
        return self.cfg.eval_T

    def with_temporal_transitions(self) -> "HerdOs":
        """Return a copy of this environment that does temporal transitions on step."""
        cfg_new = copy.deepcopy(self.cfg)
        cfg_new.do_temporal_transition = True
        return HerdOs(cfg_new)
