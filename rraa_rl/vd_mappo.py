import functools as ft
import pathlib
# from typing import Any, Self
from typing import Any

import einops as ei
import flax
import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax_dataclasses as jdc
import numpy as np
import optax
from attrs import define
from cyclopts import Parameter
from flax import struct
from jaxtyping import PRNGKeyArray
from loguru import logger
from typing_extensions import Self
from valtr.reachability import (DAGAvoid, DAGConst, DAGGUMinN, DAGGUSingle, DAGId, DAGMaxN, DAGMinN, DAGNegate,
                                DAGReach, DAGReachAvoid, DAGVar)

from rraa_rl.cfg_utils import Cfg
from rraa_rl.collector import Collector, RolloutOutput
from rraa_rl.distribution import tfd
from rraa_rl.distribution_utils import get_multidiscrete_min_entropy
from rraa_rl.evaluate_dag import evaluate_dag
from rraa_rl.gae import BellmanGUSingle, BellmanMax, BellmanMaxMin, BellmanMin, gae_generalized
from rraa_rl.jax_types import FloatScalar, bFloat
from rraa_rl.nn_modules import (BaseObsOnly, BothObs, IndexAtEnd, LearnTemporalEmbedding, MAMultiDiscretePolicy,
                                SeparateMAMultiDiscretePolicy, VDValue, VDValueShared)
from rraa_rl.src.env.general_task.env import AugObs, AugObsAutomata, Env, EnvStep, StateWithTemporalNode
from rraa_rl.train_state import ModuleDict, Params, TrainState
from rraa_rl.train_utils import compute_norm_and_clip, has_any_nan_or_inf, tree_where


@struct.dataclass
class PPOData:
    state: Any
    act: Any
    obs: jnp.ndarray
    logp: jnp.ndarray

    # Rollout advantages and Q-values (GAE'd)
    A: bFloat
    Q: bFloat

    # Which temporal node this sample corresponds to.
    temporal_idx: jnp.ndarray

    @property
    def shape(self):
        return self.Q.shape


@Parameter("*", group="AgentConfig")
@define
class VDMAPPOAgentCfg(Cfg):
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 1e-2
    clip_eps: float = 0.1

    n_envs_train: int = 1024

    n_epochs: int = 2
    n_minibatches: int = 4

    rollout_T: int = 30
    # rollout_T: int = 2

    truncate_reach_thresh: float = 0.5

    norm_adv: bool = True

    # Network parameters.
    actor_hids: tuple[int, ...] = (128, 128)
    critic_hids: tuple[int, ...] = (128, 128)

    value_shared_trunk: bool = True
    """If true, the values for all agents share a trunk"""

    actor_shared_trunk: bool = True
    """If true, the policies for all agents share a trunk"""

    actor_learn_embedding: bool = False
    """If true, and actor_shared_trunk is true, then add an additional linear layer to learn per-node embeddings."""

    max_prob: float | None = 0.9
    """Per agent, the maximum probability allowed for an action. We convert this to an entropy and use it to impose a
    minimum entropy constraint."""

    min_entropy_constr_coef: float = 5e-1
    """Coefficient on the hinge loss for minimum entropy constraint."""

    p_max_pol: float = 0.999
    """Prevent extreme probabilities in the policy, enforced by construction."""

    terminate_on_reach: bool = False
    """If true, instead of truncating on reach, we terminate so that we don't bootstrap."""


class VDMAPPOStatic:
    def __init__(self, temporal_node_alloc: np.ndarray | None = None):
        self.temporal_node_alloc = temporal_node_alloc


@ft.partial(struct.dataclass, frozen=False)
class VDMAPPOAgent:
    Cfg = VDMAPPOAgentCfg

    network: TrainState
    env: Env = struct.field(pytree_node=False)
    # Class containing static (non-pytree) data.
    static: VDMAPPOStatic = struct.field(pytree_node=False)
    cfg: VDMAPPOAgentCfg = struct.field(pytree_node=False)

    @staticmethod
    def get_agent_name() -> str:
        return "VD"

    def to_state_dict(self):
        """For saving to disk."""
        return flax.serialization.to_state_dict(self)

    @classmethod
    def create(
        cls,
        seed: int,
        cfg: VDMAPPOAgentCfg,
        env: Env,
    ):
        """Initialize the PPO agent."""
        key, init_key = jr.split(jr.key(seed))

        # Dummy data for network initialization.
        dummy_obs: AugObs | AugObsAutomata = env.get_dummy_obs()

        # Define networks.
        if cfg.value_shared_trunk:
            # 1 MLP, with a linear at the end.
            critic_def = VDValueShared(
                hidden_dims=cfg.critic_hids,
                n_out=env.n_temporal_nodes,
            )
            critic_def = BaseObsOnly(critic_def)
        else:
            # n_temporal_node separate MLPs.
            critic_def = VDValue(
                hidden_dims=cfg.critic_hids,
                n_out=env.n_temporal_nodes,
            )
            critic_def = BaseObsOnly(critic_def)

        if cfg.actor_shared_trunk:
            actor_def = MAMultiDiscretePolicy(
                hidden_dims=cfg.actor_hids, n_actions_per_agent=env.n_actions_per_agent, p_max=cfg.p_max_pol
            )
            if cfg.actor_learn_embedding:
                actor_def = LearnTemporalEmbedding(actor_def, n_temporal_nodes=env.n_temporal_nodes)
            else:
                actor_def = BothObs(actor_def)
        else:
            actor_def = SeparateMAMultiDiscretePolicy(
                hidden_dims=cfg.actor_hids,
                n_actions_per_agent=env.n_actions_per_agent,
                n_out=env.n_temporal_nodes,
                p_max=cfg.p_max_pol,
                out_axes=-2,
            )
            actor_def = IndexAtEnd(actor_def, n_out=env.n_temporal_nodes, index_pos=-2)

        if not dummy_obs.base_is_array():
            critic_def = env.add_obs_preprocessor(critic_def)
            actor_def = env.add_obs_preprocessor(actor_def)

        network_info = dict(
            critic=(critic_def, (dummy_obs,)),
            actor=(actor_def, (dummy_obs,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        # For the shared optimizer
        network_tx = optax.multi_transform(
            {
                "actor": optax.adamw(cfg.actor_lr),
                "critic": optax.adamw(cfg.critic_lr),
            },
            {
                "modules_actor": "actor",
                "modules_critic": "critic",
            },
        )

        network_def = ModuleDict(networks)
        network_params = network_def.init(init_key, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)
        static = VDMAPPOStatic()
        return cls(network=network, env=env, static=static, cfg=cfg)

    def evaluate_dag(
        self,
        node_idx: DAGId,
        t_V: jnp.ndarray,
        predicates: dict[str, jnp.ndarray],
        scratch: dict[DAGId, jnp.ndarray] | None = None,
        allow_const: bool = False,
    ) -> jnp.ndarray:
        V_dict = {}
        for temporal_idx, dag_node_idx in enumerate(self.env.temporal_nodes):
            V_dict[dag_node_idx] = t_V[..., temporal_idx]

        return evaluate_dag(self.env.dag_nodes, node_idx, predicates, V_dict, scratch=scratch, allow_const=allow_const)

    # def evaluate_dag(
    #     self,
    #     node_idx: DAGId,
    #     t_V: jnp.ndarray,
    #     predicates: dict[str, jnp.ndarray],
    #     scratch: dict[DAGId, jnp.ndarray] | None = None,
    #     allow_const: bool = False,
    # ) -> jnp.ndarray:
    #     """Can be called with batch of data."""
    #
    #     # Check if already computed.
    #     if scratch is None:
    #         scratch = {}
    #
    #     if node_idx in scratch:
    #         return scratch[node_idx]
    #
    #     batch_shape = t_V.shape[:-1]
    #
    #     dag_node = self.env.dag_nodes[node_idx]
    #     match dag_node:
    #         case DAGConst(value=value):
    #             if allow_const:
    #                 big_float = 6.969e7
    #                 value_float = {True: big_float, False: -big_float}[value]
    #                 out = jnp.full(batch_shape, value_float)
    #             else:
    #                 raise ValueError("Const nodes should have been removed")
    #         case DAGVar(name=name):
    #             out = predicates[name]
    #             # logger.debug("Var(%{}) = {}".format(node_idx, out))
    #         case DAGNegate(arg=arg):
    #             out = -self.evaluate_dag(arg, t_V, predicates, scratch)
    #             # logger.debug("Negate(%{}) = {}".format(node_idx, out))
    #         case DAGMinN(args=args):
    #             args_vals = jnp.stack(
    #                 [self.evaluate_dag(arg, t_V, predicates, scratch) for arg in args],
    #                 axis=0,
    #             )
    #             out = jnp.min(args_vals, axis=0)
    #         case DAGMaxN(args=args):
    #             args_vals = jnp.stack(
    #                 [self.evaluate_dag(arg, t_V, predicates, scratch) for arg in args],
    #                 axis=0,
    #             )
    #             out = jnp.max(args_vals, axis=0)
    #         case _:
    #             # Temporal nodes. Use the value function.
    #             temporal_idx = self.env.temporal_nodes.index(node_idx)
    #             V = t_V[..., temporal_idx]
    #             out = V
    #
    #     scratch[node_idx] = out
    #     return out

    def compute_A_Q(self, Tb_rollout: RolloutOutput, debug: bool = False):
        """Compute GAE advantages and Q-values from rollout."""
        T, b = Tb_rollout.shape

        # (batch, T, n_temporal)
        Tbt_V = self.network.select("critic")(Tb_rollout.obs_now, params=self.network.params)
        Tbt_V = jax.lax.stop_gradient(Tbt_V)
        bTt_V = ei.rearrange(Tbt_V, "T b t -> b T t")

        Tbt_V_next = self.network.select("critic")(Tb_rollout.obs_next, params=self.network.params)
        Tbt_V_next = jax.lax.stop_gradient(Tbt_V_next)
        bTt_V_next = ei.rearrange(Tbt_V_next, "T b t -> b T t")

        value_lo, value_hi = self.env.value_lims
        bTt_V = jnp.clip(bTt_V, value_lo, value_hi)
        bTt_V_next = jnp.clip(bTt_V_next, value_lo, value_hi)

        bT_term = Tb_rollout.term.T
        bT_trunc = Tb_rollout.trunc.T
        # Next step is from a different episode (due to reset) if either terminate or truncate
        bT_next_diff = bT_term | bT_trunc

        bT_A_list = []
        bT_Q_list = []
        bT_temporal_list = []

        # Use self.static.temporal_node_alloc to index into the correct V-values.
        assert self.static.temporal_node_alloc is not None, "temporal_node_alloc must be set before computing A and Q."
        start_idx = 0
        for temporal_node_idx, n_node in enumerate(self.static.temporal_node_alloc):
            end_idx = start_idx + n_node
            cTt_V = bTt_V[start_idx:end_idx]
            cTt_V_next = bTt_V_next[start_idx:end_idx]
            cT_predicates = jtu.tree_map(lambda Tb_arr: Tb_arr.T[start_idx:end_idx], Tb_rollout.predicates_next)

            cT_term = bT_term[start_idx:end_idx]
            cT_next_diff = bT_next_diff[start_idx:end_idx]

            cT_temporal_idx = jnp.full((n_node, T), temporal_node_idx, dtype=jnp.int32)

            cT_V = cTt_V[:, :, temporal_node_idx]
            cT_V_next = cTt_V_next[:, :, temporal_node_idx]

            # Use the DAG to compute the correct arguments.
            dag_node_idx = self.env.temporal_nodes[temporal_node_idx]
            dag_node = self.env.dag_nodes[dag_node_idx]

            match dag_node:
                case DAGAvoid(avoid=stay_idx):
                    logger.info("temporal_idx={} | Avoid for {}:{}".format(temporal_node_idx, start_idx, end_idx))
                    cT_q = self.evaluate_dag(stay_idx, cTt_V, cT_predicates)
                    cT_A, cT_Q = self.compute_A_Q_avoid(cT_q, cT_V, cT_V_next, cT_term, cT_next_diff)
                case DAGReach(reach=reach_idx):
                    logger.info("temporal_idx={} | Reach for {}:{}".format(temporal_node_idx, start_idx, end_idx))
                    cT_r = self.evaluate_dag(reach_idx, cTt_V, cT_predicates)
                    cT_A, cT_Q = self.compute_A_Q_reach(cT_r, cT_V, cT_V_next, cT_term, cT_next_diff)
                case DAGReachAvoid(reach=reach_idx, avoid=stay_idx):
                    logger.info("temporal_idx={} | ReachAvoid for {}:{}".format(temporal_node_idx, start_idx, end_idx))
                    cT_r = self.evaluate_dag(reach_idx, cTt_V, cT_predicates)
                    cT_q = self.evaluate_dag(stay_idx, cTt_V, cT_predicates)
                    cT_A, cT_Q = self.compute_A_Q_reachavoid(cT_q, cT_r, cT_V, cT_V_next, cT_term, cT_next_diff)
                case DAGGUSingle(reach=reach_idx, avoid=stay_idx):
                    AS_REACH = False
                    if AS_REACH:
                        cT_r = self.evaluate_dag(reach_idx, cTt_V, cT_predicates)
                        cT_q = self.evaluate_dag(stay_idx, cTt_V, cT_predicates, allow_const=True)
                        cT_A, cT_Q = self.compute_A_Q_reachavoid(cT_q, cT_r, cT_V, cT_V_next, cT_term, cT_next_diff)
                    else:
                        parent_dag_node_idx = self.env.node_parent_dict[dag_node_idx]
                        parent_node = self.env.dag_nodes[parent_dag_node_idx]
                        assert isinstance(parent_node, DAGGUMinN)
                        gu_singles = parent_node.args

                        gu_idx: int = gu_singles.index(dag_node_idx)

                        logger.info(
                            "temporal_idx={} | GUSingle for {}:{}".format(temporal_node_idx, start_idx, end_idx)
                        )
                        cT_r = self.evaluate_dag(reach_idx, cTt_V, cT_predicates)

                        # GU is the only place where we allow const nodes (for the q part).
                        cT_q = self.evaluate_dag(stay_idx, cTt_V, cT_predicates, allow_const=True)

                        if len(gu_singles) == 1:
                            # No other (r OR q) we need to satisfy.
                            cT_q_tilde = cT_q
                        else:
                            # We also need to respect (r OR q) for all the other GUSingles.
                            cT_q_mins = []
                            for other_gu_idx, other_dag_node_idx in enumerate(gu_singles):
                                if other_gu_idx == gu_idx:
                                    continue

                                other_node = self.env.dag_nodes[other_dag_node_idx]
                                assert isinstance(other_node, DAGGUSingle)

                                cT_r_other = self.evaluate_dag(other_node.reach, cTt_V, cT_predicates)
                                cT_q_other = self.evaluate_dag(other_node.avoid, cTt_V, cT_predicates, allow_const=True)

                                cT_r_or_q = jnp.maximum(cT_r_other, cT_q_other)
                                cT_q_mins.append(cT_r_or_q)

                            cT_q_tilde = jnp.stack(cT_q_mins, axis=-1).min(axis=-1)

                        # Get the temporal_node_idx corresponding to the "next" GUSingle we need to achieve.
                        # Could be the same node if there is only one GU.
                        gu_idx_nextGU = (gu_idx + 1) % len(gu_singles)
                        dag_idx_nextGU = gu_singles[gu_idx_nextGU]
                        temporal_node_idx_nextGU = self.env.temporal_nodes.index(dag_idx_nextGU)

                        logger.debug(
                            "[GUSingle] temporal_idx={}, next temporal_idx={}".format(
                                temporal_node_idx, temporal_node_idx_nextGU
                            )
                        )

                        cT_V_nextGU = cTt_V[:, :, temporal_node_idx_nextGU]
                        cT_V_nextGU_next = cTt_V_next[:, :, temporal_node_idx_nextGU]
                        assert cT_V_nextGU.shape == cT_V_nextGU_next.shape == cT_r.shape

                        # ------
                        gamma, lam = self.cfg.gamma, self.cfg.gae_lambda

                        gae_fn = ft.partial(gae_generalized, gamma=gamma, lam=lam)
                        c_bellman = BellmanGUSingle(
                            T_r=cT_r, T_q=cT_q_tilde, T_V_nextGU=cT_V_nextGU, T_V_nextGU_next=cT_V_nextGU_next
                        )
                        cT_Q_gae = jax.vmap(gae_fn)(cT_V_next, cT_term, cT_next_diff, c_bellman)
                        cT_A_gae = cT_Q_gae - cT_V

                        cT_A, cT_Q = cT_A_gae, cT_Q_gae

                        if debug:
                            c_Q_max = jnp.max(cT_Q, axis=1)
                            Q_max = c_Q_max.max()
                            logger.debug("In GUSingle. Q_max: {}".format(Q_max))

                            if Q_max >= 1.5:
                                # find an index where T_Q >= T_V
                                n_higher = jnp.sum(cT_Q > cT_V, axis=1)
                                idx = jnp.argmax(n_higher)

                                T_r = cT_r[idx]
                                T_q = cT_q[idx]
                                T_V_nextGU = cT_V_nextGU[idx]
                                T_V_nextGU_next = cT_V_nextGU_next[idx]
                                T_V_next = cT_V_next[idx]
                                T_term = cT_term[idx]
                                T_next_diff = cT_next_diff[idx]

                                bellman = BellmanGUSingle(
                                    T_r=T_r, T_q=T_q, T_V_nextGU=T_V_nextGU, T_V_nextGU_next=T_V_nextGU_next
                                )
                                T_Q_gae = gae_fn(T_V_next, T_term, T_next_diff, bellman)

                                T_V = cT_V[idx]

                                logger.debug("T_V: {}".format(T_V))
                                logger.debug("T_Q: {}".format(T_V))

                                ipdb.set_trace()
                case _:
                    raise ValueError(f"Unknown temporal node type: {type(dag_node)}")

            bT_A_list.append(cT_A)
            bT_Q_list.append(cT_Q)
            bT_temporal_list.append(cT_temporal_idx)

            start_idx = end_idx

        bT_A = jnp.concatenate(bT_A_list, axis=0)
        bT_Q = jnp.concatenate(bT_Q_list, axis=0)
        bT_temporal_idx = jnp.concatenate(bT_temporal_list, axis=0)

        assert bT_A.shape == bT_Q.shape == (b, T)
        return bT_A, bT_Q, bT_temporal_idx

    def compute_A_Q_avoid(
        self,
        bT_q: jnp.ndarray,
        bT_V: jnp.ndarray,
        bT_V_next: jnp.ndarray,
        bT_term: jnp.ndarray,
        bT_next_diff: jnp.ndarray,
    ) -> tuple[bFloat, bFloat]:
        gamma, lam = self.cfg.gamma, self.cfg.gae_lambda
        gae_fn = ft.partial(gae_generalized, gamma=gamma, lam=lam)
        b_bellman = BellmanMin(T_q=bT_q)
        bT_Q_gae = jax.vmap(gae_fn)(bT_V_next, bT_term, bT_next_diff, b_bellman)
        bT_A_gae = bT_Q_gae - bT_V
        return bT_A_gae, bT_Q_gae

    def compute_A_Q_reach(
        self,
        bT_r: jnp.ndarray,
        bT_V: jnp.ndarray,
        bT_V_next: jnp.ndarray,
        bT_term: jnp.ndarray,
        bT_next_diff: jnp.ndarray,
    ) -> tuple[bFloat, bFloat]:
        gamma, lam = self.cfg.gamma, self.cfg.gae_lambda
        gae_fn = ft.partial(gae_generalized, gamma=gamma, lam=lam)
        b_bellman = BellmanMax(T_r=bT_r)
        bT_Q_gae = jax.vmap(gae_fn)(bT_V_next, bT_term, bT_next_diff, b_bellman)
        bT_A_gae = bT_Q_gae - bT_V
        return bT_A_gae, bT_Q_gae

    def compute_A_Q_reachavoid(
        self,
        bT_q: jnp.ndarray,
        bT_r: jnp.ndarray,
        bT_V: jnp.ndarray,
        bT_V_next: jnp.ndarray,
        bT_term: jnp.ndarray,
        bT_next_diff: jnp.ndarray,
    ) -> tuple[bFloat, bFloat]:
        gamma, lam = self.cfg.gamma, self.cfg.gae_lambda
        gae_fn = ft.partial(gae_generalized, gamma=gamma, lam=lam)
        b_bellman = BellmanMaxMin(T_r=bT_r, T_q=bT_q)
        bT_Q_gae = jax.vmap(gae_fn)(bT_V_next, bT_term, bT_next_diff, b_bellman)
        bT_A_gae = bT_Q_gae - bT_V
        return bT_A_gae, bT_Q_gae

    def get_Tb_data(self, Tb_rollout: RolloutOutput) -> PPOData:
        bT_A, bT_Q, bT_temporal_idx = self.compute_A_Q(Tb_rollout)
        Tb_data = PPOData(
            state=Tb_rollout.state_now,
            act=Tb_rollout.act,
            obs=Tb_rollout.obs_now,
            logp=Tb_rollout.logprob,
            A=bT_A.T,
            Q=bT_Q.T,
            temporal_idx=bT_temporal_idx.T,
        )
        return Tb_data

    def construct_flattened_rollout(self, Tb_rollout: RolloutOutput) -> PPOData:
        """Construct flattened PPO rollout with advantages and Q-values."""
        Tb_data = self.get_Tb_data(Tb_rollout)

        # Flatten batch and time dimensions
        T, b = Tb_rollout.shape
        b_data = jtu.tree_map(lambda x: x.reshape((b * T,) + x.shape[2:]), Tb_data)

        return b_data

    def update(self, Tb_rollout: RolloutOutput, key: PRNGKeyArray) -> tuple[Self, dict]:
        # Assumption: temporal_node_idx is ascending in the batch dimension.
        # Move the temporal node information into the static field, since our update function depends on it.
        Tb_state_now: StateWithTemporalNode = Tb_rollout.state_now
        b_temporal_node_idx = jax.device_get(Tb_state_now.temporal_node_idx[0])
        # Count how many of each temporal node we have in the batch.
        temporal_node_alloc = []
        for idx in range(self.env.n_temporal_nodes):
            n_idx = np.sum(b_temporal_node_idx == idx)

            # Make sure that it is ascending:
            start_idx = sum(temporal_node_alloc)
            end_idx = start_idx + n_idx
            assert np.all(b_temporal_node_idx[start_idx:end_idx] == idx), (
                f"Temporal node indices are not ascending in batch dimension. "
                f"Node {idx} has {n_idx} samples, but they are not all in a contiguous block."
            )

            temporal_node_alloc.append(n_idx)
        temporal_node_alloc = np.array(temporal_node_alloc)

        if self.static.temporal_node_alloc is None:
            # Set the temporal_node_alloc in static data so that the _update function can use it.
            self.static.temporal_node_alloc = temporal_node_alloc
        else:
            assert np.array_equal(self.static.temporal_node_alloc, temporal_node_alloc), (
                f"Temporal node allocation changed between updates: "
                f"was {self.static.temporal_node_alloc}, now {temporal_node_alloc}"
            )

        self_new, info = self._update(Tb_rollout, key)
        return self_new, info

    def permute_for_minibatch(self, b_data: PPOData):
        (batch_size,) = b_data.shape
        assert (
            batch_size % self.cfg.n_minibatches == 0
        ), f"Batch size {batch_size} not divisible by n_minibatches {self.cfg.n_minibatches}"
        mb_size = batch_size // self.cfg.n_minibatches

        mb_data = jtu.tree_map(lambda b_arr: ei.rearrange(b_arr, "(m mb) ... -> m mb ...", mb=mb_size), b_data)
        return mb_data

    @ft.partial(jax.jit, donate_argnums=0)
    def _update(self, Tb_rollout: RolloutOutput, key: PRNGKeyArray) -> tuple[Self, dict]:
        b_data = self.construct_flattened_rollout(Tb_rollout)

        def loop(carry, inps):
            network: TrainState
            (network,) = carry
            (key_,) = inps

            permute_key, update_key = jr.split(key_, 2)
            m_update_keys = jr.split(update_key, self.cfg.n_minibatches)
            # Permute data to form minibatches.
            mb_data = self.permute_for_minibatch(b_data)

            def update(network_: TrainState, inp_):
                b_data_, update_key_ = inp_
                return self._update_network(network_, b_data_, update_key_)

            carry0_ = network
            inp0_ = mb_data, m_update_keys
            network_new_, infos_ = jax.lax.scan(update, carry0_, inp0_)

            # Take the last
            info_ = jtu.tree_map(lambda x: x[-1], infos_)

            # We want to compute the max for these infos
            info_max_ = {
                "critic/clipped_grad_norm max": jnp.max(infos_["critic/clipped_grad_norm"]),
                "critic/grad_norm max": jnp.max(infos_["critic/grad_norm"]),
                "critic/grad bad": jnp.max(infos_["critic/grad_bad"]),
                #
                "actor/clipped_grad_norm max": jnp.max(infos_["actor/clipped_grad_norm"]),
                "actor/grad_norm max": jnp.max(infos_["actor/grad_norm"]),
                "actor/grad bad": jnp.max(infos_["actor/grad_bad"]),
            }

            return (network_new_,), (info_, info_max_)

        e_keys = jr.split(key, self.cfg.n_epochs)
        carry0 = (self.network,)
        (new_network,), (infos, infos_max) = jax.lax.scan(
            loop,
            carry0,
            (e_keys,),
        )

        # Only keep last step info
        info = jtu.tree_map(lambda x: x[-1], infos)
        info_max = jtu.tree_map(lambda x: jnp.max(x, axis=0), infos_max)
        info = info | info_max
        info["debug/b_data"] = b_data

        return self.replace(network=new_network), info

    def _update_network(self, network: TrainState, b_data: PPOData, key: PRNGKeyArray) -> tuple[TrainState, dict]:
        network, info_critic = self._update_critic(network, b_data)
        network, info_actor = self._update_actor(network, b_data, key)
        info_critic = {f"critic/{k}": v for k, v in info_critic.items()}
        info_actor = {f"actor/{k}": v for k, v in info_actor.items()}
        info = info_critic | info_actor
        return network, info

    def _critic_loss(self, b_data: PPOData, params: Params):
        """Compute MSE loss for the value function."""
        bt_V = self.network.select("critic")(b_data.obs, params=params)
        b_temporal_idx = b_data.temporal_idx
        batch_size = len(b_temporal_idx)
        b_arange = jnp.arange(batch_size)

        b_V = bt_V[b_arange, b_temporal_idx]
        assert b_V.shape == (batch_size,)

        loss = jnp.mean((b_data.Q - b_V) ** 2)

        explained_variance = 1 - jnp.var(b_data.Q - b_V) / (jnp.var(b_data.Q) + 1e-8)

        info = {
            "Loss": loss,
            "V_mean": jnp.mean(b_V),
            "Q_mean": jnp.mean(b_data.Q),
            "Explained Variance": explained_variance,
        }
        return loss, info

    def _actor_loss(self, b_data: PPOData, params: Params, key: PRNGKeyArray):
        """Compute the actor loss. We are trying to maximize reward."""
        b_obs = b_data.obs
        b_act = b_data.act
        batch_size = len(b_data.Q)

        # (batch, n_agents)
        bn_logp_old = b_data.logp

        # Compute logprob and entropies.
        b_key = jr.split(key, batch_size)
        bn_logp, bn_entropy = jax.vmap(ft.partial(self.pol_logp_entropy, params=params))(b_obs, b_act, b_key)

        # Compute ratios.
        bn_logratio = bn_logp - bn_logp_old
        bn_is_ratio = jnp.exp(bn_logratio)
        approx_kl = ((bn_is_ratio - 1) - bn_logratio).mean()

        b_A = b_data.A
        if self.cfg.norm_adv:
            A_mean = jnp.mean(b_A)
            A_std = jnp.std(b_A)
            b_A = (b_A - A_mean) / (A_std + 1e-8)

        bn_is_ratio_clip = jnp.clip(bn_is_ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)

        bn_clip_fraction = (bn_is_ratio < (1 - self.cfg.clip_eps)) | (bn_is_ratio > (1 + self.cfg.clip_eps))
        clip_fraction = jnp.mean(bn_clip_fraction)

        # Negative sign because we are trying to maximize reward => minimize negative reward.
        bn_loss1 = -bn_is_ratio * b_A[:, None]
        bn_loss2 = -bn_is_ratio_clip * b_A[:, None]
        loss_pg = jnp.maximum(bn_loss1, bn_loss2).mean()

        entropy_mean = jnp.mean(bn_entropy)
        loss_entropy = -entropy_mean

        loss = loss_pg + self.cfg.entropy_coef * loss_entropy

        info = {
            "Loss": loss,
            "Loss_pg": loss_pg,
            "Entropy": entropy_mean,
            "Entropy Min": jnp.min(bn_entropy),
            "Approx KL": approx_kl,
            "Clip Frac": clip_fraction,
        }

        if self.cfg.max_prob is not None:
            # Impose a state-wise minimum entropy constraint, for each agent.

            # The entropy of a Multi-Discrete distribution, where for each dimension, one action has probability p_max,
            # and the rest share the remaining probability mass equally.
            n_min_entropy = np.array(
                [
                    get_multidiscrete_min_entropy(n_actions, self.cfg.max_prob)
                    for n_actions in self.env.n_actions_per_agent
                ]
            )
            # Positive loss if bn_entropy < n_min_entropy
            bn_loss_min_entropy = jnp.maximum(0.0, n_min_entropy - bn_entropy)
            loss_min_entropy = bn_loss_min_entropy.mean()

            loss = loss + self.cfg.min_entropy_constr_coef * loss_min_entropy

            info["Loss Min Entropy"] = loss_min_entropy
            info["Min Entropy Violate Frac"] = jnp.mean(bn_entropy < n_min_entropy)

        # Compute the fraction of maximum entropy, if available, to make it easier to interpret.
        max_entropy = self.env.max_entropy
        if max_entropy is not None:
            info["Entropy Frac"] = entropy_mean / max_entropy

        return loss, info

    def _update_critic(self, network: TrainState, b_data: PPOData) -> tuple[TrainState, dict]:
        critic_loss = ft.partial(self._critic_loss, b_data)
        grad, info = jax.grad(critic_loss, has_aux=True)(network.params)

        grad_bad = has_any_nan_or_inf(grad)
        grad, grad_norm, clipped_grad_norm = compute_norm_and_clip(grad, self.cfg.max_grad_norm)
        grad = tree_where(grad_bad, jtu.tree_map(jnp.zeros_like, grad), grad)

        info["clipped_grad_norm"] = clipped_grad_norm
        info["grad_norm"] = grad_norm
        info["grad_bad"] = grad_bad

        network_new = network.apply_gradients(grads=grad)
        return network_new, info

    def _update_actor(self, network: TrainState, b_data: PPOData, key: PRNGKeyArray) -> tuple[TrainState, dict]:
        actor_loss = ft.partial(self._actor_loss, b_data)
        grad, info = jax.grad(actor_loss, has_aux=True)(network.params, key)

        grad_bad = has_any_nan_or_inf(grad)
        grad, grad_norm, clipped_grad_norm = compute_norm_and_clip(grad, self.cfg.max_grad_norm)
        grad = tree_where(grad_bad, jtu.tree_map(jnp.zeros_like, grad), grad)

        info["clipped_grad_norm"] = clipped_grad_norm
        info["grad_norm"] = grad_norm
        info["grad_bad"] = grad_bad

        network_new = network.apply_gradients(grads=grad)
        return network_new, info

    def pol_logp_entropy(self, obs: jnp.ndarray, act: Any, key: PRNGKeyArray, params: Params):
        act_dist: tfd.JointDistributionSequential = self.network.select("actor")(obs, params=params)
        logp_list = act_dist.log_prob_parts(act)
        n_logp = jnp.stack(logp_list, axis=0)
        entropies_list = [dist.entropy() for dist in act_dist.model]
        n_entropy = jnp.stack(entropies_list, axis=0)
        return n_logp, n_entropy

    def sample_action(self, obs: Any, key: PRNGKeyArray) -> tuple[Any, FloatScalar]:
        """Sample a stochastic action from the policy given an observation."""
        act_dist = self.network.select("actor")(obs)
        act = act_dist.sample(seed=key)
        logp_list = act_dist.log_prob_parts(act)
        assert isinstance(logp_list, list)
        logp = jnp.stack(logp_list, axis=0)
        assert logp.shape == (self.env.n_agents,)
        return act, logp

    def det_action(self, obs: Any) -> Any:
        """Get the deterministic action (mode) from the policy given an observation."""
        act_dist = self.network.select("actor")(obs)
        act = act_dist.mode()
        return act

    @ft.partial(jax.jit, static_argnames=("rollout_T", "agent_truncate"), donate_argnames=("collector",))
    def collect_batch(
        self, collector: Collector, rollout_T: int, agent_truncate: bool = True
    ) -> tuple[Collector, RolloutOutput, dict]:
        """Collect a batch of data using stochastic policy."""
        logger.debug("jitting collect_batch...")
        if agent_truncate:
            truncate_fn = self.should_truncate
        else:
            truncate_fn = None
        out = collector.collect_batch(self.sample_action, rollout_T, reset_fn=None, truncate_fn=truncate_fn)
        logger.debug("done jitting collect_batch.")
        return out

    @ft.partial(jax.jit, static_argnames=("rollout_T", "temporal_transitions"))
    def collect_eval_with_states(
        self, collector: Collector, b_state0: Any, rollout_T: int, temporal_transitions: bool = True
    ) -> tuple[RolloutOutput, dict]:
        """Collect full eval trajectories using deterministic policy given eval keys."""
        logger.debug("jitting collect_eval_with_states...")
        # Reset all envs before eval
        new_collector = collector.reset_with_state(b_state0)
        switch_fn = self.temporal_switch_fn if temporal_transitions else None
        _, Tb_rollout, collect_info = new_collector.collect_full_traj_det(
            self.det_action, rollout_T, switch_fn=switch_fn
        )
        logger.debug("done jitting collect_eval_with_states.")
        return Tb_rollout, collect_info

    def get_t_reach_val(self, obs: Any, predicates: dict):
        """Compute the temporal reach values for all temporal nodes. Used for truncation."""

        # The returned obs is for the next state.
        obs_next = obs
        t_V_next = self.network.select("critic")(obs_next, params=self.network.params)
        pred_next = predicates

        # Compute the satisfaction of all temporal predicates, including the value function.
        t_reach_val = []
        for temporal_node_idx, dag_node_idx in enumerate(self.env.temporal_nodes):
            node = self.env.dag_nodes[dag_node_idx]
            match node:
                case DAGReach(reach=reach_idx):
                    reach_val = self.evaluate_dag(reach_idx, t_V_next, pred_next)
                case DAGReachAvoid(reach=reach_idx, avoid=stay_idx):
                    reach_val = self.evaluate_dag(reach_idx, t_V_next, pred_next)
                case DAGAvoid(avoid=stay_idx):
                    # Avoid is not a reach, we never truncate.
                    reach_val = np.array(-np.inf)
                case DAGGUSingle(reach=reach_idx, avoid=stay_idx):
                    # Treat it as a reachavoid.
                    reach_val = self.evaluate_dag(reach_idx, t_V_next, pred_next)
                # case DAGGU(args=args_idx):
                #     reach_val = np.array(-np.inf)
                # raise NotImplementedError("GU not implemented yet")
                case _:
                    raise ValueError(f"Unknown temporal node type: {type(node)}")

            t_reach_val.append(reach_val)
        t_reach_val = jnp.stack(t_reach_val, axis=-1)
        return t_reach_val

    def should_truncate(self, env: Env, b_key: PRNGKeyArray, b_step: EnvStep) -> EnvStep:
        """For all reach or reach-avoid nodes, truncate if we reach the goal."""
        batch_size = len(b_step.term)
        bt_reach_val = jax.vmap(self.get_t_reach_val)(b_step.obs, b_step.predicates)
        b_state: StateWithTemporalNode = b_step.envstate
        b_temporal_node_idx = b_state.temporal_node_idx
        assert b_temporal_node_idx.shape == (batch_size,)

        b_reach_val = bt_reach_val[jnp.arange(batch_size), b_temporal_node_idx]
        assert b_reach_val.shape == (batch_size,)

        # Additionally truncate if reach_val >= threshold.
        b_trunc = b_reach_val >= self.cfg.truncate_reach_thresh
        b_step = b_step._replace(trunc=b_trunc | b_step.trunc)
        return b_step

    def temporal_switch_fn(self, env: Env, key: PRNGKeyArray, step: EnvStep):
        if len(env.temporal_nodes) == 1:
            # No need to switch if only one temporal node.
            return step

        state_next: StateWithTemporalNode = step.envstate
        temporal_node_idx = state_next.temporal_node_idx
        predicates_next = step.predicates
        obs_next = step.obs
        t_value_next = self.network.select("critic")(obs_next, params=self.network.params)
        temporal_node_idx_new = env.transition_temporal_node(predicates_next, t_value_next, temporal_node_idx)

        with jdc.copy_and_mutate(state_next) as envstate_new:
            envstate_new.temporal_node_idx = temporal_node_idx_new

        obs_new = env.get_obs(envstate_new)
        step_new = step._replace(envstate=envstate_new, obs=obs_new)
        return step_new
