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
from rraa_rl.cmdp_wrapper import CMDPEnvWrapper
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
class CMDPMAPPOAgentCfg(Cfg):
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

    norm_adv: bool = True

    # Network parameters.
    actor_hids: tuple[int, ...] = (128, 128)
    critic_hids: tuple[int, ...] = (128, 128)

    value_shared_trunk: bool = True
    """If true, the values for all agents share a trunk"""

    max_prob: float | None = 0.95
    """Per agent, the maximum probability allowed for an action. We convert this to an entropy and use it to impose a
    minimum entropy constraint."""

    min_entropy_constr_coef: float = 5e-1
    """Coefficient on the hinge loss for minimum entropy constraint."""

    p_max_pol: float = 0.999
    """Prevent extreme probabilities in the policy, enforced by construction."""


@ft.partial(struct.dataclass, frozen=False)
class CMDPMAPPOAgent:
    Cfg = CMDPMAPPOAgentCfg

    network: TrainState
    env: Env = struct.field(pytree_node=False)
    cfg: CMDPMAPPOAgentCfg = struct.field(pytree_node=False)

    @staticmethod
    def get_agent_name() -> str:
        return "CMDP"

    def to_state_dict(self):
        """For saving to disk."""
        return flax.serialization.to_state_dict(self)

    @classmethod
    def create(
        cls,
        seed: int,
        cfg: CMDPMAPPOAgentCfg,
        env: CMDPEnvWrapper,
    ):
        key, init_key = jr.split(jr.key(seed))

        # Dummy data for network initialization.
        dummy_obs: AugObs | AugObsAutomata = env.get_dummy_obs()

        # Define networks.
        if cfg.value_shared_trunk:
            # 1 MLP, with a linear at the end.
            critic_def = VDValueShared(
                hidden_dims=cfg.critic_hids,
                n_out=env.n_conjunctions,
            )
            critic_def = BaseObsOnly(critic_def)
        else:
            # n_temporal_node separate MLPs.
            critic_def = VDValue(
                hidden_dims=cfg.critic_hids,
                n_out=env.n_conjunctions,
            )
            critic_def = BaseObsOnly(critic_def)

        actor_def = MAMultiDiscretePolicy(
            hidden_dims=cfg.actor_hids, n_actions_per_agent=env.n_actions_per_agent, p_max=cfg.p_max_pol
        )
        actor_def = BaseObsOnly(actor_def)

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
        return cls(network=network, env=env, cfg=cfg)

    def