import functools as ft
from typing import Self, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax
from attrs import define
from cyclopts import Parameter
from flax import struct
from jaxtyping import PRNGKeyArray
from optax.tree_utils import tree_where

from rraa_rl.collector import RolloutOutput
from rraa_rl.jax_types import bFloat
from rraa_rl.nn_modules import MAMultiDiscretePolicy, VDValue
from rraa_rl.src.env.general_task.herd_os import HerdOs
from rraa_rl.train_utils import ModuleDict, TrainState


@struct.dataclass
class PPORolloutOutput:
    rollout: RolloutOutput

    # Rollout advantages and Q-values (GAE'd)
    T_A: bFloat
    T_Q: bFloat


@Parameter("*", group="AgentConfig")
@define
class VDMAPPOAgentCfg:
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 1e-3
    clip_eps: float = 0.1
    updates_per_step: int = 10

    # Network parameters.
    actor_hids: tuple[int, ...] = (128, 128, 128)
    critic_hids: tuple[int, ...] = (128, 128, 128)


@struct.dataclass
class VDMAPPOAgent:
    Cfg = VDMAPPOAgentCfg

    network: TrainState
    cfg: VDMAPPOAgentCfg = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        cfg: VDMAPPOAgentCfg,
        env: HerdOs,
        **kwargs,
    ):
        """Initialize the PPO agent."""
        key, init_key = jr.split(jr.key(seed))

        # Dummy data for network initialization.
        dummy_obs = env.get_dummy_obs()

        # Define networks.
        critic_def = VDValue(
            hidden_dims=cfg.critic_hids,
            n_out=env.n_temporal_nodes,
        )
        actor_def = MAMultiDiscretePolicy(
            hidden_dims=cfg.actor_hids,
            n_actions_per_agent=env.n_actions_per_agent,
        )
        network_info = dict(
            critic=(critic_def, (dummy_obs,)),
            actor=(actor_def, (dummy_obs,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        # For the shared optimizer
        network_tx = optax.multi_transform(
            {
                "actor": optax.adamw(cfg.actor_lr, weight_decay=1e-6),
                "critic": optax.adamw(cfg.critic_lr, weight_decay=1e-6),
            },
            {
                "modules_actor": "actor",
                "modules_critic": "critic",
            },
        )

        network_def = ModuleDict(networks)
        network_params = network_def.init(init_key, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(network=network, cfg=cfg)

    def compute_A_Q(self, bT_rollout: RolloutOutput):
        """Compute GAE advantages and Q-values from rollout."""
        b, T = bT_rollout.shape

        # (batch, T, n_temporal)
        bTt_V = self.network.select("critic")(bT_rollout.obs_now, params=self.network.params)
        bTt_V = jax.lax.stop_gradient(bTt_V)
        bTt_V_next = self.network.select("critic")(bT_rollout.obs_next, params=self.network.params)
        bTt_V_next = jax.lax.stop_gradient(bTt_V_next)

        bT_term = bT_rollout.term

    def construct_flattened_rollout(self, bT_rollout: RolloutOutput) -> PPORolloutOutput:
        """Construct flattened PPO rollout with advantages and Q-values."""
        bT_A, bT_Q = self.compute_A_Q(bT_rollout)
        bT_rollout_ppo = PPORolloutOutput(
            rollout=bT_rollout,
            T_A=bT_A,
            T_Q=bT_Q,
        )

        # Flatten batch and time dimensions
        b, T = bT_rollout.T_rew.shape
        bT_flat_rollout_ppo = jtu.tree_map(lambda x: x.reshape((b * T,) + x.shape[2:]), bT_rollout_ppo)

        return bT_flat_rollout_ppo

    @ft.partial(jax.jit, donate_argnums=0)
    def update(self, bT_rollout: RolloutOutput, key: PRNGKeyArray) -> tuple[Self, dict]:
        bT_flat_rollout_ppo = self.construct_flattened_rollout(bT_rollout)

        def loss_fn_(params, loss_key):
            total_loss_, info_ = self.total_loss(bT_flat_rollout_ppo, params, loss_key)
            return total_loss_, info_

        def loop(carry, inps):
            (network,) = carry
            (_, update_key) = inps
            # loss_fn = ft.partial(loss_fn_, loss_key=loss_key)
            grad, info = jax.grad(loss_fn_, has_aux=True)(network.params, update_key)
            grad_ill = has_any_nan_or_inf(grad)
            grad, grad_norm, clipped_grad_norm = compute_norm_and_clip(grad, self.cfg.max_grad_norm)
            grad = tree_where(grad_ill, jtu.tree_map(jnp.zeros_like, grad), grad)
            new_network = network.apply_gradients(grads=grad)
            return (new_network,), (grad_norm, clipped_grad_norm, grad_ill, info)

        update_idxs = jnp.arange(self.cfg.updates_per_step)
        key, skey = jax.random.split(key)
        update_keys = jr.split(skey, self.cfg.updates_per_step)
        (new_network,), (grad_norm, clipped_grad_norm, grad_ill, info) = jax.lax.scan(
            loop,
            (self.network,),
            (update_idxs, update_keys),
        )

        # Only keep last step info
        clipped_grad_norm = clipped_grad_norm[-1]
        grad_norm = grad_norm[-1]
        grad_ill = grad_ill[-1]
        info = jtu.tree_map(lambda x: x[-1], info)
        info = info | {
            "total/clipped_grad_norm": clipped_grad_norm,
            "total/grad_norm": grad_norm,
            "total/grad_ill": grad_ill,
        }

        return self.replace(network=new_network), info
