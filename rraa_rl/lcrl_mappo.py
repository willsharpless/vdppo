import functools as ft
# from typing import Any, Self
from typing import Any

import einops as ei
import flax
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax
from attrs import define
from cyclopts import Parameter
from flax import struct
from jaxtyping import PRNGKeyArray
from loguru import logger
from typing_extensions import Self

from rraa_rl.cfg_utils import Cfg
from rraa_rl.collector import Collector, RolloutOutput
from rraa_rl.distribution import tfd
from rraa_rl.gae import sum_gae
from rraa_rl.jax_types import FloatScalar, bFloat
from rraa_rl.lcrl.lcrl_wrapper import LCRLWrapper
from rraa_rl.nn_modules import BaseObsOnly, IndexAtEnd, SeparateMAMultiDiscretePolicy, VDValue
from rraa_rl.src.env.general_task.env import Env
from rraa_rl.train_state import ModuleDict, Params, TrainState
from rraa_rl.train_utils import compute_norm_and_clip, has_any_nan_or_inf, tree_where


@struct.dataclass
class LCRLData:
    # state: Any
    act: Any
    obs: jnp.ndarray
    logp: jnp.ndarray

    # Rollout advantages and Q-values (GAE'd)
    A: bFloat
    Q: bFloat

    # Which automata state this sample corresponds to.
    automata_idx: jnp.ndarray

    # If an epsilon transition was taken.
    # If so, only update the epsilon actor (last one), otherwise update the base actors.
    epsilon_taken: jnp.ndarray

    @property
    def shape(self):
        return self.Q.shape


@Parameter("*", group="AgentConfig")
@define
class LCRLMAPPOAgentCfg(Cfg):
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

    unsat_penalty: float = 0.0
    """Penalty for not changing the frontier in LCRL."""

    sink_penalty: float = -1.0
    """Penalty for entering the sink state in LCRL."""

    # Network parameters.
    actor_hids: tuple[int, ...] = (128, 128)
    critic_hids: tuple[int, ...] = (128, 128)


@ft.partial(struct.dataclass, frozen=False)
class LCRLMAPPOAgent:
    Cfg = LCRLMAPPOAgentCfg

    network: TrainState
    env: Env = struct.field(pytree_node=False)
    # Class containing static (non-pytree) data.
    cfg: LCRLMAPPOAgentCfg = struct.field(pytree_node=False)

    def to_state_dict(self):
        """For saving to disk."""
        return flax.serialization.to_state_dict(self)

    @classmethod
    def create(
        cls,
        seed: int,
        cfg: LCRLMAPPOAgentCfg,
        env: LCRLWrapper,
    ):
        """Initialize the PPO agent."""
        key, init_key = jr.split(jr.key(seed))

        # Dummy data for network initialization.
        dummy_obs = env.get_dummy_obs()

        # Define networks.
        # n_temporal_node separate MLPs.
        critic_def = VDValue(
            hidden_dims=cfg.critic_hids,
            n_out=env.ldba.n_states,
        )
        critic_def = BaseObsOnly(critic_def)

        actor_def = SeparateMAMultiDiscretePolicy(
            hidden_dims=cfg.actor_hids, n_actions_per_agent=env.n_actions_per_agent, n_out=env.ldba.n_states
        )
        actor_def = IndexAtEnd(actor_def, n_out=env.ldba.n_states)

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

    def get_LCRL_reward(self, frontier_changed: jnp.ndarray, into_sink: jnp.ndarray):
        rew_frontier = jnp.where(frontier_changed, 1.0, self.cfg.unsat_penalty)
        rew_into_sink = jnp.where(into_sink, self.cfg.sink_penalty, 0.0)
        rew = rew_frontier + rew_into_sink
        return rew

    def compute_A_Q(self, Tb_rollout: RolloutOutput):
        """Compute GAE advantages and Q-values from rollout."""
        T, b = Tb_rollout.shape

        Tb_state: LCRLWrapper.State = Tb_rollout.state_now
        Tb_automata_idx = Tb_state.ldba_state.state
        bT_automata_idx = Tb_automata_idx.T

        Tb_state_next: LCRLWrapper.State = Tb_rollout.state_next
        Tb_automata_idx_next = Tb_state_next.ldba_state.state
        bT_automata_idx_next = Tb_automata_idx_next.T

        Tb_frontier_changed = Tb_rollout.info["has_frontier_changed"]
        bT_frontier_changed = Tb_frontier_changed.T

        Tb_into_sink = Tb_rollout.info["into_sink"]
        bT_into_sink = Tb_into_sink.T

        bT_rew = self.get_LCRL_reward(bT_frontier_changed, bT_into_sink)

        # (batch, T, n_automata)
        Tbt_V = self.network.select("critic")(Tb_rollout.obs_now, params=self.network.params)
        Tbt_V = jax.lax.stop_gradient(Tbt_V)
        bTt_V = ei.rearrange(Tbt_V, "T b t -> b T t")
        # Use bT_automata_idx to index into the correct temporal node value.
        bT_V = bTt_V[jnp.arange(b)[:, None], jnp.arange(T)[None, :], bT_automata_idx]

        Tbt_V_next = self.network.select("critic")(Tb_rollout.obs_next, params=self.network.params)
        Tbt_V_next = jax.lax.stop_gradient(Tbt_V_next)
        bTt_V_next = ei.rearrange(Tbt_V_next, "T b t -> b T t")
        bT_V_next = bTt_V_next[jnp.arange(b)[:, None], jnp.arange(T)[None, :], bT_automata_idx_next]

        assert bT_V.shape == bT_V_next.shape == (b, T)

        bT_term = Tb_rollout.term.T
        bT_trunc = Tb_rollout.trunc.T
        # Next step is from a different episode (due to reset) if either terminate or truncate
        bT_next_diff = bT_term | bT_trunc

        gamma, gae_lambda = self.cfg.gamma, self.cfg.gae_lambda
        gae_fn = ft.partial(sum_gae, gamma=gamma, gae_lambda=gae_lambda)
        bT_A, bT_Q = jax.vmap(gae_fn)(bT_V, bT_V_next, bT_term, bT_next_diff, bT_rew)
        return bT_A, bT_Q

    def get_Tb_data(self, Tb_rollout: RolloutOutput) -> LCRLData:
        Tb_state: LCRLWrapper.State = Tb_rollout.state_now
        Tb_automata_idx = Tb_state.ldba_state.state

        Tb_epsilon_taken = Tb_rollout.info["epsilon_taken"]

        bT_A, bT_Q = self.compute_A_Q(Tb_rollout)
        Tb_data = LCRLData(
            act=Tb_rollout.act,
            obs=Tb_rollout.obs_now,
            logp=Tb_rollout.logprob,
            A=bT_A.T,
            Q=bT_Q.T,
            automata_idx=Tb_automata_idx,
            epsilon_taken=Tb_epsilon_taken,
        )
        return Tb_data

    def construct_flattened_rollout(self, Tb_rollout: RolloutOutput) -> LCRLData:
        """Construct flattened PPO rollout with advantages and Q-values."""
        Tb_data = self.get_Tb_data(Tb_rollout)

        # Flatten batch and time dimensions
        T, b = Tb_rollout.shape
        b_data = jtu.tree_map(lambda x: x.reshape((b * T,) + x.shape[2:]), Tb_data)

        return b_data

    def update(self, Tb_rollout: RolloutOutput, key: PRNGKeyArray) -> tuple[Self, dict]:
        self_new, info = self._update(Tb_rollout, key)
        return self_new, info

    def permute_for_minibatch(self, b_data: LCRLData):
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

    def _update_network(self, network: TrainState, b_data: LCRLData, key: PRNGKeyArray) -> tuple[TrainState, dict]:
        network, info_critic = self._update_critic(network, b_data)
        network, info_actor = self._update_actor(network, b_data, key)
        info_critic = {f"critic/{k}": v for k, v in info_critic.items()}
        info_actor = {f"actor/{k}": v for k, v in info_actor.items()}
        info = info_critic | info_actor
        return network, info

    def _critic_loss(self, b_data: LCRLData, params: Params):
        """Compute MSE loss for the value function."""
        bt_V = self.network.select("critic")(b_data.obs, params=params)
        b_automata_idx = b_data.automata_idx
        batch_size = len(b_automata_idx)
        b_arange = jnp.arange(batch_size)

        b_V = bt_V[b_arange, b_automata_idx]
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

    def _actor_loss(self, b_data: LCRLData, params: Params, key: PRNGKeyArray):
        """Compute the actor loss. We are trying to maximize reward."""
        b_obs = b_data.obs
        b_act = b_data.act
        batch_size = len(b_data.Q)

        b_epsilon_taken = b_data.epsilon_taken

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
        bn_loss = jnp.maximum(bn_loss1, bn_loss2)

        # If epsilon was taken, only update the epsilon actor (last one), i.e.,
        bn_loss_base = bn_loss[:, :-1]
        b_loss_pg_base = jnp.mean(bn_loss_base, axis=1)
        b_loss_pg_epsilon = bn_loss[:, -1]
        b_loss_pg = jnp.where(b_epsilon_taken, b_loss_pg_epsilon, b_loss_pg_base)
        loss_pg = jnp.mean(b_loss_pg)

        bn_entropy_base = bn_entropy[:, :-1]
        b_entropy_base = jnp.mean(bn_entropy_base, axis=1)
        b_entropy_epsilon = bn_entropy[:, -1]
        b_entropy = jnp.where(b_epsilon_taken, b_entropy_epsilon, b_entropy_base)
        entropy_mean = jnp.mean(b_entropy)
        loss_entropy = -entropy_mean

        loss = loss_pg + self.cfg.entropy_coef * loss_entropy

        info = {
            "Loss": loss,
            "Loss_pg": loss_pg,
            "Entropy": entropy_mean,
            "Approx KL": approx_kl,
            "Clip Frac": clip_fraction,
        }

        # Compute the fraction of maximum entropy, if available, to make it easier to interpret.
        max_entropy = self.env.max_entropy
        if max_entropy is not None:
            info["Entropy Frac"] = entropy_mean / max_entropy

        return loss, info

    def _update_critic(self, network: TrainState, b_data: LCRLData) -> tuple[TrainState, dict]:
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

    def _update_actor(self, network: TrainState, b_data: LCRLData, key: PRNGKeyArray) -> tuple[TrainState, dict]:
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
        # Last one is for the epsilon action.
        assert logp.shape == (self.env.n_agents + 1,)
        return act, logp

    def det_action(self, obs: Any) -> Any:
        """Get the deterministic action (mode) from the policy given an observation."""
        act_dist = self.network.select("actor")(obs)
        act = act_dist.mode()
        return act

    @ft.partial(jax.jit, static_argnums=(2,))
    def collect_batch(self, collector: Collector, rollout_T: int) -> tuple[Collector, RolloutOutput, dict]:
        """Collect a batch of data using stochastic policy."""
        logger.debug("jitting collect_batch...")
        out = collector.collect_batch(self.sample_action, rollout_T, reset_fn=None)
        logger.debug("done jitting collect_batch.")
        return out

    @ft.partial(jax.jit, static_argnames=("rollout_T",))
    def collect_eval_with_states(
        self, collector: Collector, b_state0: Any, rollout_T: int
    ) -> tuple[RolloutOutput, dict]:
        """Collect full eval trajectories using deterministic policy given eval keys."""
        logger.debug("jitting collect_eval_with_states...")
        # Reset all envs before eval
        new_collector = collector.reset_with_state(b_state0)
        _, Tb_rollout, collect_info = new_collector.collect_full_traj_det(self.det_action, rollout_T)
        logger.debug("done jitting collect_eval_with_states.")
        return Tb_rollout, collect_info
