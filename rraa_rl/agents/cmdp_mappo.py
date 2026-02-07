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
from rraa_rl.cmdp_wrapper import CMDPEnvWrapper, CMDPAvoid, CMDPWeakUntil, CMDPReachChain, CMDPFG, CMDPGF, CMDPAugState
from rraa_rl.collector import Collector, RolloutOutput
from rraa_rl.distribution import tfd
from rraa_rl.evaluate_dag import evaluate_dag
from rraa_rl.gae import sum_gae
from rraa_rl.jax_types import FloatScalar, bFloat
from rraa_rl.nn_modules import (BaseObsOnly, MAMultiDiscretePolicy,
                                VDValue, VDValueShared, PositiveConstant)
from rraa_rl.src.env.general_task.env import AugObs, AugObsAutomata
from rraa_rl.train_state import ModuleDict, Params, TrainState
from rraa_rl.train_utils import compute_norm_and_clip, has_any_nan_or_inf, tree_where


@struct.dataclass
class PPOData:
    act: Any
    obs: jnp.ndarray
    logp: jnp.ndarray

    # Rollout advantages and Q-values (GAE'd)
    t_A: bFloat
    t_Q: bFloat

    @property
    def shape(self):
        return self.t_A.shape[:-1]


@Parameter("*", group="AgentConfig")
@define
class CMDPMAPPOAgentCfg(Cfg):
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    lambda_lr: float = 1e-3

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

    thresh_avoid: float = 0.0
    """Avoid constrant: sum_k violations_k >= thresh_avoid"""

    thresh_weak_until: float = 0.0
    """Weak Until constraint: sum_k violations_k >= thresh_weak_until"""

    thresh_reach: float = 0.5
    """Reach constraint: sum_k 1 >= thresh_reach"""

    thresh_fg : float = 0.0
    """FG constraint: sum_k stay >= thresh_fg."""

    thresh_gf: float = -1.0
    """GF constraint: sum_k reach >= thresh_gf. Since reach is always >=0, setting to -1.0 means no constraint."""

@ft.partial(struct.dataclass, frozen=False)
class CMDPMAPPOAgent:
    Cfg = CMDPMAPPOAgentCfg

    network: TrainState
    env: CMDPEnvWrapper = struct.field(pytree_node=False)
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

        lambd_def = PositiveConstant(env.n_conjunctions, init_value=1.0)

        network_info = dict(
            critic=(critic_def, (dummy_obs,)),
            actor=(actor_def, (dummy_obs,)),
            lambd=(lambd_def, ())
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        # For the shared optimizer
        network_tx = optax.multi_transform(
            {
                "actor": optax.adamw(cfg.actor_lr),
                "critic": optax.adamw(cfg.critic_lr),
                "lambd": optax.adamw(cfg.lambda_lr)
            },
            {
                "modules_actor": "actor",
                "modules_critic": "critic",
                "modules_lambd": "lambd"
            },
        )

        network_def = ModuleDict(networks)
        network_params = network_def.init(init_key, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)
        return cls(network=network, env=env, cfg=cfg)

    def compute_A_Q(self, Tb_rollout: RolloutOutput, debug: bool = False):
        """Compute GAE advantages and Q-values from rollout.
        There should be one advantage and Q-value per CMDP computation node.
        """
        T, batch_size = Tb_rollout.shape
        # (batch, T, n_temporal)
        Tbt_V = self.network.select("critic")(Tb_rollout.obs_now, params=self.network.params)
        Tbt_V = jax.lax.stop_gradient(Tbt_V)
        bTt_V = ei.rearrange(Tbt_V, "T b t -> b T t")

        Tbt_V_next = self.network.select("critic")(Tb_rollout.obs_next, params=self.network.params)
        Tbt_V_next = jax.lax.stop_gradient(Tbt_V_next)
        bTt_V_next = ei.rearrange(Tbt_V_next, "T b t -> b T t")

        bT_reachflags_now = {k: ei.rearrange(v, "T b ... -> b T ...") for k, v in Tb_rollout.state_now.reach_flags.items()}
        bT_reachflags_next = {k: ei.rearrange(v, "T b ... -> b T ...") for k, v in Tb_rollout.state_next.reach_flags.items()}

        bT_pred = {k: v.T for k, v in Tb_rollout.predicates_next.items()}

        bT_term = Tb_rollout.term.T
        bT_trunc = Tb_rollout.trunc.T
        # Next step is from a different episode (due to reset) if either terminate or truncate
        bT_next_diff = bT_term | bT_trunc

        bTt_A_list = []
        bTt_Q_list = []

        scratch = {}
        for ii, operation in enumerate(self.env.cmdp_info.operations):
            bT_V = bTt_V[:, :, ii]
            bT_V_next = bTt_V_next[:, :, ii]

            match operation:
                case CMDPAvoid(avoid=avoid_id):
                    # Use q as the "cost".
                    bT_avoid_val = evaluate_dag(self.env.dag_nodes_notrans, avoid_id, bT_pred, scratch)
                    # rew = q<0: q, q>=0: 0.
                    bT_reward = jnp.minimum(bT_avoid_val, 0.0)
                case CMDPWeakUntil(stay=stay_id, reach=reach_id):
                    # Same as Avoid, but if the reachflag is true then force zero cost.
                    bT_stay_val = evaluate_dag(self.env.dag_nodes_notrans, stay_id, bT_pred, scratch)
                    bT_reached = bT_reachflags_next[reach_id]
                    bT_reward = jnp.minimum(bT_avoid_val, 0.0)
                    bT_reward = jnp.where(bT_reached, 0.0, bT_reward)
                case CMDPReachChain(reach=reach_id, condition=condition_ids):
                    # If reachflag goes from 0 to 1, give a reward of +1.
                    bT_reached_prev = bT_reachflags_now[reach_id]
                    bT_reached_next = bT_reachflags_next[reach_id]
                    bT_into_reach = (~bT_reached_prev) & (bT_reached_next)
                    bT_reward = jnp.where(bT_into_reach, 1.0, 0.0)
                case CMDPFG(stay=stay_id):
                    # Same as Avoid, but if we haven't made the epsilon move then give it zero cost.
                    bT_stay_val = evaluate_dag(self.env.dag_nodes_notrans, stay_id, bT_pred, scratch)
                    raise NotImplementedError("")
                case CMDPGF(reach=reach_id):
                    # This will be the objective. Sum up if r>0.
                    bT_reach_val = evaluate_dag(self.env.dag_nodes_notrans, reach_id, bT_pred, scratch)
                    bT_reward = jnp.maximum(bT_reach_val, 0.0)
                case _:
                    raise ValueError(f"Unknown CMDP operation {operation}")

            # Compute GAE advantages and Q-values for this operation.
            gamma, gae_lambda = self.cfg.gamma, self.cfg.gae_lambda
            gae_fn = ft.partial(sum_gae, gamma=gamma, gae_lambda=gae_lambda)
            bT_A, bT_Q = jax.vmap(gae_fn)(bT_V, bT_V_next, bT_term, bT_next_diff, bT_reward)

            bTt_A_list.append(bT_A)
            bTt_Q_list.append(bT_Q)

        bTt_A = jnp.stack(bTt_A_list, axis=-1)
        bTt_Q = jnp.stack(bTt_Q_list, axis=-1)

        assert bTt_A.shape == (batch_size, T, self.env.n_conjunctions)

        return bTt_A, bTt_Q

    def get_Tb_data(self, Tb_rollout: RolloutOutput) -> PPOData:
        bTt_A, bTt_Q = self.compute_A_Q(Tb_rollout)
        Tbt_A = ei.rearrange(bTt_A, "b T t -> T b t")
        Tbt_Q = ei.rearrange(bTt_Q, "b T t -> T b t")

        Tb_data = PPOData(
            act=Tb_rollout.act,
            obs=Tb_rollout.obs_now,
            logp=Tb_rollout.logprob,
            t_A=Tbt_A,
            t_Q=Tbt_Q,
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
        network, info_lambd = self._update_lambd(network, b_data)
        info_critic = {f"critic/{k}": v for k, v in info_critic.items()}
        info_actor = {f"actor/{k}": v for k, v in info_actor.items()}
        info_lambd = {f"lambd/{k}": v for k, v in info_lambd.items()}
        info = info_critic | info_actor | info_lambd
        return network, info

    def _critic_loss(self, b_data: PPOData, params: Params):
        """Compute MSE loss for the value function."""
        bt_V = self.network.select("critic")(b_data.obs, params=params)
        loss = jnp.mean((b_data.t_Q - bt_V) ** 2)

        explained_variance = 1 - jnp.var(b_data.t_Q - bt_V) / (jnp.var(b_data.t_Q.flatten()) + 1e-8)

        info = {
            "Loss": loss,
            "V_mean": jnp.mean(bt_V),
            "Q_mean": jnp.mean(b_data.t_Q),
            "Explained Variance": explained_variance,
        }
        return loss, info

    def _actor_loss(self, b_data: PPOData, params: Params, key: PRNGKeyArray):
        """Compute the actor loss. We are trying to maximize reward."""
        b_obs = b_data.obs
        b_act = b_data.act
        batch_size, _ = b_data.t_Q.shape

        # (batch, n_agents)
        bn_logp_old = b_data.logp

        # Compute logprob and entropies.
        b_key = jr.split(key, batch_size)
        bn_logp, bn_entropy = jax.vmap(ft.partial(self.pol_logp_entropy, params=params))(b_obs, b_act, b_key)

        # Compute ratios.
        bn_logratio = bn_logp - bn_logp_old
        bn_is_ratio = jnp.exp(bn_logratio)
        approx_kl = ((bn_is_ratio - 1) - bn_logratio).mean()

        t_lambda = jax.lax.stop_gradient(self.network.select("lambd")(params=params))

        # Weighted sum with t_lambda.
        bt_A = b_data.t_A
        b_A = jnp.sum(bt_A * t_lambda, axis=-1)
        assert b_A.shape == (batch_size,)

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

        b_loss_pg = jnp.mean(bn_loss, axis=1)
        loss_pg = jnp.mean(b_loss_pg)

        b_entropy = jnp.mean(bn_entropy, axis=1)
        entropy_mean = jnp.mean(b_entropy)
        loss_entropy = -entropy_mean

        logprob_mean = jnp.mean(bn_logp)

        loss = loss_pg + self.cfg.entropy_coef * loss_entropy

        info = {
            "Loss": loss,
            "Loss_pg": loss_pg,
            "Entropy": entropy_mean,
            "Mean logprob": logprob_mean,
            "Approx KL": approx_kl,
            "Clip Frac": clip_fraction,
        }

        # Compute the fraction of maximum entropy, if available, to make it easier to interpret.
        max_entropy = self.env.max_entropy
        if max_entropy is not None:
            info["Entropy Frac"] = entropy_mean / max_entropy

        return loss, info

    def _lambd_loss(self, b_data: PPOData, params: Params):
        bt_Q = b_data.t_Q

        t_lambd = self.network.select("lambd")(params=params)

        bt_loss_list = []

        for ii, op in enumerate(self.env.cmdp_info.operations):
            b_Q = bt_Q[:, ii]
            # Constraint: Q >= thresh, so penalize if Q < thresh  <=>  thresh - Q > 0
            match op:
                case CMDPAvoid(avoid=avoid_id):
                    b_loss = jnp.maximum(0.0, self.cfg.thresh_avoid - b_Q)
                case CMDPWeakUntil(stay=stay_id, reach=reach_id):
                    b_loss = jnp.maximum(0.0, self.cfg.thresh_weak_until - b_Q)
                case CMDPReachChain(reach=reach_id, condition=condition_ids):
                    b_loss = jnp.maximum(0.0, self.cfg.thresh_reach - b_Q)
                case CMDPFG(stay=stay_id):
                    b_loss = jnp.maximum(0.0, self.cfg.thresh_fg - b_Q)
                case CMDPGF(reach=reach_id):
                    b_loss = jnp.maximum(0.0, self.cfg.thresh_gf - b_Q)
                case _:
                    raise ValueError(f"Unknown CMDP operation {op}")
            bt_loss_list.append(b_loss)
        bt_loss = jnp.stack(bt_loss_list, axis=-1)
        b_loss = jnp.sum(bt_loss * t_lambd, axis=-1)
        loss = jnp.mean(b_loss)
        print("loss.dtype: ", loss.dtype)

        bt_violated = jnp.where(bt_loss > 0.0, 1, 0)
        b_all_satisfy = jnp.all(bt_violated == 0, axis=1)

        info = {"Loss": loss, "all satisfy": jnp.mean(b_all_satisfy), "violate prob": jnp.mean(bt_violated)}
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

    def _update_lambd(self, network: TrainState, b_data: PPOData) -> tuple[TrainState, dict]:
        lambd_loss = ft.partial(self._lambd_loss, b_data)
        grad, info = jax.grad(lambd_loss, has_aux=True)(network.params)

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
        if self.env.cmdp_info.has_epsilon_move:
            assert logp.shape == (self.env.n_agents + 1,)
        else:
            assert logp.shape == (self.env.n_agents,)

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
