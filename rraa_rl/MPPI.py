import pickle
from typing import Protocol

import jax
import jax.random as jr
import jax.numpy as jnp
import jax.tree_util as jtu
import jax_dataclasses as jdc
from jaxtyping import Bool, Float, PRNGKeyArray
import jax.lax as lax
from flax import struct

import numpy as np
import tqdm
import yaml
from attrs import define
from cyclopts import Parameter
from loguru import logger
from typing import Any, Callable, Protocol, Tuple
from typing_extensions import Self

import wandb
from rraa_rl.cfg_utils import Cfg
from rraa_rl.collector import Collector, extract_info_from_rollout
from rraa_rl.rollout_temporal_analysis import evaluate_ltl_finite
from rraa_rl.rollout_utils import extract_rollouts_eval
from rraa_rl.run import Run
from rraa_rl.src.env.general_task.env import Env, EnvStep, StateWithTemporalNode
from rraa_rl.vd_mappo import VDMAPPOAgent
from rraa_rl.jax_utils import switch01, tree_where_dim0

@struct.dataclass
class RolloutOutput:
    state_now: Any
    state_next: Any
    obs_now: Any
    obs_next: Any
    act: jnp.ndarray

    predicates_next: dict

    term: jnp.ndarray
    """Termination flags after taking action."""

    trunc: jnp.ndarray
    """Truncation flags after taking action."""

    # logprob: jnp.ndarray
    # """Log probabilities of the actions taken."""

    info: dict
    """Additional info from the environment."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Get n_envs and n_steps."""
        return self.term.shape

    def switch01(self) -> "RolloutOutput":
        return jtu.tree_map(switch01, self)

    @staticmethod
    def from_rollout(
        b_state: Any,
        b_obs: Any,
        b_step_result: EnvStep,
        b_act: jnp.ndarray,
    ) -> "RolloutOutput":
        return RolloutOutput(
            state_now=b_state,
            state_next=b_step_result.envstate,
            obs_now=b_obs,
            obs_next=b_step_result.obs,
            act=b_act,
            predicates_next=b_step_result.predicates,
            term=b_step_result.term,
            trunc=b_step_result.trunc,
            # logprob=1., # NOTE DUMMY
            info=b_step_result.info,
        )

    @property
    def temporal_node_idx(self):
        assert hasattr(self.state_now, "temporal_node_idx")
        return self.state_now.temporal_node_idx

@define
class CallbackProps:
    run: Run

    train_step: int
    agent: VDMAPPOAgent
    bT_test_rollouts: list[RolloutOutput]
    bT_test_rollout: RolloutOutput
    test_trigger_dict: dict[tuple[str, str], np.ndarray]
    temporal_values_dict: dict[int, np.ndarray]

    collector_train: Collector
    Tb_rollout: RolloutOutput
    info_update: dict

    @property
    def env(self):
        return self.agent.env


class Callback(Protocol):
    def __call__(self, p: CallbackProps) -> None: ...

@Parameter("*", group="MPPI")
@define
class MPPICfg(Cfg):
    eval_every: int = 5_000
    log_every: int = 100
    save_every: int = 5_000

    n_envs: int = 128

    # mppi_samples: int = 1_000
    # mppi_horizon_H: int = 10
    # mppi_iterations: int = 50

    mppi_samples: int = 2
    mppi_horizon_H: int = 2
    mppi_iterations: int = 2

    mppi_noise_sd_init: float = 0.1
    mppi_lambda_init: float = 50.
    mppi_shrink_factor: float = 0.5

class MPPI:
    Cfg = MPPICfg
    env: Env
    key: PRNGKeyArray

    def __init__(self, env: Env, cfg: MPPICfg, key: PRNGKeyArray):
        self.cfg = cfg
        self.env = env
        self.b_state0 = None
        self.key = key

    def reset_with_state(self, b_state: Any) -> Self:
        with jdc.copy_and_mutate(self) as self_new:
            b_state = b_state
            b_obs = jax.vmap(self.env.get_obs)(b_state)

        return self_new

    def step_single_fn_mppi(self, state: Any, control_guess: Any, key:jr.PRNGKey):
        control, updated_control_guess, JHK_rollout_imag = self.mppi_policy(
            state, control_guess, key
        )
        step_result = self.env.step_control(state, control)
        return step_result, control, updated_control_guess, JHK_rollout_imag
    
    def step_single_fn(self, state: Any, control: Any):
        step_result = self.env.step_control(state, control)
        return step_result, control

    def mppi_policy(self, state:Any, control_guess_traj: Any, key:jr.PRNGKey) -> Any:
        """MPPI control policy: 
        iter:
            1. sample noise, add to control guess
            2. roll out
            3. compute cost (neg robustness score)
            4. compute weights, w_k = exp(-S^(k) / lambda)
            5. update control_guess_traj with weighted noise average
            6. shrink 
        out: use first control, shift control_guess_traj
        """

        # TODO, a dumb discrete action version, which just does iterative sampling no reweighting, is seriously worth a try
        def mppi_iter(carry, _):
            key, control_guess_traj, noise_sd_curr, lambda_curr = carry

            # Sample K control perturbations
            key, new_key = jr.split(key)
            KHNA_sample_control_perturbation = jax.random.normal(key, (self.cfg.mppi_samples, *control_guess_traj.shape)) * noise_sd_curr
            KHNA_control_guess_traj_repeated = jnp.broadcast_to(control_guess_traj, (self.cfg.mppi_samples, *control_guess_traj.shape))
            KHNA_perturbed_control_traj = KHNA_control_guess_traj_repeated + KHNA_sample_control_perturbation

            # Bound sampled control inputs, self.env.cfg.acc_maxs
            KHNA_perturbed_control_traj = jnp.clip(KHNA_perturbed_control_traj, jnp.array(self.env.base.control_lim_lo), jnp.array(self.env.base.control_lim_hi))

            # vmap over K samples
            # state is replicated K times, perturbed_control_traj: (K, H, action_dim)
            def expand_state_for_samples(b_state: StateWithTemporalNode, k_samples: int) -> StateWithTemporalNode:
                """Expand a batched state (batch_size,) to (batch_size, k_samples) for MPPI."""
                return jtu.tree_map(
                    lambda x: jnp.broadcast_to(x[None, ...], (k_samples, *x.shape)),
                    b_state
                )
            K_state_0 = expand_state_for_samples(state, self.cfg.mppi_samples)

            # rollout the k batch
            def imag_rollout(hk_state, hk_control):
                hk_step_result, _ = jax.vmap(self.step_single_fn)(hk_state, hk_control)
                hk_obs = jax.vmap(self.env.get_obs)(hk_state)
                out = RolloutOutput.from_rollout(hk_state, hk_obs, hk_step_result, hk_control)
                return hk_step_result.envstate, out

            HKNA_perturbed_control_traj = jnp.swapaxes(KHNA_perturbed_control_traj, 0, 1)
            _, HK_rollout_imag = lax.scan(imag_rollout, K_state_0, HKNA_perturbed_control_traj)

            # Compute costs as negative robustness score
            KH_rollout_imag = HK_rollout_imag.switch01()
            K_robustness = jax.vmap(
                lambda predicates_next: evaluate_ltl_finite(self.env, predicates_next, which=jnp)
            )(KH_rollout_imag.predicates_next)
            costs = -1 * K_robustness[0] # TODOD FIXME which node, first or last?

            # Compute weights
            weights = jnp.exp(-costs / lambda_curr)
            # weights = jnp.exp(-(costs - costs.min(axis=0)) / mppi_lambda_curr) # TODO test: advantage-based, best (Althoff)
            # weights = jnp.exp(-(costs - jnp.mean(costs, axis=0)) / mppi_lambda_curr) # TODO test: advantage-based, mean
            weights = weights / jnp.sum(weights)

            updated_control_guess = control_guess_traj + jnp.sum(weights[:, None, None, None] * KHNA_perturbed_control_traj, axis=0)

            # Bound updated control guess
            updated_control_guess = jnp.clip(updated_control_guess, jnp.array(self.env.base.control_lim_lo), jnp.array(self.env.base.control_lim_hi))

            # Anneal lambda and noise sd
            lambda_curr = lambda_curr * self.cfg.mppi_shrink_factor
            noise_sd_curr = noise_sd_curr * self.cfg.mppi_shrink_factor

            return (new_key, updated_control_guess, noise_sd_curr, lambda_curr), KH_rollout_imag

        carry0 = (key, control_guess_traj, self.cfg.mppi_noise_sd_init, self.cfg.mppi_lambda_init)
        (_, updated_control_guess, _, _), JKH_rollout_imag = lax.scan(mppi_iter, carry0, xs=None, length=self.cfg.mppi_iterations)

        control = updated_control_guess[0, ...]  # Take the first control in the trajectory
        updated_control_guess_shifted = jnp.roll(updated_control_guess, shift=-1, axis=0)
        return control, updated_control_guess_shifted, JKH_rollout_imag

    def collect_full_traj(
        self,
        b_state_0: Any,
        T: int,
        key_eval: jr.PRNGKey,
    ) -> tuple[Self, RolloutOutput, dict]:
        
        def rollout(carry, _):
            (b_state, b_control_guess_traj, key) = carry
            key, new_key = jr.split(key)
            b_keys = jr.split(key, self.cfg.n_envs)

            b_obs = jax.vmap(self.env.get_obs)(b_state)
            b_step_result, b_act, b_updated_control_guess_traj, BJHK_rollout_imag = jax.vmap(self.step_single_fn_mppi)(b_state, b_control_guess_traj, b_keys)
            out = (RolloutOutput.from_rollout(b_state, b_obs, b_step_result, b_act), BJHK_rollout_imag)

            carry_new = (b_step_result.envstate, b_updated_control_guess_traj, new_key)
            return carry_new, out

        control_guess_0 = jnp.zeros((self.cfg.n_envs, self.cfg.mppi_horizon_H, self.env.n_agents, self.env.base.action_dim))
        # TODO could also try random
        carry0 = (b_state_0, control_guess_0, key_eval)
        carry_out, both_Tb_rollouts = lax.scan(rollout, carry0, xs=None, length=T)

        return both_Tb_rollouts

    def eval(
        self, 
        run: Run, 
        key_eval: jr.PRNGKey,
        eval_cbs: list[Callback] = None
    ):
        env = self.env

        if self.b_state0 is None:
            self.b_state0 = env.get_eval_states(self.cfg.n_envs)

        both_Tb_rollouts = self.collect_full_traj(
            self.b_state0, env.eval_T, key_eval
        )

        # TBJKH_rollout_imag = both_Tb_rollouts[1], but # TODO FIXME need to use tree_index or smth
        Tb_rollout, TBJKH_rollout_mppi = both_Tb_rollouts

        Tb_rollout = jax.device_get(Tb_rollout)
        bT_rollout = Tb_rollout.switch01()

        TBJKH_rollout_mppi = jax.device_get(TBJKH_rollout_mppi)

        # Extract each rollout
        trajs = extract_rollouts_eval(bT_rollout)

        # Evaluate the LTL satisfaction over each trajectory.
        temporal_node_values_l: dict[int, list[float]] = {}
        for traj in trajs:
            T_temporal_node_idx: np.ndarray = traj.temporal_node_idx
            temporal_node_idx = T_temporal_node_idx[0]
            dag_node_idx = env.temporal_nodes[temporal_node_idx]
            dag_value = evaluate_ltl_finite(env, traj.predicates_next, which=np)[dag_node_idx]

            temporal_node_value = temporal_node_values_l.get(temporal_node_idx, [])
            temporal_node_value.append(dag_value)
            temporal_node_values_l[temporal_node_idx] = temporal_node_value
        temporal_node_values: dict[int, np.ndarray] = {k: np.array(v) for k, v in temporal_node_values_l.items()}

        # Compute the average satisfaction rate for each temporal node
        info_satisfaction = {}
        for temporal_node_idx, temporal_node_value in temporal_node_values.items():
            node_name = env.temporal_node_names[temporal_node_idx]
            # Satisfy if positive.
            info_satisfaction[f"Eval/Satisfy/{node_name}"] = float(np.mean(temporal_node_value > 0.1))

        # # Evaluate the satisfaction of each temporal node.
        # trigger_dict = evaluate_triggers(env, trajs)
        # # Compute the average satisfaction rate for each trigger
        # info_trigger = {f"Eval/Triggers/{k[0]}->{k[1]}": float(np.mean(v)) for k, v in trigger_dict.items()}
        trigger_dict = {}
        info = info_satisfaction

        # info = info_trigger | info_satisfaction

        info["debug/temporal_values_dict"] = temporal_node_values

        # Output

        # Save configs.
        cfg_to_save = {
            "mppi": self.cfg.asdict(),
        }
        # Save as yaml.
        yaml_path = run.run_dir / "config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(cfg_to_save, f)
        logger.success("Saved config to {}".format(yaml_path))

        ag_cfg = VDMAPPOAgent.Cfg()
        dummy_agent = VDMAPPOAgent.create(0, ag_cfg, env)

        cb_props = CallbackProps(run, -1, dummy_agent, None, None, None, None, None, None, None)
        cb_props.train_step = 0
        cb_props.agent = dummy_agent
        cb_props.bT_test_rollouts = trajs
        cb_props.bT_test_rollout = bT_rollout
        cb_props.test_trigger_dict = trigger_dict
        cb_props.temporal_values_dict = temporal_node_values
        cb_props.collector_train = None

        # cb_props.TBJKH_rollout_mppi = TBJKH_rollout_mppi # for plotting MPPI its

        for cb in eval_cbs:
            cb_name = cb.__name__ if hasattr(cb, "__name__") else str(type(cb))
            print(f"Running eval callback {cb_name}")
            cb(cb_props)

        # can export Tb_rollout_imagss for debug too
