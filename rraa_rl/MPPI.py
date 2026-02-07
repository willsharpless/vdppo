import jax
import jax.random as jr
import jax.numpy as jnp
import jax.tree_util as jtu
import jax_dataclasses as jdc
from jaxtyping import PRNGKeyArray
import jax.lax as lax
from flax import struct

import numpy as np
import yaml
from attrs import define
from cyclopts import Parameter
from loguru import logger
from typing import Any, Protocol
from typing_extensions import Self

from rraa_rl.cfg_utils import Cfg
from rraa_rl.collector import Collector
from rraa_rl.rollout_temporal_analysis import evaluate_ltl_finite
from rraa_rl.rollout_utils import extract_rollouts_eval
from rraa_rl.run import Run
from rraa_rl.src.env.general_task.env import Env, EnvStep, StateWithTemporalNode
from rraa_rl.agents.vd_mappo import VDMAPPOAgent
from rraa_rl.jax_utils import switch01
from rraa_rl.trainer import CallbackProps
from rraa_rl.src.env.general_task.get_env import get_env_and_cbs

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

# @define
# class CallbackProps:
#     run: Run

#     train_step: int
#     agent: VDMAPPOAgent
#     bT_test_rollouts: list[RolloutOutput]
#     bT_test_rollout: RolloutOutput
#     test_trigger_dict: dict[tuple[str, str], np.ndarray]
#     temporal_values_dict: dict[int, np.ndarray]

#     collector_train: Collector
#     Tb_rollout: RolloutOutput
#     info_update: dict

#     @property
#     def env(self):
#         return self.agent.env


class Callback(Protocol):
    def __call__(self, p: CallbackProps) -> None: ...

@Parameter("*", group="MPPI")
@define
class MPPICfg(Cfg):
    mode: str = "planned"  # "rhc" or "planned"

    eval_every: int = 5_000
    log_every: int = 100
    save_every: int = 5_000

    n_envs: int = 128

    # Works for one-shot or rhc, F target_dense0 & G(!obstacles) & G(!oob)
    mppi_samples: int = 1_000
    mppi_horizon_H: int = 100
    mppi_iterations: int = 20

    mppi_noise_sd_init: float = 10.
    mppi_lambda_init: float = 50.
    mppi_shrink_factor: float = 0.6

class MPPI:
    Cfg = MPPICfg
    env: Env
    key: PRNGKeyArray

    def __init__(self, env: Env, cfg: MPPICfg, key: PRNGKeyArray=jr.PRNGKey(0)):
        self.cfg = cfg
        self.env = env
        self.b_state0 = None
        self.key = key

    def reset_with_state(self, b_state: Any) -> Self:
        with jdc.copy_and_mutate(self) as self_new:
            b_state = b_state
            b_obs = jax.vmap(self.env.get_obs)(b_state)

        return self_new

    def step_single_fn_mppi_rhc(self, state: Any, control_guess: Any, key:jr.PRNGKey):
        control, updated_control_guess = self.mppi_policy(state, control_guess, key)
        step_result = self.env.step_control(state, control)
        return step_result, control, jnp.roll(updated_control_guess, shift=-1, axis=0)
    
    def step_single_fn_mppi_planned(self, state: Any, control_plan: Any, key:jr.PRNGKey):
        step_result = self.env.step_control(state, control_plan[0, ...])
        return step_result, control_plan[0, ...], jnp.roll(control_plan, shift=-1, axis=0)

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

            # Compute weights (DOESNT WORK)
            # costs = 1. * K_robustness[0]
            # costs = K_robustness[0]
            # weights = jnp.exp(-costs / lambda_curr)
            # weights = jnp.exp(-(costs - costs.min(axis=0)) / mppi_lambda_curr) # TODO test: advantage-based, best (Althoff)
            # weights = jnp.exp(-(costs - jnp.mean(costs, axis=0)) / mppi_lambda_curr) # TODO test: advantage-based, mean
            # weights = weights / jnp.sum(weights)
            # updated_control_guess = control_guess_traj + jnp.sum(weights[:, None, None, None] * KHNA_sample_control_perturbation, axis=0)

            # Take the best sample
            updated_control_guess = KHNA_perturbed_control_traj[jnp.argmax(K_robustness[self.env.dag_root]), ...]  # Take the best sample

            # Bound updated control guess
            updated_control_guess = jnp.clip(updated_control_guess, jnp.array(self.env.base.control_lim_lo), jnp.array(self.env.base.control_lim_hi))

            # Anneal lambda and noise sd
            lambda_curr = lambda_curr * self.cfg.mppi_shrink_factor
            noise_sd_curr = noise_sd_curr * self.cfg.mppi_shrink_factor

            return (new_key, updated_control_guess, noise_sd_curr, lambda_curr), None

        carry0 = (key, control_guess_traj, self.cfg.mppi_noise_sd_init, self.cfg.mppi_lambda_init)
        (_, updated_control_guess, _, _), _ = lax.scan(mppi_iter, carry0, xs=None, length=self.cfg.mppi_iterations)

        control = updated_control_guess[0, ...]  # Take the first control in the trajectory
        return control, updated_control_guess

    def collect_full_traj(
        self,
        b_state_0: Any,
        T: int,
        key_eval: jr.PRNGKey,
    ) -> tuple[Self, RolloutOutput, dict]:
        
        # Compute Receding Horizon Style MPPI 
        if self.cfg.mode == "rhc":
            step_fn = self.step_single_fn_mppi_rhc
            control_traj_0 = jnp.zeros((self.cfg.n_envs, self.cfg.mppi_horizon_H, self.env.n_agents, self.env.base.action_dim))

        # One-shot MPPI Trajectory Optimization
        elif self.cfg.mode == "planned":
            step_fn = self.step_single_fn_mppi_planned

            # Plan MPPI control traj
            control_guess_0 = jnp.zeros((self.cfg.n_envs, self.cfg.mppi_horizon_H, self.env.n_agents, self.env.base.action_dim))
            key, new_key = jr.split(key_eval)
            b_keys = jr.split(key, self.cfg.n_envs)
            _, control_traj_0 = jax.vmap(self.mppi_policy)(b_state_0, control_guess_0, b_keys)

        else:
            raise ValueError(f"Unknown MPPI mode: {self.cfg.mode}")

        def rollout(carry, _):
            (b_state, b_control_guess_traj, key) = carry
            key, new_key = jr.split(key)
            b_keys = jr.split(key, self.cfg.n_envs)

            b_obs = jax.vmap(self.env.get_obs)(b_state)
            b_step_result, b_act, b_updated_control_guess_traj = jax.vmap(step_fn)(b_state, b_control_guess_traj, b_keys)
            out = RolloutOutput.from_rollout(b_state, b_obs, b_step_result, b_act)

            carry_new = (b_step_result.envstate, b_updated_control_guess_traj, new_key)
            return carry_new, out
        
        # TODO could also try random
        carry0 = (b_state_0, control_traj_0, key_eval)
        carry_out, Tb_rollout = lax.scan(rollout, carry0, xs=None, length=T)

        return Tb_rollout

    ## DEBUG versions (looping instead of scanning for plots)

    def step_single_fn_mppi_rhc_debug(self, state: Any, control_guess: Any, key:jr.PRNGKey):
        control, updated_control_guess, JHK_rollout_imag = self.mppi_policy_debug(
            state, control_guess, key
        )
        step_result = self.env.step_control(state, control)
        return step_result, control, updated_control_guess, JHK_rollout_imag

    def mppi_policy_debug(self, state:Any, control_guess_traj: Any, key:jr.PRNGKey) -> Any:
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

        def mppi_iter_body(key, control_guess_traj, noise_sd_curr, lambda_curr):
            # Sample K control perturbations
            key, new_key = jr.split(key)
            KHNA_sample_control_perturbation = jax.random.normal(key, (self.cfg.mppi_samples, *control_guess_traj.shape)) * noise_sd_curr
            KHNA_control_guess_traj_repeated = jnp.broadcast_to(control_guess_traj, (self.cfg.mppi_samples, *control_guess_traj.shape))
            KHNA_perturbed_control_traj = KHNA_control_guess_traj_repeated + KHNA_sample_control_perturbation

            # Bound sampled control inputs
            KHNA_perturbed_control_traj = jnp.clip(KHNA_perturbed_control_traj, jnp.array(self.env.base.control_lim_lo), jnp.array(self.env.base.control_lim_hi))

            def expand_state_for_samples(b_state: StateWithTemporalNode, k_samples: int) -> StateWithTemporalNode:
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

            # # Compute weights
            # costs = -1. * K_robustness[1]
            # weights = jnp.exp(-costs / lambda_curr)
            # # weights = jnp.exp(-(costs - costs.min(axis=0)) / mppi_lambda_curr) # TODO test: advantage-based, best (Althoff)
            # # weights = jnp.exp(-(costs - jnp.mean(costs, axis=0)) / mppi_lambda_curr) # TODO test: advantage-based, mean
            # weights = weights / jnp.sum(weights)
            # updated_control_guess = control_guess_traj + jnp.sum(weights[:, None, None, None] * KHNA_sample_control_perturbation, axis=0)

            # Take the best sample
            top_node = jnp.array(list(K_robustness.keys())).max().item()
            updated_control_guess = KHNA_perturbed_control_traj[jnp.argmax(K_robustness[top_node]), ...]  # Take the best sample

            # Bound updated control guess
            updated_control_guess = jnp.clip(updated_control_guess, jnp.array(self.env.base.control_lim_lo), jnp.array(self.env.base.control_lim_hi))

            # Anneal lambda and noise sd
            lambda_curr = lambda_curr * self.cfg.mppi_shrink_factor
            noise_sd_curr = noise_sd_curr * self.cfg.mppi_shrink_factor

            return new_key, updated_control_guess, noise_sd_curr, lambda_curr, KH_rollout_imag

        # Python for loop instead of scan - outputs moved to device between iterations
        noise_sd_curr = self.cfg.mppi_noise_sd_init
        lambda_curr = self.cfg.mppi_lambda_init
        rollout_imags = []
        
        for _ in range(self.cfg.mppi_iterations):
            print("    mppi iter", _)
            key, control_guess_traj, noise_sd_curr, lambda_curr, KH_rollout_imag = mppi_iter_body(
                key, control_guess_traj, noise_sd_curr, lambda_curr
            )
            # Move to CPU immediately to free GPU memory
            rollout_imags.append(jax.device_get(KH_rollout_imag))

        # Stack the collected rollouts: list of KH -> JKH
        JKH_rollout_imag = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *rollout_imags)

        control = control_guess_traj[0, ...]
        updated_control_guess_shifted = jnp.roll(control_guess_traj, shift=-1, axis=0)
        return control, updated_control_guess_shifted, JKH_rollout_imag

    def collect_full_traj_debug(
        self,
        b_state_0: Any,
        T: int,
        n_envs: int,
        key_eval: jr.PRNGKey,
    ) -> tuple[Self, RolloutOutput, dict]:

        control_guess_0 = jnp.zeros((n_envs, self.cfg.mppi_horizon_H, self.env.n_agents, self.env.base.action_dim))

        b_state = b_state_0
        b_control_guess_traj = control_guess_0
        key = key_eval
        
        rollout_outputs = []
        mppi_rollouts = []
        
        for _ in range(T):
            print("iter", _)
            key, new_key = jr.split(key)
            b_keys = jr.split(key, n_envs)

            b_obs = jax.vmap(self.env.get_obs)(b_state)
            b_step_result, b_act, b_updated_control_guess_traj, BJHK_rollout_imag = jax.vmap(self.step_single_fn_mppi_rhc_debug)(b_state, b_control_guess_traj, b_keys)
            
            rollout_out = RolloutOutput.from_rollout(b_state, b_obs, b_step_result, b_act)
            
            # Move to CPU immediately
            rollout_outputs.append(jax.device_get(rollout_out))
            mppi_rollouts.append(jax.device_get(BJHK_rollout_imag))
            
            b_state = b_step_result.envstate
            b_control_guess_traj = b_updated_control_guess_traj
            key = new_key

        # Stack: list of b -> Tb
        Tb_rollout = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *rollout_outputs)
        TBJKH_rollout_mppi = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *mppi_rollouts)

        return Tb_rollout, TBJKH_rollout_mppi

    def eval(
        self, 
        run: Run, 
        key_eval: jr.PRNGKey,
        eval_cbs: list[Callback] = None,
        debug: bool = False,
    ):
        env = self.env
        eval_T = 3 if debug else env.eval_T
        eval_n_envs = 1 if debug else self.cfg.n_envs

        if self.b_state0 is None:
            self.b_state0 = env.get_eval_states(eval_n_envs)

        mppi_rollouts = None
        if debug:
            Tb_rollout, mppi_rollouts = self.collect_full_traj_debug(
                self.b_state0, eval_T, eval_n_envs, key_eval
            )
            mppi_rollouts = jax.device_get(mppi_rollouts)
            
        else:
            Tb_rollout = self.collect_full_traj(
                self.b_state0, eval_T, key_eval
            )
            
        Tb_rollout = jax.device_get(Tb_rollout)
        bT_rollout = Tb_rollout.switch01()

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

        agent_cfg = VDMAPPOAgent.Cfg()
        # agent_cfg = get_vd_agent_cfg("Delivery")
        # dummy_agent = VDMAPPOAgent.create(0, agent_cfg, env)
        class DummyAgent:
            def __init__(self, agent_cfg, env):
                self.seed = 0
                self.cfg = agent_cfg
                self.env = env
        dummy_agent = DummyAgent(agent_cfg, env)

        cb_props = CallbackProps(run, -1, dummy_agent, None, None, None, None, None, None, None)
        cb_props.train_step = 0
        cb_props.agent = dummy_agent
        cb_props.bT_test_rollouts = trajs
        cb_props.bT_test_rollout = bT_rollout
        cb_props.test_trigger_dict = trigger_dict
        cb_props.temporal_values_dict = temporal_node_values
        cb_props.collector_train = None

        if debug:  # for plotting MPPI its
            cb_props.mppi_rollouts = mppi_rollouts

        for cb in eval_cbs:
            cb_name = cb.__name__ if hasattr(cb, "__name__") else str(type(cb))
            print(f"Running eval callback {cb_name}")
            cb(cb_props)

    def collect_eval_with_states(
        self, 
        collector: Collector,
        b_state0: Any, 
        rollout_T: int,
        temporal_transitions: bool = False,
        debug: bool = False,
    ):
        key_eval = self.key # self.key should also work
        env = self.env
        rollout_T = 3 if debug else rollout_T
        eval_n_envs = 1 if debug else self.cfg.n_envs

        if b_state0 is None:
            b_state0 = env.get_eval_states(eval_n_envs)

        mppi_rollouts = None
        if debug:
            Tb_rollout, mppi_rollouts = self.collect_full_traj_debug(
                b_state0, rollout_T, eval_n_envs, key_eval
            )
            mppi_rollouts = jax.device_get(mppi_rollouts)
            
        else:
            Tb_rollout = self.collect_full_traj(
                b_state0, rollout_T, key_eval
            )
        return Tb_rollout, mppi_rollouts
    
def init_mppi(env_name:str, seed:jr.PRNGKey=jr.PRNGKey(0), n_spec:int=1, n_agent:int=1):
    env, _, _ = get_env_and_cbs(env_name, agent_name='mppi', n_spec=n_spec, n_agent=n_agent, dense=True)
    run = Run.create(env_name=env_name, name=f"mppi_{env_name}", agent_name="MPPI")
    agent = MPPI(env=env, cfg=MPPI.Cfg(), key=seed)
    return run, agent, env, {}