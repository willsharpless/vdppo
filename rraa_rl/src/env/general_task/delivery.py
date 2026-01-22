import jax.numpy as jnp
import jax.random as jr
import jax_dataclasses as jdc
from attrs import define
from jaxtyping import PRNGKeyArray
from loguru import logger

from rraa_rl.jax_types import BoolScalar
from rraa_rl.src.env.general_task.delivery_base import DeliveryBase, DeliveryBaseCfg, DeliveryBaseState
from rraa_rl.src.env.general_task.env import (EnvCfg, EnvUsingBase, StateWithTemporalNode, StaticTemporalNodeMixin,
                                              StaticTemporalNodeMixinCfg)


@define(slots=False)
class DeliveryCfg(EnvCfg, StaticTemporalNodeMixinCfg):
    # specification: str = "F target0 && F target1 && G(!oob) && G(!obstacles) && (!collide)"
    specification: str = "F target0 && F target1 && G(!obstacles)"
    # specification: str = "F G herd_herded"
    base: DeliveryBaseCfg = DeliveryBaseCfg()


class Delivery(StaticTemporalNodeMixin, EnvUsingBase):
    """
    Delivery env -- made from herd env (eg. num agents = n_herders) to use same callbacks/plotting/utils. Agents move

    Also, in case "dummy" agents (herded) are desired (moving obstacles). Otherwise, just a multi-agent env designed for multi-reach-avoiding.

    Predicates include:
        - reaching targets (delivery locs)
        - avoiding obstacles (city)

    Additionally, one may instantiate a 'base' agent, which is slower agent which the other agents may need to revisit.

    In the discrete action setup, each agent is a double-integrator that can accelerate / decelerate in either axis. Herd agents are single-integrators with built in policies.
    action: (n_herders, 2): int, {0, 1, 2} for each axis, where 0 = -accel, 1 = no accel, 2 = accel
    """

    Cfg = DeliveryCfg
    State = StateWithTemporalNode[DeliveryBaseState]

    def __init__(self, cfg: DeliveryCfg = DeliveryCfg()):
        self.cfg = cfg
        base_env = DeliveryBase(cfg.base, should_term_fn=self.should_terminate)
        EnvUsingBase.__init__(self, cfg, self.specification, base_env)
        StaticTemporalNodeMixin.__init__(self, cfg)
        self.base = base_env

    @staticmethod
    def create(cfg: DeliveryCfg) -> "Delivery":
        return Delivery(cfg)

    def reset_batch(self, key: PRNGKeyArray, batch_size: int, init: bool = False) -> StateWithTemporalNode:
        key_reset, key_steps = jr.split(key)
        b_state = super().reset_batch(key, batch_size)

        if init:
            # Randomize the initial timestep.
            with jdc.copy_and_mutate(b_state) as b_state_new:
                # b_state_new_base: HerdingHerd.State = b_state_new.base
                # b_state_new_base.steps = jr.randint(key_steps, (batch_size,), 0, self.base.cfg.trunc_steps)
                b_state_new.base.steps = jr.randint(key_steps, (batch_size,), 0, self.base.cfg.trunc_steps)
        else:
            b_state_new = b_state

        return b_state_new

    @property
    def specification(self):
        return self.cfg.specification

    def should_terminate(self, predicates: dict[str, jnp.ndarray]) -> BoolScalar:
        eps = 0.1

        # Terminate when leaving the allowed area.
        # is_oob = predicates["oob"] > eps
        # should_term = is_oob

        is_oob = predicates["oob"] > eps
        is_in_obst = predicates["obstacles"] > eps
        should_term = is_oob | is_in_obst

        # NOTE, both induce reset
        # terminating is suffix will not affect value (reached end-point)
        # truncate bootstraps thru reset!

        return should_term
