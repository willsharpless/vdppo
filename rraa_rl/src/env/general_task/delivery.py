import jax.numpy as jnp
from attrs import define

from rraa_rl.jax_types import BoolScalar
from rraa_rl.src.env.general_task.env import (EnvCfg, EnvUsingBase, StateWithTemporalNode, StaticTemporalNodeMixin,
                                              StaticTemporalNodeMixinCfg)
from rraa_rl.src.env.general_task.delivery_base import DeliveryBase, DeliveryBaseCfg, DeliveryBasePlay, DeliveryBasePlayCfg


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
    State = StateWithTemporalNode

    def __init__(self, cfg: DeliveryCfg = DeliveryCfg()):
        self.cfg = cfg
        base_env = DeliveryBase(cfg.base, should_term_fn=self.should_terminate)
        EnvUsingBase.__init__(self, cfg, self.specification, base_env)
        StaticTemporalNodeMixin.__init__(self, cfg)
        self.base = base_env

    @staticmethod
    def create(cfg: DeliveryCfg) -> "Delivery":
        return Delivery(cfg)

    @property
    def specification(self):
        return self.cfg.specification

    def should_terminate(self, predicates: dict[str, jnp.ndarray]) -> BoolScalar:
        eps = 0.1

        # Terminate when leaving the allowed area.
        is_oob = predicates["oob"] > eps
        should_term = is_oob

        return should_term


@define(slots=False)
class DeliveryPlayCfg(EnvCfg, StaticTemporalNodeMixinCfg):
    specification: str = "F target0 && F target1 && G(!oob) && G(!obstacles) && (!collide)"

    base: DeliveryBasePlayCfg = DeliveryBasePlayCfg()


class DeliveryPlay(StaticTemporalNodeMixin, EnvUsingBase):
    """Delivery but for playing around."""

    Cfg = DeliveryCfg
    State = StateWithTemporalNode

    def __init__(self, cfg: DeliveryPlayCfg = DeliveryPlayCfg()):
        self.cfg = cfg
        base_env = DeliveryBasePlay(cfg.base, should_term_fn=self.should_terminate)
        EnvUsingBase.__init__(self, cfg, self.specification, base_env)
        StaticTemporalNodeMixin.__init__(self, cfg)

    @property
    def specification(self):
        # return "F(G(herd_herded)) && G( !herder_collide ) && G( !herder_oob )"
        # return "( !herder_collide && ! herder_oob ) U ( G(herd_herded && !herder_collide && ! herder_oob) )"
        return self.cfg.specification

    def should_terminate(self, predicates: dict[str, jnp.ndarray]) -> BoolScalar:
        eps = 0.1

        # Terminate when leaving the allowed area.
        is_oob = predicates["oob"] > eps
        should_term = is_oob

        return should_term
