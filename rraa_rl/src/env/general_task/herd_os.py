import jax
import jax.numpy as jnp
import jax.random as jr
import jax_dataclasses as jdc
from attrs import define
from jaxtyping import PRNGKeyArray
from loguru import logger

from rraa_rl.jax_types import BoolScalar
from rraa_rl.src.env.general_task.env import (EnvCfg, EnvUsingBase, StateWithTemporalNode, StaticTemporalNodeMixin,
                                              StaticTemporalNodeMixinCfg)
from rraa_rl.src.env.general_task.herd_base import (HerdBase, HerdBaseCfg, HerdBasePlay, HerdBasePlayCfg, HerdingHerd,
                                                    HerdingHerdCfg)


@define(slots=False)
class HerdOsCfg(EnvCfg, StaticTemporalNodeMixinCfg):
    specification: str = "F G herd_herded"
    base: HerdingHerdCfg = HerdBaseCfg()


class HerdOs(StaticTemporalNodeMixin, EnvUsingBase):
    """Herding environment with one or more herders and a herd of agents. The herd moves according to some fixed policy.
    The herders can influence the herd by moving around them.

    Each herd agent is a single-integrator that minimizes the soft minimum distance to the herders, the obstacles,
    and other herd agents, where the distances are scaled such that herders have larger influence.
    If the distance is large enough, the herd agents stay still.

    In the discrete action setup, each herder is a double-integrator that can accelerate / decelerate in either axis.

    action: (n_herders, 2): int, {0, 1, 2} for each axis, where 0 = -accel, 1 = no accel, 2 = accel
    """

    Cfg = HerdOsCfg
    State = StateWithTemporalNode[HerdingHerd.State]

    def __init__(self, cfg: HerdOsCfg = HerdOsCfg()):
        self.cfg = cfg
        base_env = HerdingHerd(cfg.base, should_term_fn=self.should_terminate)
        EnvUsingBase.__init__(self, cfg, self.specification, base_env)
        StaticTemporalNodeMixin.__init__(self, cfg)
        self.base = base_env

    def reset_batch(self, key: PRNGKeyArray, batch_size: int, init: bool = False) -> StateWithTemporalNode:
        key_reset, key_steps = jr.split(key)
        b_state: StateWithTemporalNode[HerdingHerd.State] = super().reset_batch(key, batch_size)

        if init:
            # Randomize the initial timestep.
            with jdc.copy_and_mutate(b_state) as b_state_new:
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
        is_oob = predicates["herder_oob"] > eps
        should_term = is_oob

        return should_term


@define(slots=False)
class HerdOsPlayCfg(EnvCfg, StaticTemporalNodeMixinCfg):
    specification: str = "F herder_c1 && F herder_c2 && G(!herder_oob) && G(!herder_collide)"

    base: HerdBasePlayCfg = HerdBasePlayCfg()


class HerdOsPlay(StaticTemporalNodeMixin, EnvUsingBase):
    """HerdOs but for playing around."""

    Cfg = HerdOsCfg
    State = StateWithTemporalNode

    def __init__(self, cfg: HerdOsPlayCfg = HerdOsPlayCfg()):
        self.cfg = cfg
        base_env = HerdBasePlay(cfg.base, should_term_fn=self.should_terminate)
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
        is_oob = predicates["herder_oob"] > eps
        should_term = is_oob

        return should_term
