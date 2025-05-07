from .reach_avoid.grid_avoid import GridAvoid
from .reach_avoid.grid_constraint import GridConstraint
from .reach_avoid.pendulum_constraint import PendulumConstraint
from .reach_avoid.hopper_avoid_ceiling import HopperAvoidCeiling, HopperAvoidCeilingDeterministic, HopperAvoidCeilingWall
from .reach_avoid.hopper_avoid_ceiling import HopperReachReach, HopperReachReachDeterministic
from .reach_avoid.wind_field import WindField
from .reach_avoid.half_cheetah_avoid import HalfCheetahAvoid, HalfCheetahAvoidDeterministic
from .reach_avoid.safety_gym_avoid import PointAvoid
from .baseline.pendulum_constraint_baseline import PendulumConstraintBaseline
from .baseline.hopper_avoid_ceiling_baseline import HopperAvoidCeilingBaseline
from .baseline.wind_field_baseline import WindFieldBaseline
from .baseline.half_cheetah_avoid_baseline import HalfCheetahAvoidBaseline

from .wrappers import TransformObservation

from functools import partial
import jax.numpy as jnp

def transform_observation(mean, variance, obs):
    return (obs - mean) / variance

def get_env(config):
    if config["EXP_NAME"] == 'GridConstraint':
        trans = partial(transform_observation, jnp.array([0., 0., 100.]), jnp.array([1., 1., 100.]))
        env = GridConstraint()
        env = TransformObservation(env, trans)
    elif config["EXP_NAME"] == 'GridAvoid':
        trans = partial(transform_observation, jnp.array([0., 0., 0., 200.]), jnp.array([1., 1., 1., 200.]))
        env = GridAvoid()
        env = TransformObservation(env, trans)
    elif config["EXP_NAME"] == 'PendulumConstraint':
        trans = partial(transform_observation, jnp.array([0., 0., 0., 400.]), jnp.array([1., 1., 1., 400.]))
        env = PendulumConstraint()
        env = TransformObservation(env, trans)
    elif config["EXP_NAME"] == 'HopperAvoidCeiling' and config["TEST_MODE"] == False:
        vec1 = jnp.zeros(14, dtype=jnp.float32)
        vec1 = vec1.at[0].set(1.)
        vec1 = vec1.at[-1].set(400.)
        vec2 = jnp.ones(14, dtype=jnp.float32)
        vec2 = vec2.at[-1].set(400.)
        trans = partial(transform_observation, vec1, vec2)
        env = HopperAvoidCeiling()
        env = TransformObservation(env, trans)
    elif config["EXP_NAME"] == 'HopperAvoidCeiling' and config["TEST_MODE"] == True:
        vec1 = jnp.zeros(14, dtype=jnp.float32)
        vec1 = vec1.at[0].set(1.)
        vec1 = vec1.at[-1].set(400.)
        vec2 = jnp.ones(14, dtype=jnp.float32)
        vec2 = vec2.at[-1].set(400.)
        trans = partial(transform_observation, vec1, vec2)
        env = HopperAvoidCeilingDeterministic()
        env = TransformObservation(env, trans)
    elif config["EXP_NAME"] == 'HopperReachReach' and config["TEST_MODE"] == False:
        vec1 = jnp.zeros(14, dtype=jnp.float32)
        vec1 = vec1.at[0].set(1.)
        vec1 = vec1.at[-1].set(400.)
        vec2 = jnp.ones(14, dtype=jnp.float32)
        vec2 = vec2.at[-1].set(400.)
        trans = partial(transform_observation, vec1, vec2)
        env = HopperReachReach()
        env = TransformObservation(env, trans)
    elif config["EXP_NAME"] == 'HopperReachReach' and config["TEST_MODE"] == True:
        vec1 = jnp.zeros(14, dtype=jnp.float32)
        vec1 = vec1.at[0].set(1.)
        vec1 = vec1.at[-1].set(400.)
        vec2 = jnp.ones(14, dtype=jnp.float32)
        vec2 = vec2.at[-1].set(400.)
        trans = partial(transform_observation, vec1, vec2)
        env = HopperReachReachDeterministic()
        env = TransformObservation(env, trans)
    elif config["EXP_NAME"] == 'HopperAvoidCeilingWall':
        vec1 = jnp.zeros(14, dtype=jnp.float32)
        vec1 = vec1.at[0].set(1.)
        vec1 = vec1.at[-1].set(400.)
        vec2 = jnp.ones(14, dtype=jnp.float32)
        vec2 = vec2.at[-1].set(400.)
        trans = partial(transform_observation, vec1, vec2)
        env = HopperAvoidCeilingWall()
        env = TransformObservation(env, trans)
    # TODO DEFINE OTHERS
    elif config["EXP_NAME"] == 'HalfCheetahAvoid':
        vec1 = jnp.zeros(20, dtype=jnp.float32)
        vec1 = vec1.at[0].set(2.5)
        vec1 = vec1.at[-1].set(400.)
        vec2 = jnp.ones(20, dtype=jnp.float32)
        vec2 = vec2.at[0].set(3.)
        vec2 = vec2.at[-1].set(400.)
        trans = partial(transform_observation, vec1, vec2)
        env = HalfCheetahAvoid()
        env = TransformObservation(env, trans)
    elif config["EXP_NAME"] == 'WindField':
        vec1 = jnp.zeros(14, dtype=jnp.float32)
        vec1 = vec1.at[-1].set(400.)
        vec2 = jnp.ones(14, dtype=jnp.float32)
        vec2 = vec2.at[-1].set(400.)
        vec2 = vec2.at[0].set(3.)
        vec2 = vec2.at[1].set(3.)
        vec2 = vec2.at[2].set(2.)
        trans = partial(transform_observation, vec1, vec2)
        env = WindField()
        env = TransformObservation(env, trans)
    elif config["EXP_NAME"] == 'F16Avoid':
        from .reach_avoid.F16_avoid import F16Avoid
        vec1 = jnp.zeros(26, dtype=jnp.float32)
        vec1 = vec1.at[-1].set(400.)
        vec2 = jnp.ones(26, dtype=jnp.float32)
        vec2 = vec2.at[-1].set(400.)
        trans = partial(transform_observation, vec1, vec2)
        env = F16Avoid()
        env = TransformObservation(env, trans)
    elif config["EXP_NAME"] == 'PointAvoid':
        vec1 = jnp.zeros(9, dtype=jnp.float32)
        vec1 = vec1.at[-1].set(400.)
        vec2 = jnp.ones(9, dtype=jnp.float32)
        vec2 = vec2.at[-1].set(400.)
        vec2 = vec2.at[0].set(2.)
        vec2 = vec2.at[1].set(2.)
        trans = partial(transform_observation, vec1, vec2)
        env = PointAvoid()
        env = TransformObservation(env, trans)
    elif config["EXP_NAME"] == 'PendulumConstraintBaseline':
        env = PendulumConstraintBaseline()
    elif config["EXP_NAME"] == 'HopperAvoidCeilingBaseline':
        vec1 = jnp.zeros(13, dtype=jnp.float32)
        vec1 = vec1.at[0].set(1.)
        vec2 = jnp.ones(13, dtype=jnp.float32)
        trans = partial(transform_observation, vec1, vec2)
        env = HopperAvoidCeilingBaseline()
        env = TransformObservation(env, trans)
    elif config["EXP_NAME"] == 'WindFieldBaseline':
        env = WindFieldBaseline()
        vec1 = jnp.zeros(13, dtype=jnp.float32)
        vec2 = jnp.ones(13, dtype=jnp.float32)
        vec2 = vec2.at[0].set(3.)
        vec2 = vec2.at[1].set(3.)
        vec2 = vec2.at[2].set(2.)
        trans = partial(transform_observation, vec1, vec2)
        env = TransformObservation(env, trans)
    elif config["EXP_NAME"] == 'HalfCheetahAvoidBaseline':
        vec1 = jnp.zeros(19, dtype=jnp.float32)
        vec1 = vec1.at[0].set(2.5)
        vec2 = jnp.ones(19, dtype=jnp.float32)
        vec2 = vec2.at[0].set(3.)
        trans = partial(transform_observation, vec1, vec2)
        env = HalfCheetahAvoidBaseline()
        env = TransformObservation(env, trans)
    elif config["EXP_NAME"] == 'PointAvoidBaseline':
        vec1 = jnp.zeros(8, dtype=jnp.float32)
        vec2 = jnp.ones(8, dtype=jnp.float32)
        vec2 = vec2.at[0].set(2.)
        vec2 = vec2.at[1].set(2.)
        trans = partial(transform_observation, vec1, vec2)
        env = PointAvoidBaseline()
        env = TransformObservation(env, trans)
    elif config["EXP_NAME"] == 'F16AvoidBaseline':
        from .baseline.F16_avoid_baseline import F16AvoidBaseline
        env = F16AvoidBaseline()
    else:
        raise Exception("No Given Environment")
    return env