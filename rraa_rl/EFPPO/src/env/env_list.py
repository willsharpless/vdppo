from .reach_avoid.grid_avoid import GridAvoid
from .reach_avoid.grid_constraint import GridConstraint
from .reach_avoid.pendulum_constraint import PendulumConstraint
from .reach_avoid.hopper_avoid_ceiling import HopperAvoidCeiling, HopperAvoidCeilingDeterministic, HopperAvoidCeilingWallEnergy, HopperAvoidCeilingWallEnergyDeterministic
from .reach_avoid.hopper_avoid_ceiling import HopperReachReach, HopperReachReachDeterministic, HopperReach1, HopperReach2, HopperReach1Deterministic, HopperReach2Deterministic, \
    HopperAvoidOnly, HopperReachAvoid
from .reach_avoid.wind_field import WindField
from .reach_avoid.half_cheetah_avoid import HalfCheetahAvoid, HalfCheetahAvoidDeterministic
from .reach_avoid.half_cheetah_RAA import HalfCheetahReachAvoid, HalfCheetahAvoidOnly
from .reach_avoid.half_cheetah_RR import HalfCheetahReachReach, HalfCheetahReach1, HalfCheetahReach2
from .reach_avoid.safety_gym_avoid import PointAvoid
from .baseline.pendulum_constraint_baseline import PendulumConstraintBaseline
from .baseline.hopper_avoid_ceiling_baseline import HopperAvoidCeilingBaseline, HopperReachAlwaysAvoidBaseline_augmented, HopperReachReachBaseline_augmented_max, \
    HopperReachReachBaseline_augmented_sum, HopperReachReachBaseline_reward_cost_separated
from .baseline.wind_field_baseline import WindFieldBaseline
from .baseline.half_cheetah_avoid_baseline import HalfCheetahAvoidBaseline
from .baseline.half_cheetah_RAA_baseline import HalfCheetahReachAlwaysAvoidBaseline_augmented
from .baseline.half_cheetah_RR_baseline import HalfCheetahReachReachBaseline_augmented

from .baseline.F16_RAA_baseline import F16ReachAvoidBaseline
from .baseline.F16_RR_baseline import F16ReachReachBaseline

from .wrappers import TransformObservation

from functools import partial
import jax.numpy as jnp

def transform_observation(mean, variance, obs):
    return (obs - mean) / variance

def untransform_observation(mean, variance, obs):
    return obs * variance + mean

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
        # vec1 = vec1.at[-1].set(400.)
        vec2 = jnp.ones(14, dtype=jnp.float32)
        # vec2 = vec2.at[-1].set(400.) # INIT ENERGY?
        trans = partial(transform_observation, vec1, vec2)
        untrans = partial(untransform_observation, vec1, vec2)

        env = HopperReachReach()
        env = TransformObservation(env, trans)
        
        env1 = HopperReach1() # TODO make determinstic
        env1 = TransformObservation(env1, trans)
        env2 = HopperReach2() # TODO make determinstic
        env2 = TransformObservation(env2, trans)

        env.set_untransform_obs(untrans)
        env1.set_untransform_obs(untrans)
        env2.set_untransform_obs(untrans)
        return (env, env1, env2)
    elif config["EXP_NAME"] == 'HopperReachReach' and config["TEST_MODE"] == True:
        vec1 = jnp.zeros(14, dtype=jnp.float32)
        vec1 = vec1.at[0].set(1.)
        vec2 = jnp.ones(14, dtype=jnp.float32)
        trans = partial(transform_observation, vec1, vec2)
        untrans = partial(untransform_observation, vec1, vec2)

        env = HopperReachReachDeterministic()
        env = TransformObservation(env, trans)

        env1 = HopperReach1Deterministic() # TODO make determinstic
        env1 = TransformObservation(env1, trans)
        env2 = HopperReach2Deterministic() # TODO make determinstic
        env2 = TransformObservation(env2, trans)

        env.set_untransform_obs(untrans)
        env1.set_untransform_obs(untrans)
        env2.set_untransform_obs(untrans)
        return (env, env1, env2)

    elif config["EXP_NAME"] == 'HopperReachReach' and config["TEST_MODE"] == False:
        vec1 = jnp.zeros(14, dtype=jnp.float32)
        vec1 = vec1.at[0].set(1.)
        # vec1 = vec1.at[-1].set(400.)
        vec2 = jnp.ones(14, dtype=jnp.float32)
        # vec2 = vec2.at[-1].set(400.) # INIT ENERGY?
        trans = partial(transform_observation, vec1, vec2)
        untrans = partial(untransform_observation, vec1, vec2)

        env = HopperReachReach()
        env = TransformObservation(env, trans)
        
        env1 = HopperReach1() # TODO make determinstic
        env1 = TransformObservation(env1, trans)
        env2 = HopperReach2() # TODO make determinstic
        env2 = TransformObservation(env2, trans)

        env.set_untransform_obs(untrans)
        env1.set_untransform_obs(untrans)
        env2.set_untransform_obs(untrans)
        return (env, env1, env2)
    elif config["EXP_NAME"] == 'HopperReachReach' and config["TEST_MODE"] == True:
        vec1 = jnp.zeros(14, dtype=jnp.float32)
        vec1 = vec1.at[0].set(1.)
        vec2 = jnp.ones(14, dtype=jnp.float32)
        trans = partial(transform_observation, vec1, vec2)
        untrans = partial(untransform_observation, vec1, vec2)

        env = HopperReachReachDeterministic()
        env = TransformObservation(env, trans)

        env1 = HopperReach1Deterministic() # TODO make determinstic
        env1 = TransformObservation(env1, trans)
        env2 = HopperReach2Deterministic() # TODO make determinstic
        env2 = TransformObservation(env2, trans)

        env.set_untransform_obs(untrans)
        env1.set_untransform_obs(untrans)
        env2.set_untransform_obs(untrans)
        return (env, env1, env2)
    
    elif config["EXP_NAME"] == "HopperReachReachDecomposed":
        vec1 = jnp.zeros(12, dtype=jnp.float32)
        vec1 = vec1.at[0].set(1.)
        # vec1 = vec1.at[-1].set(400.)
        vec2 = jnp.ones(12, dtype=jnp.float32)
        # vec2 = vec2.at[-1].set(400.) # INIT ENERGY?
        trans = partial(transform_observation, vec1, vec2)
        untrans = partial(untransform_observation, vec1, vec2)

        from .baseline.hopper_avoid_ceiling_baseline import HopperRR, HopperR1, HopperR2
        if config["TEST_MODE"] == False:
            env = HopperRR()
            env1 = HopperR1()
            env2 = HopperR2()
        else:
            env = HopperRR(deterministic=True)
            env1 = HopperR1(deterministic=True)
            env2 = HopperR2(deterministic=True)
        env = TransformObservation(env, trans)
        env1 = TransformObservation(env1, trans)
        env2 = TransformObservation(env2, trans)
        env.set_untransform_obs(untrans)
        env1.set_untransform_obs(untrans)
        env2.set_untransform_obs(untrans)
        return (env, env1, env2)
        
    elif config["EXP_NAME"] == "HopperReachAlwaysAvoid": 
        # TODO: Add a determinist and random version after you create the environments - change based on mode? 
        vec1 = jnp.zeros(14, dtype=jnp.float32)
        vec1 = vec1.at[0].set(1.)
        vec2 = jnp.ones(14, dtype=jnp.float32)
        
        trans = partial(transform_observation, vec1, vec2)
        untrans = partial(untransform_observation, vec1, vec2)

        if config["TEST_MODE"] == False:
            env = HopperReachAvoid()
            env_avoid = HopperAvoidOnly() 
        else:
            env = HopperReachAvoid(deterministic=True)
            env_avoid = HopperAvoidOnly(deterministic=True)
        env = TransformObservation(env, trans)
        env_avoid = TransformObservation(env_avoid, trans)

        env.set_untransform_obs(untrans)
        env_avoid.set_untransform_obs(untrans)
        return (env, env_avoid)
    
    elif config["EXP_NAME"] == "HopperReachAlwaysAvoid_CPPO" or config["EXP_NAME"] == "HopperReachAlwaysAvoid_RCPPO":
        obs_dim = 12 + 1
        vec1 = jnp.zeros(obs_dim, dtype=jnp.float32)
        vec1 = vec1.at[0].set(1.)
        vec2 = jnp.ones(obs_dim, dtype=jnp.float32)
        
        trans = partial(transform_observation, vec1, vec2)
        untrans = partial(untransform_observation, vec1, vec2)

        env = HopperReachAlwaysAvoidBaseline_augmented()
        env = TransformObservation(env, trans)
        env.set_untransform_obs(untrans)

        return (env)
    
    elif config["EXP_NAME"] == "HopperReachReach_max_CPPO":
        obs_dim = 12 + 2
        vec1 = jnp.zeros(obs_dim, dtype=jnp.float32)
        vec1 = vec1.at[0].set(1.)
        vec2 = jnp.ones(obs_dim, dtype=jnp.float32)
        
        trans = partial(transform_observation, vec1, vec2)
        untrans = partial(untransform_observation, vec1, vec2)

        env = HopperReachReachBaseline_augmented_max(cost_type=config["ENV_COST_TYPE"], 
                                                     use_stl=config["USE_STL"], 
                                                     cost_fn=config["ENV_COST_FN"])
        env = TransformObservation(env, trans)
        env.set_untransform_obs(untrans)

        return (env)
    
    elif config["EXP_NAME"] == "HopperReachReach_sum_CPPO" or config["EXP_NAME"] == "HopperReachReach_sum_RCPPO":
        obs_dim = 12 + 2
        vec1 = jnp.zeros(obs_dim, dtype=jnp.float32)
        vec1 = vec1.at[0].set(1.)
        vec2 = jnp.ones(obs_dim, dtype=jnp.float32)
        
        trans = partial(transform_observation, vec1, vec2)
        untrans = partial(untransform_observation, vec1, vec2)

        env = HopperReachReachBaseline_augmented_sum(cost_type=config["ENV_COST_TYPE"], 
                                                     use_stl=config["USE_STL"], 
                                                     cost_fn=config["ENV_COST_FN"])
        env = TransformObservation(env, trans)
        env.set_untransform_obs(untrans)

        return (env)
    
    elif config["EXP_NAME"] == "HopperReachReach_separated_CPPO":
        obs_dim = 12 + 2
        vec1 = jnp.zeros(obs_dim, dtype=jnp.float32)
        vec1 = vec1.at[0].set(1.)
        vec2 = jnp.ones(obs_dim, dtype=jnp.float32)
        
        trans = partial(transform_observation, vec1, vec2)
        untrans = partial(untransform_observation, vec1, vec2)

        env = HopperReachReachBaseline_reward_cost_separated(cost_type=config["ENV_COST_TYPE"], 
                                                     use_stl=config["USE_STL"], 
                                                     cost_fn=config["ENV_COST_FN"])
        env = TransformObservation(env, trans)
        env.set_untransform_obs(untrans)

        return (env)
    
    elif config["EXP_NAME"] == "HopperReachAvoid":
        vec1 = jnp.zeros(14, dtype=jnp.float32)
        vec1 = vec1.at[0].set(1.)
        vec2 = jnp.ones(14, dtype=jnp.float32)
        trans = partial(transform_observation, vec1, vec2)
        if config["TEST_MODE"] == False:
            env = HopperReachAvoid()
        else:
            env = HopperReachAvoid(deterministic=True)
        env = TransformObservation(env, trans)

        return (env)

    elif config["EXP_NAME"] == 'HopperAvoidCeilingWallEnergy' and config["TEST_MODE"] == False:
        vec1 = jnp.zeros(14, dtype=jnp.float32)
        vec1 = vec1.at[0].set(1.)
        vec1 = vec1.at[-1].set(400.)
        vec2 = jnp.ones(14, dtype=jnp.float32)
        vec2 = vec2.at[-1].set(400.)
        trans = partial(transform_observation, vec1, vec2)
        env = HopperAvoidCeilingWallEnergy()
        env = TransformObservation(env, trans)
    elif config["EXP_NAME"] == 'HopperAvoidCeilingWallEnergy' and config["TEST_MODE"] == True:
        vec1 = jnp.zeros(14, dtype=jnp.float32)
        vec1 = vec1.at[0].set(1.)
        vec1 = vec1.at[-1].set(400.)
        vec2 = jnp.ones(14, dtype=jnp.float32)
        vec2 = vec2.at[-1].set(400.)
        trans = partial(transform_observation, vec1, vec2)
        env = HopperAvoidCeilingWallEnergyDeterministic()
        env = TransformObservation(env, trans)
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

    elif config["EXP_NAME"] == "HalfCheetahReachAlwaysAvoid": 
        vec1 = jnp.zeros(20, dtype=jnp.float32)
        vec1 = vec1.at[0].set(2.5)
        # vec1 = vec1.at[-1].set(0.)
        # vec1 = vec1.at[-2].set(0.)
        vec2 = jnp.ones(20, dtype=jnp.float32)
        vec2 = vec2.at[0].set(3.)
        # vec2 = vec2.at[-1].set(1.)
        # vec2 = vec2.at[-2].set(1.)
        trans = partial(transform_observation, vec1, vec2)
        untrans = partial(untransform_observation, vec1, vec2)

        env = HalfCheetahReachAvoid()
        env_avoid = HalfCheetahAvoidOnly() 
        env = TransformObservation(env, trans)
        env_avoid = TransformObservation(env_avoid, trans)

        env.set_untransform_obs(untrans)
        env_avoid.set_untransform_obs(untrans)
        return (env, env_avoid)
    
    elif config["EXP_NAME"] == "HalfCheetahReachAlwaysAvoid_CPPO":
        obs_dim = 18 + 1
        vec1 = jnp.zeros(obs_dim, dtype=jnp.float32)
        vec1 = vec1.at[0].set(2.5)
        vec2 = jnp.ones(obs_dim, dtype=jnp.float32)
        vec2 = vec2.at[0].set(3.)
        
        trans = partial(transform_observation, vec1, vec2)
        untrans = partial(untransform_observation, vec1, vec2)

        env = HalfCheetahReachAlwaysAvoidBaseline_augmented()
        env = TransformObservation(env, trans)
        env.set_untransform_obs(untrans)

        return (env)

    elif config["EXP_NAME"] == "HalfCheetahReachReach": 
        vec1 = jnp.zeros(20, dtype=jnp.float32)
        vec1 = vec1.at[0].set(2.5)
        # vec1 = vec1.at[-1].set(0.)
        # vec1 = vec1.at[-2].set(0.)
        vec2 = jnp.ones(20, dtype=jnp.float32)
        vec2 = vec2.at[0].set(3.)
        # vec2 = vec2.at[-1].set(1.)
        # vec2 = vec2.at[-2].set(1.)
        trans = partial(transform_observation, vec1, vec2)
        untrans = partial(untransform_observation, vec1, vec2)

        env = HalfCheetahReachReach()
        env = TransformObservation(env, trans)
        
        env1 = HalfCheetahReach1() # TODO make determinstic
        env1 = TransformObservation(env1, trans)
        env2 = HalfCheetahReach2() # TODO make determinstic
        env2 = TransformObservation(env2, trans)

        env.set_untransform_obs(untrans)
        env1.set_untransform_obs(untrans)
        env2.set_untransform_obs(untrans)
        return (env, env1, env2)
    
    elif config["EXP_NAME"] == "HalfCheetahReachReach_CPPO":
        obs_dim = 18 + 2
        vec1 = jnp.zeros(obs_dim, dtype=jnp.float32)
        vec1 = vec1.at[0].set(2.5)
        vec2 = jnp.ones(obs_dim, dtype=jnp.float32)
        vec2 = vec2.at[0].set(3.)
        
        trans = partial(transform_observation, vec1, vec2)
        untrans = partial(untransform_observation, vec1, vec2)

        env = HalfCheetahReachReachBaseline_augmented(use_stl=config["USE_STL"])
        env = TransformObservation(env, trans)
        env.set_untransform_obs(untrans)

        return (env)
        
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

    elif config["EXP_NAME"] == "F16ReachAvoid":
        from .reach_avoid.F16_RAA import F16ReachAvoid
        vec1 = jnp.zeros(26, dtype=jnp.float32)
        vec1 = vec1.at[-1].set(400.)
        vec2 = jnp.ones(26, dtype=jnp.float32)
        vec2 = vec2.at[-1].set(400.)
        trans = partial(transform_observation, vec1, vec2)

        env = F16ReachAvoid()
        env = TransformObservation(env, trans)

        return (env)


    elif config["EXP_NAME"] == "F16ReachReachDecomposed":
        from .reach_avoid.F16_RR import F16ReachReachBaseline, F16Reach1Baseline, F16Reach2Baseline
        vec1 = jnp.zeros(24, dtype=jnp.float32)
        vec1 = vec1.at[-1].set(400.)
        vec2 = jnp.ones(24, dtype=jnp.float32)
        vec2 = vec2.at[-1].set(400.)
        trans = partial(transform_observation, vec1, vec2)

        env = F16ReachReachBaseline()
        env = TransformObservation(env, trans)

        env1 = F16Reach1Baseline()
        env1 = TransformObservation(env1, trans)

        env2 = F16Reach2Baseline()
        env2 = TransformObservation(env2, trans)

        return (env, env1, env2)
    
    elif config["EXP_NAME"] == 'F16ReachAlwaysAvoid':
        from .reach_avoid.F16_RAA import F16ReachAvoid, F16AvoidOnly
        vec1 = jnp.zeros(26, dtype=jnp.float32)
        vec1 = vec1.at[-1].set(80.)
        vec1 = vec1.at[-2].set(80.)
        vec2 = jnp.ones(26, dtype=jnp.float32)
        vec2 = vec2.at[-1].set(80.)
        vec2 = vec2.at[-2].set(80.)
        trans = partial(transform_observation, vec1, vec2)

        env = F16ReachAvoid()
        env_avoid = F16AvoidOnly()

        env = TransformObservation(env, trans)
        env_avoid = TransformObservation(env_avoid, trans)

        # env.set_untransform_obs(untrans) # Not implemented
        # env_avoid.set_untransform_obs(untrans)
        return (env, env_avoid)
    
    elif config["EXP_NAME"] == 'F16ReachAlwaysAvoid_CPPO':
        from .reach_avoid.F16_RAA import F16ReachAvoid, F16AvoidOnly

        obs_dim = 26 - 2 + 1

        vec1 = jnp.zeros(obs_dim, dtype=jnp.float32)
        vec1 = vec1.at[-1].set(400.)
        vec2 = jnp.ones(obs_dim, dtype=jnp.float32)
        vec2 = vec2.at[-1].set(400.)
        trans = partial(transform_observation, vec1, vec2)

        env = F16ReachAvoidBaseline()

        env = TransformObservation(env, trans)

        # env.set_untransform_obs(untrans) # Not implemented
        return env
    
    elif config["EXP_NAME"] == 'F16ReachReach':
        from .reach_avoid.F16_RR import F16ReachReach, F16Reach1, F16Reach2
        vec1 = jnp.zeros(26, dtype=jnp.float32)
        vec1 = vec1.at[-1].set(80.)
        vec1 = vec1.at[-2].set(80.)
        vec2 = jnp.ones(26, dtype=jnp.float32)
        vec2 = vec2.at[-1].set(80.)
        vec2 = vec2.at[-2].set(80.)
        trans = partial(transform_observation, vec1, vec2)

        env = F16ReachReach()
        env1 = F16Reach1()
        env2 = F16Reach2()

        env = TransformObservation(env, trans)
        env1= TransformObservation(env1, trans)
        env2= TransformObservation(env2, trans)

        # env.set_untransform_obs(untrans) # Not implemented
        # env_avoid.set_untransform_obs(untrans)
        return (env, env1, env2)

    elif config["EXP_NAME"] == 'F16ReachReach_CPPO':
        from .baseline.F16_RR_baseline import F16ReachReachBaseline
        
        vec1 = jnp.zeros(26, dtype=jnp.float32)
        vec1 = vec1.at[-1].set(400.)
        vec2 = jnp.ones(26, dtype=jnp.float32)
        vec2 = vec2.at[-1].set(400.)
        trans = partial(transform_observation, vec1, vec2)

        env = F16ReachReachBaseline()

        env = TransformObservation(env, trans)

        # env.set_untransform_obs(untrans) # Not implemented
        # env_avoid.set_untransform_obs(untrans)
        return env

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