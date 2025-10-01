import os
import sys
import time

import wandb
sys.path.append("/home/mepear_gc")

import jax
import jax.numpy as jnp

from flax.training import train_state
from flax.training import checkpoints
from typing import Any

from rraa_rl.src.rl.utils.arguments import get_args
from functools import partial

from rraa_rl.src.env.env_list import get_env
from rraa_rl.src.model.actorcritic import Policy_Network, Policy_Network_Discrete, Value_Network, ActorCritic_Discrete
from rraa_rl.src.rl.utils.utils import optimizer, get_BuRd, tree_index1, tree_index2
from rraa_rl.src.rl.utils.gae import calculate_gae
from rraa_rl.src.rl.utils.alg_utils import _env_step_respo, _respo_update

class TrainState(train_state.TrainState):
    lambda_coef: Any

def train(env, env_params, config, rng):
    def _train(update_state, ent):

        train_state_policy, train_state_value, train_state_prob, train_state_cost, rng = update_state

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state_policy, train_state_value, train_state_prob, train_state_cost, env_state, obsv, _rng)

        # COLLECT TRAJECTORY
        runner_state, traj_batch = jax.lax.scan(
            env_step, runner_state, None, config["NUM_STEPS"]
        )

        # CALCULATE ADVANTAGE
        train_state_policy, train_state_value, train_state_prob, train_state_cost, env_state, last_obs, rng = runner_state
        last_val = train_state_value.apply_fn(train_state_value.params, last_obs)
        last_prob = train_state_prob.apply_fn(train_state_prob.params, last_obs)
        last_cost = train_state_cost.apply_fn(train_state_cost.params, last_obs)
        advantages_value, targets_value = calculate_gae(config["GAMMA_ENERGY"], config["GAE_LAMBDA"], traj_batch.value,
                                                        traj_batch.reward, traj_batch.done, last_val)
        advantages_cost, targets_cost = calculate_gae(config["GAMMA_ENERGY"], config["GAE_LAMBDA"], traj_batch.value_cost,
                                                        traj_batch.cost, traj_batch.done, last_cost)


        # UPDATE NETWORK
        update_state = (train_state_policy, train_state_value, train_state_prob, train_state_cost, traj_batch,
                        advantages_value, targets_value, advantages_cost, targets_cost, rng)
        xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent
        update_state, loss_info = jax.lax.scan(
            update_epoch, update_state, xs, config["UPDATE_EPOCHS"]
        )

        return (update_state[0], update_state[1],
                update_state[2], update_state[3], update_state[-1]), {"batch_info": traj_batch, "loss_info": loss_info}

    update_epoch = partial(_respo_update, config)
    env_step = partial(_env_step_respo, env, env_params)
    training = jax.jit(_train)

    tx = optimizer(config)

    # INIT NETWORK
    if config["DISCRETE"] == False:
        policy_network = Policy_Network(
            env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
        )
    else:
        policy_network = Policy_Network_Discrete(
            env.action_space(env_params).n, activation=config["ACTIVATION"]
        )

    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(env.observation_space(env_params).shape)
    network_params_policy = policy_network.init(_rng, init_x)
    train_state_policy = TrainState.create(
        apply_fn=policy_network.apply,
        params=network_params_policy,
        tx=tx,
        lambda_coef=0.,
    )

    value_network = Value_Network(activation=config["ACTIVATION"])
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(env.observation_space(env_params).shape)
    network_params_value = value_network.init(_rng, init_x)
    train_state_value = TrainState.create(
        apply_fn=value_network.apply,
        params=network_params_value,
        tx=tx,
        lambda_coef=0.,
    )

    value_network_prob = Value_Network(activation=config["ACTIVATION"])
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(env.observation_space(env_params).shape)
    network_params_prob = value_network_prob.init(_rng, init_x)
    train_state_prob = TrainState.create(
        apply_fn=value_network_prob.apply,
        params=network_params_prob,
        tx=tx,
        lambda_coef=0.,
    )

    value_network_cost = Value_Network(activation=config["ACTIVATION"])
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(env.observation_space(env_params).shape)
    network_params_cost = value_network_cost.init(_rng, init_x)
    train_state_cost = TrainState.create(
        apply_fn=value_network_cost.apply,
        params=network_params_cost,
        tx=tx,
        lambda_coef=0.,
    )

    total_timesteps = config["NUM_UPDATES"] // config["STEP_SCAN"]

    for timestep in range(config["NUM_UPDATES"] // config["STEP_SCAN"]):

        t0 = time.time()

        if config['ANNEAL_ENT'] == True:
            xs = jnp.ones(config["STEP_SCAN"]) * config["ENT_COEF"] * (total_timesteps - timestep) / total_timesteps
        else:
            xs = jnp.ones(config["STEP_SCAN"]) * config["ENT_COEF"]

        update_state, result = jax.lax.scan(
            training, (train_state_policy, train_state_value, train_state_prob, train_state_cost, rng), xs, config["STEP_SCAN"]
        )

        train_state_policy, train_state_value, train_state_prob, train_state_cost, rng = update_state

        loss_info = result['loss_info']

        traj_batch = tree_index1(result['batch_info'], 0)


        checkpoints.save_checkpoint(ckpt_dir=os.path.abspath('model/{}/'.format(config["DIR"])),
                                    target={"policy_network": train_state_policy, "value_network": train_state_value,
                                            "prob_network": train_state_prob, "cost_network": train_state_cost},
                                    step=timestep,
                                    overwrite=True,
                                    keep=2)

        t1 = time.time()

        wandb.log({"reaching goal": jnp.sum(traj_batch.done),
                   "average total return": -jnp.sum(traj_batch.reward) / jnp.sum(traj_batch.done),
                   "average cost": jnp.sum(traj_batch.cost) / jnp.sum(traj_batch.done),
                   "actor_loss": jnp.mean(loss_info["actor_loss"]), "entropy_loss": jnp.mean(loss_info["entropy_loss"]),
                   "value_loss": jnp.mean(loss_info["value_loss"]), "cost_loss": jnp.mean(loss_info["cost_loss"]),
                   "lambda": jnp.mean(loss_info['lambda'])})
        print("Earliest Reach {}: reach {} reward {} cost {}".format(timestep, jnp.sum(traj_batch.done),
                                                                     -jnp.sum(traj_batch.reward) / jnp.sum(
                                                                         traj_batch.done), jnp.sum(traj_batch.cost)
                                                                     / jnp.sum(traj_batch.done)))
        print("Time {}".format(t1-t0))

    return

if __name__ == "__main__":
    config = vars(get_args(sys.argv[1:]))
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = int(
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    folder = os.path.exists("model/{}".format(config['DIR']))
    if not folder:
        os.makedirs("model/{}".format(config['DIR']))
        os.makedirs("model/{}/reach".format(config['DIR']))
        os.makedirs("model/{}/value_target".format(config['DIR']))
        os.makedirs("model/{}/traj".format(config['DIR']))
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA_USE']
    env = get_env(config)
    env_params = env.default_params
    env_params = env_params.replace(gamma=config["GAMMA_ENERGY"])
    wandb.init(project='RESPO-{}'.format(config["EXP_NAME"]), name=config["NAME"], config=config)
    rng = jax.random.PRNGKey(20)
    out = train(env, env_params, config, rng)