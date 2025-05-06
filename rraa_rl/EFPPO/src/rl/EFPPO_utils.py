import jax
import jax.numpy as jnp
from rraa_rl.EFPPO.src.rl.gae import Transition_reach, Transition_cppo, Transition_sac

def _env_step(env, env_params, runner_state, _):
    (train_state_policy, train_state_energy, train_state_h,
     last_env_state, last_obs, rng) = runner_state

    # SELECT ACTION
    rng, _rng = jax.random.split(rng)
    pi = train_state_policy.apply_fn(train_state_policy.params, last_obs)
    value = train_state_energy.apply_fn(train_state_energy.params, last_obs)
    value_h = train_state_h.apply_fn(train_state_h.params, last_obs)

    action = pi.sample(seed=_rng)
    log_prob = pi.log_prob(action)

    # STEP ENV
    rng, _rng = jax.random.split(rng)
    env_num = last_obs.shape[0]
    rng_step = jax.random.split(_rng, env_num)
    obsv, env_state, reward, done, info = jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(rng_step, last_env_state, action, env_params)

    transition = Transition_reach(
        done, action, value, value_h, reward, last_env_state.energy, log_prob, last_obs, info,
        last_env_state.reach
    )
    runner_state = (train_state_policy, train_state_energy, train_state_h,
                    env_state, obsv, rng)
    return runner_state, transition

def _env_step_cppo(env, env_params, runner_state, _):
    train_state_policy, train_state_value, train_state_cost, last_env_state, last_obs, rng = runner_state

    # SELECT ACTION
    rng, _rng = jax.random.split(rng)
    pi = train_state_policy.apply_fn(train_state_policy.params, last_obs)
    value = train_state_value.apply_fn(train_state_value.params, last_obs)
    value_cost = train_state_cost.apply_fn(train_state_cost.params, last_obs)

    action = pi.sample(seed=_rng)
    log_prob = pi.log_prob(action)

    # STEP ENV
    rng, _rng = jax.random.split(rng)
    env_num = last_obs.shape[0]
    rng_step = jax.random.split(_rng, env_num)
    obsv, env_state, reward, done, info = jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(rng_step, last_env_state, action, env_params)

    transition = Transition_cppo(
        (env_state.reach < -1.) | (env_state.avoid == -1), action, value, value_cost, reward, env_state.cost,
        log_prob, last_obs, info, last_env_state.reach, env_state.avoid
    )
    runner_state = (train_state_policy, train_state_value, train_state_cost, env_state, obsv, rng)
    return runner_state, transition

def _env_step_sac(env, env_params, runner_state, _):
    train_state, last_env_state, last_obs, rng = runner_state

    # SELECT ACTION
    rng, _rng = jax.random.split(rng)
    action, log_prob = train_state.apply_fn(train_state.params, last_obs[:, :-1], _rng)

    # STEP ENV
    rng, _rng = jax.random.split(rng)
    env_num = last_obs.shape[0]
    rng_step = jax.random.split(_rng, env_num)
    obsv, env_state, reward, done, info = jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(rng_step, last_env_state, action, env_params)
    transition = Transition_sac(
        done, action, reward, reward, log_prob, last_obs, info, last_env_state.reach
    )
    runner_state = (train_state, env_state, obsv, rng)
    return runner_state, transition

def _env_step_deterministic(env, env_params, runner_state, _):
    (train_state_policy, last_env_state, last_obs, rng) = runner_state

    # SELECT ACTION
    rng, _rng = jax.random.split(rng)
    pi = train_state_policy.apply_fn(train_state_policy.params, last_obs)

    action = pi.loc
    log_prob = pi.log_prob(action)

    # STEP ENV
    rng, _rng = jax.random.split(rng)
    env_num = last_obs.shape[0]
    rng_step = jax.random.split(_rng, env_num)
    obsv, env_state, reward, done, info = jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(rng_step, last_env_state, action, env_params)
    transition = Transition_reach(
        done, action, 0., 0., reward, last_env_state.energy, log_prob, last_obs, info,
        last_env_state.reach
    )
    runner_state = (train_state_policy, env_state, obsv, rng)

    return runner_state, transition


def _env_step_test(env, env_params, runner_state, _):
    (train_state_policy, last_env_state, last_obs, rng) = runner_state

    # SELECT ACTION
    rng, _rng = jax.random.split(rng)
    pi = train_state_policy.apply_fn(train_state_policy.params, last_obs[:, :-1])

    action = pi.sample(seed=_rng)
    log_prob = pi.log_prob(action)

    # STEP ENV
    rng, _rng = jax.random.split(rng)
    env_num = last_obs.shape[0]
    rng_step = jax.random.split(_rng, env_num)
    obsv, env_state, reward, done, info = jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(rng_step, last_env_state, action, env_params)
    transition = Transition_reach(
        done, action, 0., 0., reward, last_env_state.energy, log_prob, last_obs, info,
        last_env_state.reach
    )
    runner_state = (train_state_policy, env_state, obsv, rng)

    return runner_state, transition


def _ecefppo_update(config, update_state, ent):
    (train_state_policy, train_state_energy, train_state_h, traj_batch,
     advantages_h, targets_h, advantages_V, targets_V, advantages_total, rng) = update_state
    rng, _rng = jax.random.split(rng)

    def _update_minbatch(train_state, batch_info):
        train_state_policy, train_state_energy, train_state_h = train_state
        traj_batch, advantages_h, targets_h, advantages_V, targets_V, advantages_total = batch_info

        def _loss_fn_reach(params, traj_batch, targets_h):
            # RERUN NETWORK
            value_h = train_state_h.apply_fn(params, traj_batch.obs)

            # CALCULATE VALUE LOSS FOR REACH FUNCTION
            value_pred_clipped_reach = traj_batch.value_reach + (
                    value_h - traj_batch.value_reach
            ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
            value_losses_reach = jnp.square(value_h - targets_h)
            value_losses_clipped_reach = jnp.square(value_pred_clipped_reach - targets_h)
            value_loss_reach = (
                    0.5 * jnp.maximum(value_losses_reach, value_losses_clipped_reach).mean()
            )

            total_loss = config["VF_COEF"] * value_loss_reach
            return total_loss, value_loss_reach

        def _loss_fn_energy(params, traj_batch, targets_V):
            # RERUN NETWORK
            value = train_state_energy.apply_fn(params, traj_batch.obs)

            # CALCULATE VALUE LOSS FOR NORMAL VALUE FUNCTION
            value_pred_clipped = traj_batch.value + (
                    value - traj_batch.value
            ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
            value_losses = jnp.square(value - targets_V)
            value_losses_clipped = jnp.square(value_pred_clipped - targets_V)
            value_loss_V = (
                    0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
            )

            total_loss = config["VF_COEF"] * value_loss_V
            return total_loss, value_loss_V

        def _loss_fn_policy(params, traj_batch, gae):
            # RERUN NETWORK
            pi = train_state_policy.apply_fn(params, traj_batch.obs)
            log_prob = pi.log_prob(traj_batch.action)

            # CALCULATE ACTOR LOSS
            ratio = jnp.exp(log_prob - traj_batch.log_prob)
            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
            loss_actor1 = ratio * gae
            loss_actor2 = (
                    jnp.clip(
                        ratio,
                        1.0 - config["CLIP_EPS"],
                        1.0 + config["CLIP_EPS"],
                    )
                    * gae
            )
            loss_actor = jnp.maximum(loss_actor1, loss_actor2)
            loss_actor = loss_actor.mean()
            entropy = pi.entropy().mean()

            total_loss = (
                    loss_actor
                    - ent * entropy
            )
            return total_loss, (loss_actor, entropy)

        grad_fn = jax.value_and_grad(_loss_fn_policy, has_aux=True)
        total_loss_policy, grads = grad_fn(
            train_state_policy.params, traj_batch, advantages_total
        )
        train_state_policy = train_state_policy.apply_gradients(grads=grads)

        grad_fn = jax.value_and_grad(_loss_fn_energy, has_aux=True)
        total_loss_energy, grads = grad_fn(
            train_state_energy.params, traj_batch, targets_V
        )
        train_state_energy = train_state_energy.apply_gradients(grads=grads)

        grad_fn = jax.value_and_grad(_loss_fn_reach, has_aux=True)
        total_loss_h, grads = grad_fn(
            train_state_h.params, traj_batch, targets_h
        )
        train_state_h = train_state_h.apply_gradients(grads=grads)

        return (train_state_policy, train_state_energy, train_state_h), {"actor_loss": total_loss_policy[1][0],
                                                                         "entropy_loss": total_loss_policy[1][1],
                                                                         "energy_loss": total_loss_energy[1],
                                                                         "reach_loss": total_loss_h[1]}


    rng, _rng = jax.random.split(rng)
    batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
    assert (
            batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
    ), "batch size must be equal to number of steps * number of envs"
    permutation = jax.random.permutation(_rng, batch_size)
    batch = (traj_batch, advantages_h, targets_h, advantages_V, targets_V, advantages_total)
    batch = jax.tree_util.tree_map(
        lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
    )
    shuffled_batch = jax.tree_util.tree_map(
        lambda x: jnp.take(x, permutation, axis=0), batch
    )
    minibatches = jax.tree_util.tree_map(
        lambda x: jnp.reshape(
            x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
        ),
        shuffled_batch,
    )
    (train_state_policy, train_state_energy, train_state_h), total_loss = jax.lax.scan(
        _update_minbatch, (train_state_policy, train_state_energy, train_state_h), minibatches
    )
    update_state = (train_state_policy, train_state_energy, train_state_h,
                    traj_batch, advantages_h, targets_h, advantages_V, targets_V, advantages_total, rng)
    return update_state, total_loss

def _cppo_update(config, update_state, ent):
    (train_state_policy, train_state_value, train_state_cost, traj_batch,
     advantages_value, targets_value, advantages_cost, targets_cost, rng) = update_state
    rng, _rng = jax.random.split(rng)

    def _update_minbatch(train_state, batch_info):
        train_state_policy, train_state_value, train_state_cost = train_state
        traj_batch, advantages_value, targets_value, advantages_cost, targets_cost = batch_info

        def _loss_fn_cost(params, traj_batch, targets_cost):
            # RERUN NETWORK
            value_cost = train_state_cost.apply_fn(params, traj_batch.obs)

            # CALCULATE VALUE LOSS FOR REACH FUNCTION
            value_pred_clipped_cost = traj_batch.value_cost + (
                    value_cost - traj_batch.value_cost
            ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
            value_losses_cost = jnp.square(value_cost - targets_cost)
            value_losses_clipped_cost = jnp.square(value_pred_clipped_cost - targets_cost)
            value_loss_cost = (
                    0.5 * jnp.maximum(value_losses_cost, value_losses_clipped_cost).mean()
            )
            lambda_new = jnp.clip(targets_cost.mean() - config['THRESHOLD_CPPO'], 0) * config['K_P']

            total_loss = config["VF_COEF"] * value_loss_cost
            return total_loss, (value_loss_cost, lambda_new)

        def _loss_fn_value(params, traj_batch, targets_value):
            # RERUN NETWORK
            value = train_state_value.apply_fn(params, traj_batch.obs)

            # CALCULATE VALUE LOSS FOR NORMAL VALUE FUNCTION
            value_pred_clipped = traj_batch.value + (
                    value - traj_batch.value
            ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
            value_losses = jnp.square(value - targets_value)
            value_losses_clipped = jnp.square(value_pred_clipped - targets_value)
            value_loss_value = (
                    0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
            )

            total_loss = config["VF_COEF"] * value_loss_value
            return total_loss, value_loss_value

        def _loss_fn_policy(params, traj_batch, gae):
            # RERUN NETWORK
            pi = train_state_policy.apply_fn(params, traj_batch.obs)
            log_prob = pi.log_prob(traj_batch.action)

            # CALCULATE ACTOR LOSS
            ratio = jnp.exp(log_prob - traj_batch.log_prob)
            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
            loss_actor1 = ratio * gae
            loss_actor2 = (
                    jnp.clip(
                        ratio,
                        1.0 - config["CLIP_EPS"],
                        1.0 + config["CLIP_EPS"],
                    )
                    * gae
            )
            loss_actor = jnp.maximum(loss_actor1, loss_actor2)
            loss_actor = loss_actor.mean()
            entropy = pi.entropy().mean()

            total_loss = (
                    loss_actor
                    - ent * entropy
            )
            return total_loss, (loss_actor, entropy)

        grad_fn = jax.value_and_grad(_loss_fn_policy, has_aux=True)
        total_loss_policy, grads = grad_fn(
            train_state_policy.params, traj_batch, (advantages_value + train_state_policy.lambda_coef * advantages_cost)
                                                   / (1 + train_state_policy.lambda_coef)
        )
        train_state_policy = train_state_policy.apply_gradients(grads=grads)

        grad_fn = jax.value_and_grad(_loss_fn_value, has_aux=True)
        total_loss_value, grads = grad_fn(
            train_state_value.params, traj_batch, targets_value
        )
        train_state_value = train_state_value.apply_gradients(grads=grads)

        grad_fn = jax.value_and_grad(_loss_fn_cost, has_aux=True)
        total_loss_cost, grads = grad_fn(
            train_state_cost.params, traj_batch, targets_cost
        )
        train_state_cost = train_state_cost.apply_gradients(grads=grads)
        lambda_change = jnp.where(config['FIX_LAMBDA'], config['LAMBDA_REACH'], total_loss_cost[1][1])
        train_state_policy = train_state_policy.replace(lambda_coef=lambda_change)

        return (train_state_policy, train_state_value, train_state_cost), {"actor_loss": total_loss_policy[1][0],
                                                                         "entropy_loss": total_loss_policy[1][1],
                                                                         "value_loss": total_loss_value[1],
                                                                         "cost_loss": total_loss_cost[1][0],
                                                                        "lambda": lambda_change}


    rng, _rng = jax.random.split(rng)
    batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
    assert (
            batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
    ), "batch size must be equal to number of steps * number of envs"
    permutation = jax.random.permutation(_rng, batch_size)
    batch = (traj_batch, advantages_value, targets_value, advantages_cost, targets_cost)
    batch = jax.tree_util.tree_map(
        lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
    )
    shuffled_batch = jax.tree_util.tree_map(
        lambda x: jnp.take(x, permutation, axis=0), batch
    )
    minibatches = jax.tree_util.tree_map(
        lambda x: jnp.reshape(
            x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
        ),
        shuffled_batch,
    )
    (train_state_policy, train_state_value, train_state_cost), total_loss = jax.lax.scan(
        _update_minbatch, (train_state_policy, train_state_value, train_state_cost), minibatches
    )
    update_state = (train_state_policy, train_state_value, train_state_cost, traj_batch,
                    advantages_value, targets_value, advantages_cost, targets_cost, rng)
    return update_state, total_loss

def _respo_update(config, update_state, ent):
    (train_state_policy, train_state_value, train_state_cost, traj_batch,
     advantages_value, targets_value, advantages_cost, targets_cost, rng) = update_state
    rng, _rng = jax.random.split(rng)

    def _update_minbatch(train_state, batch_info):
        train_state_policy, train_state_value, train_state_cost = train_state
        traj_batch, advantages_value, targets_value, advantages_cost, targets_cost = batch_info

        def _loss_fn_cost(params, traj_batch, targets_cost):
            # RERUN NETWORK
            value_cost = train_state_cost.apply_fn(params, traj_batch.obs)

            # CALCULATE VALUE LOSS FOR REACH FUNCTION
            value_pred_clipped_cost = traj_batch.value_cost + (
                    value_cost - traj_batch.value_cost
            ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
            value_losses_cost = jnp.square(value_cost - targets_cost)
            value_losses_clipped_cost = jnp.square(value_pred_clipped_cost - targets_cost)
            value_loss_cost = (
                    0.5 * jnp.maximum(value_losses_cost, value_losses_clipped_cost).mean()
            )
            lambda_new = jnp.clip(targets_cost.mean() - config['THRESHOLD_CPPO'], 0) * config['K_P']

            total_loss = config["VF_COEF"] * value_loss_cost
            return total_loss, (value_loss_cost, lambda_new)

        def _loss_fn_value(params, traj_batch, targets_value):
            # RERUN NETWORK
            value = train_state_value.apply_fn(params, traj_batch.obs)

            # CALCULATE VALUE LOSS FOR NORMAL VALUE FUNCTION
            value_pred_clipped = traj_batch.value + (
                    value - traj_batch.value
            ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
            value_losses = jnp.square(value - targets_value)
            value_losses_clipped = jnp.square(value_pred_clipped - targets_value)
            value_loss_value = (
                    0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
            )

            total_loss = config["VF_COEF"] * value_loss_value
            return total_loss, value_loss_value

        def _loss_fn_policy(params, traj_batch, gae):
            # RERUN NETWORK
            pi = train_state_policy.apply_fn(params, traj_batch.obs)
            log_prob = pi.log_prob(traj_batch.action)

            # CALCULATE ACTOR LOSS
            ratio = jnp.exp(log_prob - traj_batch.log_prob)
            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
            loss_actor1 = ratio * gae
            loss_actor2 = (
                    jnp.clip(
                        ratio,
                        1.0 - config["CLIP_EPS"],
                        1.0 + config["CLIP_EPS"],
                    )
                    * gae
            )
            loss_actor = jnp.maximum(loss_actor1, loss_actor2)
            loss_actor = loss_actor.mean()
            entropy = pi.entropy().mean()

            total_loss = (
                    loss_actor
                    - ent * entropy
            )
            return total_loss, (loss_actor, entropy)

        grad_fn = jax.value_and_grad(_loss_fn_policy, has_aux=True)
        total_loss_policy, grads = grad_fn(
            train_state_policy.params, traj_batch, (advantages_value + train_state_policy.lambda_coef * advantages_cost)
                                                   / (1 + train_state_policy.lambda_coef)
        )
        train_state_policy = train_state_policy.apply_gradients(grads=grads)

        grad_fn = jax.value_and_grad(_loss_fn_value, has_aux=True)
        total_loss_value, grads = grad_fn(
            train_state_value.params, traj_batch, targets_value
        )
        train_state_value = train_state_value.apply_gradients(grads=grads)

        grad_fn = jax.value_and_grad(_loss_fn_cost, has_aux=True)
        total_loss_cost, grads = grad_fn(
            train_state_cost.params, traj_batch, targets_cost
        )
        train_state_cost = train_state_cost.apply_gradients(grads=grads)
        lambda_change = jnp.where(config['FIX_LAMBDA'], config['LAMBDA_REACH'], total_loss_cost[1][1])
        train_state_policy = train_state_policy.replace(lambda_coef=lambda_change)

        return (train_state_policy, train_state_value, train_state_cost), {"actor_loss": total_loss_policy[1][0],
                                                                         "entropy_loss": total_loss_policy[1][1],
                                                                         "value_loss": total_loss_value[1],
                                                                         "cost_loss": total_loss_cost[1][0],
                                                                         "lambda": lambda_change}


    rng, _rng = jax.random.split(rng)
    batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
    assert (
            batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
    ), "batch size must be equal to number of steps * number of envs"
    permutation = jax.random.permutation(_rng, batch_size)
    batch = (traj_batch, advantages_value, targets_value, advantages_cost, targets_cost)
    batch = jax.tree_util.tree_map(
        lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
    )
    shuffled_batch = jax.tree_util.tree_map(
        lambda x: jnp.take(x, permutation, axis=0), batch
    )
    minibatches = jax.tree_util.tree_map(
        lambda x: jnp.reshape(
            x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
        ),
        shuffled_batch,
    )
    (train_state_policy, train_state_value, train_state_cost), total_loss = jax.lax.scan(
        _update_minbatch, (train_state_policy, train_state_value, train_state_cost), minibatches
    )
    update_state = (train_state_policy, train_state_value, train_state_cost, traj_batch,
                    advantages_value, targets_value, advantages_cost, targets_cost, rng)
    return update_state, total_loss

def _ppo_update(config, update_state, ent):
    def _update_minbatch(train_state, batch_info):
        traj_batch, advantages, targets = batch_info

        def _loss_fn(params, traj_batch, gae, targets):
            # RERUN NETWORK
            pi, value = train_state.apply_fn(params, traj_batch.obs)
            log_prob = pi.log_prob(traj_batch.action)

            # CALCULATE VALUE LOSS
            value_pred_clipped = traj_batch.value + (
                    value - traj_batch.value
            ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
            value_losses = jnp.square(value - targets)
            value_losses_clipped = jnp.square(value_pred_clipped - targets)
            value_loss = (
                    0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
            )

            # CALCULATE ACTOR LOSS
            ratio = jnp.exp(log_prob - traj_batch.log_prob)
            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
            loss_actor1 = ratio * gae
            loss_actor2 = (
                    jnp.clip(
                        ratio,
                        1.0 - config["CLIP_EPS"],
                        1.0 + config["CLIP_EPS"],
                    )
                    * gae
            )
            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
            loss_actor = loss_actor.mean()
            entropy = pi.entropy().mean()

            total_loss = (
                    loss_actor
                    + config["VF_COEF"] * value_loss
                    - ent * entropy
            )
            return total_loss, (loss_actor, value_loss, entropy)

        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        total_loss, grads = grad_fn(
            train_state.params, traj_batch, advantages, targets
        )
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, {"actor_loss": total_loss[1][0],
                             "entropy_loss": total_loss[1][2],
                             "value_loss": total_loss[1][1]}

    train_state, traj_batch, advantages, targets, rng = update_state
    rng, _rng = jax.random.split(rng)
    batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
    assert (
            batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
    ), "batch size must be equal to number of steps * number of envs"
    permutation = jax.random.permutation(_rng, batch_size)
    batch = (traj_batch, advantages, targets)
    batch = jax.tree_util.tree_map(
        lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
    )
    shuffled_batch = jax.tree_util.tree_map(
        lambda x: jnp.take(x, permutation, axis=0), batch
    )
    minibatches = jax.tree_util.tree_map(
        lambda x: jnp.reshape(
            x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
        ),
        shuffled_batch,
    )
    train_state, total_loss = jax.lax.scan(
        _update_minbatch, train_state, minibatches
    )
    update_state = (train_state, traj_batch, advantages, targets, rng)
    return update_state, total_loss