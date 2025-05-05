import argparse
import functools
import os
import sys
import wandb

sys.path.append("/home/mepear_gc")

import flax
import jax
import numpy as np
import optax
import gymnax
from flax.training.train_state import TrainState
from jax import numpy as jnp
from jax.numpy import einsum
from optax import sigmoid_binary_cross_entropy
from tqdm import tqdm

from EFPPO.src.model.actorcritic import Actor_Network_SAC, Representation_Network_SAC
from EFPPO.src.env.env_list import get_env

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--EXP_NAME", type=str, default="PendulumConstraintBaseline")
    parser.add_argument("--NAME", type=str, default="test_pendulum_constraint")
    parser.add_argument("--NUM_ENVS", type=int, default=64)
    parser.add_argument("--NUM_STEPS", type=int, default=200)
    parser.add_argument("--TOTAL_TIMESTEPS", type=int, default=200_000)
    parser.add_argument("--BATCH_SIZE", type=int, default=256)
    parser.add_argument("--BUFFER_SIZE", type=int, default=4096)
    parser.add_argument("--LR", type=float, default=3e-4)
    parser.add_argument("--GAMMA", type=float, default=0.99)
    parser.add_argument("--TAU", type=float, default=0.005)
    parser.add_argument("--ALPHA", type=float, default=0.0)
    parser.add_argument("--LEARNING_START", type=int, default=20_000)
    parser.add_argument("--POLICY_FREQUENCY", type=int, default=16)
    parser.add_argument("--SECTION", type=int, default=0)
    parser.add_argument("--DIR", type=str, default='pendulum')
    parser.add_argument("--SEED", type=int, default=42)

    args = parser.parse_args()

    return args


@functools.partial(jax.jit, static_argnums=0)
def actor_output(apply_fn, params, state, key):
    return apply_fn(params, state, key)


@functools.partial(jax.jit, static_argnums=0)
def critic_output(apply_fn, params, state, action, future_state):
    return apply_fn(params, state, action, future_state)


@functools.partial(jax.jit, static_argnums=(4, 5))
def critic_train_step(critic_train_state_1, critic_train_state_2, actor_train_state, batch, gamma, alpha, key):
    states, actions, future_states, _, _, _ = batch
    batch_size = states.shape[0]

    def loss_fn(params, apply_fn):
        qsa, qg = critic_output(apply_fn, params, states, actions, future_states[:, :-1])
        logits = einsum('ik, jk -> ij', qsa, qg)
        I = jnp.eye(batch_size)
        loss = sigmoid_binary_cross_entropy(logits, I)
        return jnp.mean(loss)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(critic_train_state_1.params, critic_train_state_1.apply_fn)
    critic_train_state_1 = critic_train_state_1.apply_gradients(grads=grads)

    loss, grads = grad_fn(critic_train_state_2.params, critic_train_state_2.apply_fn)
    critic_train_state_2 = critic_train_state_2.apply_gradients(grads=grads)

    return critic_train_state_1, critic_train_state_2, loss


@functools.partial(jax.jit, static_argnums=4)
def actor_train_step(actor_train_state, critic_train_state_1, critic_train_state_2, batch, alpha, key):
    states, _, _, _, _, _ = batch
    batch_size = states.shape[0]

    def loss_fn(params):
        action, log_pi = actor_output(actor_train_state.apply_fn, params, states, key)
        goal = jnp.array([[1., 0.]])
        goals = goal.repeat(batch_size, axis=0)
        qsa_1, qg_1 = critic_output(critic_train_state_1.apply_fn, critic_train_state_1.params, states, action, goals)
        qsa_2, qg_2 = critic_output(critic_train_state_2.apply_fn, critic_train_state_2.params, states, action, goals)
        q_1 = einsum('ik, ik -> i', qsa_1, qg_1)
        q_2 = einsum('ik, ik -> i', qsa_2, qg_2)
        min_qf_pi = jnp.minimum(q_1, q_2)
        return jnp.mean(alpha * log_pi - min_qf_pi)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(actor_train_state.params)
    actor_train_state = actor_train_state.apply_gradients(grads=grads)

    return actor_train_state, loss


class TrainStateNew(TrainState):
    target_params: flax.core.FrozenDict


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, num_envs, env_steps, observation_shape, action_shape, gamma):
        self.states = np.zeros((buffer_size, env_steps, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, env_steps, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, env_steps,), dtype=np.float32)
        self.flags = np.zeros((buffer_size, env_steps,), dtype=np.float32)

        assert buffer_size % num_envs == 0

        self.num_envs = num_envs
        self.env_steps = env_steps
        self.batch_size = batch_size
        self.max_size = buffer_size * env_steps // num_envs
        self.idx = 0
        self.size = 0
        self.gamma = gamma

    def push(self, state, action, reward, flag):
        x_idx = self.idx // self.env_steps
        y_idx = self.idx % self.env_steps
        self.states[x_idx * self.num_envs: x_idx * self.num_envs + self.num_envs, y_idx] = state
        self.actions[x_idx * self.num_envs: x_idx * self.num_envs + self.num_envs, y_idx] = action
        self.rewards[x_idx * self.num_envs: x_idx * self.num_envs + self.num_envs, y_idx] = reward
        self.flags[x_idx * self.num_envs: x_idx * self.num_envs + self.num_envs, y_idx] = flag

        self.idx = (self.idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, rng):
        rng, _rng = jax.random.split(rng)
        max_x_idx = ((self.size - 1) // self.env_steps) * self.num_envs
        x_idxs = jax.random.randint(_rng, shape=(self.batch_size, ), minval=0, maxval=max_x_idx)
        rng, _rng = jax.random.split(rng)
        y_idxs_low = jax.random.randint(rng, (self.batch_size, ), minval=0, maxval=self.env_steps - 1)
        rng, _rng = jax.random.split(rng)

        arange = jnp.expand_dims(jnp.arange(self.env_steps), axis=0)
        arange_vec = arange.repeat(self.batch_size, axis=0)
        y_idxs_low_expand = jnp.expand_dims(y_idxs_low, axis=1)
        log_prob = jnp.log(self.gamma ** (arange_vec - y_idxs_low_expand))
        log_prob_new = jnp.where(arange_vec < y_idxs_low_expand, -jnp.inf, log_prob)
        y_idxs_high = jax.random.categorical(rng, log_prob_new)


        return (
            self.states[x_idxs, y_idxs_low],
            self.actions[x_idxs, y_idxs_low],
            self.states[x_idxs, y_idxs_high],
            self.rewards[x_idxs, y_idxs_low],
            self.states[x_idxs, y_idxs_low+1],
            self.flags[x_idxs, y_idxs_low],
        )


def train(config):

    wandb.init(
        project="Contrastive_RL_{}".format(config["EXP_NAME"]),
        name=config['NAME'],
        config=config,
        monitor_gym=True,
        save_code=True,
    )

    # Create vectorized environment
    # env = get_env(config)
    # env_params = env.default_params
    env, env_params = gymnax.make('Pendulum-v1')

    # Metadata about the environment
    observation_shape = env.observation_space(env_params).shape
    action_shape = env.action_space(env_params).shape
    action_dim = np.prod(action_shape)
    action_low = env.action_space(env_params).low
    action_high = env.action_space(env_params).high

    rng, actor_rng, critic_rng = jax.random.split(jax.random.PRNGKey(config["SEED"]), 3)

    # Create the networks and the optimizer
    action_scale = (action_high - action_low) / 2.0
    action_bias = (action_high + action_low) / 2.0

    actor_net = Actor_Network_SAC(action_dim=action_dim, action_scale=action_scale, action_bias=action_bias)
    actor_init_params = actor_net.init(actor_rng, jnp.zeros(observation_shape), rng)

    critic_net = Representation_Network_SAC()
    critic_init_params = critic_net.init(critic_rng, jnp.zeros(observation_shape),
                                         jnp.zeros(action_shape), jnp.zeros(observation_shape[0] - 1))

    optimizer = optax.adam(learning_rate=config["LR"])

    actor_train_state = TrainStateNew.create(
        apply_fn=actor_net.apply,
        params=actor_init_params,
        target_params=actor_init_params,
        tx=optimizer,
    )

    critic_train_state_1 = TrainStateNew.create(
        apply_fn=critic_net.apply,
        params=critic_init_params,
        target_params=critic_init_params,
        tx=optimizer,
    )

    critic_train_state_2 = TrainStateNew.create(
        apply_fn=critic_net.apply,
        params=critic_init_params,
        target_params=critic_init_params,
        tx=optimizer,
    )

    alpha = config["ALPHA"]

    # Create the replay buffer
    replay_buffer = ReplayBuffer(config["BUFFER_SIZE"], config["BATCH_SIZE"], config["NUM_ENVS"], config["NUM_STEPS"],
                                 observation_shape, action_shape, config['GAMMA'])

    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])

    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

    rng, _rng = jax.random.split(rng)

    episode_return_list = []

    episode_return = np.zeros(config["NUM_ENVS"])
    episode_reach = np.zeros(config["NUM_ENVS"], dtype=bool)

    # Main loop
    for global_step in tqdm(range(config["TOTAL_TIMESTEPS"])):

        if global_step * config["NUM_ENVS"] < config["LEARNING_START"]:
            action = jax.random.uniform(_rng, shape=(config["NUM_ENVS"], action_shape[0]),
                                        minval=action_low, maxval=action_high)
            rng, _rng = jax.random.split(_rng)
        else:
            action, _ = actor_output(actor_train_state.apply_fn, actor_train_state.params, obsv, _rng)
            rng, _rng = jax.random.split(_rng)

        # Perform action
        step_rng = jax.random.split(_rng, config["NUM_ENVS"])
        next_obsv, next_env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(step_rng, env_state, action, env_params)
        rng, _rng = jax.random.split(_rng)

        # Store transition in the replay buffer
        flag = 1.0 - done
        replay_buffer.push(obsv, action, reward, flag)

        normalized_theta = (env_state.theta + jnp.pi) % (2 * jnp.pi) - jnp.pi
        reach = normalized_theta * (normalized_theta + 0.05 * env_state.theta_dot) < 0

        episode_return += (1 - episode_reach) * reward
        episode_reach += reach

        env_state = next_env_state
        obsv = next_obsv

        if global_step % config["NUM_STEPS"] == config["NUM_STEPS"] - 1 and global_step * config["NUM_ENVS"] > config["LEARNING_START"]:
            episode_return_list.append(episode_return)
            episode_return = np.zeros(config["NUM_ENVS"])
            wandb.log({"not reach goal": config["NUM_ENVS"] - np.sum(episode_reach)})
            print("Not reaching goal: {}".format(config["NUM_ENVS"] - np.sum(episode_reach)))
            episode_reach = np.zeros(config["NUM_ENVS"], dtype=bool)

            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

            wandb.log({"Episode Return Mean": jnp.mean(episode_return_list[-1])})
            rng, _rng = jax.random.split(_rng)

        # Perform training step
        if global_step * config["NUM_ENVS"] > config["LEARNING_START"] and global_step % config["POLICY_FREQUENCY"] != 0:

            # Sample replay buffer
            batch = replay_buffer.sample(_rng)
            rng, _rng = jax.random.split(_rng)

            critic_train_state_1, critic_train_state_2, critic_loss = critic_train_step(
                critic_train_state_1,
                critic_train_state_2,
                actor_train_state,
                batch,
                config["GAMMA"],
                alpha,
                _rng,
            )
            rng, _rng = jax.random.split(_rng)
            wandb.log({"critic_loss": critic_loss})

            critic_train_state_1 = critic_train_state_1.replace(
                target_params=optax.incremental_update(
                    critic_train_state_1.params,
                    critic_train_state_1.target_params,
                    config["TAU"],
                ),
            )
            critic_train_state_2 = critic_train_state_2.replace(
                target_params=optax.incremental_update(
                    critic_train_state_2.params,
                    critic_train_state_2.target_params,
                    config["TAU"],
                ),
            )

            actor_train_state, actor_loss = actor_train_step(
                actor_train_state,
                critic_train_state_1,
                critic_train_state_2,
                batch,
                alpha,
                _rng,
            )
            rng, _rng = jax.random.split(_rng)
            wandb.log({"actor loss": actor_loss})
    return


if __name__ == "__main__":
    config = vars(parse_args())

    # Create run directory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    mean_train_return = train(config)
    # print(f"Training - Mean returns achieved: {mean_train_return}.")