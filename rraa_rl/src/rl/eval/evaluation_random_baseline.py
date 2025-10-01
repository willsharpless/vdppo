import os
import optax
import jax
import jax.numpy as jnp
import sys
import numpy as np

sys.path.append("/home/mepear_gc")

from functools import partial
from flax.training.train_state import TrainState
from flax.training import checkpoints

from rraa_rl.src.rl.utils.arguments import get_args
from rraa_rl.src.env.env_list import get_env
from rraa_rl.src.model.actorcritic import ActorCritic_Continuous
from rraa_rl.src.rl.utils.alg_utils import _env_step_ppo


def save_consumption(traj_batch, config):
    reach_idx = (traj_batch.reach < 0).argmax(axis=0)
    energy = []
    for i in range(reach_idx.shape[0]):

        if reach_idx[i] == 0 and traj_batch.reach[0, i] > 0:
            energy.append(1000.)
        else:
            energy.append(-np.sum(traj_batch.reward[0: reach_idx[i], i]))
    energy = np.array(energy)
    np.save('model/{}/{}.npy'.format(config['DIR'], config['NAME']), energy)
    print(energy)
    return


def test(env, env_params, config, rng):

    env_step = partial(_env_step_ppo, env, env_params)

    network = ActorCritic_Continuous(
        env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
    )

    raw_restored = checkpoints.restore_checkpoint(
        ckpt_dir=os.path.abspath('model/{}/{}'.format(config["DIR"], config["DIR_MODEL"])),
        target=None
        )

    train_state = TrainState.create(
        apply_fn=network.apply,
        params=raw_restored['actor_critic_network']['params'],
        tx=optax.sgd(0.01, 0.99),
    )

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

    rng, _rng = jax.random.split(rng)
    runner_state = (train_state, env_state, obsv, _rng)

    # COLLECT TRAJECTORY
    _, traj_batch = jax.lax.scan(
        env_step, runner_state, None, config["NUM_STEPS"]
    )

    save_consumption(traj_batch, config)

    return


if __name__ == "__main__":
    config = vars(get_args(sys.argv[1:]))
    env = get_env(config)
    env_params = env.default_params
    rng = jax.random.PRNGKey(23)
    folder = os.path.exists("model/{}/traj".format(config['DIR']))
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    test(env, env_params, config, rng)