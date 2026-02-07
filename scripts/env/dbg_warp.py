import ipdb
import jax.numpy as jnp

import jax
import jax.random as jr

from rraa_rl.envs.scene import ManipScene


def main():
    cfg = ManipScene.Cfg()
    env = ManipScene(cfg)

    n_envs_train = 16
    bs = n_envs_train

    key = jr.PRNGKey(0)
    b_state = env.reset_batch(key, batch_size=bs, init=False)
    b_action = [jnp.zeros((bs, 5), dtype=jnp.int32)]

    vmap_step_fn = jax.jit(jax.vmap(env.step))
    b_out = vmap_step_fn(b_state, b_action)

    print(b_out.envstate.temporal_node_idx)

if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()