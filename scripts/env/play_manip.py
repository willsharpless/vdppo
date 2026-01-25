import os

import ipdb
import jax.numpy as jnp
from loguru import logger

from rraa_rl.envs.manipspace.scene_env_jax import ManipStep, SceneEnvJax

os.environ["XLA_FLAGS"] = "--xla_gpu_graph_min_graph_size=1"

import copy
import logging
import time
from typing import Sequence

import jax
import jax.random as jr
import mujoco
import mujoco.viewer
# import mujoco_warp as mjw
import warp as wp
from absl import app, flags
from jax import numpy as jp
from mujoco import mjx

from rraa_rl.envs.manipspace.manipspace_env_jax import ManipSpaceEnvJax

_VIEWER_GLOBAL_STATE = {
    "running": True,
}


def key_callback(key: int) -> None:
    if key == 32:  # Space bar
        _VIEWER_GLOBAL_STATE["running"] = not _VIEWER_GLOBAL_STATE["running"]
        logging.info("RUNNING = %s", _VIEWER_GLOBAL_STATE["running"])


def main():
    jax.config.update('jax_debug_nans', True)

    # impl = "warp"
    # env = ManipSpaceEnvJax.create()
    env = SceneEnvJax()
    # state, obs = env.reset(jr.PRNGKey(0))

    print(f"Default backend: {jax.default_backend()}")
    step_fn = mjx.step

    def set_data_fn(dx, ctrl, act, xfrc_applied, qpos, qvel, time_):
        return dx.tree_replace(
            {
                "ctrl": jp.array(ctrl),
                "act": jp.array(act),
                "xfrc_applied": jp.array(xfrc_applied),
                "qpos": jp.array(qpos),
                "qvel": jp.array(qvel),
                "time": jp.array(time_),
            }
        )

    m = env.mj_model
    mx = env.mjx_model

    d = mujoco.MjData(m)
    # dx = mjx.put_data(m, d, impl="warp", naconmax=None, njmax=None)
    dx = mjx.make_data(m, impl="warp", naconmax=None, njmax=None)
    # dx = state.mjx_data

    start = time.time()
    # step_fn = jax.jit(step_fn, donate_argnums=(1,), keep_unused=True).lower(mx, dx).compile()

    action = jnp.array(env.config.action_high)
    env_step_fn = env.step
    # env_step_fn = jax.jit(env_step_fn, keep_unused=True).lower(state, action).compile()
    elapsed = time.time() - start
    print(f"Compilation took {elapsed}s.")
    # set_model_fn = (
    #     jax.jit(set_model_fn, donate_argnums=(0,), keep_unused=True)
    #     .lower(mx, m.opt.gravity, m.opt.tolerance, m.opt.ls_tolerance, m.opt.timestep)
    #     .compile()
    # )
    set_data_fn = (
        jax.jit(set_data_fn, donate_argnums=(0,), keep_unused=True)
        .lower(dx, d.ctrl, d.act, d.xfrc_applied, d.qpos, d.qvel, d.time)
        .compile()
    )

    viewer = mujoco.viewer.launch_passive(m, d, key_callback=key_callback)

    # if _IMPL.value == 'warp':
    #     # TODO(btaba): use put_data.
    #     dx = mjx.make_data(
    #         m, impl=_IMPL.value, naconmax=_NACONMAX.value, njmax=_NJMAX.value
    #     )
    # else:

    idx = 0
    action = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
    # action = action.at[idx].set(env.config.action_high[idx])

    with viewer:
        opt = copy.copy(m.opt)
        n_iters = 0
        while True:
            logger.debug("Iteration {}".format(n_iters))
            start = time.time()

            dx = set_data_fn(dx, d.ctrl, d.act, d.xfrc_applied, d.qpos, d.qvel, d.time)
            # state = state._replace(mjx_data=dx)

            # if m.opt != opt:
            #     opt = copy.copy(m.opt)
            #     mx = set_model_fn(mx, m.opt.gravity, m.opt.tolerance, m.opt.ls_tolerance, m.opt.timestep)

            if _VIEWER_GLOBAL_STATE["running"]:
                print("Running!")

                dx = step_fn(mx, dx)

                # # dx = step_fn(mx, dx)
                # out: ManipStep = env_step_fn(state, action)
                # dx = out.next_state.mjx_data
                # state = out.next_state

            # Copy only the fields needed for visualization to avoid shape mismatch
            # errors with sparse flex fields (flexedge_J has shape (0, nv) in MJX
            # but (0,) in MuJoCo).
            d.qpos[:] = jax.device_get(dx.qpos)
            d.qvel[:] = jax.device_get(dx.qvel)
            d.time = jax.device_get(dx.time)
            mujoco.mj_forward(m, d)
            viewer.sync()

            elapsed = time.time() - start
            if elapsed < m.opt.timestep:
                time.sleep(m.opt.timestep - elapsed)

            n_iters += 1


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
