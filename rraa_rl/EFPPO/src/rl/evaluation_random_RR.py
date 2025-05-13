import os
import optax
import jax
from jax import lax
import sys
import numpy as np

from functools import partial
from flax.training.train_state import TrainState
from flax.training import checkpoints
import jax.numpy as jnp

import matplotlib.pyplot as plt
from PIL import Image
import imageio

from rraa_rl.EFPPO.src.rl.arguments import get_args
from rraa_rl.EFPPO.src.env.env_list import get_env
from rraa_rl.EFPPO.src.model.actorcritic import Policy_Network, Value_Network, ActorCritic_Continuous, Policy_Network_Discrete
from rraa_rl.EFPPO.src.rl.EFPPO_utils import _env_step_rr_vanilla, _env_step_rr_deterministic
# from rraa_rl.EFPPO.src.rl.plot_utils import calculate_reachreach
from rraa_rl.EFPPO.src.rl.root_finding import Bisection
from rraa_rl.EFPPO.src.rl.utils import tree_index1, tree_index2, optimizer

def calculate_reachreach(traj_batch, reach_type="both"):
    
    # Compute first reaching idx
    reach_idx_1 = (traj_batch.reach1 < 0).argmax(axis=0) if reach_type in ["both", "1"] else None
    reach_idx_2 = (traj_batch.reach2 < 0).argmax(axis=0) if reach_type in ["both", "2"] else None
    reach_idx_1 = jnp.where(jnp.any((traj_batch.reach1 < 0) == 1, axis=0), reach_idx_1, jnp.inf) if reach_type in ["both", "1"] else None
    reach_idx_2 = jnp.where(jnp.any((traj_batch.reach2 < 0) == 1, axis=0), reach_idx_2, jnp.inf) if reach_type in ["both", "2"] else None
    reach_idx = jnp.maximum(reach_idx_1, reach_idx_2) if reach_type in ["both"] else None

    # Compute
    reach_1_perc = (reach_idx_1 < jnp.inf).sum() / reach_idx_1.__len__() if reach_type in ["both", "1"] else None
    reach_2_perc = (reach_idx_2 < jnp.inf).sum() / reach_idx_2.__len__() if reach_type in ["both", "2"] else None
    reach_perc = (reach_idx < jnp.inf).sum() / reach_idx.__len__() if reach_type in ["both"] else None

    reach_percs = (reach_1_perc.item(), reach_2_perc.item(), reach_perc.item())
    reach_idxs = (reach_idx_1, reach_idx_2, reach_idx)
    return reach_percs, reach_idxs

def plot_traj_sample(traj_batch, traj_batch_d, config, sample_size=10, make_video=False):

    if config['EXP_NAME'] == 'HopperReachReach':

        plt.figure(figsize=(20, 5), constrained_layout=True)
        fig, axes = plt.subplots(sample_size, 2)

        def draw_hopper_rr(info, title, ax, target_type="both", plot_until="success", plot_freq=50, make_video=False, video_step=0):
            reach_idx_1 = info['reach_index_1']
            reach_idx_2 = info['reach_index_2']

            if plot_until == "success":
                full_len = jnp.maximum(reach_idx_1, reach_idx_2)
                full_len = info['head_pos'].shape[0] if full_len.item() == jnp.inf else int(full_len.item())
            else:
                full_len = info['head_pos'].shape[0]
            reach_idx_1 = int(reach_idx_1.item()) if reach_idx_1.item() != jnp.inf else -1
            reach_idx_2 = int(reach_idx_2.item()) if reach_idx_2.item() != jnp.inf else -1

            draw_circle = plt.Circle((2.0, 1.4), 0.1, edgecolor="green", linewidth=2, fill=False)
            # draw_circle2 = plt.Circle((-2.0, 1.4), 0.1, edgecolor="blue", linewidth=2, fill=False)
            draw_circle2 = plt.Circle((0., 1.4), 0.1, edgecolor="blue", linewidth=2, fill=False)
            
            if target_type == "both":
                ax.add_patch(draw_circle)
                ax.add_patch(draw_circle2)
            elif target_type == "1":
                ax.add_patch(draw_circle)
            elif target_type == "2":
                ax.add_patch(draw_circle2)

            def draw_body(ax, info, i, alpha, color_mode="normal"):
                if color_mode == "R1":
                    c1, c2, c3, c4, c5 = 'g', 'g', 'g', 'g', 'g'
                    linewidth=3
                elif color_mode == "R2":
                    c1, c2, c3, c4, c5 = 'b', 'b', 'b', 'b', 'b'
                    linewidth=3
                else:
                    c1, c2, c3, c4, c5 = 'r', 'g', 'b', 'b', 'm'
                    linewidth=1
                ax.plot(np.array([info['head_pos'][i, 0], info['jaw_pos'][i, 0]]),
                        np.array([info['head_pos'][i, 1], info['jaw_pos'][i, 1]]), c=c1, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['jaw_pos'][i, 0], info['thg_pos'][i, 0]]),
                        np.array([info['jaw_pos'][i, 1], info['thg_pos'][i, 1]]), c=c2, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['thg_pos'][i, 0], info['leg_pos'][i, 0]]),
                        np.array([info['thg_pos'][i, 1], info['leg_pos'][i, 1]]), c=c3, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['leg_pos'][i, 0], info['foot_front_pos'][i, 0]]),
                        np.array([info['leg_pos'][i, 1], info['foot_front_pos'][i, 1]]), c=c4, alpha=alpha, linewidth=linewidth)
                ax.plot(np.array([info['leg_pos'][i, 0], info['foot_back_pos'][i, 0]]),
                        np.array([info['leg_pos'][i, 1], info['foot_back_pos'][i, 1]]), c=c5, alpha=alpha, linewidth=linewidth)
            
            if not make_video:
                for i in range(0, full_len, plot_freq):
                    alpha = 0.3 + 0.3 * (i/full_len)
                    draw_body(ax, info, i, alpha)

                if reach_idx_1 > -1 and (target_type == "both" or target_type == "1"):
                    draw_body(ax, info, reach_idx_1, 0.9, color_mode = "R1")

                if reach_idx_2 > -1 and (target_type == "both" or target_type == "2"):
                    draw_body(ax, info, reach_idx_2, 0.9, color_mode = "R2")

            else:
                alpha = 0.9
                draw_body(ax, info, video_step, alpha)

                if video_step > reach_idx_1 and reach_idx_1 > -1 and (target_type == "both" or target_type == "1"):
                    draw_body(ax, info, reach_idx_1, alpha, color_mode = "R1")

                if video_step > reach_idx_2 and reach_idx_2 > -1 and (target_type == "both" or target_type == "2"):
                    draw_body(ax, info, reach_idx_2, alpha, color_mode = "R2")

            ax.set_xlim((-2.5, 2.5))
            ax.set_xlim((-0.5, 2.5))
            ax.set_ylim((-0.1, 1.6))
            # ax.set_aspect('equal')
            
            ax.set_title(title)

        _, reach_idxs  = calculate_reachreach(traj_batch)
        (reach_idx_1, reach_idx_2, reach_idx) = reach_idxs
        _, reach_idxs_d = calculate_reachreach(traj_batch_d)   
        (reach_idx_1_d, reach_idx_2_d, reach_idx_d) = reach_idxs_d
        
        for k in range(sample_size):
            info = tree_index2(traj_batch.info, k)
            info_d = tree_index2(traj_batch_d.info, k)
            info['reach_index_1'], info['reach_index_2'] = reach_idx_1[k], reach_idx_2[k]
            info_d['reach_index_1'], info_d['reach_index_2'] = reach_idx_1_d[k], reach_idx_2_d[k]
            title_1 = "Reach Reach - Stochastic" if k == 0 else ""
            title_2 = "Reach Reach - Deterministic" if k == 0 else ""
            draw_hopper_rr(info, title_1, axes[k, 0], plot_freq=50)
            draw_hopper_rr(info_d, title_2, axes[k, 1], plot_freq=50)

        plt.savefig(f"model/{config['DIR']}/test/{config['DIR_MODEL']}_trajectory_plot", dpi=300)
        plt.close()

        if make_video:
            frames = []
            full_len = info['head_pos'].shape[0]
            num_frames = full_len//2
            indices = np.linspace(0, full_len, num_frames, dtype=int)

            for step_n in indices:

                plt.figure(figsize=(20, 5), constrained_layout=True, dpi=100)
                fig, axes = plt.subplots(sample_size, 2)

                for k in range(sample_size):
                    info = tree_index2(traj_batch.info, k)
                    info_d = tree_index2(traj_batch_d.info, k)
                    info['reach_index_1'], info['reach_index_2'] = reach_idx_1[k], reach_idx_2[k]
                    info_d['reach_index_1'], info_d['reach_index_2'] = reach_idx_1_d[k], reach_idx_2_d[k]
                    title_1 = "Reach Reach - Stochastic" if k == 0 else ""
                    title_2 = "Reach Reach - Deterministic" if k == 0 else ""
                    draw_hopper_rr(info, title_1, axes[k, 0], make_video=True, video_step=step_n)
                    draw_hopper_rr(info_d, title_2, axes[k, 1], make_video=True, video_step=step_n)
                
                fig.canvas.draw()
                frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # RGBA (4 channels)
                frames.append(frame)
                
                plt.close(fig)
                plt.close("all")
            
            frames = [Image.fromarray(frame) for frame in frames]

            video_path = f"model/{config['DIR']}/test/{config['DIR_MODEL']}_trajectory_video.mp4"
            imageio.mimsave(video_path, frames, fps=30)

        return fig
    
def plot_RR_value(traj_batch, traj_batch_d, config):

    plt.figure(figsize=(5, 10), constrained_layout=True)
    fig, axes = plt.subplots(1, 2)

    def draw_vals(batch, title, ax):

        # Cumulative min over time for each batch column
        cummin1 = lax.associative_scan(jnp.minimum, batch.reach1, axis=0)  # [T, B]
        cummin2 = lax.associative_scan(jnp.minimum, batch.reach2, axis=0)  # [T, B]
        # score = jnp.maximum(cummin1, cummin2)  # [T, B]

        # Compute summary statistics over batch for ribbon plot
        median_1 = jnp.median(cummin1, axis=1)
        q25_1 = jnp.percentile(cummin1, 25, axis=1)
        q75_1 = jnp.percentile(cummin1, 75, axis=1)

        median_2 = jnp.median(cummin2, axis=1)
        q25_2 = jnp.percentile(cummin2, 25, axis=1)
        q75_2 = jnp.percentile(cummin2, 75, axis=1)

        # Plot
        timesteps = jnp.arange(cummin1.shape[0])
        ax.fill_between(timesteps, q25_1, q75_1, alpha=0.3, label="min<t l1", color="green")
        ax.fill_between(timesteps, q25_2, q75_2, alpha=0.3, label="min<t l2", color="blue")
        ax.plot(timesteps, median_1, color="green")
        ax.plot(timesteps, median_2, color="blue")
        ax.set_xlabel("Trajectory Step")
        ax.set_title(title)
        ax.legend()

    draw_vals(traj_batch, "Stochastic Policy", axes[0])
    draw_vals(traj_batch_d, "Deterministic Policy", axes[1])

    plt.savefig(f"model/{config['DIR']}/test/{config['DIR_MODEL']}_value_plot", dpi=300)
    return fig

def test(envs, env_paramss, config, rng_og):

    env, env_reach_1, env_reach_2 = envs # COMPOSED (RR) + 2 DECOMPOSED (R1 + R2)
    env_params, env_params_reach_1, env_params_reach_2 = env_paramss

    # DEFINE ENV STEP WRAPPERS
    env_step = partial(_env_step_rr_vanilla, env, env_params)
    env_step_d = partial(_env_step_rr_deterministic, env, env_params)
    tx = optimizer(config)

    ## LOAD POLICY NETWORKS
    if config["DISCRETE"] == False:
        policy_network = Policy_Network(
            env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
        )
        policy_network_reach1 = Policy_Network(
            env_reach_1.action_space(env_params_reach_1).shape[0], activation=config["ACTIVATION"]
        )
        policy_network_reach2 = Policy_Network(
            env_reach_2.action_space(env_params_reach_2).shape[0], activation=config["ACTIVATION"]
        )
    else:
        raise NotImplementedError()

    raw_restored = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath('model/{}/{}'.format(
        config["DIR"], config["DIR_MODEL"])), target=None)

    train_state_policy = TrainState.create(
        apply_fn=policy_network.apply,
        params=raw_restored['policy_network']['params'],
        tx=tx,
    )    
    train_state_policy_reach1 = TrainState.create(
        apply_fn=policy_network_reach1.apply,
        params=raw_restored['policy_reach1_network']['params'],
        tx=tx,
    )
    train_state_policy_reach2 = TrainState.create(
        apply_fn=policy_network_reach2.apply,
        params=raw_restored['policy_reach2_network']['params'],
        tx=tx,
    )

    ## LOAD VALUE NETWORKS
    value_network = Value_Network(activation=config["ACTIVATION"])
    train_state_value = TrainState.create(
        apply_fn=value_network.apply,
        params=raw_restored['value_network']['params'],
        tx=tx,
    )
    value_network_reach1 = Value_Network(activation=config["ACTIVATION"])
    train_state_value_reach1 = TrainState.create(
        apply_fn=value_network_reach1.apply,
        params=raw_restored['value_reach1_network']['params'],
        tx=tx,
    )
    value_network_reach2 = Value_Network(activation=config["ACTIVATION"])
    train_state_value_reach2 = TrainState.create(
        apply_fn=value_network_reach2.apply,
        params=raw_restored['value_reach2_network']['params'],
        tx=tx,
    )

    ## ROLL OUT TRAJECTORY (STOCHASTIC)
    rng, _rng = jax.random.split(rng_og)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
    rng, _rng = jax.random.split(rng)
    runner_state_standard = (train_state_policy, train_state_value, env_state, obsv, _rng)
    
    # SPECIAL DECOMPOSED STATES
    decomposed_state = (train_state_policy_reach1, train_state_value_reach1, train_state_policy_reach2, train_state_value_reach2)
    policy_controls = (False, False, False)
    runner_state = (*runner_state_standard, decomposed_state, policy_controls)

    # COLLECT TRAJECTORY COMPOSED
    runner_state, traj_batch = jax.lax.scan(
        env_step, runner_state, None, config["NUM_STEPS"]
    )

    ## ROLL OUT TRAJECTORY (DETERMINISTIC)
    rng, _rng = jax.random.split(rng_og)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obsv_d, env_state_d = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
    rng, _rng = jax.random.split(rng)
    runner_state_standard_d = (train_state_policy, train_state_value, env_state_d, obsv_d, _rng)
    runner_state_d = (*runner_state_standard_d, decomposed_state, policy_controls)

    # COLLECT TRAJECTORY COMPOSED
    runner_state_d, traj_batch_d = jax.lax.scan(
        env_step_d, runner_state_d, None, config["NUM_STEPS"]
    )
    
    return traj_batch, traj_batch_d

if __name__ == "__main__":
    config = vars(get_args(sys.argv[1:]))

    debug = False
    if debug:
        config["EXP_NAME"]="HopperReachReach"
        config["DIR"]="hopper_reachreach_halfwidth_R10"
        config["DIR_MODEL"]="checkpoint_975"
        config["NUM_ENVS"]=128
        config["NUM_STEPS"]=500
        config["ACTIVATION"]="tanh"

    envs = get_env(config)
    env, env_1, env_2 = envs
    env_paramss = (env.default_params, env_1.default_params, env_2.default_params)
    rng = jax.random.PRNGKey(20)
    folder = os.path.exists("model/{}/traj".format(config['DIR']))

    # if config['EXP_NAME'] == 'WindField': 
    #     env_params = env_params.replace(index=config['SECTION'])
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    (result_traj_batch, result_traj_batch_deterministic) = test(envs, env_paramss, config, rng)

    ((reach_1_perc, reach_2_perc, reach_perc), 
     (reach_idx_1, reach_idx_2, reach_idx)) = calculate_reachreach(result_traj_batch)
    
    ((reach_1_perc_d, reach_2_perc_d, reach_perc_d), 
     (reach_idx_1_d, reach_idx_2_d, reach_idx_d)) = calculate_reachreach(result_traj_batch_deterministic)
    
    print("\nSCORES")
    print(f" STOCH - REACH-REACH : {100*reach_perc:0.1f}%")
    print(f" STOCH - REACH-1     : {100*reach_1_perc:0.1f}%")
    print(f" STOCH - REACH-2     : {100*reach_2_perc:0.1f}%")
    print(f" DETER - REACH-REACH : {100*reach_perc_d:0.1f}%")
    print(f" DETER - REACH-1     : {100*reach_1_perc_d:0.1f}%")
    print(f" DETER - REACH-2     : {100*reach_2_perc_d:0.1f}%")
    print("")

    os.makedirs(f"model/{config['DIR']}/test", exist_ok=True)
    val_fig = plot_RR_value(result_traj_batch, result_traj_batch_deterministic, config)
    traj_fig = plot_traj_sample(result_traj_batch, result_traj_batch_deterministic, config, sample_size=5, make_video=True)
