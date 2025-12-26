"""
Multi Point RRAA File for General Task PPO training by Value Decomposition (VDPPO).
"""

import sys
import os
import wandb
import jax
import jax.numpy as jnp
import numpy as np
import pdb
import matplotlib.pyplot as plt

from rraa_rl.src.rl.utils.arguments import get_args
from functools import partial

from rraa_rl.src.rl.utils.utils import optimizer, get_BuRd, tree_index1, tree_index2

from rraa_rl.src.env.general_task.multi_safety_gym import MultiPointGeneralTask
from rraa_rl.src.env.wrappers import TransformObservation

from rraa_rl.src.env.reach_avoid.safety_gym_RRAA import PointRRAA
from jax import jit

from rraa_rl.src.rl.VDPPO import process_dag, train

from valtr.dag_graphviz import visualize_dag
from valtr.dag_passes import PassFoldConstBool, PassDuplicateMixedPolarity, PassDuplicateMixedRole
from valtr.ir_builder import IRBuilder
from valtr.ir_pass import PassCombineGloballySegments, PassFinallyToUntil
from valtr.lowering import Lowerer
from valtr.reachability import dag_to_str, lower_ir_to_dag
from valtr.tl_lexer import TLLexer
from valtr.tl_parser import TLParser

from PIL import Image
import imageio
from time import time 

#########################################################################################################################################

## ARGS (FIXED)

config = vars(get_args(sys.argv[1:]))

config["N_AGENGTS"] = 2
config["TASK_SOURCE"] = "F reach1_any && F reach2_any && G !obstacles"

config["EXP_NAME"]="MultiPointValDec"
config["MODEL_DIR"] = 'model_valdec'
config["NAME"]=config["DIR"]="multi_point_rraa_debug_{N_AGENTS}ag".format(N_AGENTS=config["N_AGENGTS"])
config["LR"]=3e-4
config["NUM_ENVS"]=128
config["NUM_STEPS"]=400
config["TOTAL_TIMESTEPS"]=200_000_000
config["STEP_SCAN"]=8
config["UPDATE_EPOCHS"]=10
config["NUM_MINIBATCHES"]=32
config["GAMMA_ENERGY"]=1.0
config["GAMMA_REACH_INIT"]=0.999
config["GAMMA_REACH_FINAL"]=0.9999
config["GAE_LAMBDA"]=0.95
config["CLIP_EPS"]=0.2
config["ENT_COEF"]=0.01
config["VF_COEF"]=0.5
config["MAX_GRAD_NORM"]=0.5
config["ACTIVATION"]="tanh"
config["CUDA_USE"]="1"
config["ANNEAL_LR"]=True
config["ANNEAL_ENT"]=True
config["LOAD_DECOMPOSED"] = False
config['VIDEO_FREQ'] = 25
config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
config["MINIBATCH_SIZE"] = int(config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"])

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA_USE']

config["USE_WANDB"] = True
if config["USE_WANDB"]:
    wandb.init(project='valdec-{}-{}'.format(config["EXP_NAME"], config["WANDB_GROUP"]), name=config["NAME"], config=config, entity='valdec')


#########################################################################################################################################

def main():

    rng = jax.random.PRNGKey(config["SEED"])

    ## MAKE THE VALUE DAG

    # TL source -> AST
    lexer = TLLexer()
    tokens = list(lexer.tokenize(config["TASK_SOURCE"]))
    ast = TLParser(tokens).parse()

    # AST -> IR
    ir = IRBuilder()
    lowerer = Lowerer(builder=ir)
    ir_root_id = lowerer.lower(ast)

    # IR passes
    passes = [PassFinallyToUntil, PassCombineGloballySegments]
    for p_cls in passes:
        p = p_cls(ir)
        ir_root_id, ir = p.run(ir_root_id)

    # IR -> DAG
    value_dag, dag_root = lower_ir_to_dag(ir, ir_root_id)

    # DAG passes
    passes = [PassDuplicateMixedPolarity, PassDuplicateMixedRole, PassFoldConstBool]
    for p_cls in passes:
        p = p_cls(value_dag)
        dag_root, value_dag, changed = p.run(dag_root)

    # Visualize DAG
    dot_dag = visualize_dag(value_dag, dag_root, filename="dags/value_dag", view=False)

    #########################################################################################################################################

    ## PROCESS VALUE DAG FOR JAX-PPO AUXILIARIES

    value_dag = process_dag(value_dag, dag_root, reported_nodes="all")

    #########################################################################################################################################

    ## MAKE MULTI POINT ENV

    # env = get_env(config) # DEBUG FIXME
    # env_params = env.default_params

    n_agents = 2
    env = MultiPointGeneralTask(
        active_predicates=value_dag.predicates, 
        negated_predicate_mask=value_dag.negated_predicate_mask,
        n_agents=n_agents
    )

    ## Define transformation vectors

    def transform_observation(mean, variance, obs):
        return (obs - mean) / variance
    def untransform_observation(mean, variance, obs):
        return obs * variance + mean

    obs_size = env._env.observation_size + env.n_active_predicates,
    vec1 = jnp.zeros(obs_size, dtype=jnp.float32)
    vec2 = jnp.ones(obs_size, dtype=jnp.float32)
    for i in range(n_agents):
        vec2 = vec2.at[i * env.obs_size_per_agent].set(2.)
        vec2 = vec2.at[i * env.obs_size_per_agent + 1].set(2.)
    # TODO define mean/var for predicate values?

    trans = partial(transform_observation, vec1, vec2)
    untrans = partial(untransform_observation, vec1, vec2)
    env = TransformObservation(env, trans)
    env.set_untransform_obs(untrans)
    env_params = env.default_params

    #########################################################################################################################################

    ## RUN TRAINING

    out = train(env, env_params, value_dag, config, rng, plot_function=plot_multipoint_rraa) 

#########################################################################################################################################

## PLOT FUNCTIONS

def plot_multipoint_rraa(value_dag, config, result, scores, timestep, total_timesteps, idx=0):
    pos_rraa, pos_raa1, pos_raa2, pos_a = 0, 1, 2, 3

    # MAKE DIAGNOSTIC PLOTS -- TODO GENERAL TASK LOGIC PLOTTING

    traj_batches = tree_index1(result['traj_batches'], idx) # first scan step

    traj_batch_rraa = tree_index1(traj_batches, pos_rraa)
    traj_batch_raa1 = tree_index1(traj_batches, pos_raa1)
    traj_batch_raa2 = tree_index1(traj_batches, pos_raa2)
    traj_batch_a    = tree_index1(traj_batches, pos_a)
    scores_rraa = tree_index1(scores, pos_rraa)
    scores_raa1 = tree_index1(scores, pos_raa1)
    scores_raa2 = tree_index1(scores, pos_raa2)
    scores_a    = tree_index1(scores, pos_a)

    info_rraa = tree_index2(traj_batch_rraa.info, idx)
    info_raa1 = tree_index2(traj_batch_raa1.info, idx)
    info_raa2 = tree_index2(traj_batch_raa2.info, idx)
    info_a = tree_index2(traj_batch_a.info, idx)

    info_rraa['reach_index_1'], info_rraa['reach_index_2'] = scores_rraa["each_reach_idx"][:, 0][idx], scores_rraa["each_reach_idx"][:, 1][idx]
    info_raa1['reach_index_1'], info_raa1['reach_index_2'] = scores_raa1["reach_idx"][idx], np.array(-1)
    info_raa2['reach_index_1'], info_raa2['reach_index_2'] = np.array(-1), scores_raa2["reach_idx"][idx]
    info_a['reach_index_1'], info_a['reach_index_2'] = np.array(-1), np.array(-1)

    info_rraa['crash_index'] = scores_rraa["crash_idx"][idx]
    info_raa1['crash_index'] = scores_raa1["crash_idx"][idx]
    info_raa2['crash_index'] = scores_raa2["crash_idx"][idx]
    info_a['crash_index'] = scores_a["crash_idx"][idx]

    reset_indices = result["reset_indices"]
    policy_decision_sample = traj_batch_rraa.current_value_node[:,idx]
    fig = plot_contour_RRAA_old((info_rraa, info_raa1, info_raa2, info_a), timestep, config, policy_decision_sample=policy_decision_sample)
    # fig2 = plot_policy_decision(policy_decision_sample, timestep, config)

    if config["USE_WANDB"]:
        if "F16" not in config["EXP_NAME"]: # TODO make f16 methods uniform
            wandb.log({
                'trajectory_sample':wandb.Image(fig),
                # 'policy_decision_sample':wandb.Image(fig2),
            }, step=timestep)
        
    # Save video of trajectory 
    if "F16" not in config["EXP_NAME"]:
        if timestep % config['VIDEO_FREQ'] == 0 or timestep == total_timesteps - 1: 
            video_frames = plot_video_contour_RRAA_old((info_rraa, info_raa1, info_raa2, info_a), timestep, config, save_video=True, log_wandb=config["USE_WANDB"])

    # return fig, fig2
    return fig

def _get_contour_data():
    """Precompute contour data for reach and avoid predicates."""
    x = np.linspace(-3.1, 3.1, 400)
    y = np.linspace(-3.1, 3.1, 400)
    X, Y = np.meshgrid(x, y)
    positions = np.stack([X, Y], axis=-1)
    
    model = PointRRAA()
    reach1_values = np.array(jit(model.is_reach1)(positions))
    reach2_values = np.array(jit(model.is_reach2)(positions))
    avoid_values = np.array(jit(model.is_avoid)(positions))
    
    return X, Y, reach1_values, reach2_values, avoid_values

def _plot_contours(ax, X, Y, reach1_values, reach2_values, avoid_values, mode="rraa"):
    """Plot contours based on mode."""
    if mode == "rraa":
        ax.contourf(X, Y, np.maximum(np.maximum(reach1_values, reach2_values), avoid_values), alpha=0.3, levels=20)
        ax.contourf(X, Y, np.maximum(reach1_values, avoid_values), levels=[reach1_values.min(), 0], colors=['green'], alpha=0.4)
        ax.contourf(X, Y, np.maximum(reach2_values, avoid_values), levels=[reach2_values.min(), 0], colors=['blue'], alpha=0.4)
    elif mode == "raa1":
        ax.contourf(X, Y, np.maximum(reach1_values, avoid_values), levels=[reach1_values.min(), 0], colors=['green'], alpha=0.4)
        ax.contourf(X, Y, np.maximum(reach1_values, avoid_values), alpha=0.3, levels=20)
    elif mode == "raa2":
        ax.contourf(X, Y, np.maximum(reach2_values, avoid_values), levels=[reach2_values.min(), 0], colors=['blue'], alpha=0.4)
        ax.contourf(X, Y, np.maximum(reach2_values, avoid_values), alpha=0.3, levels=20)
    else:
        ax.contourf(X, Y, avoid_values, alpha=0.3, levels=20)
    ax.contourf(X, Y, avoid_values, levels=[0, avoid_values.max()], colors=['red'], alpha=0.4)

def _get_agent_keys(info):
    """Extract agent-specific keys from info dict."""
    x_keys = sorted([k for k in info.keys() if k.startswith('x_') and k[2:].isdigit()])
    return [(k, k.replace('x_', 'y_')) for k in x_keys]

def _draw_agents(ax, info, step_idx, alpha, reach1_values, reach2_values, avoid_values, mode="rraa"):
    """Draw all agents at a given step with appropriate coloring."""
    model = PointRRAA()
    is_reach1_np, is_reach2_np, is_avoid_np = jit(model.is_reach1), jit(model.is_reach2), jit(model.is_avoid)
    color_dict = {"R1": 'g', "R2": 'b', "A": 'r', "normal": 'k'}
    agent_colors = ['yellow', 'magenta', 'cyan', 'orange', 'purple', 'brown']
    
    agent_keys = _get_agent_keys(info)
    if not agent_keys:
        # Fallback to single agent (original behavior)
        agent_keys = [('x', 'y')]
    
    for agent_idx, (x_key, y_key) in enumerate(agent_keys):
        x_val, y_val = info[x_key][step_idx].item(), info[y_key][step_idx].item()
        
        # Determine color based on predicate satisfaction
        reach1_val = is_reach1_np(np.array([x_val, y_val]))
        reach2_val = is_reach2_np(np.array([x_val, y_val]))
        avoid_val = is_avoid_np(np.array([x_val, y_val]))
        
        if avoid_val > 0.:
            color_mode = "A"
        elif reach1_val < 0. and mode in ["rraa", "raa1"]:
            color_mode = "R1"
        elif reach2_val < 0. and mode in ["rraa", "raa2"]:
            color_mode = "R2"
        else:
            color_mode = "normal"
        
        # Plot agent with border to distinguish multiple agents
        agent_color = agent_colors[agent_idx % len(agent_colors)]
        ax.scatter(x_val, y_val, color=color_dict[color_mode], alpha=alpha, s=100, edgecolors=agent_color, linewidths=2)

def plot_contour_RRAA_old(multi_info, epoch, config, policy_decision_sample=None):
    info_rraa, info_raa1, info_raa2, info_a = multi_info
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    
    # Precompute contour data once
    X, Y, reach1_values, reach2_values, avoid_values = _get_contour_data()
    
    def draw_point_rraa(info, title, ax, mode="a"):
        # Handle infinity and NaN values for reach indices
        def safe_int_convert(val, default=-1):
            if hasattr(val, 'item'):
                val = val.item()
            if np.isinf(val) or np.isnan(val):
                return default
            try:
                return int(val)
            except (ValueError, OverflowError):
                return default
        
        reach1_idx = safe_int_convert(info.get('reach_index_1', -1))
        reach2_idx = safe_int_convert(info.get('reach_index_2', -1))
        crash_idx = safe_int_convert(info.get('crash_index', -1))
        
        agent_keys = _get_agent_keys(info)
        if not agent_keys:
            agent_keys = [('x', 'y')]
        full_len = info[agent_keys[0][0]].shape[0]
        
        # Plot contours
        _plot_contours(ax, X, Y, reach1_values, reach2_values, avoid_values, mode)
        
        # Draw trajectory
        indices = np.linspace(0, full_len - 1, 11, dtype=int)
        for step_n, i in enumerate(indices):
            alpha = (step_n + 1) / 11
            _draw_agents(ax, info, i, alpha, reach1_values, reach2_values, avoid_values, mode)
        
        # Highlight special events
        if reach1_idx > -1 and mode in ["rraa", "raa1"]:
            _draw_agents(ax, info, reach1_idx, 0.5, reach1_values, reach2_values, avoid_values, mode)
        if reach2_idx > -1 and mode in ["rraa", "raa2"]:
            _draw_agents(ax, info, reach2_idx, 0.5, reach1_values, reach2_values, avoid_values, mode)
        if crash_idx > -1:
            _draw_agents(ax, info, crash_idx, 0.5, reach1_values, reach2_values, avoid_values, mode)
        
        ax.set_xlim((-3.1, 3.1))
        ax.set_ylim((-3.1, 3.1))
        ax.set_aspect('equal')
        ax.set_title(title)
    
    draw_point_rraa(info_rraa, "RRAA", axes[0, 0], mode="rraa")
    draw_point_rraa(info_raa1, "RAA 1", axes[0, 1], mode="raa1")
    draw_point_rraa(info_raa2, "RAA 2", axes[1, 0], mode="raa2")
    draw_point_rraa(info_a, "A", axes[1, 1], mode="a")
    
    plt.tight_layout()
    plt.savefig('{}/{}/reach/trajectory_{:0>4d}'.format(config["MODEL_DIR"], config["DIR"], epoch), dpi=300)
    return fig

def plot_video_contour_RRAA_old(multi_info, epoch, config, save_video=False, prefix="", log_wandb=True):
    start_time = time()
    info_rraa, info_raa1, info_raa2, info_a = multi_info
    
    # Precompute contour data once
    X, Y, reach1_values, reach2_values, avoid_values = _get_contour_data()
    
    def draw_point_rraa(step, info, title, ax, mode="rraa"):
        # Handle infinity and NaN values for reach indices
        def safe_int_convert(val, default=-1):
            if hasattr(val, 'item'):
                val = val.item()
            if np.isinf(val) or np.isnan(val):
                return default
            try:
                return int(val)
            except (ValueError, OverflowError):
                return default
        
        reach1_idx = safe_int_convert(info.get('reach_index_1', -1))
        reach2_idx = safe_int_convert(info.get('reach_index_2', -1))
        crash_idx = safe_int_convert(info.get('crash_index', -1))
        
        # Plot contours
        _plot_contours(ax, X, Y, reach1_values, reach2_values, avoid_values, mode)
        
        # Draw current agents
        _draw_agents(ax, info, step, 0.9, reach1_values, reach2_values, avoid_values, mode)
        
        # Highlight special events if they've occurred
        if reach1_idx > -1 and step >= reach1_idx and mode in ["rraa", "raa1"]:
            _draw_agents(ax, info, reach1_idx, 0.5, reach1_values, reach2_values, avoid_values, mode)
        if reach2_idx > -1 and step >= reach2_idx and mode in ["rraa", "raa2"]:
            _draw_agents(ax, info, reach2_idx, 0.5, reach1_values, reach2_values, avoid_values, mode)
        if crash_idx > -1 and step >= crash_idx:
            _draw_agents(ax, info, crash_idx, 0.5, reach1_values, reach2_values, avoid_values, mode)
        
        ax.set_xlim((-3.1, 3.1))
        ax.set_ylim((-3.1, 3.1))
        ax.set_aspect('equal')
        ax.set_title(title)
    
    # Get trajectory length
    agent_keys = _get_agent_keys(info_rraa)
    if not agent_keys:
        agent_keys = [('x', 'y')]
    full_len = info_rraa[agent_keys[0][0]].shape[0]
    
    # Generate video frames
    frames = []
    num_frames = full_len // 2
    indices = np.linspace(0, full_len - 1, num_frames, dtype=int)
    
    for step_n in indices:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=100)
        
        draw_point_rraa(step_n, info_rraa, "RRAA", axes[0, 0], mode="rraa")
        draw_point_rraa(step_n, info_raa1, "RAA 1", axes[0, 1], mode="raa1")
        draw_point_rraa(step_n, info_raa2, "RAA 2", axes[1, 0], mode="raa2")
        draw_point_rraa(step_n, info_a, "A", axes[1, 1], mode="a")
        
        plt.tight_layout()
        
        # Render to frame
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(frame)
        
        plt.close(fig)
        plt.close("all")
    
    # Save video
    if frames:
        frames = [Image.fromarray(frame) for frame in frames]
        if save_video:
            prefix_underscore = prefix.replace("/", "_")
            video_path = '{}/{}/reach/trajectory_{}{:0>4d}.mp4'.format(
                config["MODEL_DIR"], config["DIR"], prefix_underscore, epoch
            )
            print("\n\nSaving video to: ", video_path)
            imageio.mimsave(video_path, frames, fps=30)
            
            if log_wandb:
                wandb_name = f"{prefix}trajectory video"
                print("Logging video to wandb: ", wandb_name)
                try:
                    wandb.log({wandb_name: wandb.Video(video_path, format="mp4")}, step=epoch)
                except:
                    print("Error logging video to wandb")
        
        end_time = time()
        print("Time taken to plot and push video: ", end_time - start_time)
    
    return frames
    
if __name__ == "__main__":
    main()