"""
Multi Point GU (RA-Loop) File for General Task PPO training by Value Decomposition (VDPPO).
"""

import sys
import os
import wandb
import jax
import jax.numpy as jnp
import numpy as np
import pdb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from rraa_rl.src.rl.utils.arguments import get_args
from functools import partial

from rraa_rl.src.rl.utils.utils import optimizer, get_BuRd, tree_index1, tree_index2

from rraa_rl.src.env.general_task.multi_safety_gym import MultiPointGeneralTask
from rraa_rl.src.env.general_task.multi_safety_gym_dynamic_pred import MultiPointDynamicGeneralTask, static_dummy_dynamics, constant_dynamics_with_random_reset, circular_motion_dynamics, obstacle_weave_dynamics
from rraa_rl.src.env.general_task.gym_herding import HerdEnv
from rraa_rl.src.env.wrappers import TransformObservation
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

N_AGENTS = 5
REL_ACEL = [0.25, 1.5, 0.075, 0.075, 0.075]  # relative acceleration limits for each agent
EVADERS = [2, 3, 4]  # agent index of evaders

config = vars(get_args(sys.argv[1:]))

config["REACH_AVOID_LOOP_GAP"] = 3
config["FIXED_VELOCITY"] = None
fxd_vel_tag = '' if not config["FIXED_VELOCITY"] else f"_fxdvel"
pred_tag = 'RAA' if config["DEBUG_JUST_RAA"] else 'RALoop'

config["TASK_SOURCE"] = "G(F reach3_static) && G !obstacles" # DEBUG FIXME not used

# config["EXP_NAME"]="GUtest_MultiPointValDec_GapTest"
config["EXP_NAME"]="GUtest_MultiPointValDec_DynamicPredicate"
config["MODEL_DIR"] = 'model_valdec'
# config["NAME"]=config["DIR"]=f"GU_multi_point_dynpred_constrandreset_{config['N_AGENTS']}ag_RAAbaseline_fxdvel_augnorm"
# config["NAME"]=config["DIR"]=f"GU_multi_point_dynpred_circ_{config['N_AGENTS']}ag_RAAbaseline_fxdvel_augnorm_longeval"
# config["NAME"]=config["DIR"]=f"GU_multi_point{fxd_vel_tag}_dynpred_{config['REACH3_DYNAMIC_PRED_TYPE']}_{config['N_AGENTS']}ag_{pred_tag}"
# config["NAME"]=config["DIR"]="GU_multi_point_{}ag_RAALoop_fxdvel_gap{}_expval_sd{}".format(config["N_AGENTS"], config["REACH_AVOID_LOOP_GAP"], config["SEED"])
config["LR"]=3e-4
config["NUM_STEPS"]=400
config["NUM_EPISODE"]=2000 # DEBUG FIXME not used other than in long eval

if True: # DEBUG FIXME
    config["NUM_ENVS"]=128
    config["TOTAL_TIMESTEPS"]=200_000_000
    config["STEP_SCAN"]=4
    config["UPDATE_EPOCHS"]=10
    config["NUM_MINIBATCHES"]=32
    config["GAMMA_REACH_INIT"]=0.995
    config["GAMMA_REACH_FINAL"]=0.9995
    config["GAE_LAMBDA"]=0.95
    config["CLIP_EPS"]=0.2
    config["ENT_COEF"]=0.005
    config["VF_COEF"]=2.0
    config["MAX_GRAD_NORM"]=0.5
    config["ANNEAL_ENT"]=True
else:
    config["NUM_ENVS"]=256
    # config["TOTAL_TIMESTEPS"]=1_000_000_000
    # config["STEP_SCAN"]=40 # making it a dynamic parameter
    config["UPDATE_EPOCHS"]=10
    config["NUM_MINIBATCHES"]=64
    config["GAMMA_REACH_INIT"]=0.995
    config["GAMMA_REACH_FINAL"]=0.9975
    config["GAE_LAMBDA"]=0.95
    config["CLIP_EPS"]=0.2
    # config["ENT_COEF"]=0.0001
    config["VF_COEF"]=2.0
    config["MAX_GRAD_NORM"]=0.5

config["NAME"]=config["DIR"]=f"herd_init_5ag_1fast1slow"

config["ACTIVATION"]="tanh"
config["CUDA_USE"]="0"
config["ANNEAL_LR"]=True
config['VIDEO_FREQ'] = 25
config["LOAD_DECOMPOSED"] = False
config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
config["MINIBATCH_SIZE"] = int(config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"])

config["EVAL"] = True
config['EVAL_FREQ'] = 25
config["EVAL_HORIZON"] = 2000
assert config["EVAL_HORIZON"] >= config["NUM_EPISODE"], "EVAL_HORIZON must be greater than or equal NUM_EPISODE"
    
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA_USE']

config["USE_WANDB"] = False
if config["USE_WANDB"]:
    wandb.init(project='valdec-{}-{}'.format(config["EXP_NAME"], config["WANDB_GROUP"]), name=config["NAME"], config=config, entity='valdec')


#########################################################################################################################################

def main():

    rng = jax.random.PRNGKey(config["SEED"])

    ## MAKE THE VALUE DAG # DEBUG FIXME, expand valtr

    # # TL source -> AST
    # lexer = TLLexer()
    # tokens = list(lexer.tokenize(config["TASK_SOURCE"]))
    # ast = TLParser(tokens).parse()

    # # AST -> IR
    # ir = IRBuilder()
    # lowerer = Lowerer(builder=ir)
    # ir_root_id = lowerer.lower(ast)

    # # IR passes
    # passes = [PassFinallyToUntil, PassCombineGloballySegments]
    # for p_cls in passes:
    #     p = p_cls(ir)
    #     ir_root_id, ir = p.run(ir_root_id)

    # # IR -> DAG
    # value_dag, dag_root = lower_ir_to_dag(ir, ir_root_id)

    # # DAG passes
    # passes = [PassDuplicateMixedPolarity, PassDuplicateMixedRole, PassFoldConstBool]
    # for p_cls in passes:
    #     p = p_cls(value_dag)
    #     dag_root, value_dag, changed = p.run(dag_root)

    # # Visualize DAG
    # dot_dag = visualize_dag(value_dag, dag_root, filename="dags/value_dag", view=False)

    #########################################################################################################################################

    ## PROCESS VALUE DAG FOR JAX-PPO AUXILIARIES

    # value_dag = process_dag(value_dag, dag_root, reported_nodes="all")

    ## DUMMY DAG FOR NOW

    class DummyRALoopDAG: # DEBUG FIXME placeholder, fake node numbers
        def __init__(self, just_RAA=False):
            trigger_value = 0 if not just_RAA else 1
            node_type = 2 if not just_RAA else 0
            
            self.temporal_nodes = jnp.array([0, 1])
            self.parent_pos_padded = jnp.array([
                [-1],  # node 0 (pos 0) root
                [ 0],  # node 1 (pos 1) parent: 0(pos0)
            ])
            self.trigger_predicate_map = jnp.array([
                [ trigger_value, -1],  # (LOOP): reach vacuously triggers (std RAA): reach_ triggers avoid
                [-1, -1],              # A is a terminal node
            ])
            self.node_types = jnp.array([ # 0: reach-avoid, 1: avoid-only, TODO: 2: reach-only, 3: GF 4: release?
                node_type, 
                1
            ]) 
            self.predicate_types = jnp.array([ # 0: reach, 1: avoid
                0, 
                1
            ])
            self.predicates = ["together", "collisions"]
            self.predicate_ids = jnp.array([0, 1])
            self.negated_predicate_mask=jnp.array([1, 0])
            self.node_index = {0: 0, 1: 1}
            self.reported_nodes = [0, 1]

    value_dag = DummyRALoopDAG(just_RAA=config["DEBUG_JUST_RAA"]) # RAA debugging
    # value_dag = DummyRALoopDAG(just_RAA=False)

    #########################################################################################################################################   

    ## MAKE ENV

    env = HerdEnv(
        active_predicates=value_dag.predicates, 
        negated_predicate_mask=value_dag.negated_predicate_mask,
        n_agents=N_AGENTS,
        rel_acel=REL_ACEL,
        evaders=EVADERS,
        dynamic_predicate_names=None,
    )

    ## Define transformation vectors

    def transform_observation(mean, variance, obs):
        return (obs - mean) / variance
    def untransform_observation(mean, variance, obs):
        return obs * variance + mean

    obs_size = env.observation_space(None).shape[0]
    obs_mean = jnp.zeros(obs_size, dtype=jnp.float32)
    obs_std = jnp.ones(obs_size, dtype=jnp.float32)
    for i in range(config["N_AGENTS"]):
        obs_std = obs_std.at[i * env.obs_size_per_agent].set(2.)
        obs_std = obs_std.at[i * env.obs_size_per_agent + 1].set(2.)

    obs_mean = obs_mean.at[env._env.observation_size].set(60)  # together mean # DEBUG FIXME track and check!
    obs_mean = obs_mean.at[env._env.observation_size+1].set(-20)  # obst
    obs_std = obs_std.at[env._env.observation_size].set(15000)  # together std
    obs_std = obs_std.at[env._env.observation_size+1].set(5000)  # obst

    trans = partial(transform_observation, obs_mean, obs_std)
    untrans = partial(untransform_observation, obs_mean, obs_std)
    env = TransformObservation(env, trans)
    env.set_untransform_obs(untrans)
    env_params = env.default_params

    #########################################################################################################################################

    ## RUN TRAINING

    out = train(env, env_params, value_dag, config, rng, plot_function=plot_multipoint_raa_loop) 

#########################################################################################################################################

## PLOT FUNCTIONS

def plot_multipoint_raa_loop(value_dag, config, traj_batches, scores, timestep, total_timesteps, idx=0):
    pos_raa, pos_a = 0, 1

    # MAKE DIAGNOSTIC PLOTS -- TODO GENERAL TASK LOGIC PLOTTING

    traj_batch_raa = tree_index1(traj_batches, pos_raa)
    traj_batch_a   = tree_index1(traj_batches, pos_a)
    scores_raa = tree_index1(scores, pos_raa)
    scores_a   = tree_index1(scores, pos_a)

    info_raa = tree_index2(traj_batch_raa.info, idx)
    info_a = tree_index2(traj_batch_a.info, idx)

    # DEBUG FIXME just for testing with old plotting
    info_raa['reach_index_1'], info_raa['reach_index_2'], info_raa['reach_index_3'] = np.array(-1), np.array(-1), scores_raa["reach_idx"][idx]
    info_a['reach_index_1'], info_a['reach_index_2'], info_a['reach_index_3'] = np.array(-1), np.array(-1), np.array(-1)

    info_raa['crash_index'] = scores_raa["crash_idx"][idx]
    info_a['crash_index'] = scores_a["crash_idx"][idx]

    policy_decision_sample = traj_batch_raa.current_value_node[:,idx]
    fig = plot_contour_multipoint((info_raa, info_a), timestep, config, policy_decision_sample=policy_decision_sample)
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
            video_frames = plot_video_contour_raa_loop((info_raa, info_a), timestep, config, save_video=True, log_wandb=config["USE_WANDB"])

    # return fig, fig2
    return fig

def _plot_walls(ax):
    """Plot wall boundaries."""
    from rraa_rl.src.env.general_task.gym_herding import SAFETYGYM_RAA_BOX_CUSHION_RADIUS as BOX_RADIUS
    # Plot walls as thick black lines
    ax.plot([-BOX_RADIUS, BOX_RADIUS], [BOX_RADIUS, BOX_RADIUS], 'k-', linewidth=3)  # North
    ax.plot([-BOX_RADIUS, BOX_RADIUS], [-BOX_RADIUS, -BOX_RADIUS], 'k-', linewidth=3)  # South
    ax.plot([BOX_RADIUS, BOX_RADIUS], [-BOX_RADIUS, BOX_RADIUS], 'k-', linewidth=3)  # East
    ax.plot([-BOX_RADIUS, -BOX_RADIUS], [-BOX_RADIUS, BOX_RADIUS], 'k-', linewidth=3)  # West

def _get_agent_keys(info):
    """Extract agent-specific keys from info dict."""
    # Get all x_ keys and sort them by agent index
    x_keys = [k for k in info.keys() if k.startswith('x_') and k[2:].isdigit()]
    x_keys = sorted(x_keys, key=lambda k: int(k.split('_')[1]))
    return [(k, k.replace('x_', 'y_')) for k in x_keys]

def _check_collision(info, step_idx):
    """Check if any pursuer is in collision at this timestep.
    
    Returns:
        Array of booleans indicating which agents are in collision
    """
    from rraa_rl.src.env.general_task.gym_herding import HerdEnv, HERDING_COLLISION_RADIUS, SAFETYGYM_RAA_BOX_CUSHION_RADIUS
    from brax.envs.base import State
    
    # Extract positions for all agents
    agent_keys = _get_agent_keys(info)
    if not agent_keys:
        return jnp.array([False])
    
    positions = []
    for x_key, y_key in agent_keys:
        # info[x_key] is an array of positions across timesteps
        if isinstance(info[x_key], (list, np.ndarray, jnp.ndarray)):
            x_val = float(info[x_key][step_idx])
            y_val = float(info[y_key][step_idx])
        else:
            x_val = float(info[x_key])
            y_val = float(info[y_key])
        positions.append([x_val, y_val])
    positions = jnp.array(positions)
    
    # Create dummy state with these positions
    n_agents = len(positions)
    obs_list = []
    for pos in positions:
        # Double integrator obs: [x, y, vx, vy, ax, ay]
        obs_list.append(jnp.array([pos[0], pos[1], 0., 0., 0., 0.]))
    dummy_obs = jnp.concatenate(obs_list)
    
    dummy_pipeline_state = jnp.zeros((1, 5))
    reward, done = jnp.zeros(2)
    metrics = {}
    dummy_state = State(dummy_pipeline_state, dummy_obs, reward, done, metrics)
    
    # Check collisions for each pursuer
    evaders = EVADERS  # Hard-coded from config
    collision_status = []
    
    for i in range(n_agents):
        if i in evaders:
            collision_status.append(False)
            continue
        
        # Check collision with other agents
        has_collision = False
        for j in range(n_agents):
            if i == j:
                continue
            dist = jnp.linalg.norm(positions[i] - positions[j])
            if dist < HERDING_COLLISION_RADIUS:
                has_collision = True
                break
        
        # Check collision with walls
        if jnp.abs(positions[i][0]) > SAFETYGYM_RAA_BOX_CUSHION_RADIUS or \
           jnp.abs(positions[i][1]) > SAFETYGYM_RAA_BOX_CUSHION_RADIUS:
            has_collision = True
        
        collision_status.append(has_collision)
    
    return jnp.array(collision_status)

def _draw_evader_connections(ax, info, step_idx):
    """Draw lines between evader agents, colored based on distance.
    
    Blue if distance < HERD_TARGET_RADIUS, gray otherwise.
    """
    from rraa_rl.src.env.general_task.gym_herding import HERD_TARGET_RADIUS
    
    evaders = EVADERS  # Hard-coded from config # DEBUG FIXME
    agent_keys = _get_agent_keys(info)
    
    # Get evader positions
    evader_positions = []
    for evader_idx in evaders:
        if evader_idx < len(agent_keys):
            x_key, y_key = agent_keys[evader_idx]
            if isinstance(info[x_key], (list, np.ndarray, jnp.ndarray)):
                x_val = float(info[x_key][step_idx])
                y_val = float(info[y_key][step_idx])
            else:
                x_val = float(info[x_key])
                y_val = float(info[y_key])
            evader_positions.append((x_val, y_val))
    
    # Draw lines between all pairs of evaders
    for i in range(len(evader_positions)):
        for j in range(i + 1, len(evader_positions)):
            pos_i = evader_positions[i]
            pos_j = evader_positions[j]
            
            # Calculate distance
            dist = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
            
            # Color based on distance
            if dist < HERD_TARGET_RADIUS:
                color = 'blue'
                linewidth = 2
                alpha = 0.8
            else:
                color = 'gray'
                linewidth = 1
                alpha = 0.3
            
            ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 
                   color=color, linewidth=linewidth, alpha=alpha)

def _draw_agents(ax, info, step_idx, alpha):
    """Draw all agents at a given step with appropriate coloring for herding task.
    
    Pursuers (agents 0, 1) are drawn with colored fill:
    - Red if in collision
    - Black otherwise
    
    Evaders (agents 2, 3, 4) are drawn with gray fill.
    """
    evaders = EVADERS  # Hard-coded from config
    pursuers = [0, 1]
    agent_colors = ['yellow', 'magenta', 'cyan', 'orange', 'purple', 'brown']
    
    # Check collision status for all agents
    collision_status = _check_collision(info, step_idx)
    
    agent_keys = _get_agent_keys(info)
    if not agent_keys:
        agent_keys = [('x', 'y')]
    
    for agent_idx, (x_key, y_key) in enumerate(agent_keys):
        # Handle both scalar and array values
        if isinstance(info[x_key], (list, np.ndarray, jnp.ndarray)):
            x_val = float(info[x_key][step_idx])
            y_val = float(info[y_key][step_idx])
        else:
            x_val = float(info[x_key])
            y_val = float(info[y_key])
        
        # Determine fill color based on agent type and status
        if agent_idx in evaders:
            face_color = 'gray'  # Evaders are gray
        elif agent_idx in pursuers:
            # Pursuers are red if in collision, black otherwise
            face_color = 'red' if collision_status[agent_idx] else 'black'
        else:
            face_color = 'black'
        
        # Edge color to distinguish agents
        edge_color = agent_colors[agent_idx % len(agent_colors)]
        
        # Plot agent as circle marker
        ax.plot(x_val, y_val, 'o', markersize=12, 
               markerfacecolor=face_color, 
               markeredgecolor=edge_color,
               markeredgewidth=2, 
               alpha=alpha)
    
    # Draw lines between evader agents
    _draw_evader_connections(ax, info, step_idx)

def plot_contour_multipoint(multi_info, epoch, config, policy_decision_sample=None, just_RAA=False):
    info_raa, info_a = multi_info
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    def draw_point_herding(info, title, ax):
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
        reach3_idx = safe_int_convert(info.get('reach_index_3', -1))
        crash_idx = safe_int_convert(info.get('crash_index', -1))
        
        agent_keys = _get_agent_keys(info)
        if not agent_keys:
            agent_keys = [('x', 'y')]
        full_len = info[agent_keys[0][0]].shape[0]
        
        # Plot walls only
        _plot_walls(ax)
        
        # Draw trajectory
        indices = np.linspace(0, full_len - 1, 11, dtype=int)
        for step_n, i in enumerate(indices):
            alpha = (step_n + 1) / 11
            _draw_agents(ax, info, i, alpha)
        
        # Highlight special events
        if reach3_idx > -1:
            _draw_agents(ax, info, reach3_idx, 0.5)
        if crash_idx > -1:
            _draw_agents(ax, info, crash_idx, 0.5)
        
        ax.set_xlim((-3.1, 3.1))
        ax.set_ylim((-3.1, 3.1))
        ax.set_aspect('equal')
        ax.set_title(title)
    
    title = "RA-Loop" if not config["DEBUG_JUST_RAA"] else "RAA"
    draw_point_herding(info_raa, title, axes[0])
    draw_point_herding(info_a, "A", axes[1])
    
    plt.tight_layout()
    plt.savefig('{}/{}/reach/trajectory_{:0>4d}'.format(config["MODEL_DIR"], config["DIR"], epoch), dpi=300)
    return fig

def plot_video_contour_raa_loop(multi_info, epoch, config, save_video=False, prefix="", log_wandb=True):
    start_time = time()
    info_raa, info_a = multi_info
    
    # Get trajectory length
    agent_keys = _get_agent_keys(info_raa)
    if not agent_keys:
        agent_keys = [('x', 'y')]
    full_len = info_raa[agent_keys[0][0]].shape[0]

    # LIMIT number of frames to avoid hanging - max 200 frames
    num_frames = min(full_len // 2, 200)
    indices = np.linspace(0, full_len - 1, num_frames, dtype=int)
    
    def draw_point_herding(step, info, title, ax):
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
        reach3_idx = safe_int_convert(info.get('reach_index_3', -1))
        crash_idx = safe_int_convert(info.get('crash_index', -1))
        
        # Plot walls only
        _plot_walls(ax)
        
        # Draw current agents
        _draw_agents(ax, info, step, 0.9)
        
        # Highlight special events if they've occurred
        if reach3_idx > -1 and step >= reach3_idx:
            _draw_agents(ax, info, reach3_idx, 0.5)
        if crash_idx > -1 and step >= crash_idx:
            _draw_agents(ax, info, crash_idx, 0.5)
        
        ax.set_xlim((-3.1, 3.1))
        ax.set_ylim((-3.1, 3.1))
        ax.set_aspect('equal')
        ax.set_title(title)
    
    # Generate video frames
    frames = []
    for frame_idx, step_n in enumerate(indices):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=100)
        
        title = "RA-Loop" if not config["DEBUG_JUST_RAA"] else "RAA"
        draw_point_herding(step_n, info_raa, title, axes[0])
        draw_point_herding(step_n, info_a, "A", axes[1])

        plt.tight_layout()
        
        # Render to frame
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(frame[:, :, :3])  # Take only RGB channels
        
        plt.close(fig)
    
    # Save video
    if frames:
        frames = [Image.fromarray(frame) for frame in frames]
        if save_video:
            prefix_underscore = prefix.replace("/", "_")
            video_path = '{}/{}/reach/trajectory_{}{:0>4d}.mp4'.format(
                config["MODEL_DIR"], config["DIR"], prefix_underscore, epoch
            )
            print(f"\nSaving video to: {video_path}")
            imageio.mimsave(video_path, frames, fps=30)
            
            if log_wandb:
                wandb_name = f"{prefix}trajectory video"
                print(f"Logging video to wandb: {wandb_name}")
                try:
                    wandb.log({wandb_name: wandb.Video(video_path, format="mp4")}, step=epoch)
                except Exception as e:
                    print(f"Error logging video to wandb: {e}")
        
        end_time = time()
        print(f"Time taken to plot and push video: {end_time - start_time:.2f}s")
    
    return frames
    
if __name__ == "__main__":
    main()