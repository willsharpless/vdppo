"""
File for General Task PPO training by Value Decomposition.
"""
from statistics import variance
import sys
# sys.path.append("/home/mepear_gc")

import os
import time
import wandb
import jax
import jax.numpy as jnp
import numpy as np
import pdb
import matplotlib.pyplot as plt

from flax.training import train_state
from flax.training import checkpoints

from rraa_rl.src.rl.utils.arguments import get_args
from functools import partial
from typing import Any

from rraa_rl.src.rl.utils.alg_utils import _ppo_vanilla_update, _env_step_general_task
from rraa_rl.src.env.env_list import get_env
from rraa_rl.src.model.actorcritic import Policy_Network, Value_Network, Policy_Network_Discrete, MoGPolicy_Network
from rraa_rl.src.rl.utils.plot_utils import plot_contour_RRAA, plot_policy_decision, plot_video_contour_RRAA
from rraa_rl.src.rl.utils.utils import optimizer, get_BuRd, tree_index1, tree_index2
from rraa_rl.src.rl.utils.gae import calculate_gae_reachavoid4, calculate_gae_avoid4, calculate_indexs3_rr

from rraa_rl.src.env.general_task.safety_gym import PointGeneralTask
from rraa_rl.src.env.wrappers import TransformObservation

from rraa_rl.src.rl.VDPPO import train

config = vars(get_args(sys.argv[1:]))

config["EXP_NAME"]="PointValDec"
config["MODEL_DIR"] = 'model_valdec'
config["DIR"]="point_raa_debug_stage3_scan8_newscore_cleaned"
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
config["CUDA_USE"]="0"
config["ANNEAL_LR"]=True
config["ANNEAL_ENT"]=True
config["NAME"]="point_raa_debug_stage3_scan8_newscore_cleaned"

config["NUM_UPDATES"] = int(
    config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
)
config["MINIBATCH_SIZE"] = int(
    config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA_USE']

folder = os.path.exists("{}/{}".format(config["MODEL_DIR"], config['DIR']))
if not folder:
    os.makedirs("{}/{}".format(config["MODEL_DIR"], config['DIR']))
    os.makedirs("{}/{}/reach".format(config["MODEL_DIR"], config['DIR']))
    os.makedirs("{}/{}/policy".format(config["MODEL_DIR"], config['DIR']))
    os.makedirs("{}/{}/value".format(config["MODEL_DIR"], config['DIR']))
    os.makedirs("{}/{}/total".format(config["MODEL_DIR"], config['DIR']))
    os.makedirs("{}/{}/target".format(config["MODEL_DIR"], config['DIR']))
    os.makedirs("{}/{}/value_target".format(config["MODEL_DIR"], config['DIR']))
    os.makedirs("{}/{}/state_traj".format(config["MODEL_DIR"], config['DIR']))

# env = get_env(config) # DEBUG FIXME
# env_params = env.default_params

## MAKE THE VALUE DAG
# value_dag = valt.make_value_dag(config)

class DummyRRAAdag: # DEBUG FIXME placeholder, fake node numbers
    def __init__(self):
        # fixed topological order: [RRAA(0), RAA1(1), RAA2(2), A(3)]
        self.nodes = jnp.array([0, 1, 2, 3])
        self.node_tags = {0: 'RRAA', 1: 'RAA1', 2: 'RAA2', 3: 'A'}
        # parent positions (padded with -1 to fixed width 2)
        self.parent_pos_padded = jnp.array([
            [-1, -1],  # node 0 (pos 0) root
            [ 0, -1],  # node 1 (pos 1) parent: 0(pos0)
            [ 0, -1],  # node 2 (pos 2) parent: 0(pos0)
            [ 1,  2],  # node 3 (pos 3) parents: 1(pos1), 2(pos2)
        ])
        # trigger predicate index for each node (in same order as nodes)
        self.trigger_predicate_map = jnp.array([
            [ 2,  1, -1],  # RRAA can reach1 (-> RAA2) or reach2 (-> RAA1)
            [ 3, -1, -1],  # RAA1 can reach1 (-> A)
            [-1,  3, -1],  # RAA2  can reach2 (-> A)
            [-1, -1, -1],  # A is a terminal node
        ]) # (N, P) where entry is the child pos we switch to upon satisfaction of that predicate, or -1 if none
        self.node_types = jnp.array([ # 0: reach-avoid, 1: avoid-only, TODO: 2: reach-only, 3: GF 4: release?
            0, 
            0, 
            0, 
            1
        ]) 
        self.predicate_types = jnp.array([ # 0: reach, 1: avoid
            0, 
            0, 
            1
        ])
        self.predicates = ["reach1", "reach2", "obstacles"]
        self.negated_predicate_mask=jnp.array([1, 1, 0]) # NOTE this is used in env directly, < 0 triggers reach-preds, > 0 triggers avoid-preds
        self.node_index = {0: 0, 1: 1, 2: 2, 3: 3}
        self.reported_nodes = [0, 1, 2, 3]
value_dag = DummyRRAAdag()

## DEBUG FIXME PLACEHOLDER for env/env_params modifications
def transform_observation(mean, variance, obs):
    return (obs - mean) / variance
def untransform_observation(mean, variance, obs):
    return obs * variance + mean

vec1 = jnp.zeros(10, dtype=jnp.float32)
vec2 = jnp.ones(10, dtype=jnp.float32)
vec2 = vec2.at[0].set(2.)
vec2 = vec2.at[1].set(2.)
trans = partial(transform_observation, vec1, vec2)
untrans = partial(untransform_observation, vec1, vec2)
env = PointGeneralTask(
    active_predicates=value_dag.predicates, 
    negated_predicate_mask=value_dag.negated_predicate_mask
)
env = TransformObservation(env, trans)
env.set_untransform_obs(untrans)
env_params = env.default_params

if config['EXP_NAME'] == 'WindField':
    env_params = env_params.replace(index=config['SECTION'])

def plot_rraa(value_dag, config, result, scores, timestep, total_timesteps, idx=0):
    pos_rraa, pos_raa1, pos_raa2, pos_a = 0, 1, 2, 3

    # MAKE DIAGNOSTIC PLOTS -- FIXME for GENERAL TASK LOGIC

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

    # DEBUG FIXME just for testing with old plotting
    info_rraa['reach_index_1'], info_rraa['reach_index_2'] = scores_rraa["each_reach_idx"][:, 0][idx], scores_rraa["each_reach_idx"][:, 1][idx]
    info_raa1['reach_index_1'], info_raa1['reach_index_2'] = scores_raa1["reach_idx"][idx], np.array(-1)
    info_raa2['reach_index_1'], info_raa2['reach_index_2'] = np.array(-1), scores_raa2["reach_idx"][idx]
    info_a['reach_index_1'], info_a['reach_index_2'] = np.array(-1), np.array(-1)

    info_rraa['crash_index'] = scores_rraa["crash_idx"][idx]
    info_raa1['crash_index'] = scores_raa1["crash_idx"][idx]
    info_raa2['crash_index'] = scores_raa2["crash_idx"][idx]
    info_a['crash_index'] = scores_a["crash_idx"][idx]

    if config['EXP_NAME'] == 'WindField' or config['EXP_NAME'] == 'WindFieldFull':
        info_rraa['u_air'] = env_params.u_air
        info_rraa['v_air'] = env_params.v_air
        info_rraa['obs'] = env_params.obstacle

    # policy_decision_sample = traj_batch_rraa.policy_taken[:,idx]
    reset_indices = result["reset_indices"]
    policy_decision_sample = traj_batch_rraa.current_value_node[:,idx] ## DEBUG FIXME
    fig = plot_contour_RRAA((info_rraa, info_raa1, info_raa2, info_a), timestep, config, policy_decision_sample=policy_decision_sample)
    fig2 = plot_policy_decision(policy_decision_sample, timestep, config)

    if config["USE_WANDB"]:
        if "F16" not in config["EXP_NAME"]: # FIXME make f16 methods uniform
            wandb.log({
                'trajectory_sample':wandb.Image(fig),
                'policy_decision_sample':wandb.Image(fig2),
            }, step=timestep)
        
    # Save video of trajectory 
    if "F16" not in config["EXP_NAME"] and config["USE_WANDB"]:
        if timestep % config['VIDEO_FREQ'] == 0 or timestep == total_timesteps - 1: 
            video_frames = plot_video_contour_RRAA((info_rraa, info_raa1, info_raa2, info_a), timestep, config, save_video=True, log_wandb=config["USE_WANDB"])

    return fig, fig2

config["USE_WANDB"] = True #not debug # False for debugging
if config["USE_WANDB"]:
    wandb.init(project='valdec-DEBUG-{}-{}'.format(config["EXP_NAME"], config["WANDB_GROUP"]), name=config["NAME"], config=config,
                entity='valdec')

config["LOAD_DECOMPOSED"] = False
# if config["LOAD_DECOMPOSED"]:
#     config["LOAD_DEC_DIR"] ="hopper_reachreach_idxsMAX_switchfix_augstate_obsfix_long"
#     config["LOAD_DEC_DIR_MODEL"] ="checkpoint_859"

if 'VIDEO_FREQ' not in config.keys():
    if 'Humanoid' in config['EXP_NAME']:
        config['VIDEO_FREQ'] = 200
    else:
        config['VIDEO_FREQ'] = 25

rng = jax.random.PRNGKey(config["SEED"])
out = train(env, env_params, value_dag, config, rng, plot_function=plot_rraa) 