"""
Point RRAA Test File for General Task PPO training by Value Decomposition (VDPPO).
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

from rraa_rl.src.rl.utils.plot_utils import plot_contour_RRAA, plot_policy_decision, plot_video_contour_RRAA
from rraa_rl.src.rl.utils.utils import optimizer, get_BuRd, tree_index1, tree_index2

from rraa_rl.src.env.general_task.safety_gym import PointGeneralTask
from rraa_rl.src.env.wrappers import TransformObservation

from rraa_rl.src.rl.VDPPO import process_dag, train

from valtr.dag_graphviz import visualize_dag
from valtr.dag_passes import PassFoldConstBool, PassDuplicateMixedPolarity, PassDuplicateMixedRole
from valtr.ir_builder import IRBuilder
from valtr.ir_pass import PassCombineGloballySegments, PassFinallyToUntil
from valtr.lowering import Lowerer
from valtr.reachability import dag_to_str, lower_ir_to_dag
from valtr.tl_lexer import TLLexer
from valtr.tl_parser import TLParser

#########################################################################################################################################

## ARGS (FIXED)

config = vars(get_args(sys.argv[1:]))

config["EXP_NAME"]="PointValDec"
config["MODEL_DIR"] = 'model_valdec'
config["DIR"]="point_raa_debug_stage3_auto"
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
config["NAME"]="point_raa_debug_stage3_auto"

config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
config["MINIBATCH_SIZE"] = int(config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"])

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA_USE']

config["USE_WANDB"] = True
if config["USE_WANDB"]:
    wandb.init(project='valdec-DEBUG-{}-{}'.format(config["EXP_NAME"], config["WANDB_GROUP"]), name=config["NAME"], config=config, entity='valdec')

#########################################################################################################################################

## MAKE THE VALUE DAG

TASK_SOURCE = "F reach1 && F reach2 && G !obstacles"

# TL source -> AST
lexer = TLLexer()
tokens = list(lexer.tokenize(TASK_SOURCE))
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

## MAKE POINT ENV

# env = get_env(config) # DEBUG FIXME
# env_params = env.default_params

## DEBUG FIXME, could generalize
def transform_observation(mean, variance, obs):
    return (obs - mean) / variance
def untransform_observation(mean, variance, obs):
    return obs * variance + mean

vec1 = jnp.zeros(7 + len(value_dag.predicates), dtype=jnp.float32)
vec2 = jnp.ones(7 + len(value_dag.predicates), dtype=jnp.float32)
vec2 = vec2.at[0].set(2.)
vec2 = vec2.at[1].set(2.)
# TODO define mean/var for predicate values
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

#########################################################################################################################################

## PLOT FUNCTION

def plot_rraa(value_dag, config, result, scores, timestep, total_timesteps, idx=0):
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

    if config['EXP_NAME'] == 'WindField' or config['EXP_NAME'] == 'WindFieldFull':
        info_rraa['u_air'] = env_params.u_air
        info_rraa['v_air'] = env_params.v_air
        info_rraa['obs'] = env_params.obstacle

    reset_indices = result["reset_indices"]
    policy_decision_sample = traj_batch_rraa.current_value_node[:,idx]
    fig = plot_contour_RRAA((info_rraa, info_raa1, info_raa2, info_a), timestep, config, policy_decision_sample=policy_decision_sample)
    fig2 = plot_policy_decision(policy_decision_sample, timestep, config)

    if config["USE_WANDB"]:
        if "F16" not in config["EXP_NAME"]: # TODO make f16 methods uniform
            wandb.log({
                'trajectory_sample':wandb.Image(fig),
                'policy_decision_sample':wandb.Image(fig2),
            }, step=timestep)
        
    # Save video of trajectory 
    if "F16" not in config["EXP_NAME"] and config["USE_WANDB"]:
        if timestep % config['VIDEO_FREQ'] == 0 or timestep == total_timesteps - 1: 
            video_frames = plot_video_contour_RRAA((info_rraa, info_raa1, info_raa2, info_a), timestep, config, save_video=True, log_wandb=config["USE_WANDB"])

    return fig, fig2

#########################################################################################################################################

## RUN TRAINING

rng = jax.random.PRNGKey(config["SEED"])
out = train(env, env_params, value_dag, config, rng, plot_function=plot_rraa) 