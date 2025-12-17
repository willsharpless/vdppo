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
from rraa_rl.src.rl.utils.plot_utils import (calculate_minimal_reach, calculate_consumption, 
                                             calculate_reachreach, calculate_reachalwaysavoid, calculate_reachavoid,
                                             plot_target, plot_value_target, plot_contour, plot_contour_RRAA, 
                                             plot_policy_decision, plot_video_contour_RRAA, calculate_rraa, calculate_rraa_TEMP_VDPPO)
from rraa_rl.src.rl.utils.utils import optimizer, get_BuRd, tree_index1, tree_index2
from rraa_rl.src.rl.utils.gae import (Transition_reach,
                              calculate_gae, calculate_gae2, calculate_gae3,
                              calculate_gae_reach, calculate_gae_reach2, calculate_gae_reach3, calculate_gae_reach4, calculate_gae_reachavoid4, calculate_gae_avoid4,
                              calculate_indexs, calculate_indexs2, calculate_indexs3, calculate_indexs3_rr, calculate_indexs_rr)

from rraa_rl.src.env.general_task.safety_gym import PointGeneralTask
from rraa_rl.src.env.wrappers import TransformObservation

from rraa_rl.src.env.reach_avoid.humanoid_RR import HUMANOID_TORSO_MIN_Z, HUMANOID_TORSO_MAX_Z

class TrainState(train_state.TrainState):
    mean: Any
    variance: Any
    count: Any

### SCRIPTING TODO/FIXME/NOTE
# - valtr dag creation
# - dag transition fn
# - iterative rollouts
# - iterative gae comps
# - iterative model updates
# - generalized scoring method
# - generalized reset mechanism
# - FIX the done-reset mechanism
# - FIX sequential-policy mask -> reset instead (true seq rollout in eval)

## OPEN QUESTIONS
# - one vs multiple representations? (one per node vs shared)

def train(env, env_params, value_dag, config, rng):
    def _train(train_state_total, ent_gamma):
        
        ####################################################################################################################
        # ROLLOUT BATCHES FOR EACH NODE

        # Pre-allocate functional buffers to carry parent data across node rollouts
        num_nodes = value_dag.nodes.shape[0]
        obs_shape = env.observation_space(env_params).shape
        num_predicates = len(value_dag.predicates)

        def roll_out_node(carry, node_pos):
            (train_state_policies,
                train_state_values,
                rng,
                timestep,
                obs_buffer,              # [num_nodes, config["NUM_STEPS"], config["NUM_ENVS"], *obs_shape]
                predicate_extrema_buffer          # [num_nodes, config["NUM_STEPS"], config["NUM_ENVS"], num_predicates]
            ) = carry

            # node = value_dag.nodes[node_pos]
            parent_positions = value_dag.parent_pos_padded[node_pos]  # padded with -1
            trigger_predicate_ix = value_dag.trigger_predicate_ix_arr[node_pos]

            def coupled_reset(env, env_params, config, rng):
                # choose a parent per-env and sample a time index from its trajectory
                rng, _rng_par = jax.random.split(rng)
                # number of candidate parents (padded)
                max_parents = parent_positions.shape[0]
                valid_mask = parent_positions >= 0
                valid_count = jnp.sum(valid_mask)
                # random parent choice over valid entries only
                rand_parent_choice = jax.random.randint(_rng_par, shape=(config["NUM_ENVS"],), minval=0, maxval=jnp.maximum(valid_count, 1))
                chosen_parent_pos = parent_positions[rand_parent_choice]  # [config["NUM_ENVS"]]

                def sample_from_parent(per_env_rng, p_pos, env_ix):
                    # Compute time index based on trigger of chosen valid parent
                    rng1, rng2 = jax.random.split(per_env_rng)
                    pred_series = predicate_extrema_buffer[p_pos, :, env_ix, trigger_predicate_ix]
                    random_index_pre = jax.random.randint(rng1, shape=(), minval=0, maxval=config["NUM_STEPS"])
                    trigger_hit = (pred_series < 0)
                    trigger_any = jnp.any(trigger_hit)
                    trigger_idx_pre = jnp.argmax(trigger_hit)
                    trigger_idx = jnp.where(trigger_any, trigger_idx_pre, config["NUM_STEPS"])
                    random_index = jnp.where(trigger_any, trigger_idx, random_index_pre)
                    # select untransformed observation at that time
                    obs_series = obs_buffer[p_pos, :, env_ix, ...]  # [config["NUM_STEPS"], *obs_shape]
                    sel_obs = jax.lax.dynamic_slice_in_dim(obs_series, random_index, slice_size=1, axis=0)
                    sel_obs = jnp.squeeze(sel_obs, axis=0)
                    return sel_obs

                # rngs for each env
                rng, _rng = jax.random.split(rng)
                reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
                # gather per-env untransformed obs from chosen parents
                untrans_traj_batch_observations = jax.vmap(sample_from_parent, in_axes=(0, 0, 0))(reset_rng, chosen_parent_pos, jnp.arange(config["NUM_ENVS"]))
                # reset env to those observations
                obsv, env_state = jax.vmap(env.reset_toinput, in_axes=(0, 0, None))(
                    reset_rng, untrans_traj_batch_observations, env_params
                )
                return obsv, env_state
            
            def standard_reset(env, env_params, config, rng):
                rng, _rng = jax.random.split(rng)
                reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
                return obsv, env_state

            # root node if all parents are -1
            is_root_node = jnp.all(parent_positions < 0)
            obsv, env_state = jax.lax.cond(
                is_root_node,
                lambda r: standard_reset(env, env_params, config, r),
                lambda r: standard_reset(env, env_params, config, r),
                # lambda r: coupled_reset(env, env_params, config, r),
                rng
            )
            
            # COLLECT TRAJECTORY
            rng, _rng = jax.random.split(rng)
            # pass node position (not node id) so value_transition and env_step work with position indexing
            initial_value_node = jnp.ones((config["NUM_ENVS"],), dtype=jnp.int32) * node_pos
            runner_state = (train_state_policies, train_state_values, env_state, obsv, initial_value_node, _rng)
            runner_state, traj_batch = jax.lax.scan(
                env_step, runner_state, None, config["NUM_STEPS"]
            )

            # UPDATE BUFFERS FOR CHILDREN

            # store untransformed observations
            untrans_obs_full = env.untransform_obs(traj_batch.obs)
            obs_buffer = jax.lax.dynamic_update_slice_in_dim(
                obs_buffer, 
                untrans_obs_full[None, ...], 
                node_pos, 
                axis=0
            )
            
            # store predicate maxima over time (as produced during rollout)
            predicate_extrema_buffer = jax.lax.dynamic_update_slice_in_dim(
                predicate_extrema_buffer, 
                traj_batch.predicate_history_extrema[None, ...], 
                node_pos, 
                axis=0
            )

            new_carry = (
                train_state_policies,
                train_state_values,
                rng,
                timestep,
                obs_buffer,
                predicate_extrema_buffer,
            )

            return new_carry, (runner_state, traj_batch)

        # initialize buffers for parent-conditioned resets
        obs_buf0 = jnp.zeros((num_nodes, config["NUM_STEPS"], config["NUM_ENVS"]) + tuple(obs_shape), dtype=jnp.float32)
        pred_max_buf0 = jnp.zeros((num_nodes, config["NUM_STEPS"], config["NUM_ENVS"], num_predicates), dtype=jnp.float32)

        initial_carry = (*train_state_total, obs_buf0, pred_max_buf0)
        (train_state_total, (runner_states, traj_batches)) = jax.lax.scan(
            roll_out_node, initial_carry, jnp.arange(num_nodes), 
        )

        train_state_policies, train_state_values, rng, timestep, _, _ = train_state_total

        ## DEBUG FIXME: Fixed for 4-node RRAA DAG (use node positions)
        
        pos_rraa = int(value_dag.node_index[9])
        pos_raa1 = int(value_dag.node_index[6])
        pos_raa2 = int(value_dag.node_index[3])
        pos_a    = int(value_dag.node_index[0])

        pred_pos_r1 = 0
        pred_pos_r2 = 1
        pred_pos_a  = 2

        runner_state_rraa = tree_index1(runner_states, pos_rraa)
        runner_state_raa1 = tree_index1(runner_states, pos_raa1)
        runner_state_raa2 = tree_index1(runner_states, pos_raa2)
        runner_state_a    = tree_index1(runner_states, pos_a)

        train_state_policy_rraa, train_state_value_rraa = train_state_policies[pos_rraa], train_state_values[pos_rraa]
        train_state_policy_raa1, train_state_value_raa1 = train_state_policies[pos_raa1], train_state_values[pos_raa1]
        train_state_policy_raa2, train_state_value_raa2 = train_state_policies[pos_raa2], train_state_values[pos_raa2]
        train_state_policy_a,   train_state_value_a   = train_state_policies[pos_a],    train_state_values[pos_a]

        traj_batch_rraa = tree_index1(traj_batches, pos_rraa)
        traj_batch_raa1 = tree_index1(traj_batches, pos_raa1)
        traj_batch_raa2 = tree_index1(traj_batches, pos_raa2)
        traj_batch_a    = tree_index1(traj_batches, pos_a)

        ## DEBUG FIXME EVERYTHING BELOW IS OLD RRAA CODE 

        ####################################################################################################################
        # UPDATE RRAA
        
        # CALCULATE COMPOSED ADVANTAGE
        # (train_state_policy_rraa, train_state_value_rraa, env_state_rraa, last_obs, rng,
        #   decomposed_state, policy_controls) = runner_state_rraa
        (_, _, env_state_rraa, last_obs, _, rng) = runner_state_rraa

        last_val_rraa = train_state_value_rraa.apply_fn(train_state_value_rraa.params, last_obs)
        last_val_raa1 = train_state_value_raa1.apply_fn(train_state_value_raa1.params, last_obs)
        last_val_raa2 = train_state_value_raa2.apply_fn(train_state_value_raa2.params, last_obs)

        # DECOMPOSED REACH VALUES ON COMPOSED PPO ACTOR ROLL OUT        
        # V_rraa_append = jnp.concatenate((traj_batch_rraa.value, jnp.expand_dims(last_val_rraa, axis=1).T))
        # r1_append = jnp.concatenate((traj_batch_rraa.reach1, jnp.expand_dims(env_state_rraa.reach1, axis=1).T))
        # V_raa1_append = jnp.concatenate((traj_batch_rraa.value_raa1, jnp.expand_dims(last_val_raa1, axis=1).T))
        # r2_append = jnp.concatenate((traj_batch_rraa.reach2, jnp.expand_dims(env_state_rraa.reach2, axis=1).T))
        # V_raa2_append = jnp.concatenate((traj_batch_rraa.value_raa2, jnp.expand_dims(last_val_raa2, axis=1).T))
        # a_append = jnp.concatenate((traj_batch_rraa.avoid, jnp.expand_dims(env_state_rraa.avoid, axis=1).T))

        V_rraa_append = jnp.concatenate((traj_batch_rraa.value, jnp.expand_dims(last_val_rraa, axis=1).T))
        r1_append = jnp.concatenate((traj_batch_rraa.predicate_values[..., pred_pos_r1], jnp.expand_dims(env_state_rraa.predicate_values[..., pred_pos_r1], axis=1).T))
        V_raa1_append = jnp.concatenate((traj_batch_rraa.all_values[..., pos_raa1], jnp.expand_dims(last_val_raa1, axis=1).T))
        r2_append = jnp.concatenate((traj_batch_rraa.predicate_values[..., pred_pos_r2], jnp.expand_dims(env_state_rraa.predicate_values[..., pred_pos_r2], axis=1).T))
        V_raa2_append = jnp.concatenate((traj_batch_rraa.all_values[..., pos_raa2], jnp.expand_dims(last_val_raa2, axis=1).T))
        a_append = jnp.concatenate((traj_batch_rraa.predicate_values[..., pred_pos_a], jnp.expand_dims(env_state_rraa.predicate_values[..., pred_pos_a], axis=1).T))

        # SPECIAL BRT TARGET FOR BRRT PROBLEM
        l_tilde_rraa = jnp.minimum(jnp.maximum(r1_append, V_raa2_append), jnp.maximum(r2_append, V_raa1_append))

        indexs, done_rraa = calculate_indexs3_rr(ent_gamma[1], traj_batch_rraa.reward, l_tilde_rraa,
                                               jnp.expand_dims(last_val_rraa, axis=1).T) 
        
        done_rraa = done_rraa[:-1, :]

        advantages_V_rraa, targets_V_rraa = calculate_gae_reachavoid4(ent_gamma[1], 
                                                            config["GAE_LAMBDA"], 
                                                            T_ls=l_tilde_rraa, 
                                                            T_gs=a_append,
                                                            T_Vs=V_rraa_append, 
                                                            done=done_rraa)

        # UPDATE COMPOSED NETWORK
        # composed_policy_mask = jnp.where(traj_batch_rraa.policy_taken == 0, 1., 0.) 
        composed_policy_mask = jnp.where(traj_batch_raa1.current_value_node == 9, 1., 0.) #FIXME check
        # FIXME FIXME FIXME needs to include all policies now for policy_taken
        update_state_rraa = (train_state_policy_rraa, train_state_value_rraa,
                        traj_batch_rraa, advantages_V_rraa, targets_V_rraa, advantages_V_rraa, composed_policy_mask, rng)

        xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent_gamma[0]
        update_state_rraa, loss_info_rraa = jax.lax.scan(
            update_epoch, update_state_rraa, xs, config["UPDATE_EPOCHS"]
        )
        train_state_policy_rraa = update_state_rraa[0]
        train_state_value_rraa = update_state_rraa[1]
        rng = update_state_rraa[-1]

        ####################################################################################################################
        # UPDATE RAA1

        # CALCULATE DECOMPOSED ADVANTAGES - RAA 1
        (_, _, env_state_raa1, last_obs_raa1, _, rng) = runner_state_raa1

        last_val_raa1 = train_state_value_raa1.apply_fn(train_state_value_raa1.params, last_obs_raa1)
        last_val_a1 = train_state_value_a.apply_fn(train_state_value_a.params, last_obs_raa1)
        # last_val_a1 = train_state_value_a1.apply_fn(train_state_value_a1.params, last_obs_raa1)

        r1_append = jnp.concatenate((traj_batch_raa1.predicate_values[..., pred_pos_r1], jnp.expand_dims(env_state_raa1.predicate_values[..., pred_pos_r1], axis=1).T))
        V_raa1_append = jnp.concatenate((traj_batch_raa1.value, jnp.expand_dims(last_val_raa1, axis=1).T))
        a1_append = jnp.concatenate((traj_batch_raa1.predicate_values[..., pred_pos_a], jnp.expand_dims(env_state_raa1.predicate_values[..., pred_pos_a], axis=1).T))
        V_a1_append = jnp.concatenate((traj_batch_raa1.all_values[..., pos_a], jnp.expand_dims(last_val_a1, axis=1).T))

        l_tilde_raa1 = jnp.maximum(r1_append, V_a1_append)

        indexs, done_raa1 = calculate_indexs3_rr(ent_gamma[1], traj_batch_raa1.reward, l_tilde_raa1,
                                               jnp.expand_dims(last_val_raa1, axis=1).T)

        done_raa1 = done_raa1[:-1, :]

        # new_done_raa1 = jnp.zeros_like(done_raa1)
        # new_done_raa1 = new_done_raa1.at[-1, :].set(1.0) # TODO: check where this last point actually is 
        # done_raa1 = new_done_raa1

        # advantages_V_raa1, targets_V_raa1 = calculate_gae_reach4(ent_gamma[1], config["GAE_LAMBDA"], r1_append, V_raa1_append, done_raa1)

        advantages_V_raa1, targets_V_raa1 = calculate_gae_reachavoid4(ent_gamma[1], config["GAE_LAMBDA"],
                                                            T_ls=l_tilde_raa1,
                                                            T_gs=a1_append,
                                                            T_Vs=V_raa1_append,
                                                            done=done_raa1)

        # UPDATE DECOMPOSED NETWORK - 1
        # dummy_mask = jnp.ones(traj_batch_raa1.reach1.shape)
        # composed_policy_mask = jnp.where(traj_batch_raa1.policy_taken == 0, 1., 0.)
        composed_policy_mask = jnp.where(traj_batch_raa1.current_value_node == 3, 1., 0.) #FIXME check
        update_state_raa1 = (train_state_policy_raa1, train_state_value_raa1,
                        traj_batch_raa1, advantages_V_raa1, targets_V_raa1, advantages_V_raa1, composed_policy_mask, rng)

        xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent_gamma[0]
        update_state_raa1, loss_info_raa1 = jax.lax.scan(
            update_epoch, update_state_raa1, xs, config["UPDATE_EPOCHS"]
        )
        train_state_policy_raa1 = update_state_raa1[0]
        train_state_value_raa1 = update_state_raa1[1]
        rng = update_state_raa1[-1]

        ####################################################################################################################
        # UPDATE RAA2

        # CALCULATE DECOMPOSED ADVANTAGES - RAA 2
        (_, _, env_state_raa2, last_obs_raa2, _, rng) = runner_state_raa2

        last_val_raa2 = train_state_value_raa2.apply_fn(train_state_value_raa2.params, last_obs_raa2)
        last_val_a2 = train_state_value_a.apply_fn(train_state_value_a.params, last_obs_raa2)
        # last_val_a2 = train_state_value_a2.apply_fn(train_state_value_a2.params, last_obs_raa2)

        r2_append = jnp.concatenate((traj_batch_raa2.predicate_values[..., pred_pos_r2], jnp.expand_dims(env_state_raa2.predicate_values[..., pred_pos_r2], axis=1).T))
        V_raa2_append = jnp.concatenate((traj_batch_raa2.value, jnp.expand_dims(last_val_raa2, axis=1).T))
        a2_append = jnp.concatenate((traj_batch_raa2.predicate_values[..., pred_pos_a], jnp.expand_dims(env_state_raa2.predicate_values[..., pred_pos_a], axis=1).T))
        V_a2_append = jnp.concatenate((traj_batch_raa2.all_values[..., pos_a], jnp.expand_dims(last_val_a2, axis=1).T))

        l_tilde_raa2 = jnp.maximum(r2_append, V_a2_append)

        indexs, done_raa2 = calculate_indexs3_rr(ent_gamma[1], traj_batch_raa2.reward, l_tilde_raa2,
                                               jnp.expand_dims(last_val_raa2, axis=1).T)

        done_raa2 = done_raa2[:-1, :]

        # advantages_V_raa1, targets_V_raa1 = calculate_gae_reach4(ent_gamma[1], config["GAE_LAMBDA"], r1_append, V_raa1_append, done_raa1)

        advantages_V_raa2, targets_V_raa2 = calculate_gae_reachavoid4(ent_gamma[1], config["GAE_LAMBDA"],
                                                            T_ls=l_tilde_raa2,
                                                            T_gs=a2_append,
                                                            T_Vs=V_raa2_append,
                                                            done=done_raa2)

        # UPDATE DECOMPOSED NETWORK - RAA 2
        # dummy_mask = jnp.ones(traj_batch_raa2.reach2.shape)
        # composed_policy_mask = jnp.where(traj_batch_raa2.policy_taken == 0, 1., 0.)
        composed_policy_mask = jnp.where(traj_batch_raa1.current_value_node == 6, 1., 0.) #FIXME check
        update_state_raa2 = (train_state_policy_raa2, train_state_value_raa2,
                        traj_batch_raa2, advantages_V_raa2, targets_V_raa2, advantages_V_raa2, composed_policy_mask, rng)

        xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent_gamma[0]
        update_state_raa2, loss_info_raa2 = jax.lax.scan(
            update_epoch, update_state_raa2, xs, config["UPDATE_EPOCHS"]
        )
        train_state_policy_raa2 = update_state_raa2[0]
        train_state_value_raa2 = update_state_raa2[1]
        rng = update_state_raa2[-1]

        ####################################################################################################################
        # UPDATE A

        # CALCULATE COMPOSED ADVANTAGE - A1
        (_, _, env_state_a, last_obs_a, _, rng) = runner_state_a

        last_val_a = train_state_value_a.apply_fn(train_state_value_a.params, last_obs_a)

        a_append = jnp.concatenate((traj_batch_a.predicate_values[..., pred_pos_a], jnp.expand_dims(env_state_a.predicate_values[..., pred_pos_a], axis=1).T)) # avoid function
        V_a_append = jnp.concatenate((traj_batch_a.value, jnp.expand_dims(last_val_a, axis=1).T)) # avoid value function

        indexs, done_a = calculate_indexs3_rr(ent_gamma[1], traj_batch_a.reward, a_append,
                                               jnp.expand_dims(last_val_a, axis=1).T) # NOTE are we totally sure this works, I dont really get og usage,
        done_a = done_a[:-1, :]
        # # Temp override: done is only the last step
        # new_done_a = jnp.zeros_like(done_a)
        # new_done_a = new_done_a.at[-1, :].set(1.0) # TODO: check where this last point actually is
        # done_a = new_done_a

        advantages_V_a, targets_V_a = calculate_gae_avoid4(ent_gamma[1], config["GAE_LAMBDA"],
                                                            T_hs=a_append,
                                                            T_Vhs=V_a_append,
                                                            done=done_a)
        
        # UPDATE DECOMPOSED NETWORK - AVOID
        dummy_mask = jnp.ones(traj_batch_a.predicate_values[..., pred_pos_a].shape)
        update_state_a = (train_state_policy_a, train_state_value_a,
                           traj_batch_a, advantages_V_a, targets_V_a, advantages_V_a, dummy_mask, rng)
        xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent_gamma[0]
        update_state_a, loss_info_a = jax.lax.scan(
            update_epoch, update_state_a, xs, config["UPDATE_EPOCHS"]
        )
        train_state_policy_a = update_state_a[0]
        train_state_value_a = update_state_a[1]
        rng = update_state_a[-1]

        ####################################################################################################################
        # Output

        # train_state_total_out = (train_state_policy_rraa, train_state_value_rraa,
        #     train_state_policy_raa1, train_state_value_raa1,
        #     train_state_policy_raa2, train_state_value_raa2, 
        #     train_state_policy_a, train_state_value_a,
        #     rng, timestep)
        
        # DEBUG FIXME: Fixed for 4-node RRAA DAG
        # train_state_policies_out = train_state_policies[pos_rraa].set(train_state_policy_rraa)
        # train_state_policies_out = train_state_policies_out[pos_raa1].set(train_state_policy_raa1)
        # train_state_policies_out = train_state_policies_out[pos_raa2].set(train_state_policy_raa2)
        # train_state_policies_out = train_state_policies_out[pos_a].set(train_state_policy_a)

        # train_state_values_out = train_state_values[pos_rraa].set(train_state_value_rraa)
        # train_state_values_out = train_state_values_out[pos_raa1].set(train_state_value_raa1)
        # train_state_values_out = train_state_values_out[pos_raa2].set(train_state_value_raa2)
        # train_state_values_out = train_state_values_out[pos_a].set(train_state_value_a)

        train_state_policies = [train_state_policy_rraa, train_state_policy_raa1, train_state_policy_raa2, train_state_policy_a]
        train_state_values = [train_state_value_rraa, train_state_value_raa1, train_state_value_raa2, train_state_value_a]

        train_state_total_out = (train_state_policies, train_state_values, rng, timestep)
        
        return (train_state_total_out,
                {"batch_info_rraa": (traj_batch_rraa, targets_V_rraa, done_rraa), "loss_info_rraa": loss_info_rraa,
                 "batch_info_raa1": (traj_batch_raa1, targets_V_raa1, done_raa1), "loss_info_raa1": loss_info_raa1,
                 "batch_info_raa2": (traj_batch_raa2, targets_V_raa2, done_raa2), "loss_info_raa2": loss_info_raa2,
                 "batch_info_a": (traj_batch_a, targets_V_a, done_a), "loss_info_a": loss_info_a,
                 "reach_gamma": ent_gamma[1], "entropy_weight": ent_gamma[0]})
    
    ########################################################################################################################

    # MAKE THE VALUE TRANSITION FUNCTION FROM THE DAG
    # value_transition = make_value_transition_fn(value_dag, config)

    # FIXME TODO: Allows only one node transition per step, what about multiple satisfactions?
    def _value_transition(value_dag, last_value_node, last_env_state):
        # last_value_node: (NUM_ENVS,) positions
        # last_env_state.predicate_values: expected shape (NUM_ENVS, P)
        pred_vals = last_env_state.predicate_values  # (E, P)
        trigger_pred_ix_arr = value_dag.trigger_predicate_ix_arr  # (N,)
        children_pos_padded = value_dag.children_pos_padded       # (N, K)

        def per_env_transition(cur_pos_e, pred_vals_e):
            # Children positions for this env’s current node
            child_pos = children_pos_padded[cur_pos_e]                  # (K,)
            child_exists = child_pos >= 0                               # (K,), means child
            child_pos_safe = jnp.where(child_exists, child_pos, 0)      # (K,)

            # Trigger predicate index for each child position
            trig_ix = trigger_pred_ix_arr[child_pos_safe]               # (K,)
            trigger_exists = trig_ix >= 0                               # (K,)

            # Select each child’s trigger predicate value
            pred_sel = jnp.take(pred_vals_e, jnp.where(trigger_exists, trig_ix, 0), axis=0)  # (K,)
            triggers = (pred_sel < 0) & trigger_exists & child_exists

            # Pick first triggered child or stay
            any_trig = jnp.any(triggers)
            first_idx = jnp.argmax(triggers)
            sel_child_pos = jnp.take(child_pos_safe, first_idx, axis=0)
            next_pos_e = jnp.where(any_trig, sel_child_pos, cur_pos_e)
            return next_pos_e

        current_value_node = jax.vmap(per_env_transition, in_axes=(0, 0))(last_value_node, pred_vals)
        return current_value_node

    value_transition = partial(_value_transition, value_dag)

    # INIT JAX WRAPPERS
    update_epoch = partial(_ppo_vanilla_update, config)
    env_step = partial(_env_step_general_task, env, env_params, value_transition)
    training = jax.jit(_train)

    tx = optimizer(config)

    def create_train_state(value_dag, config, env, env_params, rng):

        train_state_policies_list = []
        train_state_values_list = []
        node_tags = {}
        for pos in range(len(value_dag.nodes)):
            node = int(value_dag.nodes[pos])

            # INIT POLICY NETWORK
            if config["DISCRETE"] == False:
                policy_network = MoGPolicy_Network( # Gaussian Mixture (TODO does this really work better?)
                # policy_network_rraa = Policy_Network(
                    env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
                )
            else:
                policy_network = Policy_Network_Discrete(
                    env.action_space(env_params).n, activation=config["ACTIVATION"]
                )

            # INIT Actor
            rng, _rng = jax.random.split(rng)
            init_x = jnp.zeros(env.observation_space(env_params).shape)
            network_params_policy = policy_network.init(_rng, init_x)
            train_state_policy = TrainState.create(
                apply_fn=policy_network.apply,
                params=network_params_policy,
                tx=tx,
                mean=jnp.zeros(env.observation_space(env_params).shape),
                variance=jnp.zeros(env.observation_space(env_params).shape),
                count=1e-4,
            )
            train_state_policies_list.append(train_state_policy)
            
            # INIT VALUE NETWORK
            value_network = Value_Network(activation=config["ACTIVATION"])
            rng, _rng = jax.random.split(rng)
            init_x = jnp.zeros(env.observation_space(env_params).shape)
            network_params = value_network.init(_rng, init_x)
            train_state_value = TrainState.create(
                apply_fn=value_network.apply,
                params=network_params,
                tx=tx,
                mean=jnp.zeros(env.observation_space(env_params).shape),
                variance=jnp.zeros(env.observation_space(env_params).shape),
                count=1e-4,
            )
            train_state_values_list.append(train_state_value)

            node_tags[node] = value_dag.node_tags[node] if isinstance(value_dag.node_tags, dict) and node in value_dag.node_tags else None

        return train_state_policies_list, train_state_values_list, node_tags, rng

    train_state_policies, train_state_values, node_tags, rng = create_train_state(value_dag, config, env, env_params, rng)

    # # LOAD DECOMPOSED ACTOR AND CRITICS
    # else:
    #     raw_restored = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath('model/{}/{}'.format(
    #         config["LOAD_DEC_DIR"], config["LOAD_DEC_DIR_MODEL"])), target=None)
        
    #     train_state_policy_reach1 = TrainState.create(
    #         apply_fn=policy_network_reach1.apply,
    #         params=raw_restored['policy_reach1_network']['params'],
    #         mean=raw_restored['policy_reach1_network']["mean"],
    #         variance=raw_restored['policy_reach1_network']["variance"],
    #         count=raw_restored['policy_reach1_network']["count"],
    #         tx=tx,
    #     )
    #     train_state_policy_reach2 = TrainState.create(
    #         apply_fn=policy_network_reach2.apply,
    #         params=raw_restored['policy_reach2_network']['params'],
    #         mean=raw_restored['policy_reach2_network']["mean"],
    #         variance=raw_restored['policy_reach2_network']["variance"],
    #         count=raw_restored['policy_reach2_network']["count"],
    #         tx=tx,
    #     )

    #     value_network_reach1 = Value_Network(activation=config["ACTIVATION"])
    #     train_state_value_reach1 = TrainState.create(
    #         apply_fn=value_network_reach1.apply,
    #         params=raw_restored['value_reach1_network']['params'],
    #         mean=raw_restored['value_reach1_network']["mean"],
    #         variance=raw_restored['value_reach1_network']["variance"],
    #         count=raw_restored['value_reach1_network']["count"],
    #         tx=tx,
    #     )
    #     value_network_reach2 = Value_Network(activation=config["ACTIVATION"])
    #     train_state_value_reach2 = TrainState.create(
    #         apply_fn=value_network_reach2.apply,
    #         params=raw_restored['value_reach2_network']['params'],
    #         mean=raw_restored['value_reach2_network']["mean"],
    #         variance=raw_restored['value_reach2_network']["variance"],
    #         count=raw_restored['value_reach2_network']["count"],
    #         tx=tx,
    #     )

    # # IF TRAINING DECOMPOSED, USE PPO
    if not config["LOAD_DECOMPOSED"]:
        update_epoch_dec = partial(_ppo_vanilla_update, config)

    # # IF LOADING PRESOLVED DECOMPOSED, NO TRAINING
    # else:
    #     def _no_update(config, update_state, ent):
    #         dummy_loss = {
    #             "actor_loss": 0.0,
    #             "value_loss": 0.0,
    #             "entropy_loss": 0.0,
    #         }
    #         return update_state, dummy_loss
    #     update_epoch_dec = partial(_no_update, config)

    total_timesteps = config["NUM_UPDATES"] // config["STEP_SCAN"]

    best_score = -jnp.inf
    for timestep in range(config["NUM_UPDATES"] // config["STEP_SCAN"]):

        t0 = time.time()

        xs = jnp.zeros((config["STEP_SCAN"], 2))

        if config['ANNEAL_ENT'] == True:
            ent = jnp.ones(config["STEP_SCAN"]) * config["ENT_COEF"] * (total_timesteps - timestep) / total_timesteps
        else:
            ent = jnp.ones(config["STEP_SCAN"]) * config["ENT_COEF"]

        gamma_1 = jnp.ones(config["STEP_SCAN"]) * config["GAMMA_REACH_INIT"] + (config['GAMMA_REACH_FINAL'] - config["GAMMA_REACH_INIT"]) * timestep / total_timesteps
        gamma_2 = jnp.ones(config["STEP_SCAN"]) * jnp.minimum(config['GAMMA_REACH_FINAL'], config["GAMMA_REACH_INIT"] +
                              (config['GAMMA_REACH_FINAL'] - config["GAMMA_REACH_INIT"]) * timestep * 2 / total_timesteps)

        xs = xs.at[:, 0].set(ent)
        xs = xs.at[:, 1].set(gamma_2)

        update_state, result = jax.lax.scan(
            training, (train_state_policies, train_state_values, rng, timestep),
            xs, config["STEP_SCAN"]
        )

        (train_state_policies, train_state_values, rng, timestep) = update_state

        ## SCORING FIXME for general task logic

        loss_info_rraa = result['loss_info_rraa']
        loss_info_raa1 = result['loss_info_raa1']
        loss_info_raa2 = result['loss_info_raa2']
        loss_info_a = result['loss_info_a']

        result_traj_rraa = tree_index1(result['batch_info_rraa'], 0)
        result_traj_raa1 = tree_index1(result['batch_info_raa1'], 0)
        result_traj_raa2 = tree_index1(result['batch_info_raa2'], 0)
        result_traj_a = tree_index1(result['batch_info_a'], 0)
        
        traj_batch_rraa, targets_V_rraa, done_rraa = result_traj_rraa
        traj_batch_raa1, targets_V_raa1, done_raa1 = result_traj_raa1
        traj_batch_raa2, targets_V_raa2, done_raa2 = result_traj_raa2
        traj_batch_a, targets_V_a, done_a = result_traj_a

        (rraa_rr_perc, rraa_crash_perc, rraa_rraa_perc), rraa_reach_idxs, rraa_crash_idx = calculate_rraa_TEMP_VDPPO(traj_batch_rraa, reach_type="both")
        (raa1_r_perc, raa1_crash_perc, raa1_raa_perc), raa1_reach_idxs, raa1_crash_idx = calculate_rraa_TEMP_VDPPO(traj_batch_raa1, reach_type="1")
        (raa2_r_perc, raa2_crash_perc, raa2_raa_perc), raa2_reach_idxs, raa2_crash_idx = calculate_rraa_TEMP_VDPPO(traj_batch_raa2, reach_type="2")
        (_, a_crash_perc, _),  _, a_crash_idx = calculate_rraa_TEMP_VDPPO(traj_batch_a, reach_type="none")

        idx = 0
        info_rraa = tree_index2(traj_batch_rraa.info, idx)
        info_raa1 = tree_index2(traj_batch_raa1.info, idx)
        info_raa2 = tree_index2(traj_batch_raa2.info, idx)
        info_a = tree_index2(traj_batch_a.info, idx)

        info_rraa['reach_index_1'], info_rraa['reach_index_2'] = rraa_reach_idxs[0][idx], rraa_reach_idxs[1][idx]
        info_raa1['reach_index_1'], info_raa1['reach_index_2'] = raa1_reach_idxs[-1][idx], np.array(-1)
        info_raa2['reach_index_1'], info_raa2['reach_index_2'] = np.array(-1), raa2_reach_idxs[-1][idx]
        info_a['reach_index_1'], info_a['reach_index_2'] = np.array(-1), np.array(-1)

        info_rraa['crash_index'] = rraa_crash_idx[idx]
        info_raa1['crash_index'] = raa1_crash_idx[idx]
        info_raa2['crash_index'] = raa2_crash_idx[idx]
        info_a['crash_index'] = a_crash_idx[idx]

        if config['EXP_NAME'] == 'WindField' or config['EXP_NAME'] == 'WindFieldFull':
            info_rraa['u_air'] = env_params.u_air
            info_rraa['v_air'] = env_params.v_air
            info_rraa['obs'] = env_params.obstacle

        ## SAVE MODEL CHECKPOINTS

        all_training_states = {"policy_network_{}".format(int(node)): train_state_policies[int(value_dag.node_index[int(node)])] for node in value_dag.nodes}
        all_training_states.update({"value_network_{}".format(int(node)): train_state_values[int(value_dag.node_index[int(node)])] for node in value_dag.nodes})
        checkpoints.save_checkpoint(ckpt_dir=os.path.abspath(os.path.join("model", config["DIR"])),
                                    target=all_training_states,
                                    step=timestep,
                                    overwrite=True, 
                                    keep=2)
        
        if config["SAVE_MILESTONE"] and timestep in config["MILESTONES"]:
            checkpoints.save_checkpoint(ckpt_dir=os.path.abspath(os.path.join("model", config["DIR"])),
                                        target=all_training_states,
                                        step=timestep,
                                        overwrite=False,
                                        prefix="milestone_",)
        
        if rraa_rraa_perc > best_score:
            best_score = rraa_rraa_perc
            checkpoints.save_checkpoint(ckpt_dir=os.path.abspath(os.path.join("model", config["DIR"])),
                                        target=all_training_states,
                                        step=timestep,
                                        prefix="best_",
                                        overwrite=True,)
        
        # MAKE DIAGNOSTIC PLOTS -- FIXME for GENERAL TASK LOGIC

        # policy_decision_sample = traj_batch_rraa.policy_taken[:,idx]
        policy_decision_sample = traj_batch_rraa.current_value_node[:,idx] ## DEBUG FIXME
        fig = plot_contour_RRAA((info_rraa, info_raa1, info_raa2, info_a), timestep, config, policy_decision_sample=policy_decision_sample)

        fig2 = plot_policy_decision(policy_decision_sample, timestep, config)

        t1 = time.time()

        (rraa_rr_perc, rraa_crash_perc, rraa_rraa_perc)
        (raa1_r_perc, raa1_crash_perc, raa1_raa_perc)
        (raa2_r_perc, raa2_crash_perc, raa2_raa_perc)
        (_, a_crash_perc, _)

        # WRITE TO WANDB -- FIXME for GENERAL TASK LOGIC

        if config["USE_WANDB"]:
            # group into wandb subheaders
            wandb.log({
                    "Score/(RRAA) RRAA [%]": rraa_rraa_perc,
                    "Score/(RRAA) RR [%]": rraa_rr_perc,
                    "Score/(RRAA) Crashed [%]": rraa_crash_perc,
                    "Score/(RAA-1) R1 [%]": raa1_r_perc,
                    "Score/(RAA-1) Crashed [%]": raa1_crash_perc,
                    "Score/(RAA-1) RAA1 [%]": raa1_raa_perc,
                    "Score/(RAA-2) R2 [%]": raa2_r_perc,
                    "Score/(RAA-2) Crashed [%]": raa2_crash_perc,
                    "Score/(RAA-2) RAA2 [%]": raa2_raa_perc,
                    "Score/(A) Crashed [%]": a_crash_perc,
                    "Loss/actor_rraa_loss": jnp.mean(loss_info_rraa["actor_loss"]), 
                    "Loss/value_rraa_loss": jnp.mean(loss_info_rraa["value_loss"]),
                    "Loss/actor_raa1_loss": jnp.mean(loss_info_raa1["actor_loss"]), 
                    "Loss/value_raa1_loss": jnp.mean(loss_info_raa1["value_loss"]),
                    "Loss/actor_raa2_loss": jnp.mean(loss_info_raa2["actor_loss"]), 
                    "Loss/value_raa2_loss": jnp.mean(loss_info_raa2["value_loss"]),
                    "Loss/actor_a_loss": jnp.mean(loss_info_a["actor_loss"]), 
                    "Loss/value_a_loss": jnp.mean(loss_info_a["value_loss"]),
                    "Train/reach_gamma": result['reach_gamma'][0], 
                    "Train/entropy_weight": result['entropy_weight'][0],
                    }, step=timestep)
            
            if "F16" not in config["EXP_NAME"]: # FIXME make f16 methods uniform
                wandb.log({
                    'trajectory_sample':wandb.Image(fig),
                    'policy_decision_sample':wandb.Image(fig2),
                }, step=timestep)
            
        # Save video of trajectory 
        if "F16" not in config["EXP_NAME"]:
            if timestep % config['VIDEO_FREQ'] == 0 or timestep == total_timesteps - 1: 
                video_frames = plot_video_contour_RRAA((info_rraa, info_raa1, info_raa2, info_a), timestep, config, save_video=True, log_wandb=config["USE_WANDB"])

        plt.close("all")
        print(f"ITER TIME : {t1-t0:2.1f}s : (A)  {100*(1-a_crash_perc):2.1f}%  (RAA1)  {100*raa1_raa_perc:2.1f}%  (RAA2)  {100*raa2_raa_perc:2.1f}%  (RRAA)  {100*rraa_rr_perc:2.1f}%")
        # print("Time {}".format(t1-t0))

    return


# NOTES - Things to fix / check
# - Does done setting we used work better/worse? (what should it be?)
# - 1 vs 2 avoid functions (branch?)
# - Reach/Avoid value scaling?
# - Env length? (200 -> 400?)
# - More envs per batch? (32 -> 128? )
# - entropy/LR?

if __name__ == "__main__":
    config = vars(get_args(sys.argv[1:]))

    debug = True
    if debug:
        config["EXP_NAME"]="PointValDec"
        config["MODEL_DIR"] = 'model_valdec'
        config["DIR"]="point_VD_test"
        config["LR"]=3e-4
        config["NUM_ENVS"]=128
        config["NUM_STEPS"]=400
        config["TOTAL_TIMESTEPS"]=200_000_000
        config["STEP_SCAN"]=4
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
        config["NAME"]="point_VD_test"

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

    # env = get_env(config)
    # env_params = env.default_params

    ## MAKE THE VALUE DAG
    # value_dag = valt.make_value_dag(config)

    class DummyRRAAdag: # DEBUG FIXME placeholder, fake node numbers
        def __init__(self):
            # fixed topological order: [RRAA(9), RAA1(6), RAA2(3), A(0)]
            self.nodes = jnp.array([9, 6, 3, 0])
            self.node_tags = {0: 'A', 6: 'RAA1', 3: 'RAA2', 9: 'RRAA'}
            # parent positions (padded with -1 to fixed width 2)
            # positions: 9->0, 6->1, 3->2, 0->3
            # parents: 6 <- [9], 3 <- [9], 0 <- [3,6], 9 <- []
            self.parent_pos_padded = jnp.array([
                [-1, -1],  # node 9 (pos 0) root
                [ 0, -1],  # node 6 (pos 1) parent: 9(pos0)
                [ 0, -1],  # node 3 (pos 2) parent: 9(pos0)
                [ 2,  1],  # node 0 (pos 3) parents: 3(pos2), 6(pos1)
            ])
            # children positions (not used in this edit, but kept for completeness)
            self.children_pos_padded = jnp.array([
                [ 1,  2],  # pos 0 (node 9) -> children: 6(pos1), 3(pos2)
                [ 3, -1],  # pos 1 (node 6) -> child: A(pos3)
                [ 3, -1],  # pos 2 (node 3) -> child: A(pos3)
                [-1, -1],  # pos 3 (node 0) -> leaf
            ])
            # trigger predicate index for each node (in same order as nodes)
            # RRAA(9): no trigger -> -1, RAA1(6): reach1->0, RAA2(3): reach2->1, A(0): obstacles->2
            self.trigger_predicate_ix_arr = jnp.array([-1, 0, 1, 2])
            # simple helpers
            self.predicates = ["reach1", "reach2", "obstacles"]
            self.node_parents = {0: jnp.array([3,6]), 3: jnp.array([9]), 6: jnp.array([9]), 9: jnp.array([])}
            self.node_children = {0: jnp.array([]), 3: jnp.array([0]), 6: jnp.array([0]), 9: jnp.array([3,6])}
            self.trigger_predicate_ix = {0: jnp.array([2]), 6: jnp.array([0]), 3: jnp.array([1]), 9: jnp.array([])}
            # mapping node id -> position (Python dict for convenient lookup outside JIT)
            self.node_index = {9: 0, 6: 1, 3: 2, 0: 3}
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
        negated_predicate_mask=jnp.array([1, 1, 0])
    )
    env = TransformObservation(env, trans)
    env.set_untransform_obs(untrans)
    env_params = env.default_params

    if config['EXP_NAME'] == 'WindField':
        env_params = env_params.replace(index=config['SECTION'])

    config["USE_WANDB"] = False #not debug # False for debugging
    if config["USE_WANDB"]:
        wandb.init(project='valdec-{}-{}'.format(config["EXP_NAME"], config["WANDB_GROUP"]), name=config["NAME"], config=config,
                   entity='braat_brrt')

    config["LOAD_DECOMPOSED"] = False # TODO make arg
    # if config["LOAD_DECOMPOSED"]:
    #     config["LOAD_DEC_DIR"] ="hopper_reachreach_idxsMAX_switchfix_augstate_obsfix_long"
    #     config["LOAD_DEC_DIR_MODEL"] ="checkpoint_859"

    if 'VIDEO_FREQ' not in config.keys():
        if 'Humanoid' in config['EXP_NAME']:
            config['VIDEO_FREQ'] = 200
        else:
            config['VIDEO_FREQ'] = 25

    rng = jax.random.PRNGKey(config["SEED"])
    out = train(env, env_params, value_dag, config, rng) 
    # NOTE passing multiple envs (composed + decomposed)
    # TODO more elegant use one env w/ diff env_params, but this is safe for now