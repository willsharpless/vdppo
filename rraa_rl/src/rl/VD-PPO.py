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

## Why might training be different? vs. DOHJPPO
# - exposing all predicate values to obs (even in decompsed nodes)

def train(env, env_params, value_dag, config, rng):
    def _train(train_state_total, ent_gamma):
        
        ####################################################################################################################
        # ROLLOUT BATCHES FOR EACH NODE

        # get original rng for comparison
        _, _, rng_og, _ = train_state_total

        # Pre-allocate functional buffers to carry parent data across node rollouts
        num_nodes = value_dag.nodes.shape[0]
        obs_shape = env.observation_space(env_params).shape
        num_predicates = len(value_dag.predicates)

        def roll_out_node(carry, node_pos):
            (train_state_policies,
                train_state_values,
                rng,
                timestep,
                obs_buffer,                       # [num_nodes, config["NUM_STEPS"], config["NUM_ENVS"], *obs_shape]
                predicate_buffer,                 # [num_nodes, config["NUM_STEPS"], config["NUM_ENVS"], num_predicates], redundantly in obs, but cleaner?
                reset_buffer                      # [num_nodes, config["NUM_ENVS"]]
            ) = carry

            parent_positions_per_node = value_dag.parent_pos_padded[node_pos]  # padded with -1
            is_root_node = jnp.all(parent_positions_per_node < 0)

            def coupled_reset(env, env_params, config, rng):
                # Choose a parent per-env, check trigger predicate sat and sample a time index from its trajectory
                rng, _rng_par = jax.random.split(rng)
                rand_parent_choice = jax.random.randint(_rng_par, shape=(config["NUM_ENVS"],), minval=0, maxval=(parent_positions_per_node >= 0).sum())
                chosen_parent_pos = parent_positions_per_node[rand_parent_choice]

                def sample_from_parent(per_env_rng, p_pos, env_ix):
                    # Compute time index based on trigger of chosen valid parent
                    rng1, rng2 = jax.random.split(per_env_rng)
                    random_index = jax.random.randint(rng1, shape=(), minval=0, maxval=config["NUM_STEPS"])

                    trigger_ix_mask = value_dag.trigger_predicate_map[p_pos] == node_pos
                    has_trigger = jnp.any(trigger_ix_mask)
                    trigger_ix_safe = jnp.where(has_trigger, jnp.argmax(trigger_ix_mask), 0) # so tracing works for root without parents

                    pred_series = predicate_buffer[p_pos, :, env_ix, trigger_ix_safe]
                    satisfied = (pred_series < 0) # TODO old convention, neg reaching
                    satisfied_idx = jnp.where(has_trigger & jnp.any(satisfied), jnp.argmax(satisfied), config["NUM_STEPS"])
                    reset_index = jnp.where(has_trigger & jnp.any(satisfied), satisfied_idx, random_index)
                    # NOTE previously called toinput_goal coupling, could try other forms

                    obs_series = obs_buffer[p_pos, :, env_ix, ...]
                    sel_obs = jax.lax.dynamic_slice_in_dim(obs_series, reset_index, slice_size=1, axis=0).squeeze(axis=0)
                    return sel_obs, reset_index
                    # return sel_obs

                rng, _rng = jax.random.split(rng)
                reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
                untrans_parent_batch_obs, reset_indices = jax.vmap(sample_from_parent, in_axes=(0, 0, 0))(reset_rng, chosen_parent_pos, jnp.arange(config["NUM_ENVS"]))
                obsv, env_state = jax.vmap(env.reset_toinput, in_axes=(0, 0, None))(
                    reset_rng, untrans_parent_batch_obs, env_params
                )
                return obsv, env_state, reset_indices
                # return obsv, env_state
            # TODO: could check sat then sample parents to increase efficiency
            
            def standard_reset(env, env_params, config, rng):
                rng, _rng = jax.random.split(rng)
                reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
                # return obsv, env_state
            
                # dummy reset for jax conditional (type tree gets altered here somehow?)
                untrans_obs = env.untransform_obs(obsv)
                obsv_dummy, env_state_dummy = jax.vmap(env.reset_toinput, in_axes=(0, 0, None))(
                    reset_rng, untrans_obs, env_params
                )
                reset_indices = -1 * jnp.ones((config["NUM_ENVS"],), dtype=jnp.int32)
                return obsv_dummy, env_state_dummy, reset_indices

            # RESET
            obsv, env_state, reset_indices = jax.lax.cond(
                is_root_node,
                lambda r: standard_reset(env, env_params, config, r),
                lambda r: coupled_reset(env, env_params, config, r),
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
            obs_buffer = jax.lax.dynamic_update_slice_in_dim(obs_buffer, untrans_obs_full[None, ...], node_pos, axis=0)
            
            # store predicate maxima over time (as produced during rollout)
            predicate_buffer = jax.lax.dynamic_update_slice_in_dim(predicate_buffer, traj_batch.predicate_values[None, ...], node_pos, axis=0)

            # store resets (for tracking)
            reset_buffer = jax.lax.dynamic_update_slice_in_dim(reset_buffer, reset_indices[None, :], node_pos, axis=0)

            return (train_state_policies, train_state_values, rng, timestep, obs_buffer, predicate_buffer, reset_buffer), (runner_state, traj_batch)

        # initialize buffers for parent-conditioned resets
        obs_buffer_init = jnp.zeros((num_nodes, config["NUM_STEPS"], config["NUM_ENVS"]) + tuple(obs_shape), dtype=jnp.float32)
        predicate_buffer_init = jnp.zeros((num_nodes, config["NUM_STEPS"], config["NUM_ENVS"], num_predicates), dtype=jnp.float32)
        reset_buffer_init = -1 * jnp.ones((num_nodes, config["NUM_ENVS"]), dtype=jnp.int32)

        initial_carry = (*train_state_total, obs_buffer_init, predicate_buffer_init, reset_buffer_init)
        (final_carry, (runner_states, traj_batches)) = jax.lax.scan(
            roll_out_node, initial_carry, jnp.arange(num_nodes), 
        )

        train_state_policies, train_state_values, rng, timestep, obs_buffer, predicate_buffer, reset_indices = final_carry

        ####################################################################################################################
        ## COMPUTE ADVANTAGES AND TARGETS PER NODE

        # last_obs_per_node: (N, E, *obs_shape) -> node vals at each last obs (N, N, E)
        def _gather_last_vals_for_nodes(train_state_values, last_obs_per_node):
            # last_obs_per_node: (N, E, *obs_shape)
            def apply_val(i):
                v = train_state_values[i]
                values = [v.apply_fn(v.params, obs) for obs in last_obs_per_node]
                return jnp.stack(values, axis=0)
            values = [apply_val(i) for i in range(num_nodes)]
            return jnp.stack(values, axis=0)  # (N, N, E)

        # TODO: faster way?
        # def _gather_last_vals_for_nodes(train_state_values, last_obs_per_node):
        #     last_obs_stacked = jnp.stack(last_obs_per_node, axis=0)  # (N, E, *obs_shape)
        #     params_list = tuple(ts.params for ts in train_state_values)
        #     apply_fns = tuple(ts.apply_fn for ts in train_state_values)
        #     out = jax.vmap( # vmap over i (N_eval) and j (N_src)
        #             jax.vmap(lambda i, j: apply_fns[i](params_list[i], last_obs_stacked[j]),
        #                     in_axes=(None, 0), out_axes=0),
        #             in_axes=(0, None), out_axes=0
        #         )(jnp.arange(len(train_state_values)), jnp.arange(len(train_state_values)))
        #     return out  # (N, N, E)
        
        def _gather_last_preds_for_nodes(last_env_state_per_node):
            pred_vals = tuple([last_env_state_per_node[i].predicate_values for i in range(num_nodes)])  # N-tuple of (P, E)
            return jnp.stack(pred_vals, axis=0)  # (N, P, E)
        
        def _per_node_adv_targets(node_pos, last_preds_per_node, all_last_vals_per_node):
            traj_batch = tree_index1(traj_batches, node_pos)

            # DEBUG FIXME: Hard code RRAA node types for now
            node_types = jnp.array([0, 0, 0, 1]) # 0: reach-avoid, 1: avoid-only
            avoid_predicates_mask = jnp.array([
                [ 0, 0, 1], 
                [ 0, 0, 1], 
                [ 0, 0, 1], 
                [ 0, 0, 1]
            ]).astype(bool)

            node_type = node_types[node_pos]
            node_avoid_mask = avoid_predicates_mask[node_pos] # (P,)

            # Compute appended values
            last_val_pred = last_preds_per_node[node_pos][None, ...]  # (1, E, P)
            last_vals_per_node = all_last_vals_per_node[node_pos] # (N, N, E) -> (N, E)
            pred_append = jnp.concatenate([traj_batch.predicate_values, last_val_pred], axis=0)  # (T+1, E, P)
            all_val_append = jnp.concatenate([traj_batch.all_values, last_vals_per_node.T[None, ...]], axis=0)  # (T+1, E, N)
            val_append = jnp.concatenate([traj_batch.value, last_vals_per_node[node_pos][None, ...]], axis=0)  # (T+1, E)
            
            # Compute avoid target (neutralized if none)
            a_masked = jnp.where(node_avoid_mask[None, None, :], pred_append, -jnp.inf) # (T+1, E, P), NOTE old convention, neg avoiding
            a_target = jnp.max(a_masked, axis=-1)

            # General Reach-Avoid with multiple reach/avoid predicates
            def reachavoid_N_advantage_target(_):

                # Get reach trigger children for target
                child_pos_per_pred = value_dag.trigger_predicate_map[node_pos]          # (P,)
                reach_mask = (child_pos_per_pred >= 0)                                  # (P,)
                child_pos_safe = jnp.where(reach_mask, child_pos_per_pred, 0)           # (P,)
                r_append = jnp.where(reach_mask[None, None, :], pred_append, jnp.inf)   # (T+1, E, P), NOTE old convention, neg reaching
                V_child_append = jnp.take(all_val_append, child_pos_safe, axis=-1)      # (T+1, E, P)

                # Reach-Avoid target (l_tilde), (T+1, E, P) -> (T+1, E)
                ra_target = jnp.min(jnp.maximum(r_append, V_child_append), axis=-1)                                                                              

                # Compute done flags (FIXME, use resets properly)
                indexs, done = calculate_indexs3_rr(ent_gamma[1], traj_batch.reward, ra_target,
                                               last_vals_per_node[node_pos][None, ...])
                done = done[:-1, :] # FIXME

                # Compute advantage and targets
                adv, tgt = calculate_gae_reachavoid4(
                    ent_gamma[1], config["GAE_LAMBDA"], T_ls=ra_target, T_gs=a_target, T_Vs=val_append, done=done
                )
                # return adv, tgt #DEBUG FIXME
                T_ls = ra_target
                T_gs = a_target
                T_Vs = val_append
                return adv, tgt, T_ls, T_gs, T_Vs

            def avoid_N_advantage_target(_):

                # Compute done flags (FIXME, use resets properly)
                indexs, done = calculate_indexs3_rr(ent_gamma[1], traj_batch.reward, a_target,
                                               last_vals_per_node[node_pos][None, ...])
                done = done[:-1, :] # FIXME

                # Compute advantage and targets
                adv, tgt = calculate_gae_avoid4(
                    ent_gamma[1], config["GAE_LAMBDA"], T_hs=a_target, T_Vhs=val_append, done=done
                )
                # return adv, tgt #DEBUG FIXME
                T_hs = a_target
                T_Vhs = val_append
                return adv, tgt, T_hs, T_hs, T_Vhs

            # # TODO
            # def advantage_globally_wrapped(_):
            #     return adv, tgt

            branches = (reachavoid_N_advantage_target, avoid_N_advantage_target)
            return jax.lax.switch(node_type, branches, operand=None)

        def _scan_updates_over_nodes(train_state_policies, train_state_values, last_env_state_per_node, last_obs_per_node, rng):

            # Precompute last values for all nodes (N, E)
            last_preds_per_node = _gather_last_preds_for_nodes(last_env_state_per_node)
            all_last_vals_per_node = _gather_last_vals_for_nodes(train_state_values, last_obs_per_node)

            def _select_train_state(ts_tuple, idx):
                # JIT-safe select via masked leaf selection
                N = len(ts_tuple)
                oh = jnp.eye(N, dtype=jnp.int32)[idx]  # (N,)

                # Collect arrays to combine
                params_tuple   = tuple(ts.params   for ts in ts_tuple)
                optstate_tuple = tuple(ts.opt_state for ts in ts_tuple)

                def combine(*elems):
                    acc = jnp.zeros_like(elems[0])
                    for i in range(N):
                        acc = acc + oh[i] * elems[i]
                    return acc

                sel_params   = jax.tree_util.tree_map(combine, *params_tuple)
                sel_optstate = jax.tree_util.tree_map(combine, *optstate_tuple)

                # All static fields are identical across tuple -> take from index 0
                ts0 = ts_tuple[0]
                return ts0.replace(params=sel_params, opt_state=sel_optstate)
            
            def _replace_train_state(ts_tuple, idx, new_elem):
                # JIT-safe replace via masked leaf selection
                N = len(ts_tuple)
                oh = jnp.eye(N, dtype=jnp.int32)[idx]  # (N,)

                new_list = []
                for k in range(N):
                    # For each position k, blend leaves between old[k] and new_elem using one-hot scalar oh[k]
                    old_k = ts_tuple[k]
                    def blend(a_old, a_new):
                        return a_old + oh[k] * (a_new - a_old)
                    new_params   = jax.tree_util.tree_map(blend, old_k.params,   new_elem.params)
                    new_optstate = jax.tree_util.tree_map(blend, old_k.opt_state, new_elem.opt_state)
                    new_list.append(old_k.replace(params=new_params, opt_state=new_optstate))
                return tuple(new_list)

            def body(carry, node_pos):
                ts_pi, ts_v, rng = carry

                # Compute per-node advantages/targets
                adv, tgt, T_ls, T_gs, T_Vs = _per_node_adv_targets(node_pos, last_preds_per_node, all_last_vals_per_node)

                # Policy mask: update only when this node was active (FIXME, could also just reset upon trigger)
                traj_batch = tree_index1(traj_batches, node_pos)
                policy_mask = (traj_batch.current_value_node == node_pos).astype(jnp.float32)
                
                ts_pi_node = _select_train_state(tuple(ts_pi), node_pos)
                ts_v_node  = _select_train_state(tuple(ts_v), node_pos)

                # Build update state
                # upd_state = (ts_pi_node, ts_v_node, traj_batch, adv, tgt, adv, policy_mask, rng_og) # DEBUG FIXME: rng_og -> rng
                upd_state = (ts_pi_node, ts_v_node, traj_batch, adv, tgt, adv, policy_mask, rng)
                xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent_gamma[0]

                upd_state, loss_info = jax.lax.scan(update_epoch, upd_state, xs, length=config["UPDATE_EPOCHS"])
                new_pi, new_v, rng = upd_state[0], upd_state[1], upd_state[-1]

                # Write back updated states at node_pos
                ts_pi = _replace_train_state(ts_pi, node_pos, new_pi)
                ts_v  = _replace_train_state(ts_v,  node_pos, new_v)

                # return (ts_pi, ts_v, rng), (loss_info) # debug fixme
                return (ts_pi, ts_v, rng), (loss_info, adv, tgt, T_ls, T_gs, T_Vs)

            # Ensure states are tuples for static length
            ts_pi = tuple(train_state_policies)
            ts_v  = tuple(train_state_values)

            # (ts_pi, ts_v, rng), loss_infos = jax.lax.scan(body, (ts_pi, ts_v, rng), jnp.arange(len(train_state_policies))) # DEBUG FIXME
            (ts_pi, ts_v, rng), (loss_infos, advantages, targets, T_ls, T_gs, T_Vs) = jax.lax.scan(
                body, (ts_pi, ts_v, rng), jnp.arange(len(train_state_policies))
            )
            # return ts_pi, ts_v, rng, loss_infos
            return ts_pi, ts_v, rng, loss_infos, advantages, targets, T_ls, T_gs, T_Vs, all_last_vals_per_node

        # Run updates over all nodes
        last_env_state_per_node = tuple([tree_index1(runner_states, i)[2] for i in range(num_nodes)])  # tuple of (E, *env_state_shape)
        last_obs_per_node = tuple([tree_index1(runner_states, i)[3] for i in range(num_nodes)])  # tuple of (E, *obs_shape)

        # train_state_policies, train_state_values, rng, loss_infos = _scan_updates_over_nodes( #DEBUG FIXME
        train_state_policies_new, train_state_values_new, rng, loss_infos_new, advantages_new, targets_new, T_ls_new, T_gs_new, T_Vs_new, last_vals_new = _scan_updates_over_nodes(
            train_state_policies, train_state_values, last_env_state_per_node, last_obs_per_node, rng
        )

        ####################################################################################################################
        ## DEBUG FIXME: Fixed for 4-node RRAA DAG (use node positions)

        pos_rraa = 0
        pos_raa1 = 1
        pos_raa2 = 2
        pos_a    = 3

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

        ## DEBUG FIXME EVERYTHING BELOW IS OLD RRAA CODE FOR TESTING ONLY
        ####################################################################################################################
        # UPDATE RRAA
        
        # CALCULATE COMPOSED ADVANTAGE
        (_, _, env_state_rraa, last_obs, _, rng) = runner_state_rraa

        last_val_rraa = train_state_value_rraa.apply_fn(train_state_value_rraa.params, last_obs)
        last_val_raa1 = train_state_value_raa1.apply_fn(train_state_value_raa1.params, last_obs)
        last_val_raa2 = train_state_value_raa2.apply_fn(train_state_value_raa2.params, last_obs)

        # DECOMPOSED REACH VALUES ON COMPOSED PPO ACTOR ROLL OUT        
        V_rraa_append = jnp.concatenate((traj_batch_rraa.value, jnp.expand_dims(last_val_rraa, axis=1).T))
        r1_append = jnp.concatenate((traj_batch_rraa.predicate_values[..., pred_pos_r1], jnp.expand_dims(env_state_rraa.predicate_values[..., pred_pos_r1], axis=1).T))
        V_raa1_append = jnp.concatenate((traj_batch_rraa.all_values[..., pos_raa1], jnp.expand_dims(last_val_raa1, axis=1).T))
        r2_append = jnp.concatenate((traj_batch_rraa.predicate_values[..., pred_pos_r2], jnp.expand_dims(env_state_rraa.predicate_values[..., pred_pos_r2], axis=1).T))
        V_raa2_append = jnp.concatenate((traj_batch_rraa.all_values[..., pos_raa2], jnp.expand_dims(last_val_raa2, axis=1).T))
        
        a_append_rraa = jnp.concatenate((traj_batch_rraa.predicate_values[..., pred_pos_a], jnp.expand_dims(env_state_rraa.predicate_values[..., pred_pos_a], axis=1).T))

        # SPECIAL BRT TARGET FOR BRRT PROBLEM
        l_tilde_rraa = jnp.minimum(jnp.maximum(r1_append, V_raa2_append), jnp.maximum(r2_append, V_raa1_append))

        indexs, done_rraa = calculate_indexs3_rr(ent_gamma[1], traj_batch_rraa.reward, l_tilde_rraa,
                                               jnp.expand_dims(last_val_rraa, axis=1).T) 
        
        done_rraa = done_rraa[:-1, :]

        advantages_V_rraa, targets_V_rraa = calculate_gae_reachavoid4(ent_gamma[1], 
                                                            config["GAE_LAMBDA"], 
                                                            T_ls=l_tilde_rraa, 
                                                            T_gs=a_append_rraa,
                                                            T_Vs=V_rraa_append, 
                                                            done=done_rraa)

        # UPDATE COMPOSED NETWORK
        # composed_policy_mask = jnp.where(traj_batch_rraa.policy_taken == 0, 1., 0.) 
        composed_policy_mask = jnp.where(traj_batch_rraa.current_value_node == pos_rraa, 1., 0.) #FIXME check
        # FIXME FIXME FIXME needs to include all policies now for policy_taken
        update_state_rraa = (train_state_policy_rraa, train_state_value_rraa,
                        traj_batch_rraa, advantages_V_rraa, targets_V_rraa, advantages_V_rraa, composed_policy_mask, rng_og)

        xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent_gamma[0]
        update_state_rraa, loss_info_rraa = jax.lax.scan(
            update_epoch, update_state_rraa, xs, config["UPDATE_EPOCHS"]
        )
        train_state_policy_rraa = update_state_rraa[0]
        train_state_value_rraa = update_state_rraa[1]
        rng = update_state_rraa[-1]

        last_vals_rraa_old = [last_val_rraa, last_val_raa1, last_val_raa2, 0 * last_val_rraa]

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
        composed_policy_mask = jnp.where(traj_batch_raa1.current_value_node == pos_raa1, 1., 0.) #FIXME check
        update_state_raa1 = (train_state_policy_raa1, train_state_value_raa1,
                        traj_batch_raa1, advantages_V_raa1, targets_V_raa1, advantages_V_raa1, composed_policy_mask, rng_og)

        xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent_gamma[0]
        update_state_raa1, loss_info_raa1 = jax.lax.scan(
            update_epoch, update_state_raa1, xs, config["UPDATE_EPOCHS"]
        )
        train_state_policy_raa1 = update_state_raa1[0]
        train_state_value_raa1 = update_state_raa1[1]
        rng = update_state_raa1[-1]

        last_vals_raa1_old = [0 * last_val_rraa, last_val_raa1, 0 * last_val_raa2, last_val_a1]

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
        composed_policy_mask = jnp.where(traj_batch_raa2.current_value_node == pos_raa2, 1., 0.) #FIXME check
        update_state_raa2 = (train_state_policy_raa2, train_state_value_raa2,
                        traj_batch_raa2, advantages_V_raa2, targets_V_raa2, advantages_V_raa2, composed_policy_mask, rng_og)

        xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent_gamma[0]
        update_state_raa2, loss_info_raa2 = jax.lax.scan(
            update_epoch, update_state_raa2, xs, config["UPDATE_EPOCHS"]
        )
        train_state_policy_raa2 = update_state_raa2[0]
        train_state_value_raa2 = update_state_raa2[1]
        rng = update_state_raa2[-1]

        last_vals_raa2_old = [0 * last_val_rraa, 0 * last_val_raa1, last_val_raa2, last_val_a2]

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
                           traj_batch_a, advantages_V_a, targets_V_a, advantages_V_a, dummy_mask, rng_og)
        xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent_gamma[0]
        update_state_a, loss_info_a = jax.lax.scan(
            update_epoch, update_state_a, xs, config["UPDATE_EPOCHS"]
        )
        train_state_policy_a = update_state_a[0]
        train_state_value_a = update_state_a[1]
        rng = update_state_a[-1]

        last_vals_a_old = [0 * last_val_rraa, 0 * last_val_raa1, 0 * last_val_raa2, last_val_a]

        ####################################################################################################################
        ## DEBUG FIXME CHECKING
        
        # CHECK OLD AND NEW TRAIN STATES MATCH
        train_state_policies = [train_state_policy_rraa, train_state_policy_raa1, train_state_policy_raa2, train_state_policy_a]
        train_state_values = [train_state_value_rraa, train_state_value_raa1, train_state_value_raa2, train_state_value_a]
        
        diffs_policy = []
        diffs_value = []
        for i in range(len(train_state_policies)):
            diff_policy_tree = jax.tree_util.tree_map(lambda x, y: jnp.max(jnp.abs(x - y)),
                                                      train_state_policies[i].params, train_state_policies_new[i].params)
            diff_value_tree  = jax.tree_util.tree_map(lambda x, y: jnp.max(jnp.abs(x - y)),
                                                      train_state_values[i].params,  train_state_values_new[i].params)
            # Flatten leaves and reduce with jnp.max
            policy_leaves = jax.tree_util.tree_leaves(diff_policy_tree)
            value_leaves  = jax.tree_util.tree_leaves(diff_value_tree)
            max_diff_policy = jnp.max(jnp.stack([jnp.asarray(l) for l in policy_leaves]))
            max_diff_value  = jnp.max(jnp.stack([jnp.asarray(l) for l in value_leaves]))
            diffs_policy.append(max_diff_policy)
            diffs_value.append(max_diff_value)

        ####################################################################################################################
        # Output

        # train_state_total_out = (train_state_policy_rraa, train_state_value_rraa,
        #     train_state_policy_raa1, train_state_value_raa1,
        #     train_state_policy_raa2, train_state_value_raa2, 
        #     train_state_policy_a, train_state_value_a,
        #     rng, timestep)
        
        # DEBUG FIXME: Fixed for 4-node RRAA DAG
        # train_state_policies = [train_state_policy_rraa, train_state_policy_raa1, train_state_policy_raa2, train_state_policy_a]
        # train_state_values = [train_state_value_rraa, train_state_value_raa1, train_state_value_raa2, train_state_value_a]

        # train_state_total_out = (train_state_policies, train_state_values, rng, timestep)
        train_state_total_out = (train_state_policies_new, train_state_values_new, rng, timestep)

        loss_info_rraa = tree_index1(loss_infos_new, 0)
        loss_info_raa1 = tree_index1(loss_infos_new, 1)
        loss_info_raa2 = tree_index1(loss_infos_new, 2)
        loss_info_a    = tree_index1(loss_infos_new, 3)

        advantages_old = [advantages_V_rraa, advantages_V_raa1, advantages_V_raa2, advantages_V_a]
        targets_old = [targets_V_rraa, targets_V_raa1, targets_V_raa2, targets_V_a]
        T_ls_old = [l_tilde_rraa, l_tilde_raa1, l_tilde_raa2, a_append]
        T_gs_old = [a_append_rraa, a1_append, a2_append, a_append]
        T_Vs_old = [V_rraa_append, V_raa1_append, V_raa2_append, V_a_append]
        last_vals_old = [last_vals_rraa_old, last_vals_raa1_old, last_vals_raa2_old, last_vals_a_old]

        debug_arrays = {
            "max_diff_policy_per_node": jnp.stack(diffs_policy),  # (N,)
            "max_diff_value_per_node":  jnp.stack(diffs_value),   # (N,)
            "T_ls_new": T_ls_new,
            "T_gs_new": T_gs_new,
            "T_Vs_new": T_Vs_new,
            "T_ls_old": T_ls_old,
            "T_gs_old": T_gs_old,
            "T_Vs_old": T_Vs_old,
            "last_vals_new": last_vals_new,
            "last_vals_old": last_vals_old,
            "advantages_new": advantages_new, "targets_new": targets_new,
            "advantages_old": advantages_old, "targets_old": targets_old,
        }

        return (train_state_total_out,
                {
                 "batch_info_rraa": (traj_batch_rraa, targets_V_rraa, done_rraa), "loss_info_rraa": loss_info_rraa,
                 "batch_info_raa1": (traj_batch_raa1, targets_V_raa1, done_raa1), "loss_info_raa1": loss_info_raa1,
                 "batch_info_raa2": (traj_batch_raa2, targets_V_raa2, done_raa2), "loss_info_raa2": loss_info_raa2,
                 "batch_info_a": (traj_batch_a, targets_V_a, done_a), "loss_info_a": loss_info_a,
                 "reach_gamma": ent_gamma[1], "entropy_weight": ent_gamma[0], 
                 "reset_indices": reset_indices, "debug_arrays": debug_arrays, 
                 })

    ########################################################################################################################

    # MAKE THE VALUE TRANSITION FUNCTION FROM THE DAG

    # FIXME TODO: Allows only one node transition per step, what about multiple satisfactions?
    def _value_transition(value_dag, last_value_node, last_env_state):
        pred_vals = last_env_state.predicate_values
        trigger_predicate_map = value_dag.trigger_predicate_map  # (N, P) where entry is child pos or -1
        uniform_sign_predicates = -1.0 * pred_vals * (1 - 2 * value_dag.negated_predicate_mask) # TODO old conv, negative means satisfied

        def per_env_transition(cur_pos_e, pred_vals_e):
            child_pos_per_pred = trigger_predicate_map[cur_pos_e]
            satisfied = (child_pos_per_pred >= 0) & (pred_vals_e < 0)

            sel_child_pos = jnp.take(child_pos_per_pred, jnp.argmax(satisfied), axis=0)
            next_pos_e = jnp.where(jnp.any(satisfied), sel_child_pos, cur_pos_e)
            return next_pos_e

        current_value_node = jax.vmap(per_env_transition, in_axes=(0, 0))(last_value_node, uniform_sign_predicates)
        return current_value_node

    # INIT JAX WRAPPERS
    value_transition = partial(_value_transition, value_dag)
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

        return tuple(train_state_policies_list), tuple(train_state_values_list), node_tags, rng

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
        reset_indices = result["reset_indices"]
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
        if "F16" not in config["EXP_NAME"] and config["USE_WANDB"]:
            if timestep % config['VIDEO_FREQ'] == 0 or timestep == total_timesteps - 1: 
                video_frames = plot_video_contour_RRAA((info_rraa, info_raa1, info_raa2, info_a), timestep, config, save_video=True, log_wandb=config["USE_WANDB"])

        # max_diff_policy_per_node = result["debug_arrays"]["max_diff_policy_per_node"]
        # max_diff_value_per_node  = result["debug_arrays"]["max_diff_value_per_node"]
        # print("DEBUG --- Max diffs (policy):", np.array(max_diff_policy_per_node))
        # print("DEBUG --- Max diffs (value): ", np.array(max_diff_value_per_node))

        # advantages_new = result["debug_arrays"]["advantages_new"]
        # targets_new = result["debug_arrays"]["targets_new"]
        # advantages_old = result["debug_arrays"]["advantages_old"]
        # targets_old = result["debug_arrays"]["targets_old"]

        # for i in range(len(value_dag.nodes)):
        #     diff_adv = jnp.max(jnp.abs(advantages_new[0][i, :, :] - advantages_old[i][0, :, :]))
        #     diff_tgt = jnp.max(jnp.abs(targets_new[0][i, :, :] - targets_old[i][0, :, :]))

        #     print(f"DEBUG --- Node {i} ({node_tags[int(value_dag.nodes[i])]}) - Max advantage diff: {diff_adv:.6f}, Max target diff: {diff_tgt:.6f}")

        plt.close("all")
        print(f"ITER TIME : {t1-t0:2.1f}s : (A)  {100*(1-a_crash_perc):2.1f}%  (RAA1)  {100*raa1_raa_perc:2.1f}%  (RAA2)  {100*raa2_raa_perc:2.1f}%  (RRAA)  {100*rraa_rr_perc:2.1f}%")
        # print("Time {}".format(t1-t0))

    return


# NOTES - Things to fix / check
# - Does done setting we used work better/worse? (what should it be?)
# - 1 vs 2 avoid functions (branch?)
# - Reach/Avoid value scaling?
# - Env length? (200 -> 400?)
# - More envs per batch? (32 -> 128?)
# - entropy/LR?

if __name__ == "__main__":
    config = vars(get_args(sys.argv[1:]))

    debug = True
    if debug:
        config["EXP_NAME"]="PointValDec"
        config["MODEL_DIR"] = 'model_valdec'
        config["DIR"]="point_raa_debug_stage2_coupled_dyn_advs_tgts"
        config["LR"]=3e-4
        config["NUM_ENVS"]=128
        config["NUM_STEPS"]=400
        config["TOTAL_TIMESTEPS"]=200_000_000
        config["STEP_SCAN"]=1 # DEBUG FIXME (should be 4 but confusing for debugging)
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
        config["NAME"]="point_raa_debug_stage2_coupled_dyn_advs_tgts"

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
            # fixed topological order: [RRAA(0), RAA1(1), RAA2(2), A(3)]
            self.nodes = jnp.array([0, 1, 2, 3])
            self.node_tags = {0: 'RRAA', 1: 'RAA1', 2: 'RAA2', 3: 'A'}
            # parent positions (padded with -1 to fixed width 2)
            # positions: 0->0, 1->1, 2->2, 3->3
            # parents: 0 <- [], 1 <- [0], 2 <- [0], 3 <- [2,1]
            self.parent_pos_padded = jnp.array([
                [-1, -1],  # node 0 (pos 0) root
                [ 0, -1],  # node 1 (pos 1) parent: 0(pos0)
                [ 0, -1],  # node 2 (pos 2) parent: 0(pos0)
                [ 1,  2],  # node 3 (pos 3) parents: 1(pos1), 2(pos2)
            ])
            # children positions (not used in this edit, but kept for completeness)
            self.children_pos_padded = jnp.array([
                [ 1,  2],  # pos 0 (node 0) -> children: 1(pos1), 2(pos2)
                [ 3, -1],  # pos 1 (node 1) -> child: A(pos3)
                [ 3, -1],  # pos 2 (node 2) -> child: A(pos3)
                [-1, -1],  # pos 3 (node 3) -> leaf
            ])
            # trigger predicate index for each node (in same order as nodes)
            self.trigger_predicate_map = jnp.array([
                [ 2,  1, -1],  # RRAA can reach1 (-> RAA2) or reach2 (-> RAA1)
                [ 3, -1, -1],  # RAA1 can reach1 (-> A)
                [-1,  3, -1],  # RAA2  can reach2 (-> A)
                [-1, -1, -1],  # A is a terminal node
            ])
            # simple helpers
            self.predicates = ["reach1", "reach2", "obstacles"]
            self.negated_predicate_mask=jnp.array([1, 1, 0]) #TODO this is old conv, new would be jnp.array([0, 0, 1])
            # self.trigger_predicate_ix = {0: jnp.array([2]), 1: jnp.array([0]), 2: jnp.array([1]), 3: jnp.array([])}
            # mapping node id -> position (Python dict for convenient lookup outside JIT)
            self.node_index = {0: 0, 1: 1, 2: 2, 3: 3}
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

    config["USE_WANDB"] = True #not debug # False for debugging
    if config["USE_WANDB"]:
        wandb.init(project='valdec-DEBUG-{}-{}'.format(config["EXP_NAME"], config["WANDB_GROUP"]), name=config["NAME"], config=config,
                   entity='valdec')

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