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

class TrainState(train_state.TrainState):
    mean: Any
    variance: Any
    count: Any

### SCRIPTING TODO/FIXME/NOTE
# - valtr dag creation
# - generalized scoring method
# - generalized plotting method
# - FIX the done-reset mechanism
# - FIX sequential-policy mask -> reset instead (true seq rollout in eval)
# - ^related, when masking we waste data valid for other node/decomp
# - automatic reweighting of reach/avoid predicates to improve satisfaction

## OPEN QUESTIONS
# - one vs multiple representations? (one per node vs shared)

## Why might training be different? vs. DOHJPPO
# - exposing all predicate values to obs (even in decompsed nodes)

def train(env, env_params, value_dag, config, rng, plot_function=None):
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
                obs_buffer,        # [N, T, E, O]
                predicate_buffer,  # [N, T, E, P]
                reset_buffer       # [N, E]
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

            ## RESET

            obsv, env_state, reset_indices = jax.lax.cond(
                is_root_node,
                lambda r: standard_reset(env, env_params, config, r),
                lambda r: coupled_reset(env, env_params, config, r),
                rng
            )

            ## COLLECT TRAJECTORY

            rng, _rng = jax.random.split(rng)
            # pass node position (not node id) so value_transition and env_step work with position indexing
            initial_value_node = jnp.ones((config["NUM_ENVS"],), dtype=jnp.int32) * node_pos
            runner_state = (train_state_policies, train_state_values, env_state, obsv, initial_value_node, _rng)
            runner_state, traj_batch = jax.lax.scan(
                env_step, runner_state, None, config["NUM_STEPS"]
            )

            ## UPDATE BUFFERS FOR CHILDREN

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

        # CORRECT BUT SLOWER VERSION
        # last_obs_per_node: (N, E, *obs_shape) -> node vals at each last obs (N, N, E)
        def _gather_last_vals_for_nodes(train_state_values, last_obs_per_node):
            def apply_val(i):
                v = train_state_values[i]
                values = [v.apply_fn(v.params, obs) for obs in last_obs_per_node]
                return jnp.stack(values, axis=0)
            values = [apply_val(i) for i in range(num_nodes)]
            return jnp.stack(values, axis=0)  # (N, N, E)

        # # FASTER BUT SMALL DIFF IN VALUES FROM ABOVE?; 
        # # 1e-3 difference in adv/tgt and then 1e-2 diff in policy/value? tried debuggin for a while to no avail
        # def _gather_last_vals_for_nodes(train_state_values, last_obs_per_node):
        #     last_obs_stacked = jnp.stack(last_obs_per_node, axis=0) # (N, E, *obs)
        #     apply_fns   = tuple(ts.apply_fn for ts in train_state_values)
        #     params_list = tuple(ts.params   for ts in train_state_values)
        #     def eval_all_for_source(obs_j):
        #         def make_branch(i):
        #             apply_i  = apply_fns[i]
        #             params_i = params_list[i]
        #             def branch(_):
        #                 return jax.vmap(apply_i, in_axes=(None, 0))(params_i, obs_j)  # (E,)
        #             return branch
        #         branches = tuple(make_branch(i) for i in range(len(train_state_values)))
        #         def per_eval(i):
        #             return jax.lax.switch(i, branches, operand=0)  # (E,)
        #         return jax.lax.map(per_eval, jnp.arange(len(train_state_values)))   # (N_eval, E)
        #     out = jax.lax.map(eval_all_for_source, last_obs_stacked)
        #     return jnp.transpose(out, (1, 0, 2))  # (N_source, N_eval, E)

        # def _gather_last_vals_for_nodes(train_state_values, last_obs_per_node):
        #     """
        #     Returns out[i, j, e] = V_i(last_obs_of_node_j for env e)
        #     - last_obs_per_node: length N_src of arrays (E, *obs_shape)
        #     - train_state_values: length N_eval of TrainState
        #     - out: (N_eval, N_src, E)
        #     """
        #     # Stack source-node obs: (N_src, E, *obs_shape)
        #     last_obs_stacked = jnp.stack(last_obs_per_node, axis=0)

        #     # Stack evaluator params leafwise: PyTree with leading axis N_eval
        #     params_stacked = jax.tree_util.tree_map(
        #         lambda *leaves: jnp.stack(leaves, axis=0),
        #         *[ts.params for ts in train_state_values]
        #     )
        #     apply_fn = train_state_values[0].apply_fn  # identical architecture

        #     # Evaluate one evaluator's params across all source nodes with lax.map to mimic the Python loop
        #     def eval_for_params(params_i):
        #         # obs_j: (E, *obs_shape) -> value: (E,)
        #         def eval_one(obs_j):
        #             return apply_fn(params_i, obs_j)
        #         return jax.lax.map(eval_one, last_obs_stacked)  # (N_src, E)

        #     # vmap over evaluator nodes (params axis 0)
        #     out = jax.vmap(eval_for_params, in_axes=0, out_axes=0)(params_stacked)  # (N_eval, N_src, E)
        #     # return out
        #     return jnp.transpose(out, (1, 0, 2))
        
        def _gather_last_preds_for_nodes(last_env_state_per_node):
            return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *last_env_state_per_node).predicate_values   # (N, E, P)

        def _per_node_adv_targets(node_pos, node_last_preds, node_last_vals):
            traj_batch = tree_index1(traj_batches, node_pos)

            node_type = value_dag.node_types[node_pos]
            avoid_mask = value_dag.predicate_types == 1  # (P,)

            # Compute appended values
            last_pred = node_last_preds[node_pos][None, ...]  # (1, E, P)
            last_vals = node_last_vals[node_pos] # (N, N, E) -> (N, E)
            pred_append = jnp.concatenate([traj_batch.predicate_values, last_pred], axis=0)  # (T+1, E, P)
            all_val_append = jnp.concatenate([traj_batch.all_values, last_vals.T[None, ...]], axis=0)  # (T+1, E, N)
            val_append = jnp.concatenate([traj_batch.value, last_vals[node_pos][None, ...]], axis=0)  # (T+1, E)
            
            # Compute avoid target (neutralized if none)
            a_masked = jnp.where(avoid_mask[None, None, :], pred_append, -jnp.inf) # (T+1, E, P), NOTE old convention, neg avoiding
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
                                               last_vals[node_pos][None, ...])
                done = done[:-1, :] # FIXME

                # Compute advantage and targets
                adv, tgt = calculate_gae_reachavoid4(
                    ent_gamma[1], config["GAE_LAMBDA"], T_ls=ra_target, T_gs=a_target, T_Vs=val_append, done=done
                )
                # return adv, tgt #DEBUG FIXME
                T_ls = ra_target
                T_gs = a_target
                T_Vs = val_append
                return adv, tgt, T_ls, T_gs, T_Vs, done

            def avoid_N_advantage_target(_):

                # Compute done flags (FIXME, use resets properly)
                indexs, done = calculate_indexs3_rr(ent_gamma[1], traj_batch.reward, a_target,
                                               last_vals[node_pos][None, ...])
                done = done[:-1, :] # FIXME

                # Compute advantage and targets
                adv, tgt = calculate_gae_avoid4(
                    ent_gamma[1], config["GAE_LAMBDA"], T_hs=a_target, T_Vhs=val_append, done=done
                )
                # return adv, tgt #DEBUG FIXME
                T_hs = a_target
                T_Vhs = val_append
                return adv, tgt, T_hs, T_hs, T_Vhs, done

            # # TODO
            # def advantage_globally_wrapped(_):
            #     return adv, tgt

            branches = (reachavoid_N_advantage_target, avoid_N_advantage_target)
            return jax.lax.switch(node_type, branches, operand=None)

        def _updates_over_nodes(ts_pi, ts_v, nodes_last_preds, nodes_last_vals, rng):

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

                ## COMPUTE PER-NODE ADVANTAGE AND TARGETS

                adv, tgt, T_ls, T_gs, T_Vs, done = _per_node_adv_targets(node_pos, nodes_last_preds, nodes_last_vals)

                ## UPDATE

                # Policy mask: update only when this node was active (FIXME, could also just reset upon trigger)
                traj_batch = tree_index1(traj_batches, node_pos)
                policy_mask = (traj_batch.current_value_node == node_pos).astype(jnp.float32)
                
                ts_pi_node = _select_train_state(ts_pi, node_pos)
                ts_v_node  = _select_train_state(ts_v, node_pos)

                upd_state = (ts_pi_node, ts_v_node, traj_batch, adv, tgt, adv, policy_mask, rng)
                xs = jnp.ones(config["UPDATE_EPOCHS"]) * ent_gamma[0]
                upd_state, loss_info = jax.lax.scan(
                    update_epoch, upd_state, xs, length=config["UPDATE_EPOCHS"]
                )
                new_pi, new_v, rng = upd_state[0], upd_state[1], upd_state[-1]

                # Write back updated states at node_pos
                ts_pi = _replace_train_state(ts_pi, node_pos, new_pi)
                ts_v  = _replace_train_state(ts_v,  node_pos, new_v)

                return (ts_pi, ts_v, rng), (loss_info, adv, tgt, T_ls, T_gs, T_Vs, done)

            (ts_pi, ts_v, rng), (loss_infos, advantages, targets, T_ls, T_gs, T_Vs, dones) = jax.lax.scan(
                body, (ts_pi, ts_v, rng), jnp.arange(len(ts_pi))
            )

            return ts_pi, ts_v, rng, loss_infos, advantages, targets, T_ls, T_gs, T_Vs, dones

        # Precompute last values for all nodes (N, E)
        last_env_state_per_node = tuple([tree_index1(runner_states, i)[2] for i in range(num_nodes)])  # tuple of (E, *env_state_shape)
        last_obs_per_node = tuple([tree_index1(runner_states, i)[3] for i in range(num_nodes)])  # tuple of (E, *obs_shape)
        last_preds_per_node = _gather_last_preds_for_nodes(last_env_state_per_node)
        last_vals_per_node = _gather_last_vals_for_nodes(train_state_values, last_obs_per_node)

        ## RUN UPDATES OVER NODES

        (train_state_policies_new, train_state_values_new, rng, loss_infos, advantages, targets, T_ls, T_gs, T_Vs, dones
         ) = _updates_over_nodes(
                train_state_policies, train_state_values, last_preds_per_node, last_vals_per_node, rng
        )

        ####################################################################################################################

        ## OUTPUT

        train_state_total_out = (train_state_policies_new, train_state_values_new, rng, timestep)

        training_arrays = {
            "advantages": advantages, 
            "targets": targets,
            "T_ls": T_ls,
            "T_gs": T_gs,
            "T_Vs": T_Vs,
            "last_vals": last_vals_per_node,
        }

        return (train_state_total_out,
                {"traj_batches": traj_batches, 
                 "loss_infos": loss_infos, 
                 "dones": dones,
                 "reach_gamma": ent_gamma[1], 
                 "entropy_weight": ent_gamma[0],
                 "reset_indices": reset_indices,
                 "training_arrays": training_arrays,
                 })

    ########################################################################################################################

    ## MAKE THE VALUE TRANSITION FUNCTION FROM THE DAG

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

    def create_train_state(value_dag, config, env, env_params, rng, load=False):

        train_state_policies_list = []
        train_state_values_list = []
        node_tags = {}

        for pos in range(len(value_dag.nodes)):
            node = int(value_dag.nodes[pos])

            # INIT NETWORKS
            value_network = Value_Network(activation=config["ACTIVATION"])
            if config["DISCRETE"] == False:
                policy_network = MoGPolicy_Network( # Gaussian Mixture (TODO does this really work better?)
                # policy_network_rraa = Policy_Network(
                    env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
                )
            else:
                policy_network = Policy_Network_Discrete(
                    env.action_space(env_params).n, activation=config["ACTIVATION"]
                )

            # INIT OR LOAD PARAMETERS
            if not load:
                rng, _rng = jax.random.split(rng)
                init_x = mean_pol = mean_val = var_pol = var_val = jnp.zeros(env.observation_space(env_params).shape)
                network_params_pol = policy_network.init(_rng, init_x)
                
                rng, _rng = jax.random.split(rng)
                network_params_val = value_network.init(_rng, init_x)
                count_pol = count_val = 1e-4
            
            else:
                raw_restored = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath('model/{}/{}'.format(
                    config["LOAD_DEC_DIR"], config["LOAD_DEC_DIR_MODEL"])), target=None)
                
                network_params_pol = raw_restored['policy_network_{}'.format(node)]['params']
                mean_pol = raw_restored['policy_network_{}'.format(node)]["mean"]
                var_pol = raw_restored['policy_network_{}'.format(node)]["variance"]
                count_pol = raw_restored['policy_network_{}'.format(node)]["count"]
                
                network_params_val = raw_restored['value_network_{}'.format(node)]['params']
                mean_val = raw_restored['value_network_{}'.format(node)]["mean"]
                var_val = raw_restored['value_network_{}'.format(node)]["variance"]
                count_val = raw_restored['value_network_{}'.format(node)]["count"]

            # CREATE TRAIN STATES
            train_state_policy = TrainState.create(
                apply_fn=policy_network.apply,
                params=network_params_pol,
                tx=tx,
                mean=mean_pol,
                variance=var_pol,
                count=count_pol,
             )
            train_state_value = TrainState.create(
                apply_fn=value_network.apply,
                params=network_params_val,
                tx=tx,
                mean=mean_val,
                variance=var_val,
                count=count_val,
            )

            # APPEND TO LISTS
            train_state_policies_list.append(train_state_policy)
            train_state_values_list.append(train_state_value)

            node_tags[node] = value_dag.node_tags[node] if isinstance(value_dag.node_tags, dict) and node in value_dag.node_tags else None

        return tuple(train_state_policies_list), tuple(train_state_values_list), node_tags, rng

    train_state_policies, train_state_values, node_tags, rng = create_train_state(
        value_dag, config, env, env_params, rng, load=config["LOAD_DECOMPOSED"])

    # LOAD PRETRAINED DECOMPOSED IF SPECIFIED
    # if not config["LOAD_DECOMPOSED"]:
    #     update_epoch_dec = partial(_ppo_vanilla_update, config)

    # # # IF LOADING PRESOLVED DECOMPOSED, NO TRAINING
    # else:
    #     def _no_update(config, update_state, ent):
    #         dummy_loss = {
    #             "actor_loss": 0.0,
    #             "value_loss": 0.0,
    #             "entropy_loss": 0.0,
    #         }
    #         return update_state, dummy_loss
    #     update_epoch_dec = partial(_no_update, config)

    ########################################################################################################################

    ## MAIN TRAINING LOOP

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

        ## SCORING
        
        traj_batches = tree_index1(result['traj_batches'], 0) # first scan step
        loss_infos = tree_index1(result['loss_infos'], 0)   

        # Generalized scoring method
        def per_batch_success(traj_batches, node_pos, th=0.):
            traj_batch = tree_index1(traj_batches, node_pos)

            reach_predicates = value_dag.trigger_predicate_map[node_pos] > 0
            any_reaches = jnp.any(traj_batch.predicate_values < th, axis=0) & reach_predicates[None, :] # (E, P) & (P,) -> (E, P)
            reach_all = jnp.all(any_reaches == reach_predicates[None, :], axis=-1) # (E,)
            each_reach_idx = jnp.where(any_reaches, (traj_batch.predicate_values < th).argmax(axis=0), -jnp.inf)  # (E, P)
            reach_all_idx = jnp.where(reach_all, each_reach_idx.max(axis=-1), jnp.inf)  # (E,)
            each_reach_idx = jnp.where(each_reach_idx > 0., each_reach_idx, jnp.inf)  # to match old convention

            avoid_predicates = value_dag.predicate_types == 1
            any_crashes = jnp.any(traj_batch.predicate_values > th, axis=0) & avoid_predicates[None, :] # (E, P) & (P,) -> (E, P)
            any_crash = jnp.any(any_crashes, axis=-1) # (E)
            each_crash_idxs = jnp.where(any_crashes, (traj_batch.predicate_values > th).argmax(axis=0), jnp.inf)  # (E, P)
            any_crash_idx = jnp.where(any_crash, each_crash_idxs.min(axis=-1), jnp.inf)  # (E,)

            reach_all_perc = (jnp.sum(reach_all) / reach_all.__len__())
            any_crash_perc = (jnp.sum(any_crash) / any_crash.__len__())
            reach_avoid_perc = (jnp.sum((~any_crash) & reach_all) / reach_all.__len__())
            each_reach_perc = (jnp.sum(any_reaches, axis=0) / any_reaches.shape[0])
            each_crash_perc = (jnp.sum(any_crashes, axis=0) / any_crashes.shape[0])

            return traj_batches, {
                "reach_perc": reach_all_perc,
                "crash_perc": any_crash_perc,
                "reach_avoid_perc": reach_avoid_perc,
                "each_reach_perc": each_reach_perc,
                "each_crash_perc": each_crash_perc,
                "reach_idx": reach_all_idx,
                "crash_idx": any_crash_idx,
                "each_reach_idx": each_reach_idx,
                "each_crash_idx": each_crash_idxs,
            }

        traj_batches, scores = jax.lax.scan(
            per_batch_success, traj_batches, jnp.arange(len(value_dag.nodes))
        )

        t1 = time.time()

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
        
        top_node_score = tree_index1(scores, 0)["reach_avoid_perc"].item()
        if top_node_score > best_score:
            best_score = top_node_score
            checkpoints.save_checkpoint(ckpt_dir=os.path.abspath(os.path.join("model", config["DIR"])),
                                        target=all_training_states,
                                        step=timestep,
                                        prefix="best_",
                                        overwrite=True,)

        ## WRITE TO WANDB

        reported_dict={}
        reported_dict["Train/reach_gamma"] = result['reach_gamma'][0]
        reported_dict["Train/entropy_weight"] = result['entropy_weight'][0]
        for node in value_dag.reported_nodes:
            pos = value_dag.node_index[node]
            node_score = tree_index1(scores, pos)
            if value_dag.node_types[pos] == 0:  # Reach-Avoid
                reported_dict[f"Score/RA_Node_{node}_Reach[%]"] = node_score["reach_perc"].item()
                reported_dict[f"Score/RA_Node_{node}_Crash[%]"] = node_score["crash_perc"].item()
                reported_dict[f"Score/RA_Node_{node}_ReachAvoid[%]"] = node_score["reach_avoid_perc"].item()
            elif value_dag.node_types[pos] == 1:  # Avoid
                reported_dict[f"Score/A_Node_{node}_Crash[%]"] = node_score["crash_perc"].item()
            reported_dict[f"Loss/Node_{node}_actor_loss"] = jnp.mean(tree_index1(loss_infos, pos)["actor_loss"])
            reported_dict[f"Loss/Node_{node}_value_loss"] = jnp.mean(tree_index1(loss_infos, pos)["value_loss"])

        if config["USE_WANDB"]:
            wandb.log(reported_dict, step=timestep)

        ## PLOTTING

        if plot_function is not None:
            out_put_figs = plot_function(value_dag, config, result, scores, timestep, total_timesteps)
            plt.close("all")

        ## PRINT REPORTED NODES TO CONSOLE

        print_statements = [f"ITER: {timestep:<d}/{total_timesteps:<d}, TIME: {t1-t0:2.1f}s", "||"]
        for node in value_dag.reported_nodes:
            pos = value_dag.node_index[node]
            node_score = tree_index1(scores, pos)
            if value_dag.node_types[pos] == 0:  # Reach-Avoid
                print_statements.append(f"(N{node}-RA): {100*node_score['reach_avoid_perc']:.1f}%")
            elif value_dag.node_types[pos] == 1:  # Avoid
                print_statements.append(f"(N{node}-A): {100*(1-node_score['crash_perc']):.1f}%")
        print("   ".join(print_statements))

    return


#########################################################################################################################################
#########################################################################################################################################


if __name__ == "__main__": ## DEBUG
    config = vars(get_args(sys.argv[1:]))

    debug = True
    if debug:
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
            # positions: 0->0, 1->1, 2->2, 3->3
            # parents: 0 <- [], 1 <- [0], 2 <- [0], 3 <- [2,1]
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

            # simple helpers
            self.predicates = ["reach1", "reach2", "obstacles"]
            self.negated_predicate_mask=jnp.array([1, 1, 0]) # NOTE this is used in env directly, < 0 triggers reach-preds, > 0 triggers avoid-preds
            # mapping node id -> position (Python dict for convenient lookup outside JIT)
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

    config["USE_WANDB"] = False #not debug # False for debugging
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