import os
import jax
import sys
import copy

from rraa_rl.EFPPO.src.rl.arguments import get_args
from rraa_rl.EFPPO.src.env.env_list import get_env

from rraa_rl.EFPPO.src.rl.MORL_PPO_RR import sparse_replace_rr, morl_replace_rr
from rraa_rl.EFPPO.src.rl.eval_all_RR_utils import test_RR, plot_scores_RR

if __name__ == "__main__":
    config = vars(get_args(sys.argv[1:]))

    debug = True
    if debug:
        config["EXP_NAME"]="F16ReachReach"
        config["BASE_MODEL_DIR"] = "model_rebuttal_results"

        config["DIR_HJPPO"]="BASELINE_f16_rr_verttargs_cutsamp_To80m80s_tjreset_LR1e-3"
        config["DIR_MODEL_HJPPO"]="checkpoint_194"

        config["DIR_CPPOv1"]="BASELINE_f16_rr_cppo_0"
        config["DIR_MODEL_CPPOv1"]="checkpoint_193"

        config["DIR_CPPOv2"]="BASELINE_f16_rr_cppo_0"
        config["DIR_MODEL_CPPOv2"]="checkpoint_193"

        config["DIR_CPPOv3"]="BASELINE_f16_rr_cppo_cutsamp"
        config["DIR_MODEL_CPPOv3"]="best_124"

        config["DIR_DSTL"]="BASELINE_f16_reachreachdecomposed_100M"
        config["DIR_MODEL_DSTL"]="best_143"

        config["DIR_PPOLAG"]="BASELINE_f16_rr_ppolag_lam0p01"
        config["DIR_MODEL_PPOLAG"]="best_23"

        config["DIR_PPO"]="BASELINE_f16_rr_ppo"
        config["DIR_MODEL_PPO"]="best_91"

        config["DIR_RCPPO"]="BASELINE_f16_rr_rcppo"
        config["DIR_MODEL_RCPPO"]="best_160"

        config["DIR_RESPO"]="BASELINE_f16_rr_respo"
        config["DIR_MODEL_RESPO"]="best_26"

        config["DIR_MORL"]="BASELINE_f16_rr_morl"
        config["DIR_MODEL_MORL"]="best_24"

        config["DIR_SPARSE"]="BASELINE_f16_rr_sparse"
        config["DIR_MODEL_SPARSE"]="best_84"

        config["DIR_P2BPO"]="BASELINE_f16_rr_p2bpo"
        config["DIR_MODEL_P2BPO"]="best_126"

        config["DIR_LOGBAR"]="BASELINE_f16_rr_logbar"
        config["DIR_MODEL_LOGBAR"]="best_166"

        config['NAME_TAG'] = "F16_RR"

    config["NUM_ENVS"]=1000
    config["NUM_STEPS"]=200
    config["ACTIVATION"]="tanh"
    config["ENV_REWARD_TYPE"] = "accumulated" # reward
    config["ENV_COST_FN"] = "sum" # cost_fn
    config["ENV_COST_TYPE"] = "accumulated" # cost
    config["CPPO_UPDATE_TYPE"] = "mean" # update
    config["USE_STL"] = False # stl

    envs_HJPPO = get_env(config)
    env_HJPPO, env_HJPPO_1, env_HJPPO_2 = envs_HJPPO

    # "BASELINE_final_F16_rr_cppomax_raccum_cfnmax_caccum_umin_V1--LR=3e-4"
    config_CPPOv1 = copy.deepcopy(config)
    config_CPPOv1["EXP_NAME"] = "F16ReachReach_CPPO"
    env_CPPO_v1 = get_env(config_CPPOv1)

    # "BASELINE_final_F16_rr_cpposum_raccum_cfnmax_caccum_umin_V1"
    config_CPPOv2 = copy.deepcopy(config)
    config_CPPOv2["EXP_NAME"] = "F16ReachReach_CPPO"
    env_CPPO_v2 = get_env(config_CPPOv2)

    # "BASELINE_final_F16_rr_cpposum_raccum_cfnsum_caccum_umean_V2"
    config_CPPOv3 = copy.deepcopy(config)
    config_CPPOv3["EXP_NAME"] = "F16ReachReach_CPPO"
    env_CPPO_v3 = get_env(config_CPPOv3)

    config_dSTL = copy.deepcopy(config)
    config_dSTL["EXP_NAME"] = "F16ReachReachDecomposed"
    env_dSTL, env_dSTL_1, env_dSTL_2 = get_env(config_dSTL)

    ## PPO LAG
    config_PPOLAG = copy.deepcopy(config)
    config_PPOLAG["EXP_NAME"] = "F16ReachReach_CPPO"
    env_PPOLAG = get_env(config_PPOLAG)

    ## PPO
    config_PPO = copy.deepcopy(config)
    config_PPO["EXP_NAME"] = "F16ReachReach_CPPO"
    env_PPO = get_env(config_PPO)

    ## RCPPO
    config_RCPPO = copy.deepcopy(config)
    config_RCPPO["EXP_NAME"] = "F16ReachReach_RCPPO"
    env_RCPPO = get_env(config_RCPPO)

    ## RESPO
    config_RESPO = copy.deepcopy(config)
    config_RESPO["EXP_NAME"] = "F16ReachReach_RESPO" 
    env_RESPO = get_env(config_RESPO)

    ## MORL
    config_MORL = copy.deepcopy(config)
    config_MORL["EXP_NAME"] = "F16ReachReachBaseline_MORL"
    env_MORL = get_env(config_MORL)
    env_MORL = morl_replace_rr(env_MORL)

    ## SPARSE
    config_SPARSE = copy.deepcopy(config)
    config_SPARSE["EXP_NAME"] = "F16ReachReachBaseline_Sparse"
    env_SPARSE = get_env(config_SPARSE)
    env_SPARSE = sparse_replace_rr(env_SPARSE)

    ## P2BPO
    config_P2BPO = copy.deepcopy(config)
    config_P2BPO["EXP_NAME"] = "F16ReachReachBaseline_P2BPO"
    env_P2BPO = get_env(config_P2BPO)

    ## LOGBAR
    config_LOGBAR = copy.deepcopy(config)
    config_LOGBAR["EXP_NAME"] = "F16ReachReach_CPPO"
    env_LOGBAR = get_env(config_LOGBAR)

    envs = (
        env_HJPPO, env_HJPPO_1, env_HJPPO_2, 
        env_CPPO_v1, env_CPPO_v2, env_CPPO_v3, env_dSTL, env_dSTL_1, env_dSTL_2,
        env_PPOLAG, env_PPO, 
        env_RCPPO, env_RESPO, 
        env_MORL, env_SPARSE,
        env_P2BPO, env_LOGBAR
    )
    env_paramss = (
        env_HJPPO.default_params, env_HJPPO_1.default_params, env_HJPPO_2.default_params,
        env_CPPO_v1.default_params, env_CPPO_v2.default_params, env_CPPO_v3.default_params, env_dSTL.default_params, env_dSTL_1.default_params, env_dSTL_2.default_params, 
        env_PPOLAG.default_params, env_PPO.default_params, env_RCPPO.default_params, env_RESPO.default_params, env_MORL.default_params, env_SPARSE.default_params,
        env_P2BPO.default_params, env_LOGBAR.default_params
    )

    rng_1 = jax.random.PRNGKey(20)
    rng_2 = jax.random.PRNGKey(20)
    rng_3 = jax.random.PRNGKey(20)
    rng_4 = jax.random.PRNGKey(20)
    rng_5 = jax.random.PRNGKey(20)
    rng_6 = jax.random.PRNGKey(20)
    rng_7 = jax.random.PRNGKey(20)
    rng_8 = jax.random.PRNGKey(20)
    rng_9 = jax.random.PRNGKey(20)
    rng_10 = jax.random.PRNGKey(20)
    rng_11 = jax.random.PRNGKey(20)
    rng_12 = jax.random.PRNGKey(20)
    rng_13 = jax.random.PRNGKey(20)
    rng_14 = jax.random.PRNGKey(20)
    rngs = (rng_1, rng_2, rng_3, rng_4, rng_5, rng_6, rng_7, rng_8, rng_9, rng_10, rng_11, rng_12, rng_13, rng_14)

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    print("\n\nCollecting Trajectories")
    traj_batches = test_RR(envs, env_paramss, config, rngs, saving_traj=True)

    os.makedirs(f"eval/{config['EVAL_DIR']}/{config['NAME_TAG']}", exist_ok=True)

    score_plot = plot_scores_RR(traj_batches, config, title="F16-RR")
