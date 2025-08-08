import os
import jax
import sys
import copy

from rraa_rl.EFPPO.src.rl.arguments import get_args
from rraa_rl.EFPPO.src.env.env_list import get_env

from rraa_rl.EFPPO.src.rl.MORL_PPO_RAA import sparse_replace_raa, morl_replace_raa
from rraa_rl.EFPPO.src.rl.eval_all_RAA_utils import test_RAA, plot_scores_RAA

if __name__ == "__main__":
    config = vars(get_args(sys.argv[1:]))

    debug = True
    if debug:
        config["EXP_NAME"]="HalfCheetahReachAlwaysAvoid"
        config["BASE_MODEL_DIR"] = "model_rebuttal_results"

        config["DIR_HJPPO"]="BASELINE_halfcheetah_raa_resetgoalsafe_avoidv9"
        config["DIR_MODEL_HJPPO"]="best_125"#"best_126"

        config["DIR_CPPO"]="BASELINE_halfcheetah_raa_cppo_kp100"
        config["DIR_MODEL_CPPO"]="best_343"

        config["DIR_RA"]="BASELINE_halfcheetah_ra"
        config["DIR_MODEL_RA"]="best_488"

        config["DIR_PPOLAG"]="BASELINE_halfcheetah_raa_ppolag_lam0p01"
        config["DIR_MODEL_PPOLAG"]="best_422"

        config["DIR_PPO"]="BASELINE_halfcheetah_raa_ppo"
        config["DIR_MODEL_PPO"]="best_4" 

        config["DIR_RCPPO"]="BASELINE_halfcheetah_raa_rcppo"
        config["DIR_MODEL_RCPPO"]="best_171" 

        config["DIR_RESPO"]="BASELINE_halfcheetah_raa_respo_kp1"
        config["DIR_MODEL_RESPO"]="best_6"

        config["DIR_MORL"]="BASELINE_halfcheetah_raa_morl"
        config["DIR_MODEL_MORL"]="best_459"

        config["DIR_SPARSE"]="BASELINE_halfcheetah_raa_sparse"
        config["DIR_MODEL_SPARSE"]="best_461" 

        config["DIR_P2BPO"]="BASELINE_halfcheetah_raa_p2bpo"
        config["DIR_MODEL_P2BPO"]="best_1"

        config["DIR_LOGBAR"]="BASELINE_halfcheetah_raa_logbar"
        config["DIR_MODEL_LOGBAR"]="best_4"

        
        config['NAME_TAG'] = "HalfCheetah_RAA_080625"

    config["NUM_ENVS"]=1000
    config["NUM_STEPS"]=400
    config["ACTIVATION"]="tanh"
    config["ENV_REWARD_TYPE"] = "accumulated" # reward
    config["ENV_COST_FN"] = "max" # cost_fn
    config["ENV_COST_TYPE"] = "accumulated" # cost
    config["CPPO_UPDATE_TYPE"] = "min" # update
    config["USE_STL"] = False # stl

    envs_HJPPO = get_env(config)
    env_HJPPO, env_HJPPO_avoid = envs_HJPPO

    config_CPPO = copy.deepcopy(config)
    config_CPPO["EXP_NAME"] = "HalfCheetahReachAlwaysAvoid_CPPO"
    env_CPPO = get_env(config_CPPO)

    config_RA = copy.deepcopy(config)
    config_RA["EXP_NAME"] = "HalfCheetahReachAvoid"
    env_RA = get_env(config_RA)

    ## PPO LAG
    config_PPOLAG = copy.deepcopy(config)
    config_PPOLAG["EXP_NAME"] = "HalfCheetahReachAlwaysAvoid_CPPO"
    env_PPOLAG = get_env(config_PPOLAG)

    ## PPO
    config_PPO = copy.deepcopy(config)
    config_PPO["EXP_NAME"] = "HalfCheetahReachAlwaysAvoid_CPPO"
    env_PPO = get_env(config_PPO)

    ## RCPPO
    config_RCPPO = copy.deepcopy(config)
    config_RCPPO["EXP_NAME"] = "HalfCheetahReachAlwaysAvoid_CPPO"
    env_RCPPO = get_env(config_RCPPO)

    ## RESPO
    config_RESPO = copy.deepcopy(config)
    config_RESPO["EXP_NAME"] = "HalfCheetahReachAlwaysAvoid_CPPO"
    env_RESPO = get_env(config_RESPO)

    ## MORL
    config_MORL = copy.deepcopy(config)
    config_MORL["EXP_NAME"] = "HalfCheetahReachAlwaysAvoidBaseline_MORL"
    env_MORL = get_env(config_MORL)
    env_MORL = morl_replace_raa(env_MORL)

    ## SPARSE
    config_SPARSE = copy.deepcopy(config)
    config_SPARSE["EXP_NAME"] = "HalfCheetahReachAlwaysAvoidBaseline_Sparse"
    env_SPARSE = get_env(config_SPARSE)
    env_SPARSE = sparse_replace_raa(env_SPARSE)

    ## P2BPO
    config_P2BPO = copy.deepcopy(config)
    config_P2BPO["EXP_NAME"] = "HalfCheetahReachAlwaysAvoidBaseline_P2BPO"
    env_P2BPO = get_env(config_P2BPO)

    ## LOGBAR
    config_LOGBAR = copy.deepcopy(config)
    config_LOGBAR["EXP_NAME"] = "HalfCheetahReachAlwaysAvoid_CPPO"
    env_LOGBAR = get_env(config_LOGBAR)

    envs = (
        env_HJPPO, env_HJPPO_avoid, 
        env_CPPO, env_RA,
        env_PPOLAG, env_PPO, 
        env_RCPPO, env_RESPO, 
        env_MORL, env_SPARSE,
        env_P2BPO, env_LOGBAR
    )
    env_paramss = (
        env_HJPPO.default_params, env_HJPPO_avoid.default_params, 
        env_CPPO.default_params, env_RA.default_params,
        env_PPOLAG.default_params, env_PPO.default_params, 
        env_RCPPO.default_params, env_RESPO.default_params, 
        env_MORL.default_params, env_SPARSE.default_params,
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
    traj_batches = test_RAA(envs, env_paramss, config, rngs, saving_traj=True)

    os.makedirs(f"eval/{config['EVAL_DIR']}/{config['NAME_TAG']}", exist_ok=True)

    score_plot = plot_scores_RAA(traj_batches, config, title="HalfCheetah-RAA")
