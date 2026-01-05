import argparse

def get_args(args):
    parser = argparse.ArgumentParser(description='RAN-RL')
    parser.add_argument(
        '--EXP_NAME', default='GridConstraint', type=str, help='experiment environment name'
    )
    parser.add_argument(
        '--MODEL_DIR', default='model', type=str, help='experiment model directory'
    )
    parser.add_argument(
        '--DIR', default='grid_constraint', type=str, help='experiment data directory'
    )
    parser.add_argument(
        '--DIR_MODEL', default='checkpoint_625', type=str, help='experiment model name'
    )
    parser.add_argument(
        '--EVAL_DIR', default='eval_all', type=str, help='evaluation output directory'
    )
    parser.add_argument(
        '--NAME', default='experiment', type=str, help='experiment name'
    )
    parser.add_argument(
        '--LR', default=3e-4, type=float, help='learning rate'
    )
    parser.add_argument(
        "--SEED", default=20,  type=int, help="rng seed for reproducibility"
    )
    parser.add_argument(
        "--NUM_ENVS", default=16, type=int, help="number of environments in training"
    )
    parser.add_argument(
        "--NUM_STEPS", default=199, type=int, help="number of timesteps in each trajectory"
    )
    parser.add_argument(
        "--TOTAL_TIMESTEPS", default=2e6, type=int, help="number of timesteps in training"
    )
    parser.add_argument(
        "--SECTION", default=0, type=int, help="index of section in wind field"
    )
    parser.add_argument(
        "--STEP_SCAN", default=1, type=int, help="number of timesteps in scan"
    )
    parser.add_argument(
        "--UPDATE_EPOCHS", default=10, type=int, help="number of updates in each PPO updating"
    )
    parser.add_argument(
        "--NUM_MINIBATCHES", default=8,  type=int, help="index for choosing start and goal locations"
    )
    parser.add_argument(
        "--LAMBDA_REACH", default=0.0,  type=float, help="lambda for lagrangian multiplier"
    )
    parser.add_argument(
        "--THRESHOLD_CPPO", default=0.0,  type=float, help="cost threshold"
    )
    parser.add_argument(
        "--K_P", default=1.0,  type=float, help="K_P for CPPO"
    )
    parser.add_argument(
        "--GAMMA_ENERGY", default=1.0,  type=float, help="contraction rate for energy"
    )
    parser.add_argument(
        "--GAMMA_REACH_INIT", default=0.999,  type=float, help="initial contraction rate for reach function"
    )
    parser.add_argument(
        "--GAMMA_REACH_FINAL", default=0.99999,  type=float, help="final contraction rate for reach function "
    )
    parser.add_argument(
        "--GAE_LAMBDA", default=0.95,  type=float, help="GAE lambda"
    )
    parser.add_argument(
        "--CLIP_EPS", default=0.2, type=float, help="clip threshold for PPO updating"
    )
    parser.add_argument(
        "--ENT_COEF", default=0.01,  type=float, help="entropy coefficient"
    )
    parser.add_argument(
        "--VF_COEF", default=0.5,  type=float, help="value function coefficient"
    )
    parser.add_argument(
        "--VP_COEF", default=0.05,  type=float, help="value probability function coefficient"
    )
    parser.add_argument(
        "--MAX_GRAD_NORM", default=0.5, type=float, help="max gradient norm"
    )
    parser.add_argument(
        "--ACTIVATION", default='tanh', type=str, help="activation function"
    )
    parser.add_argument(
        "--CUDA_USE", default='0', type=str, help="visible cuda device"
    )
    parser.add_argument(
        "--ANNEAL_LR", action='store_true', default=False, help="whether using annealing in PPO updating"
    )
    parser.add_argument(
        "--ANNEAL_ENT", action='store_true', default=False, help="whether using annealing in entropy in PPO updating"
    )
    parser.add_argument(
        "--TEST_MODE", action='store_true', default=False, help="whether using deterministic hopper"
    )
    parser.add_argument(
        "--DISCRETE", action='store_true', default=False, help="whether the environment is discrete"
    )
    parser.add_argument(
        "--FIX_LAMBDA", action='store_true', default=False, help="whether the environment use "
                                                                 "a fixed lagrangian multiplier"
    )
    parser.add_argument(
        "--LOG_BARRIER", action='store_true', default=False, help="whether to wrap aug-lag with log barrier"
    )
    parser.add_argument(
        "--LOG_BARRIER_MU", default=5., type=float, help="log barrier mu, \psi = -log(-(relu(x)-1)/mu"
    )
    parser.add_argument(
        '--DEC_INIT_TYPE', default='toinput_goal', type=str, help='initialization type for the decomposed rollouts (determines coupling)'
    )
    parser.add_argument(
        '--SAVE_MILESTONE', action='store_true', default=False, help='whether to save subset of model checkpoints for comparison'
    )
    parser.add_argument(
        '--MILESTONES', default=[0, 1, 2, 3, 5, 10, 20, 50, 100, 150, 300], type=list, help='timesteps to save subset of model checkpoints'
    )
    parser.add_argument(
        '--USE_WANDB', action='store_true', default=True, help='whether to use wandb for logging'
    )
    parser.add_argument(
        '--WANDB_GROUP', default='default', type=str, help='WandB group name'
    )
    parser.add_argument(
        '--WANDB_PROJECT', default="", type=str, help='WandB project name (overwrites auto naming)'
    )
    parser.add_argument(
        '--NOISE_PERCENT', default=0., type=float, help='Percentage of noise added to the dynamics (from I/Cs)'
    )
    parser.add_argument(
        '--NOISY_STATES', default=[0, 1, 2, 6, 7, 8], type=list, help='Indices of states to add noise to'
    )
    parser.add_argument(
        '--REACH_AVOID_LOOP_GAP', default=1, type=int, help='Gap size for reach-avoid loop'
    )

    args = parser.parse_args(args) if args is not None else parser.parse_args()

    return args