
import os
import sys
import rraa_rl.custom_envs
from rraa_rl.algos import algorithms # SB3 + custom
from rraa_rl.utils import *
from tqdm import tqdm
import wandb
# os.environ["MUJOCO_GL"] = "egl"

def main():

    if len(sys.argv) < 2:
        print("Usage: python -m rraa_rl.train.test <path_to_model>")
        sys.exit(1)

    ## Init experiment
    CONFIG = Config()
    model_path = sys.argv[1]
    CONFIG.load(model_path)
    RENDER = False
    if len(sys.argv) > 2 and sys.argv[2] == '-r':
        RENDER = True

    ## Define environment
    env = DMCWrapper(domain_name=CONFIG.ENV, task_name=CONFIG.TASK, seed=CONFIG.SEED)
    env.reset(seed=CONFIG.SEED)

    ## Define algorithm
    model_class = algorithms[CONFIG.ALG]
    if CONFIG.ALG in ['PPO', 'SAC', 'A2C', 'DDPG']:
        model = model_class(CONFIG.POLICY_TYPE, env, seed=CONFIG.SEED)
    else:
        model = model_class(CONFIG.POLICY_TYPE, env, seed=CONFIG.SEED, problem_type=CONFIG.PROBLEM_TYPE)
    
    ## Define training buffer for rollout scores
    train_buffer = TrainBuffer(CONFIG)

    ## Training loop with checkpoints
    print(f"\n\nRRAA-RL\n\n Testing {model_path} ...\n  writing to {CONFIG.CURR_EXP_PATH} \n")

    ## Guage model rewards
    train_buffer.model_rollout(env, model, render=RENDER)
    train_buffer.save(added_name='test')

    ## Also save model as WandB artifact
    if CONFIG.WANDB:
        artifact = wandb.Artifact(f"roll_out_{model_path}", type="mp4")
        artifact.add_file(os.path.join(CONFIG.CURR_EXP_PATH, f"roll_out.mp4"))
        wandb.log_artifact(artifact)

if __name__ == "__main__":
    main()