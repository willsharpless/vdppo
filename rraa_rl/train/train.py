
import os
import rraa_rl.custom_envs
from rraa_rl.algos import algorithms # SB3 + custom
from rraa_rl.utils import *
from tqdm import tqdm
import wandb

def main():

    ## Init experiment
    CONFIG = Config()
    # script fixed args go here
    CONFIG.parse_args()
    CONFIG.save_config()

    if CONFIG.WANDB:
        wandb.init(name=CONFIG.NAME, project=CONFIG.WB_PROJECT, entity=CONFIG.WB_ENTITY, group=CONFIG.WB_GROUP, config=CONFIG)

    ## Define environment
    env = DMCWrapper(domain_name=CONFIG.ENV, task_name=CONFIG.TASK, seed=CONFIG.SEED)
    env.reset(seed=CONFIG.SEED)

    ## Define algorithm
    model_class = algorithms[CONFIG.ALG]
    if CONFIG.ALG in ['PPO', 'SAC', 'A2C', 'DDPG']:
        model = model_class(CONFIG.POLICY_TYPE, env, seed=CONFIG.SEED)
    else:
        model = model_class(CONFIG.POLICY_TYPE, env, seed=CONFIG.SEED, bellman=CONFIG.BELLMAN)
    
    ## Define training buffer for rollout scores
    train_buffer = TrainBuffer(CONFIG)

    ## Training loop with checkpoints
    print(f"\n\nRRAA-RL\n\n Learning {CONFIG.ENV}_{CONFIG.TASK} with {CONFIG.ALG}-{CONFIG.BELLMAN} ...\n  writing to {CONFIG.CURR_EXP_PATH} \n")
    for step in tqdm(range(0, CONFIG.MODEL_STEPS, CONFIG.SUB_STEPS), desc=""):
        tqdm.write(f"Training Step: {step} | Avg Reward: {sum(train_buffer.rewards)/len(train_buffer.rewards) if train_buffer.rewards else 0:0.3f}")

        ## Learn & Save Model
        if CONFIG.WANDB:
            model.learn(total_timesteps=CONFIG.SUB_STEPS, callback=WandbCallback())
        else:
            model.learn(total_timesteps=CONFIG.SUB_STEPS)
        model.save(os.path.join(CONFIG.CURR_MODEL_PATH, f'model_{step + CONFIG.SUB_STEPS}'))

        ## Guage model rewards
        train_buffer.model_rollout(env, model, render=False)

        ## Save & Plot Reward
        train_rewards = train_buffer.rewards # FIXME
        save_reward_plot(train_rewards, step, CONFIG)
        # TODO: save_rewards(train_buffer)

        ## Also save model as WandB artifact
        if CONFIG.WANDB:
            artifact = wandb.Artifact(f"model_checkpoint_{step}", type="model")
            artifact.add_file(os.path.join(CONFIG.CURR_MODEL_PATH, f"model_{step}.zip"))
            wandb.log_artifact(artifact)

if __name__ == "__main__":
    main()