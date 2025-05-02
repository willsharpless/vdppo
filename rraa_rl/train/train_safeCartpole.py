
import os
import rraa_rl.custom_envs
from rraa_rl.algos import algorithms # SB3 + custom
from rraa_rl.utils import *
from tqdm import tqdm
import wandb

def main():

    ## Init experiment
    CONFIG = Config()
    CONFIG.parse_args()
    
    ## Safe Cartpole
    CONFIG.ENV='safeCartpole'
    CONFIG.TASK='swingup'
    CONFIG.MODEL_STEPS=200_000
    CONFIG.SUB_STEPS=2_000
    CONFIG.NAME='test'
    CONFIG.save_config()

    if CONFIG.WANDB:
        wandb.init(name=CONFIG.NAME, project=CONFIG.WB_PROJECT, entity=CONFIG.WB_ENTITY, group=CONFIG.WB_GROUP, config=CONFIG)

    ## Define environment
    env = DMCWrapper(domain_name=CONFIG.ENV, task_name=CONFIG.TASK, seed=CONFIG.SEED)
    env.reset(seed=CONFIG.SEED)

    ## Define algorithm
    if CONFIG.ALG not in algorithms:
        raise ValueError(f"Algorithm {CONFIG.ALG} not recognized. Available algorithms: {list(algorithms.keys())}")
    model_class = algorithms[CONFIG.ALG]
    model = model_class(CONFIG.POLICY_TYPE, env, seed=CONFIG.SEED)
    
    ## Training loop with intermittent saving
    print(f"\n\nRRAA-RL\n\n Learning {CONFIG.ENV}_{CONFIG.TASK} with {CONFIG.ALG} ...\n  writing to {CONFIG.CURR_EXP_PATH} \n")
    rewards = []
    for step in tqdm(range(0, CONFIG.MODEL_STEPS, CONFIG.SUB_STEPS), desc=""):
        tqdm.write(f"Training Step: {step} | Avg Reward: {sum(rewards)/len(rewards) if rewards else 0:0.3f}")

        ## Learn & Save Model
        if CONFIG.WANDB:
            model.learn(total_timesteps=CONFIG.SUB_STEPS, callback=WandbCallback())
        else:
            model.learn(total_timesteps=CONFIG.SUB_STEPS)
        model.save(os.path.join(CONFIG.CURR_MODEL_PATH, f'model_{step + CONFIG.SUB_STEPS}'))

        ## Guage model rewards
        obs, _ = env.reset(seed=CONFIG.SEED)
        for _ in tqdm(range(CONFIG.SAMPLE_HORIZON), desc="Evaluating rewards", leave=False):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset(seed=CONFIG.SEED)

        ## Save & Plot Reward
        rewards.append(reward)
        save_reward_plot(rewards, step, CONFIG.SUB_STEPS, CONFIG.CURR_EXP_PATH)

        ## Also save model as WandB artifact
        if CONFIG.WANDB:
            artifact = wandb.Artifact(f"model_checkpoint_{step}", type="model")
            artifact.add_file(os.path.join(CONFIG.CURR_MODEL_PATH, f"model_{step}.zip"))
            wandb.log_artifact(artifact)

if __name__ == "__main__":
    main()