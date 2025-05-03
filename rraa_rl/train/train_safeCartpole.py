
import os
import numpy as np
import rraa_rl.custom_envs
from rraa_rl.algos import algorithms # SB3 + custom
from rraa_rl.utils import *
from tqdm import tqdm
import wandb

def main():

    ## Init experiment
    CONFIG = Config()
        
    ## Safe Cartpole
    CONFIG.ENV='safeCartpole'
    CONFIG.TASK='swingup'
    CONFIG.MODEL_STEPS=200_000
    CONFIG.SUB_STEPS=10_000
    CONFIG.NAME='test_R'
    CONFIG.ALG='PPO_RRAA'
    CONFIG.BELLMAN='R'
    CONFIG.SEED=0
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
    
    ## Define Training Buffer for Rollout Scores
    train_rewards, train_goals, train_penalties = [], [], []
    # TODO: train_buffer = TrainBuffer(CONFIG)

    ## Training loop with intermittent saving
    print(f"\n\nRRAA-RL\n\n Learning {CONFIG.ENV}_{CONFIG.TASK} with {CONFIG.ALG}-{CONFIG.BELLMAN} ...\n  writing to {CONFIG.CURR_EXP_PATH} \n")
    for step in tqdm(range(0, CONFIG.MODEL_STEPS, CONFIG.SUB_STEPS), desc=""):
        tqdm.write(f"Training Step: {step} | Avg Reward: {sum(train_rewards)/len(train_rewards) if train_rewards else 0:0.3f}")

        ## Learn & Save Model
        if CONFIG.WANDB:
            model.learn(total_timesteps=CONFIG.SUB_STEPS, callback=WandbCallback())
        else:
            model.learn(total_timesteps=CONFIG.SUB_STEPS)
        model.save(os.path.join(CONFIG.CURR_MODEL_PATH, f'model_{step + CONFIG.SUB_STEPS}'))

        ## Guage model rewards
        # TODO: train_buffer.roll_out(env, model, render=CONFIG.RENDER)
        obs, _ = env.reset(seed=CONFIG.SEED)
        rollout_obs = []
        rollout_rewards, rollout_goals, rollout_penalties = [], [], []
        for _ in tqdm(range(CONFIG.SAMPLE_HORIZON), desc="Evaluating rewards", leave=False):
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            
            rollout_obs.append(obs)
            rollout_rewards.append(reward)
            
            # if CONFIG.BELLMAN == 'R':
            #     goal = env.get_goal(obs)
            #     rollout_goals.append(goal)

            # elif CONFIG.BELLMAN == 'RA':
            #     goal = env.get_goal(obs)
            #     penalty = env.get_penalty(obs)
            #     rollout_goals.append(goal)
            #     rollout_penalties.append(penalty)

            if terminated or truncated:
                obs, _ = env.reset(seed=CONFIG.SEED)

        if CONFIG.BELLMAN == 'normal':
            reward = rollout_rewards[-1]
        # elif CONFIG.BELLMAN == 'R':
        #     reward = np.maximum(rollout_rewards)
        # elif CONFIG.BELLMAN == 'RA':
            # reward = np.maximum(rollout_rewards)
        train_rewards.append(reward)

        ## Save & Plot Reward
        save_reward_plot(train_rewards, step, CONFIG)
        # TODO: save_rewards(train_buffer)

        ## Also save model as WandB artifact
        if CONFIG.WANDB:
            artifact = wandb.Artifact(f"model_checkpoint_{step}", type="model")
            artifact.add_file(os.path.join(CONFIG.CURR_MODEL_PATH, f"model_{step}.zip"))
            wandb.log_artifact(artifact)

if __name__ == "__main__":
    main()