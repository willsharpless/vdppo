
import os
import rraa_rl.custom_envs
from rraa_rl.algos import algorithms # SB3 + custom
from rraa_rl.utils import *
import wandb

def main():

    ## Init experiment
    CONFIG = Config()
    CONFIG.parse_args()
    # script fixed args go here (overwrite parsed/default)
    CONFIG.save_config()

    if CONFIG.WANDB:
        wandb.init(name=CONFIG.NAME, project=CONFIG.WB_PROJECT, entity=CONFIG.WB_ENTITY, group=CONFIG.WB_GROUP, config=CONFIG)

    ## Define environment
    env = DMCWrapper(domain_name=CONFIG.ENV, task_name=CONFIG.TASK)
    env.seed(CONFIG.SEED)

    ## Define algorithm
    if CONFIG.ALG not in algorithms:
        raise ValueError(f"Algorithm {CONFIG.ALG} not recognized. Available algorithms: {list(algorithms.keys())}")
    model_class = algorithms[CONFIG.ALG]
    model = model_class(CONFIG.POLICY_TYPE, env, seed=CONFIG.SEED) #FIXME verbose no, tqmd someday
    
    # Training loop with intermittent saving
    rewards = []
    for timestep in range(0, CONFIG.MODEL_STEPS, CONFIG.SUB_STEPS):

        ## Learn
        if CONFIG.WANDB:
            model.learn(total_timesteps=CONFIG.SUB_STEPS, callback=WandbCallback())
        else:
            model.learn(total_timesteps=CONFIG.SUB_STEPS)
        model.save(os.path.join(CONFIG.CURR_MODEL_PATH, f'model_{timestep + CONFIG.SUB_STEPS}'))

        ## Guage model rewards
        obs, _ = env.reset()
        for _ in range(CONFIG.SAMPLE_HORIZON):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        rewards.append(reward)

        ## Save reward plot
        save_reward_plot(rewards, timestep, CONFIG.CURR_EXP_PATH)

        if CONFIG.WANDB:
            # Log the model checkpoint as a WandB artifact
            artifact = wandb.Artifact(f"model_checkpoint_{timestep}", type="model")
            artifact.add_file(os.path.join(CONFIG.CURR_MODEL_PATH, f"model_{timestep}.zip"))
            wandb.log_artifact(artifact)

if __name__ == "__main__":
    main()