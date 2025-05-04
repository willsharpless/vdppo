import numpy as np
from tqdm import tqdm

class TrainBuffer:
    def __init__(self, config):
        self.config = config
        self.rewards = []
        self.goals = [] 
        self.penalties = []
        self.obs = []
        # NOTE, these are the _last_ wrt the sampled trajectory

    def model_rollout(self, env, model, render=False):
        obs, _ = env.reset(seed=self.config.SEED)
        rollout_obs = []
        rollout_rewards, rollout_goals, rollout_penalties = [], [], []

        for _ in tqdm(range(self.config.SAMPLE_HORIZON), desc="Evaluating rewards", leave=False):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            rollout_obs.append(obs)
            rollout_rewards.append(reward)

            # TODO: env needs goals
            # if self.config.BELLMAN in ['R', 'RA']:
            #     goal = env.get_goal(obs)
            #     rollout_goals.append(goal)

            #     if self.config.BELLMAN == 'RA':
            #         penalty = env.get_penalty(obs)
            #         rollout_penalties.append(penalty)

            if terminated or truncated:
                obs, _ = env.reset(seed=self.config.SEED)

        # Save the full trajectory
        self.obs.append(rollout_obs[-1])
        self.rewards.append(rollout_rewards[-1])

        # TODO: env needs goals
        # if self.config.BELLMAN in ['R', 'RA']:
        #     self.goals.append(rollout_goals)
        # if self.config.BELLMAN == 'RA':
        #     self.penalties.append(rollout_penalties)
