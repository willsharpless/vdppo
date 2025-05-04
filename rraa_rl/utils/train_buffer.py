import numpy as np
from tqdm import tqdm
import pickle
import os
import imageio
from PIL import Image

class TrainBuffer:
    def __init__(self, CONFIG):
        self.CONFIG = CONFIG
        # NOTE, these are the _last_ wrt the sampled trajectory
        self.rewards = []
        self.goals = [] 
        self.penalties = []
        self.obs = []
        self.render_path = os.path.join(CONFIG.CURR_EXP_PATH, 'roll_out.mp4')

    def model_rollout(self, env, model, render=False):
        obs, _ = env.reset(seed=self.CONFIG.SEED)
        rollout_obs = []
        rollout_rewards, rollout_goals, rollout_penalties = [], [], []
        
        frames = []
        if render:
            frame = env.render()
            frames.append(Image.fromarray(frame))

        for _ in tqdm(range(self.CONFIG.SAMPLE_HORIZON), desc="Evaluating rewards", leave=False):
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            rollout_obs.append(obs)
            rollout_rewards.append(reward)

            # TODO: env needs goals
            # if self.CONFIG.PROBLEM_TYPE in ['R', 'RA']:
            #     goal = env.get_goal(obs)
            #     rollout_goals.append(goal)

            #     if self.CONFIG.PROBLEM_TYPE == 'RA':
            #         penalty = env.get_penalty(obs)
            #         rollout_penalties.append(penalty)

            if render:
                frame = env.render()
                frames.append(Image.fromarray(frame))

            if terminated or truncated:
                obs, _ = env.reset(seed=self.CONFIG.SEED)

        # Save the full trajectory
        self.obs.append(rollout_obs[-1])
        self.rewards.append(rollout_rewards[-1])

        # TODO: env needs goals
        # if self.CONFIG.PROBLEM_TYPE in ['R', 'RA']:
        #     self.goals.append(rollout_goals)
        # if self.CONFIG.PROBLEM_TYPE == 'RA':
        #     self.penalties.append(rollout_penalties)

        if render and frames:
            imageio.mimsave(self.render_path, frames, fps=30)  # 30 fps

    def save(self, added_name='train'):
        with open(os.path.join(self.CONFIG.CURR_EXP_PATH, added_name + '_rewards'), "wb") as f:
            pickle.dump({"rewards": self.rewards}, f)

        if self.CONFIG.PROBLEM_TYPE in ['R', 'RA']:
            with open(os.path.join(self.CONFIG.CURR_EXP_PATH, added_name + '_goals'), "wb") as f:
                pickle.dump({"goals": self.goals}, f)

        if self.CONFIG.PROBLEM_TYPE in ['A', 'RA', 'RAA']:
            with open(os.path.join(self.CONFIG.CURR_EXP_PATH, added_name + '_penalties'), "wb") as f:
                pickle.dump({"penalties": self.penalties}, f)
        
