import argparse
import os
import pickle
import sys
import shutil
from rraa_rl.algos import algorithms # SB3 + custom

class Config:
    def __init__(self):
        self.BASE_PATH = "rraa_rl/exp"

        self.NAME = "test"
        self.OVERWRITE = False
        
        self.ENV = "cartpole"
        self.TASK = "balance"
        self.MODEL_STEPS = 10_000 # total grad steps
        self.SUB_STEPS = 2_000 # checkpoint period
        self.SAMPLE_HORIZON = 2000 # roll-out length for guaging rewards
        self.POLICY_TYPE = "MlpPolicy"
        self.ALG = "PPO"
        self.BELLMAN='normal'
        self.SEED = 0

        self.WANDB = False
        self.WB_ENTITY = "braat_brrt"
        self.WB_PROJECT = "test"
        self.WB_GROUP = "all"
        self.RENDER = False

        self.set_path()

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Training script for DM Control environments")

        parser.add_argument("-n", "--NAME", type=str, default=self.NAME, help="Model name")
        parser.add_argument("-y", "--OVERWRITE", action="store_true", default=False, help="Automatically overwrite")

        parser.add_argument("--ENV", type=str, default=self.ENV, help="Domain name")
        parser.add_argument("--TASK", type=str, default=self.TASK, help="Task name")
        parser.add_argument("-ms", "--MODEL_STEPS", type=int, default=self.MODEL_STEPS, help="Total timesteps for training")
        parser.add_argument("--SUB_STEPS", type=int, default=self.SUB_STEPS, help="Interval to save model checkpoints")
        parser.add_argument("-alg", "--ALG", type=str, default=self.ALG, help="Algorithm name for stable-baselines/custom")
        parser.add_argument("--POLICY_TYPE", type=str, default=self.POLICY_TYPE, help="Policy type")
        parser.add_argument("-s","--SEED", type=int, default=self.SEED, help="Random seed for reproducibility")
        parser.add_argument("-b", "--BELLMAN", type=str, default=self.BELLMAN, help="Bellman equation to use")
        
        parser.add_argument("-wb", "--WANDB", action="store_true", default=False, help="Log to WandB")
        parser.add_argument("--WB_ENTITY", type=str, default=self.WB_ENTITY, help="WandB entity")
        parser.add_argument("-wbp", "--WB_PROJECT", type=str, default=self.WB_PROJECT, help="WandB project")
        parser.add_argument("-wbg", "--WB_GROUP", type=str, default=self.WB_GROUP, help="WandB group(?)")
        parser.add_argument("-r", "--RENDER", action="store_true", default=False, help="Render roll-out")

        args = parser.parse_args()
        
        # Update configuration with command-line arguments
        self.ENV = args.ENV
        self.TASK = args.TASK
        self.MODEL_STEPS = args.MODEL_STEPS
        self.SUB_STEPS = args.SUB_STEPS
        self.NAME = args.NAME
        self.OVERWRITE = args.OVERWRITE
        self.ALG = args.ALG
        self.POLICY_TYPE = args.POLICY_TYPE
        self.SEED = args.SEED
        self.BELLMAN = args.BELLMAN if args.ALG in ['PPO_RRAA', 'SAC_RRAA'] else 'normal'
        self.WANDB = args.WANDB
        self.WB_ENTITY = args.WB_ENTITY
        self.WB_PROJECT = args.WB_PROJECT
        self.WB_GROUP = args.WB_GROUP
        self.RENDER = args.RENDER

        self.set_path()

        if self.ALG not in algorithms:
            raise ValueError(f"Algorithm {self.ALG} not recognized. Available algorithms: {list(algorithms.keys())}")

    def set_path(self):
        self.CURR_EXP_PATH = os.path.join(self.BASE_PATH, self.ENV + '_' + self.TASK, self.ALG, self.NAME)
        self.CURR_MODEL_PATH = os.path.join(self.BASE_PATH, self.ENV + '_' + self.TASK, self.ALG, self.NAME, 'model')       

    def save(self):
        if os.path.exists(self.CURR_EXP_PATH):
            if not self.OVERWRITE and input(f"\n{self.CURR_EXP_PATH} exists. Overwrite? (y/n):").lower() != 'y':
                sys.exit("Exiting without overwriting.")  # Exit if user chooses 'n'
            shutil.rmtree(self.CURR_EXP_PATH)

        os.makedirs(self.CURR_EXP_PATH, exist_ok=True)
        os.makedirs(self.CURR_MODEL_PATH, exist_ok=True)

        with open(os.path.join(self.CURR_EXP_PATH, "config.pkl"), "wb") as f:
            pickle.dump(self, f)  # Save the entire config object as a pickle

    @classmethod
    def load(cls, model_path):
        with open(os.path.join(model_path, 'config.pkl'), "rb") as f:
            return pickle.load(f)
        
        self.set_path()