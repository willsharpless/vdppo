import argparse
import os
import pickle
import sys
import shutil

class Config:
    def __init__(self):
        
        self.ENV = "cartpole"
        self.TASK = "balance"
        self.MODEL_STEPS = 10_000 # total grad steps
        self.SUB_STEPS = 2_000 # checkpoint period
        self.SAMPLE_HORIZON = 2000 # roll-out length for guaging rewards
        self.POLICY_TYPE = "MlpPolicy"
        self.ALG = "PPO"
        self.SEED = 0

        self.NAME = "test"
        self.BASE_PATH = "rraa_rl/exp"
        self.WANDB = False
        self.WB_ENTITY = "braat_brrt"
        self.WB_PROJECT = "test"
        self.WB_GROUP = "all"

    def parse_args(self):
        # Use argparse to optionally override defaults with command-line args
        parser = argparse.ArgumentParser(description="Training script for DM Control environments")
        
        parser.add_argument("--ENV", type=str, default=self.ENV, help="Domain name")
        parser.add_argument("--TASK", type=str, default=self.TASK, help="Task name")
        parser.add_argument("-ms", "--MODEL_STEPS", type=int, default=self.MODEL_STEPS, help="Total timesteps for training")
        parser.add_argument("--SUB_STEPS", type=int, default=self.SUB_STEPS, help="Interval to save model checkpoints")
        parser.add_argument("-n", "--NAME", type=str, default=self.NAME, help="Model name")
        parser.add_argument("-alg", "--ALG", type=str, default=self.ALG, help="Algorithm name for stable-baselines/custom")
        parser.add_argument("--POLICY_TYPE", type=str, default=self.POLICY_TYPE, help="Policy type")
        parser.add_argument("-s","--SEED", type=int, default=self.SEED, help="Random seed for reproducibility")
        
        parser.add_argument("-wb", "--WANDB", action="store_true", default=False, help="Log to WandB")
        parser.add_argument("--WB_ENTITY", type=str, default=self.WB_ENTITY, help="WandB entity")
        parser.add_argument("-wbp", "--WB_PROJECT", type=str, default=self.WB_PROJECT, help="WandB project")
        parser.add_argument("-wbg", "--WB_GROUP", type=str, default=self.WB_GROUP, help="WandB group(?)")

        args = parser.parse_args()
        
        # Update configuration with command-line arguments
        self.ENV = args.ENV
        self.TASK = args.TASK
        self.MODEL_STEPS = args.MODEL_STEPS
        self.SUB_STEPS = args.SUB_STEPS
        self.NAME = args.NAME
        self.ALG = args.ALG
        self.POLICY_TYPE = args.POLICY_TYPE
        self.SEED = args.SEED
        self.WANDB = args.WANDB
        self.WB_ENTITY = args.WB_ENTITY
        self.WB_PROJECT = args.WB_PROJECT
        self.WB_GROUP = args.WB_GROUP

        # Make out folders
        self.CURR_EXP_PATH = os.path.join(self.BASE_PATH, self.ENV + '_' + self.TASK, self.ALG, self.NAME)
        self.CURR_MODEL_PATH = os.path.join(self.BASE_PATH, self.ENV + '_' + self.TASK, self.ALG, self.NAME, 'model')
        if os.path.exists(self.CURR_EXP_PATH):
            if input(f"\n{self.CURR_EXP_PATH} exists. Overwrite? (y/n):").lower() != 'y':
                sys.exit("Exiting without overwriting.")  # Exit if user chooses 'n'
            shutil.rmtree(self.CURR_EXP_PATH)
        os.makedirs(self.CURR_EXP_PATH, exist_ok=True)
        os.makedirs(self.CURR_MODEL_PATH, exist_ok=True)       

    def save_config(self):
        with open(os.path.join(self.CURR_EXP_PATH, "config.pkl"), "wb") as f:
            pickle.dump(self, f)  # Save the entire config object as a pickle

