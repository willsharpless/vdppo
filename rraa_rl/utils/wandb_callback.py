import wandb
from stable_baselines3.common.callbacks import BaseCallback

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info:
                    wandb.log({
                        "1-Score/reward": info["episode"]["r"],
                        "2-General/length": info["episode"]["l"],
                        "2-General/timesteps": self.num_timesteps
                    })
                if "c" in info["episode"].keys():
                    wandb.log({
                        "1-Score/penalty": info["episode"]["c"],
                    })
        return True
    
# To use, import, init and add to model.learn,
# wandb.init(project="my_rl_project", config={"algo": "PPO", "env": "CartPole-v1"})
# model.learn(total_timesteps=100_000, callback=WandbCallback())

# (and dont forget to authorize)