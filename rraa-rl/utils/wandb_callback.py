from stable_baselines3.common.callbacks import BaseCallback

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Example: log episode reward and loss
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info:
                    wandb.log({
                        "reward": info["episode"]["r"],
                        "length": info["episode"]["l"],
                        "timesteps": self.num_timesteps
                    })
        return True