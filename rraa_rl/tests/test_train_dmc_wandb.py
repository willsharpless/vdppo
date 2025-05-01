# test_train_dm_control_env.py
import wandb
from stable_baselines3 import PPO
from rraa_rl.utils.custom_dmc2gym import DMCWrapper
from rraa_rl.utils.wandb_callback import WandbCallback

def main():
    wandb.init(project="init_tests", entity="braat_brrt", group="rl_tests", name="dmc_cartpole_ppo", config={"algo": "PPO", "env": "CartPole-v1"})

    env = DMCWrapper(domain_name="cartpole", task_name="balance")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000, callback=WandbCallback())

    obs, _ = env.reset()
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

if __name__ == "__main__":
    main()