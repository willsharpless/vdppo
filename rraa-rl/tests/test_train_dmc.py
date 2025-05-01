# test_train_dm_control_env.py

from stable_baselines3 import PPO
from utils.custom_dmc2gym import DMCWrapper  # adjust path if needed

def main():
    env = DMCWrapper(domain_name="cartpole", task_name="balance")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)

    obs, _ = env.reset()
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

if __name__ == "__main__":
    main()