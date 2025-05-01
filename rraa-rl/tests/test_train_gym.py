# test_train_gym_env.py

from stable_baselines3 import PPO
import gym

def main():
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)

    obs, _ = env.reset()
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

if __name__ == "__main__":
    main()
