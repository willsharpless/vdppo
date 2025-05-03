from stable_baselines3 import PPO, SAC, A2C, DDPG
from rraa_rl.algos.custom_algos.ppo_rraa.PPO_RRAA import PPO_RRAA
# , SAC_rraa  # TODO!

algorithms = {
    "PPO": PPO,
    "SAC": SAC,
    "A2C": A2C,
    "DDPG": DDPG,
    "PPO_RRAA": PPO_RRAA,
    # "SAC_rraa": SAC_rraa,
}