from stable_baselines3 import PPO, SAC, A2C, DDPG
# from rraa_rl.algos.custom_algos import PPO_rraa, SAC_rraa  # TODO!

algorithms = {
    "PPO": PPO,
    "SAC": SAC,
    "A2C": A2C,
    "DDPG": DDPG,
    # "PPO_rraa": PPO_rraa,
    # "SAC_rraa": SAC_rraa,
}