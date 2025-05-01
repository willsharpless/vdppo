# rraa-rl

# deps structure
 - python=3.10?
 - dm_control = ... (for custom rraa mods)
 - gy = 0.26.1 (paired w dmc-gym gym wrapper)
 - mujoco > 2.
 - stable-baselines3 (subclassing custom algs)

 # (this leaves) to make
 - custom envs
 - custom RRAA-RL SAC/PPO algos
 - training and sim scripts

 ### structure (4/30/25):

``` .
├── README.md
├── __init__.py
├── make_env.yml
└── rraa_rl
    ├── __init__.py
    ├── tests
    │   ├── test_train_dmc.py
    │   ├── test_train_dmc_wandb.py
    │   └── test_train_gym.py
    ├── train.py
    └── utils
        ├── __init__.py
        ├── custom_dmc2gym.py
        └── wandb_callback.py```