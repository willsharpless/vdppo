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
├── make_env.yml
└── rraa_rl
    ├── tests
    │   ├── test_train_dmc.py
    │   ├── test_train_dmc_wandb.py
    │   └── test_train_gym.py
    ├── train.py
    └── utils
        ├── custom_dmc2gym.py
        └── wandb_callback.py
```