# rraa-rl

### deps
 - python = 3.10
 - torch >= 2.2
 - dm_control >= 1.0
 - gym = 0.26.1
 - mujoco >= 2.0
 - stable-baselines3 = 2.6 (subclassing custom algs)
 - wandb (optional)

### to do
 - custom envs
 - custom RRAA-RL SAC/PPO algos
 - training and sim scripts

### structure (4/30/25):

``` .
├── README.md
├── make_env.yml
└── rraa_rl
    ├── custom_envs
    │   └── safeCartpole
    ├── tests
    │   ├── test_train_dmc.py
    │   ├── test_train_dmc_wandb.py
    │   ├── test_train_gym.py
    │   └── test_train_safeCartpole.py
    ├── train
    │   └── train_safeCartpole.py
    └── utils
        ├── custom_dmc2gym.py
        └── wandb_callback.py
```

### to run, eg.
```python -m rraa_rl.train.train_safeCartpole```