# rraa-rl

### to setup
```conda env create -f make_env.yml```

### deps
 - python = 3.10
 - torch >= 2.2
 - dm_control >= 1.0
 - gym = 0.26.1
 - mujoco >= 2.0
 - stable-baselines3 = 2.6 (subclassing custom algs)
 - wandb (optional)

### to do
 - policy test / rendering scripts
 - custom RRAA-RL SAC/PPO algos
 - more custom envs

### structure (5/1/25):

```.
├── README.md
├── make_env.yml
└── rraa_rl
    ├── algos
    │   └── ... SB3 + custom
    ├── custom_envs
    ├── exp
    │   └── ... where models are saved
    ├── tests
    │   └── ... basic code tests
    ├── train
    │   └── train.py <- general training script
    └── utils
        └── ... config, dmc2gym, wandb stuff etc.
```

### to run, eg.
```python -m rraa_rl.train.train```