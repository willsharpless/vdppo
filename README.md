# rraa-rl
ucsd phdawgs

### to setup

```conda env create -f make_env_jax_sbx.yml``` (jax + SBX)

```pip install -e .```

### to run, eg.
```
./rraa_rl/EFPPO/src/script/run_hopper_reachreach.sh
./rraa_rl/EFPPO/src/script/run_hopper_reachalwaysavoid.sh
```
this learns the `Hopper` RR and RAA problems with the `RR-PPO.py` and `RAA-PPO.py` algos respectively.

### structure (5/1/25):

```.
├── README.md
├── make_env_jax_sbx.yml
├── eval
├── model
├── render
└── rraa_rl
    └── EFPPO

    ...
    old SBX/3 stuff:
    ...
    ├── algos
    │   └── ... SB3 + custom
    ├── custom_envs
    ├── exp
    │   └── ... where models are saved
    ├── tests
    │   └── ... basic code tests not roll-out test
    ├── train
    │   ├── test.py
    │   └── train.py
    └── utils
```

### deps
 - python = 3.10
 - torch >= 2.2
 - dm_control >= 1.0
 - gym = 0.26.1
 - mujoco >= 2.0
 - stable-baselines3 = 2.6 (subclassing custom algs)
 - wandb (optional)
 - oswins code needs more jax stuff, use make_env_jax.
