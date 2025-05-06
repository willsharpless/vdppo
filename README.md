# rraa-rl
by the ucsd phdawgs

### to setup
```conda env create -f make_env.yml``` (pytorch + SB3 OLD)

```conda env create -f make_env_jax_sbx.yml``` (jax + SBX)

```pip install -e .```

### to do 05/05:
 - We now have oswin's envs & baselines code ("EFPPO" dir)
    - hopper_avoid_ceiling works (~40 min, on sdsc)
    - F16_avoid works (~35 min, on sdsc)
    - half_cheetah_avoid works (~30 min, on ws2)
    - windfield_avoid works (~60 min/section=4hrs total, on ws2)
    - envs: need to clone and remove energy minimization / state augmentation
    - envs: requires slight for general RAA & RR problem (must have l(x) & g(x))

 - W: Going to start w/ hopper, 
    - run w SBX PPO to see if this works
    - then add walls on ends (RAA)

 - custom RRAA-RL algos
    - SBX vs. oswin-modifications:
      - testing SBX forms first, which workd w prev. coded PPO-RRAA
      - PPO: written but correct GAE update?
      - if fails, making mod of oswin PPO (EC-EFPPO)
    - general: simultaneous solving of decomposed problem 
    (for coupled policy learning, probably important)

 - what baselines? need to make,
    - PPO: w simple sum max{min l, max g}
    - PPO: special augmented problem -> constrained MDP (see oswin code for eg.)

 - TARGET ENVS
    - Hopper RR
    - F16 RAA (target near obstacle wall)

 - OLD
    - safeCartpole doesnt work (ditching anyway)

### to run, eg.
Oswin code
```
./rraa_rl/EFPPO/src/script/run_hopper_avoid_ceiling.sh
```
this calls the alg `EC-EFPPO`.

OLD
```
python -m rraa_rl.train.train
python -m rraa_rl.train.train_safeCartpole
```
this calls with whatever model was used.

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
    ├── rcppo_code_private (oswin's)
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
