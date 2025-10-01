# rraa-rl
ucsd phdawgs

### to setup

```conda env create -f make_env.yml``` (jax based)

```pip install -e .```

### to run, eg.
```
./rraa_rl/src/script/run_hopper_reachreach.sh
./rraa_rl/src/script/run_hopper_reachalwaysavoid.sh
```
this learns the `Hopper` RR and RAA problems with the `RR-PPO.py` and `RAA-PPO.py` algos respectively.

### structure (5/1/25):

```.
├── README.md
├── make_env.yml
├── eval
├── model
├── render
└── rraa_rl
    └── EFPPO
```
