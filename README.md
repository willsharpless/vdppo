# rraa-rl

### setup

```
conda env create -f make_env.yml  # jax env
pip install -e .
```
### run, eg.
```
./rraa_rl/src/script/run_hopper_reachreach.sh
./rraa_rl/src/script/run_hopper_reachalwaysavoid.sh
```
this solves the `Hopper` RR and RAA solutions with the `RR-PPO.py` and `RAA-PPO.py` algs respectively.

### todo
- retest a bit (cleaned a bit)
- give windfield a go with RAA
- write RRAA-PPO.py
- one representation?
- retry humanoid, certify dones/gae

### structure (10/1/25):

```.
├── README.md
├── make_env.yml
├── eval
├── model
├── render
└── rraa_rl
    ├── mj_models
    └── src
        ├── env
        ├── model
        ├── script
        └── rl
            ├── baselines
            ├── eval
            ├── utls
            ├── RAA-PPO.py
            └── RR-PPO.py
```
