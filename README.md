# vdppo

This repo contains the jax code for Value Decomposition PPO (VDPPO) from [Bellman Value Decomposition for Task Logic in Safe Optimal Control](https://willsharpless.github.io/valdec-site/) \[RSS '26\]. This package is hevaily dependent on [valtr](https://willsharpless.github.io/valtr/) for generating the decomposed Value graph (DVG).

## basic summary

This PPO variant decomposes the Value for a given temporal-logic specification into ``atomic'' Bellman equations (BEs), and learns an actor and critic for all policies and Values concurrently. Namely, for a given spec, VDPPO synthesizes the DVG (with [valtr](https://willsharpless.github.io/valtr/)) and learns an embedding (one-hot) for the graph. 

<img src="vdppo_graphic.png" alt="VDPPO graphic" style="border:10px solid white;">

This involves distributing rollouts across all nodes and using the corresponding BE for each node to compute the batched target. To compute the BE for each node, the necessary child/dependent Values (based on DVG) are bootstrapped with the current embedded approximation.

### setup
```
conda env create -f make_env.yml  # jax env
pip install -e .
```

### run, eg.
```
python scripts/train.py ALG --name test30k --debug --env-name ENV --n-train-steps 30000
```
where `ALG` is one of `[vd, lcrl, mppi, ppo]`, and  `ENV` is one of `[herdos, delivery, deliveryrealv2, ablation]`.

### testing

If you'd like to make your own env, see `herd_base.py`. Here, dynamics/step, reset, obs and basic predicates are defined. Note, there is also a wrapper for each env, eg. `herd_os.py`.

See `get_env.py` for env registration.x

If you'd like to mess with the training, see the training agent classes, eg. `vd_mappo.py`.
