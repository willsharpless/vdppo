#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# python ./rraa_rl/src/rl/RAA-PPO.py \
# --EXP_NAME=HalfCheetahReachAlwaysAvoid \
# --DIR=NOISY_halfcheetah_raa_rand_noisy_nz0 \
# --LR=3e-4 \
# --NUM_ENVS=128 \
# --NUM_STEPS=400 \
# --TOTAL_TIMESTEPS=100_000_000 \
# --STEP_SCAN=4 \
# --UPDATE_EPOCHS=10 \
# --NUM_MINIBATCHES=32 \
# --GAMMA_ENERGY=1.0 \
# --GAMMA_REACH_INIT=0.995 \
# --GAMMA_REACH_FINAL=0.9995 \
# --GAE_LAMBDA=0.95 \
# --CLIP_EPS=0.2 \
# --ENT_COEF=0.005 \
# --VF_COEF=2.0 \
# --MAX_GRAD_NORM=0.5 \
# --ACTIVATION=tanh \
# --CUDA_USE=0 \
# --ANNEAL_LR \
# --ANNEAL_ENT \
# --DEC_INIT_TYPE=toinput \
# --NOISE_PERCENT=0 \
# --WANDB_PROJECT=RAA-HalfCheetah-noise \
# --NAME=NOISY_halfcheetah_raa_rand_noisy_nz0 \
# --SEED=20

NOISE_PERCENTS=(5 10 20)
for NOISE_PERCENT in ${NOISE_PERCENTS[@]}; do
    python ./rraa_rl/src/rl/RAA-PPO.py \
    --EXP_NAME=HalfCheetahReachAlwaysAvoid \
    --DIR=NOISY_halfcheetah_raa_rand_noisy_nz${NOISE_PERCENT} \
    --LR=3e-4 \
    --NUM_ENVS=128 \
    --NUM_STEPS=400 \
    --TOTAL_TIMESTEPS=100_000_000 \
    --STEP_SCAN=4 \
    --UPDATE_EPOCHS=10 \
    --NUM_MINIBATCHES=32 \
    --GAMMA_ENERGY=1.0 \
    --GAMMA_REACH_INIT=0.995 \
    --GAMMA_REACH_FINAL=0.9995 \
    --GAE_LAMBDA=0.95 \
    --CLIP_EPS=0.2 \
    --ENT_COEF=0.005 \
    --VF_COEF=2.0 \
    --MAX_GRAD_NORM=0.5 \
    --ACTIVATION=tanh \
    --CUDA_USE=0 \
    --ANNEAL_LR \
    --ANNEAL_ENT \
    --DEC_INIT_TYPE=toinput \
    --NOISE_PERCENT=${NOISE_PERCENT} \
    --WANDB_PROJECT=RAA-HalfCheetah-noise \
    --NAME=NOISY_halfcheetah_raa_rand_noisy_nz${NOISE_PERCENT} \
    --SEED=20
done
