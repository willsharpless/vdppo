#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# #################################### SPARSE BASELINE RUN ##########################################

# loop over noise percents
for NOISE_PERCENT in 0 5 10 20; do
    python ./rraa_rl/src/rl/baselines/MORL_PPO_RR.py \
    --EXP_NAME=HalfCheetahReachReachBaseline_Sparse \
    --DIR=NOISY_halfcheetah_rr_sparse_noisy_nz${NOISE_PERCENT} \
    --LR=3e-4 \
    --NUM_ENVS=128 \
    --NUM_STEPS=400 \
    --TOTAL_TIMESTEPS=150_000_000 \
    --STEP_SCAN=4 \
    --UPDATE_EPOCHS=10 \
    --NUM_MINIBATCHES=32 \
    --GAMMA_ENERGY=1.0 \
    --GAMMA_REACH_INIT=0.995 \
    --GAMMA_REACH_FINAL=0.9995 \
    --GAE_LAMBDA=0.95 \
    --LAMBDA_REACH=0. \
    --FIX_LAMBDA \
    --K_P=100. \
    --THRESHOLD_CPPO=0. \
    --CLIP_EPS=0.2 \
    --ENT_COEF=0.005 \
    --VF_COEF=2.0 \
    --MAX_GRAD_NORM=0.5 \
    --ACTIVATION=tanh \
    --CUDA_USE=1 \
    --ANNEAL_LR \
    --ANNEAL_ENT \
    --NOISE_PERCENT=${NOISE_PERCENT} \
    --WANDB_PROJECT=RR-HalfCheetah-noise \
    --NAME=NOISY_halfcheetah_rr_sparse_noisy_nz${NOISE_PERCENT}
done

#################################### LOGBAR BASELINE RUN ##########################################

# loop over noise percents
for NOISE_PERCENT in 0 5 10 20; do
    python ./rraa_rl/src/rl/baselines/CPPO_RR.py \
    --EXP_NAME=HalfCheetahReachReach_CPPO \
    --DIR=NOISY_halfcheetah_rr_logbar_noisy_nz${NOISE_PERCENT} \
    --LR=3e-4 \
    --NUM_ENVS=128 \
    --NUM_STEPS=400 \
    --TOTAL_TIMESTEPS=150_000_000 \
    --STEP_SCAN=4 \
    --UPDATE_EPOCHS=10 \
    --NUM_MINIBATCHES=32 \
    --GAMMA_ENERGY=1.0 \
    --GAMMA_REACH_INIT=0.995 \
    --GAMMA_REACH_FINAL=0.9995 \
    --GAE_LAMBDA=0.95 \
    --LAMBDA_REACH=1. \
    --LOG_BARRIER_MU=100. \
    --LOG_BARRIER \
    --FIX_LAMBDA \
    --K_P=1. \
    --THRESHOLD_CPPO=0. \
    --CLIP_EPS=0.2 \
    --ENT_COEF=0.005 \
    --VF_COEF=2.0 \
    --MAX_GRAD_NORM=0.5 \
    --ACTIVATION=tanh \
    --CUDA_USE=1 \
    --ANNEAL_LR \
    --ANNEAL_ENT \
    --NOISE_PERCENT=${NOISE_PERCENT} \
    --WANDB_PROJECT=RR-HalfCheetah-noise \
    --NAME=NOISY_halfcheetah_rr_logbar_noisy_nz${NOISE_PERCENT}
done
