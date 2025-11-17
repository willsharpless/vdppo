#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

SEEDS=(21 22)
for SEED in ${SEEDS[@]}; do

    # #################################### RAA-PPO RUN ##########################################

    NOISE_PERCENTS=(0 5 10 20)
    for NOISE_PERCENT in ${NOISE_PERCENTS[@]}; do
        python ./rraa_rl/src/rl/RAA-PPO.py \
        --EXP_NAME=HalfCheetahReachAlwaysAvoid \
        --DIR=NOISY_halfcheetah_raa_rg_noisy_nz${NOISE_PERCENT}_sd${SEED} \
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
        --DEC_INIT_TYPE=toinput_goal \
        --NOISE_PERCENT=${NOISE_PERCENT} \
        --WANDB_PROJECT=RAA-HalfCheetah-noise \
        --NAME=NOISY_halfcheetah_raa_rg_noisy_nz${NOISE_PERCENT}_sd${SEED} \
        --SEED=${SEED}
    done

    # #################################### SPARSE BASELINE RUN ##########################################

    NOISE_PERCENTS=(0 5 10 20)
    for NOISE_PERCENT in ${NOISE_PERCENTS[@]}; do
        python ./rraa_rl/src/rl/baselines/MORL_PPO_RAA.py \
        --EXP_NAME=HalfCheetahReachAlwaysAvoidBaseline_Sparse \
        --DIR=NOISY_halfcheetah_raa_sparse_noisy_nz${NOISE_PERCENT}_sd${SEED} \
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
        --LAMBDA_REACH=0.0 \
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
        --WANDB_PROJECT=RAA-HalfCheetah-noise \
        --NAME=NOISY_halfcheetah_raa_sparse_noisy_nz${NOISE_PERCENT}_sd${SEED} \
        --SEED=${SEED}
    done

    #################################### CPPO BASELINE RUN ##########################################

    NOISE_PERCENTS=(1 5 10 20)
    for NOISE_PERCENT in ${NOISE_PERCENTS[@]}; do
        python ./rraa_rl/src/rl/baselines/CPPO_RAA.py \
        --EXP_NAME=HalfCheetahReachAlwaysAvoid_CPPO \
        --DIR=NOISY_halfcheetah_raa_cppo_kp100_noisy_nz${NOISE_PERCENT}_sd${SEED} \
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
        --LAMBDA_REACH=0.1 \
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
        --WANDB_PROJECT=RAA-HalfCheetah-noise \
        --NAME=NOISY_halfcheetah_raa_cppo_kp100_noisy_nz${NOISE_PERCENT}_sd${SEED} \
        --SEED=${SEED}
    done

    # #################################### MORL BASELINE RUN ##########################################

    NOISE_PERCENTS=(0 5 10 20)
    for NOISE_PERCENT in ${NOISE_PERCENTS[@]}; do
        python ./rraa_rl/src/rl/baselines/MORL_PPO_RAA.py \
        --EXP_NAME=HalfCheetahReachAlwaysAvoidBaseline_MORL \
        --DIR=NOISY_halfcheetah_raa_morl_noisy_nz${NOISE_PERCENT}_sd${SEED} \
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
        --LAMBDA_REACH=0.0 \
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
        --WANDB_PROJECT=RAA-HalfCheetah-noise \
        --NAME=NOISY_halfcheetah_raa_morl_noisy_nz${NOISE_PERCENT}_sd${SEED} \
        --SEED=${SEED}
    done
done
