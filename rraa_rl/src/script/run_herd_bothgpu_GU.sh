#!/bin/bash

SEED=20

# Start first two jobs on GPU 0 in a screen session
screen -dmS gpu0_jobs bash -c "
conda activate jax_sbx
export CUDA_VISIBLE_DEVICES=0

python ./rraa_rl/src/script/valdec/herding.py \
--SEED=${SEED} \
--STEP_SCAN=40 \
--ENT_COEF=0.005 \
--TOTAL_TIMESTEPS=1_000_000_000 \
--ANNEAL_ENT

python ./rraa_rl/src/script/valdec/herding.py \
--SEED=${SEED} \
--STEP_SCAN=20 \
--TOTAL_TIMESTEPS=1_000_000_000 \
--ENT_COEF=0.0001 \
--ANNEAL_ENT

python ./rraa_rl/src/script/valdec/herding.py \
--SEED=${SEED} \
--STEP_SCAN=40 \
--ENT_COEF=0.01 \
--TOTAL_TIMESTEPS=1_000_000_000 \
--ANNEAL_ENT
"

# Start last two jobs on GPU 1 in a screen session
screen -dmS gpu1_jobs bash -c "
conda activate jax_sbx
export CUDA_VISIBLE_DEVICES=1

python ./rraa_rl/src/script/valdec/herding_center.py \
--SEED=${SEED} \
--STEP_SCAN=40 \
--ENT_COEF=0.005 \
--TOTAL_TIMESTEPS=1_000_000_000 \
--ANNEAL_ENT

python ./rraa_rl/src/script/valdec/herding_center.py \
--SEED=${SEED} \
--STEP_SCAN=20 \
--TOTAL_TIMESTEPS=1_000_000_000 \
--ENT_COEF=0.0001 \
--ANNEAL_ENT

python ./rraa_rl/src/script/valdec/herding_center.py \
--SEED=${SEED} \
--STEP_SCAN=40 \
--ENT_COEF=0.01 \
--TOTAL_TIMESTEPS=1_000_000_000 \
--ANNEAL_ENT
"

echo "Jobs started in screen sessions 'gpu0_jobs' and 'gpu1_jobs'"
echo "Use 'screen -r gpu0_jobs' or 'screen -r gpu1_jobs' to attach to the sessions"
