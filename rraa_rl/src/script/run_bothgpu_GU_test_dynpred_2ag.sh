#!/bin/bash

fixed_vel=0.05
SEED=20
REACH3_DYNAMIC_PRED_TYPE="obsweave"

# Start first two jobs on GPU 0 in a screen session
screen -dmS gpu0_jobs bash -c "
export CUDA_VISIBLE_DEVICES=0

python ./rraa_rl/src/script/valdec/multipoint_dynamic_GU_test.py \
--SEED=${SEED} \
--REACH3_DYNAMIC_PRED_TYPE=${REACH3_DYNAMIC_PRED_TYPE} \
--N_AGENTS=2 \
--STEP_SCAN=40 \
--ENT_COEF=0.0001 \
--ANNEAL_ENT

python ./rraa_rl/src/script/valdec/multipoint_dynamic_GU_test.py \
--SEED=${SEED} \
--REACH3_DYNAMIC_PRED_TYPE=${REACH3_DYNAMIC_PRED_TYPE} \
--N_AGENTS=2 \
--STEP_SCAN=40 \
--ENT_COEF=0.005 \
--ANNEAL_ENT
"

# Start last two jobs on GPU 1 in a screen session
screen -dmS gpu1_jobs bash -c "
export CUDA_VISIBLE_DEVICES=1

python ./rraa_rl/src/script/valdec/multipoint_dynamic_GU_test.py \
--SEED=${SEED} \
--REACH3_DYNAMIC_PRED_TYPE=${REACH3_DYNAMIC_PRED_TYPE} \
--N_AGENTS=2 \
--STEP_SCAN=40 \
--ENT_COEF=0.0001 \

python ./rraa_rl/src/script/valdec/multipoint_dynamic_GU_test.py \
--SEED=${SEED} \
--REACH3_DYNAMIC_PRED_TYPE=${REACH3_DYNAMIC_PRED_TYPE} \
--N_AGENTS=2 \
--STEP_SCAN=40 \
--ENT_COEF=0.005 \
"

echo "Jobs started in screen sessions 'gpu0_jobs' and 'gpu1_jobs'"
echo "Use 'screen -r gpu0_jobs' or 'screen -r gpu1_jobs' to attach to the sessions"
