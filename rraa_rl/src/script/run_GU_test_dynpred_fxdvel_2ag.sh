#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

fixed_vel=0.05

# for SEED in 20; do
#     for REACH3_DYNAMIC_PRED_TYPE in "obsweave"; do

#         python ./rraa_rl/src/script/valdec/multipoint_dynamic_GU_test.py \
#         --SEED=${SEED} \
#         --FIXED_VELOCITY=${fixed_vel} \
#         --REACH3_DYNAMIC_PRED_TYPE=${REACH3_DYNAMIC_PRED_TYPE} \
#         --N_AGENTS=2

#         python ./rraa_rl/src/script/valdec/multipoint_dynamic_GU_test.py \
#         --SEED=${SEED} \
#         --FIXED_VELOCITY=${fixed_vel} \
#         --REACH3_DYNAMIC_PRED_TYPE=${REACH3_DYNAMIC_PRED_TYPE} \
#         --DEBUG_JUST_RAA \
#         --N_AGENTS=2

#     done
# done

for SEED in 20; do
    for REACH3_DYNAMIC_PRED_TYPE in "obsweave"; do

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

    done
done
