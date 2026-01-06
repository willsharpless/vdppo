#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

fixed_vel=0.05

for SEED in 21 22 23; do
    for REACH3_DYNAMIC_PRED_TYPE in "const" "conrand" "circ" "obsweave"; do

        python ./rraa_rl/src/script/valdec/multipoint_dynamic_GU_test.py \
        --SEED=${SEED} \
        --FIXED_VELOCITY=${fixed_vel} \
        --REACH3_DYNAMIC_PRED_TYPE=${REACH3_DYNAMIC_PRED_TYPE} \
        --DEBUG_JUST_RAA

        python ./rraa_rl/src/script/valdec/multipoint_dynamic_GU_test.py \
        --SEED=${SEED} \
        --FIXED_VELOCITY=${fixed_vel} \
        --REACH3_DYNAMIC_PRED_TYPE=${REACH3_DYNAMIC_PRED_TYPE}

    done
done
