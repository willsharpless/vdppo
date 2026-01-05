#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# iterate thru seed 21, 22, 23
for SEED in 21 22 23; do

    python ./rraa_rl/src/script/valdec/multipoint_GU_test.py \
    --REACH_AVOID_LOOP_GAP=1
    --SEED=${SEED}

    python ./rraa_rl/src/script/valdec/multipoint_GU_test.py \
    --REACH_AVOID_LOOP_GAP=3
    --SEED=${SEED}

    python ./rraa_rl/src/script/valdec/multipoint_GU_test.py \
    --REACH_AVOID_LOOP_GAP=10
    --SEED=${SEED}

    python ./rraa_rl/src/script/valdec/multipoint_GU_test.py \
    --REACH_AVOID_LOOP_GAP=31
    --SEED=${SEED}

    python ./rraa_rl/src/script/valdec/multipoint_GU_test.py \
    --REACH_AVOID_LOOP_GAP=100
    --SEED=${SEED}

done
