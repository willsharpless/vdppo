#!/bin/bash

GPU_ID=1

screen -dmS gpu1_job bash -c "
conda activate jaxrl
export CUDA_VISIBLE_DEVICES=${GPU_ID}

python scripts/train.py vd --env_name delivery --seed 0 --name delivery_vdppo_seed0
python scripts/train.py vd --env_name delivery --seed 1 --name delivery_vdppo_seed1
python scripts/train.py vd --env_name delivery --seed 2 --name delivery_vdppo_seed2
"

echo "jobs have begun on GPU ${GPU_ID}"