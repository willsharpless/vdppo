#!/bin/bash

alg=${1:-vd}  # Use first argument, default to 'vd'
GPU_ID=${2:-0}
env_name=${3:-delivery}

screen -dmS gpu${GPU_ID}_job bash -c "
export CUDA_VISIBLE_DEVICES=${GPU_ID}

python scripts/train.py '${alg}' --env_name '${env_name}' --seed 1 --name '${env_name}_${alg}_seed1'
python scripts/train.py '${alg}' --env_name '${env_name}' --seed 2 --name '${env_name}_${alg}_seed2'
"

echo "jobs have begun on GPU ${GPU_ID}, in screen gpu${GPU_ID}_job"
# python scripts/train.py '"${alg}"' --env_name '"${env_name}"' --seed 0 --name '"${env_name}"'_'"${alg}"'_seed0