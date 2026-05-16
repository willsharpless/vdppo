#!/bin/bash

n_spec=5
n_agent=1
alg=${1:-vdppo}  # Use first argument, default to 'vdppo'
GPU_ID=${2:-0}

seed=${3:-0}

screen -dmS gpu${GPU_ID}_job bash -c '
conda activate jaxrlnew
export CUDA_VISIBLE_DEVICES='"${GPU_ID}"'

list_of_remaining_seeds=(1 2)
list_of_remaining_specs=(6 7 8 9 10)

# iterate thru n_specs and seeds
for seed in ${list_of_remaining_seeds[@]}; do
  for spec in "${list_of_remaining_specs[@]}"; do

    python scripts/train.py '"${alg}"' \
    --env_name ablation_depth \
    --name ablation_'"${alg}"'_spc${spec}_ag'"${n_agent}"'_seed${seed} \
    --n-spec ${spec} \
    --n-agent '"${n_agent}"' \
    --seed ${seed}

  done
done
'

echo "jobs have begun on GPUs ${GPU_ID}, in screen gpu${GPU_ID}_job"
