#!/bin/bash

n_spec=5
n_agent=1
alg=${1:-vd}  # Use first argument, default to 'vd'
GPU_ID=${2:-0}

screen -dmS gpu${GPU_ID}_job bash -c '
conda activate jaxrlnew
export CUDA_VISIBLE_DEVICES='"${GPU_ID}"'

# iterate thru n_specs and seeds
for seed in $(seq 0 2); do
  for spec in $(seq 1 '"${n_spec}"'); do

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