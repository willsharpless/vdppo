#!/bin/bash

max_n_agent=5

alg=${1:-vd}  # Use first argument, default to 'vd'
GPU_ID=${2:-0}

screen -dmS gpu${GPU_ID}_job bash -c '
conda activate jaxrlnew
export CUDA_VISIBLE_DEVICES='"${GPU_ID}"'

# iterate thru n_specs and seeds
for seed in $(seq 0 2); do
  for ag in $(seq 1 '"${max_n_agent}"'); do

    python scripts/train.py '"${alg}"' \
    --env_name ablation \
    --name ablation_'"${alg}"'_spc${ag}_ag${ag}_seed${seed} \
    --n-spec ${ag} \
    --n-agent ${ag} \
    --seed ${seed}

  done
done
'

echo "jobs have begun on GPUs ${GPU_ID}, in screen gpu${GPU_ID}_job"