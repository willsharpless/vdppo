#!/bin/bash

max_n_agent=5

alg=${1:-vd}  # Use first argument, default to 'vd'
GPU_ID_1=${2:-0}

screen -dmS gpu${GPU_ID_1}_job bash -c '
conda activate jaxrl
export CUDA_VISIBLE_DEVICES='"${GPU_ID_1}"'

# iterate thru n_specs and seeds
for seed in $(seq 0 2); do
  for ag in $(seq 1 '"${max_n_agent}"'); do

    python scripts/train.py '"${alg}"' \
    --env_name ablation \
    --name ablation_vd_spc${ag}_ag${ag}_seed${seed} \
    --n-spec ${ag} \
    --n-agent ${ag} \
    --seed ${seed}

  done
done
'

# GPU_ID_2=FIXME
# screen -dmS gpu${GPU_ID_2}_job bash -c '
# conda activate jaxrl
# export CUDA_VISIBLE_DEVICES='"${GPU_ID_2}"'

# # iterate thru n_specs and seeds
# for seed in $(seq 0 2); do
#   for ag in $(seq 1 '"${max_n_agent}"'); do

#     python scripts/train.py vd \
#     --env_name ablation \
#     --name ablation_vd_spc1_ag${ag}_seed${seed} \
#     --n-spec 1 \
#     --n-agent ${ag} \
#     --seed ${seed}

#   done
# done
# '

echo "jobs have begun on GPUs ${GPU_ID_1}"