#!/bin/bash

n_spec=5
n_agent=1

alg=${1:-vd}  # Use first argument, default to 'vd'
GPU_ID_1=${2:-0}

screen -dmS gpu${GPU_ID_1}_job bash -c '
conda activate jaxrl
export CUDA_VISIBLE_DEVICES='"${GPU_ID_1}"'

list_of_seeds=(2)
list_of_remaining_specs=(2 4 5)

# iterate thru n_specs and seeds
for seed in "${list_of_seeds[@]}"; do
  for spec in "${list_of_remaining_specs[@]}"; do

    python scripts/train.py '"${alg}"' \
    --env_name ablation \
    --name ablation_vd_spc${spec}_ag'"${n_agent}"'_seed${seed} \
    --n-spec ${spec} \
    --n-agent '"${n_agent}"' \
    --seed ${seed}

  done
done
'

echo "jobs have begun on GPUs ${GPU_ID_1}"


# for seed in $(seq 1 2); do
#   for spec in $(seq 1 '"${n_spec}"'); do

#     python scripts/train.py vd \
#     --env_name ablation \
#     --name ablation_vd_spc${spec}_ag'"${n_agent}"'_seed${seed} \
#     --n-spec ${spec} \
#     --n-agent '"${n_agent}"' \
#     --seed ${seed}

#   done
# done