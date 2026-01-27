#!/bin/bash

n_spec=5
n_agent=1

GPU_ID_1=3
screen -dmS gpu${GPU_ID_1}_job bash -c '
conda activate jaxrl
export CUDA_VISIBLE_DEVICES='"${GPU_ID_1}"'

# iterate thru n_specs and seeds
for seed in $(seq 0 2); do
  for spec in $(seq 1 '"${n_spec}"'); do

    python scripts/train.py vd \
    --env_name ablation \
    --name ablation_vd_spc${spec}_ag'"${n_agent}"'_seed${seed} \
    --n-spec ${spec} \
    --n-agent '"${n_agent}"' \
    --seed ${seed}

  done
done
'

# GPU_ID_2=3
# screen -dmS gpu${GPU_ID_2}_job bash -c "
# conda activate jaxrl
# export CUDA_VISIBLE_DEVICES=${GPU_ID_2}

# # iterate thru n_specs and seeds
# for seed in $(seq 0 2); do
#   for spec in $(seq 1 ${n_spec}); do

#     python scripts/train.py vd \
#     --env_name dblint_ablation \
#     --name ablation_vd_spc${spec}_ag${n_agent}_seed${seed} \
#     --n-spec ${n_spec} \
#     --n-agent ${n_agent} \
#     --seed ${seed}

#   done
# done
# "

echo "jobs have begun on GPUs ${GPU_ID_1} and ${GPU_ID_2}"