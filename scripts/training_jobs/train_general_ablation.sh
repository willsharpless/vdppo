#!/bin/bash

alg=${1:-vd}  # Use first argument, default to 'vd'
GPU_ID=${2:-0}
env_name=${3:-herdos}
shared_trunk_actor=${4:-True}
shared_trunk_value=${5:-True}

# Determine the correct CLI flag for Cyclopts
if [[ "${shared_trunk_actor,,}" == "true" ]]; then
    actor_shared="--actor-shared-trunk"
    actor_shared_name_tag="asharedTrue"
elif [[ "${shared_trunk_actor,,}" == "false" ]]; then
    actor_shared="--no-actor-shared-trunk"
    actor_shared_name_tag="asharedFalse"
else
    actor_shared=""
    actor_shared_name_tag=""
fi

if [[ "${shared_trunk_value,,}" == "true" ]]; then
    value_shared="--value-shared-trunk"
    value_shared_name_tag="vsharedTrue"
elif [[ "${shared_trunk_value,,}" == "false" ]]; then
    value_shared="--no-value-shared-trunk"
    value_shared_name_tag="vsharedFalse"
else
    value_shared=""
    value_shared_name_tag=""
fi

screen -dmS gpu${GPU_ID}_job bash -c "
conda activate jaxrlnew
export CUDA_VISIBLE_DEVICES=${GPU_ID}

layers=(2 3 5)

for n_layer in \"\${layers[@]}\"; do
    python scripts/train.py '${alg}' --env_name '${env_name}' --seed 0 --name '${env_name}_${alg}_${actor_shared_name_tag}_${value_shared_name_tag}_seed0' --n-layers \${n_layer} ${actor_shared} ${value_shared}
    python scripts/train.py '${alg}' --env_name '${env_name}' --seed 1 --name '${env_name}_${alg}_${actor_shared_name_tag}_${value_shared_name_tag}_seed1' --n-layers \${n_layer} ${actor_shared} ${value_shared}
    python scripts/train.py '${alg}' --env_name '${env_name}' --seed 2 --name '${env_name}_${alg}_${actor_shared_name_tag}_${value_shared_name_tag}_seed2' --n-layers \${n_layer} ${actor_shared} ${value_shared}
done
"

echo "jobs have begun on GPU ${GPU_ID}, in screen gpu${GPU_ID}_job"
