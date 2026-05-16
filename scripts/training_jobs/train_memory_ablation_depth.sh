#!/bin/bash

ENV_NAME=${1:-ablation_depth}
VD_GPU_ID=${2:-0}
LCRL_GPU_ID=${3:-1}

screen -dmS gpu${VD_GPU_ID}_memory_vdppo bash -c '
conda activate jaxrlnew
export CUDA_VISIBLE_DEVICES='"${VD_GPU_ID}"'

XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform \
python scripts/paper/plot_ablation_memory.py \
  --env-name '"${ENV_NAME}"' \
  --n-specs "16, 32, 64, 128, 256" \
  --n-seeds 3 \
  --algs vdppo \
  --n-train-steps 10
'

screen -dmS gpu${LCRL_GPU_ID}_memory_lcrl bash -c '
conda activate jaxrlnew
export CUDA_VISIBLE_DEVICES='"${LCRL_GPU_ID}"'

XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform \
python scripts/paper/plot_ablation_memory.py \
  --env-name '"${ENV_NAME}"' \
  --n-specs "16, 32, 64, 128, 256" \
  --n-seeds 3 \
  --algs lcrl \
  --n-train-steps 10
'

echo "memory jobs for ${ENV_NAME} have begun on GPUs ${VD_GPU_ID} and ${LCRL_GPU_ID}, in screens gpu${VD_GPU_ID}_memory_vdppo and gpu${LCRL_GPU_ID}_memory_lcrl"
