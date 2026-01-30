#!/bin/bash

screen -dmS gpu1_job bash -c "
export CUDA_VISIBLE_DEVICES=1
python scripts/train.py lcrl --env_name ablation --seed 0 --name lcrl_spc2_ag2_seed0 --n-spec 2 --n-agent 2
python scripts/train.py lcrl --env_name ablation --seed 0 --name lcrl_spc3_ag3_seed0 --n-spec 3 --n-agent 3
python scripts/train.py lcrl --env_name ablation --seed 0 --name lcrl_spc4_ag4_seed0 --n-spec 4 --n-agent 4
python scripts/train.py lcrl --env_name ablation --seed 0 --name lcrl_spc5_ag5_seed0 --n-spec 5 --n-agent 5
"

screen -dmS gpu2_job bash -c "
export CUDA_VISIBLE_DEVICES=2
python scripts/train.py lcrl --env_name ablation --seed 1 --name lcrl_spc4_ag1_seed1 --n-spec 4 --n-agent 1
python scripts/train.py lcrl --env_name ablation --seed 1 --name lcrl_spc5_ag1_seed1 --n-spec 5 --n-agent 1
python scripts/train.py lcrl --env_name ablation_depth --seed 1 --name lcrl_spc5_ag1_seed1 --n-spec 5 --n-agent 1
"

echo "jobs have begun on GPUs 1, 2, in screens gpu1_job, gpu2_job"