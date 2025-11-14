# python ./rraa_rl/src/rl/RAA-PPO.py \
# --EXP_NAME=PointReachAlwaysAvoid \
# --DIR=point_raa_noisy_nz0 \
# --LR=3e-4 \
# --NUM_ENVS=128 \
# --NUM_STEPS=400 \
# --TOTAL_TIMESTEPS=50_000_000 \
# --STEP_SCAN=4 \
# --UPDATE_EPOCHS=10 \
# --NUM_MINIBATCHES=32 \
# --GAMMA_ENERGY=1.0 \
# --GAMMA_REACH_INIT=0.995 \
# --GAMMA_REACH_FINAL=0.9995 \
# --GAE_LAMBDA=0.95 \
# --CLIP_EPS=0.2 \
# --ENT_COEF=0.005 \
# --VF_COEF=2.0 \
# --MAX_GRAD_NORM=0.5 \
# --ACTIVATION=tanh \
# --CUDA_USE=0 \
# --ANNEAL_LR \
# --ANNEAL_ENT \
# --NAME=point_raa_noisy_nz0

python ./rraa_rl/src/rl/RAA-PPO.py \
--EXP_NAME=PointReachAlwaysAvoid \
--DIR=point_raa_noisy_nz0p1 \
--LR=3e-4 \
--NUM_ENVS=128 \
--NUM_STEPS=400 \
--TOTAL_TIMESTEPS=50_000_000 \
--STEP_SCAN=4 \
--UPDATE_EPOCHS=10 \
--NUM_MINIBATCHES=32 \
--GAMMA_ENERGY=1.0 \
--GAMMA_REACH_INIT=0.995 \
--GAMMA_REACH_FINAL=0.9995 \
--GAE_LAMBDA=0.95 \
--CLIP_EPS=0.2 \
--ENT_COEF=0.005 \
--VF_COEF=2.0 \
--MAX_GRAD_NORM=0.5 \
--ACTIVATION=tanh \
--CUDA_USE=0 \
--ANNEAL_LR \
--ANNEAL_ENT \
--NOISE_PERCENT=0.1 \
--NAME=point_raa_noisy_nz0p1

python ./rraa_rl/src/rl/RAA-PPO.py \
--EXP_NAME=PointReachAlwaysAvoid \
--DIR=point_raa_noisy_nz1 \
--LR=3e-4 \
--NUM_ENVS=128 \
--NUM_STEPS=400 \
--TOTAL_TIMESTEPS=50_000_000 \
--STEP_SCAN=4 \
--UPDATE_EPOCHS=10 \
--NUM_MINIBATCHES=32 \
--GAMMA_ENERGY=1.0 \
--GAMMA_REACH_INIT=0.995 \
--GAMMA_REACH_FINAL=0.9995 \
--GAE_LAMBDA=0.95 \
--CLIP_EPS=0.2 \
--ENT_COEF=0.005 \
--VF_COEF=2.0 \
--MAX_GRAD_NORM=0.5 \
--ACTIVATION=tanh \
--CUDA_USE=0 \
--ANNEAL_LR \
--ANNEAL_ENT \
--NOISE_PERCENT=1.0 \
--NAME=point_raa_noisy_nz1

python ./rraa_rl/src/rl/RAA-PPO.py \
--EXP_NAME=PointReachAlwaysAvoid \
--DIR=point_raa_noisy_nz10 \
--LR=3e-4 \
--NUM_ENVS=128 \
--NUM_STEPS=400 \
--TOTAL_TIMESTEPS=50_000_000 \
--STEP_SCAN=4 \
--UPDATE_EPOCHS=10 \
--NUM_MINIBATCHES=32 \
--GAMMA_ENERGY=1.0 \
--GAMMA_REACH_INIT=0.995 \
--GAMMA_REACH_FINAL=0.9995 \
--GAE_LAMBDA=0.95 \
--CLIP_EPS=0.2 \
--ENT_COEF=0.005 \
--VF_COEF=2.0 \
--MAX_GRAD_NORM=0.5 \
--ACTIVATION=tanh \
--CUDA_USE=0 \
--ANNEAL_LR \
--ANNEAL_ENT \
--NOISE_PERCENT=10.0 \
--NAME=point_raa_noisy_nz10