
# C-PPO
python ./rraa_rl/EFPPO/src/rl/CPPO_RR.py \
--EXP_NAME=PointReachReach_CPPO \
--DIR=BASELINE_point_rr_cppo_v0_faster_ts100m \
--LR=3e-4 \
--NUM_ENVS=128 \
--NUM_STEPS=400 \
--TOTAL_TIMESTEPS=100_000_000 \
--STEP_SCAN=4 \
--UPDATE_EPOCHS=10 \
--NUM_MINIBATCHES=32 \
--GAMMA_ENERGY=1.0 \
--GAMMA_REACH_INIT=0.995 \
--GAMMA_REACH_FINAL=0.9995 \
--GAE_LAMBDA=0.95 \
--LAMBDA_REACH=0.1 \
--K_P=1.0 \
--THRESHOLD_CPPO=0. \
--CLIP_EPS=0.2 \
--ENT_COEF=0.005 \
--VF_COEF=2.0 \
--MAX_GRAD_NORM=0.5 \
--ACTIVATION=tanh \
--CUDA_USE=0 \
--ANNEAL_LR \
--ANNEAL_ENT \
--NAME=BASELINE_point_rr_cppo_v0_faster_ts100m

# PPO
python ./rraa_rl/EFPPO/src/rl/CPPO_RR.py \
--EXP_NAME=PointReachReach_CPPO \
--DIR=BASELINE_point_rr_ppo_v0_faster_ts100m \
--LR=3e-4 \
--NUM_ENVS=128 \
--NUM_STEPS=400 \
--TOTAL_TIMESTEPS=100_000_000 \
--STEP_SCAN=4 \
--UPDATE_EPOCHS=10 \
--NUM_MINIBATCHES=32 \
--GAMMA_ENERGY=1.0 \
--GAMMA_REACH_INIT=0.995 \
--GAMMA_REACH_FINAL=0.9995 \
--GAE_LAMBDA=0.95 \
--LAMBDA_REACH=0. \
--FIX_LAMBDA \
--K_P=1.0 \
--THRESHOLD_CPPO=0. \
--CLIP_EPS=0.2 \
--ENT_COEF=0.005 \
--VF_COEF=2.0 \
--MAX_GRAD_NORM=0.5 \
--ACTIVATION=tanh \
--CUDA_USE=0 \
--ANNEAL_LR \
--ANNEAL_ENT \
--NAME=BASELINE_point_rr_ppo_v0_faster_ts100m

# PPO-LAG with lambda 0.01
python ./rraa_rl/EFPPO/src/rl/CPPO_RR.py \
--EXP_NAME=PointReachReach_CPPO \
--DIR=BASELINE_point_rr_ppolag_lam0p01_v0_faster_ts100m \
--LR=3e-4 \
--NUM_ENVS=128 \
--NUM_STEPS=400 \
--TOTAL_TIMESTEPS=100_000_000 \
--STEP_SCAN=4 \
--UPDATE_EPOCHS=10 \
--NUM_MINIBATCHES=32 \
--GAMMA_ENERGY=1.0 \
--GAMMA_REACH_INIT=0.995 \
--GAMMA_REACH_FINAL=0.9995 \
--GAE_LAMBDA=0.95 \
--LAMBDA_REACH=0.01 \
--FIX_LAMBDA \
--K_P=1.0 \
--THRESHOLD_CPPO=0. \
--CLIP_EPS=0.2 \
--ENT_COEF=0.005 \
--VF_COEF=2.0 \
--MAX_GRAD_NORM=0.5 \
--ACTIVATION=tanh \
--CUDA_USE=0 \
--ANNEAL_LR \
--ANNEAL_ENT \
--NAME=BASELINE_point_rr_ppolag_lam0p01_v0_faster_ts100m

# ./rraa_rl/EFPPO/src/script/run_point_RR_cppo_ppo_ppolag.sh ; ./rraa_rl/EFPPO/src/script/run_point_RR_MORL_Sparse.sh ; ./rraa_rl/EFPPO/src/script/run_point_RR_rcppo_respo.sh ; ./rraa_rl/EFPPO/src/script/run_point_RAA_cppo_ppo_ppolag.sh ; ./rraa_rl/EFPPO/src/script/run_point_RAA_MORL_Sparse.sh ; ./rraa_rl/EFPPO/src/script/run_point_RAA_rcppo_respo.sh