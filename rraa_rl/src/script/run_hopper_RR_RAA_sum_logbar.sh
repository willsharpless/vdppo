python ./rraa_rl/src/rl/baselines/CPPO_RAA.py \
--EXP_NAME=HopperReachAlwaysAvoid_CPPO \
--DIR=hopper_raa_logbar_sum_mu100_scale \
--LR=3e-4 \
--NUM_ENVS=128 \
--NUM_STEPS=400 \
--TOTAL_TIMESTEPS=50_000_000 \
--STEP_SCAN=4 \
--UPDATE_EPOCHS=10 \
--NUM_MINIBATCHES=32 \
--GAMMA_ENERGY=0.99 \
--GAMMA_REACH_INIT=0.995 \
--GAMMA_REACH_FINAL=0.9995 \
--GAE_LAMBDA=0.95 \
--LAMBDA_REACH=1.0 \
--LOG_BARRIER_MU=100. \
--LOG_BARRIER \
--FIX_LAMBDA \
--K_P=1.0 \
--THRESHOLD_CPPO=0. \
--CLIP_EPS=0.2 \
--ENT_COEF=0.0001 \
--VF_COEF=2.0 \
--MAX_GRAD_NORM=0.5 \
--ACTIVATION=tanh \
--CUDA_USE=1,2,3 \
--ANNEAL_LR \
--ANNEAL_ENT \
--NAME=hopper_raa_logbar_sum_mu100_scale

# python ./rraa_rl/src/rl/baselines/CPPO_RR.py \
# --EXP_NAME=HopperReachReach_sum_CPPO \
# --DIR=hopper_rr_logbar_scalefix_sum_mu100 \
# --LR=3e-4 \
# --NUM_ENVS=128 \
# --NUM_STEPS=400 \
# --TOTAL_TIMESTEPS=50_000_000 \
# --STEP_SCAN=4 \
# --UPDATE_EPOCHS=10 \
# --NUM_MINIBATCHES=32 \
# --GAMMA_ENERGY=0.99 \
# --GAMMA_REACH_INIT=0.995 \
# --GAMMA_REACH_FINAL=0.9995 \
# --GAE_LAMBDA=0.95 \
# --LAMBDA_REACH=1.0 \
# --LOG_BARRIER_MU=100. \
# --LOG_BARRIER \
# --FIX_LAMBDA \
# --K_P=1.0 \
# --THRESHOLD_CPPO=0. \
# --CLIP_EPS=0.2 \
# --ENT_COEF=0.0001 \
# --VF_COEF=2.0 \
# --MAX_GRAD_NORM=0.5 \
# --ACTIVATION=tanh \
# --CUDA_USE=1,2,3 \
# --ANNEAL_LR \
# --ANNEAL_ENT \
# --NAME=hopper_rr_logbar_scalefix_sum_mu100