# python ./rraa_rl/src/rl/baselines/rr_baseline_ppo.py \
# --EXP_NAME=PointReachReachDecomposed \
# --DIR=BASELINE_point_rr_decomposed \
# --LR=3e-4 \
# --NUM_ENVS=128 \
# --NUM_STEPS=400 \
# --TOTAL_TIMESTEPS=100_000_000 \
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
# --NAME=BASELINE_point_rr_decomposed

python ./rraa_rl/src/rl/baselines/RA-PPO.py \
--EXP_NAME=PointReachAvoid \
--DIR=BASELINE_point_reachavoid \
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
--NAME=BASELINE_point_reachavoid