
wandb_group="_RESET_TEST"

for seed in {21..22}; do
    python ./rraa_rl/src/rl/RAA-PPO.py \
    --EXP_NAME=HalfCheetahReachAlwaysAvoid \
    --DEC_INIT_TYPE=standard \
    --SAVE_MILESTONE \
    --DIR=RESET_halfcheetah_raa_NOreset_seed${seed} \
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
    --CLIP_EPS=0.2 \
    --ENT_COEF=0.005 \
    --VF_COEF=2.0 \
    --MAX_GRAD_NORM=0.5 \
    --ACTIVATION=tanh \
    --CUDA_USE=0 \
    --ANNEAL_LR \
    --ANNEAL_ENT \
    --NAME=RESET_halfcheetah_raa_NOreset_seed${seed} \
    --SEED=${seed} \
    --WANDB_GROUP=${wandb_group}

    python ./rraa_rl/src/rl/RAA-PPO.py \
    --EXP_NAME=HalfCheetahReachAlwaysAvoid \
    --DEC_INIT_TYPE=toinput \
    --SAVE_MILESTONE \
    --DIR=RESET_halfcheetah_raa_reset_random_seed${seed} \
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
    --CLIP_EPS=0.2 \
    --ENT_COEF=0.005 \
    --VF_COEF=2.0 \
    --MAX_GRAD_NORM=0.5 \
    --ACTIVATION=tanh \
    --CUDA_USE=0 \
    --ANNEAL_LR \
    --ANNEAL_ENT \
    --NAME=RESET_halfcheetah_raa_reset_random_seed${seed} \
    --SEED=${seed} \
    --WANDB_GROUP=${wandb_group}

    python ./rraa_rl/src/rl/RAA-PPO.py \
    --EXP_NAME=HalfCheetahReachAlwaysAvoid \
    --DEC_INIT_TYPE=toinput_goal \
    --SAVE_MILESTONE \
    --DIR=RESET_halfcheetah_raa_reset_goal_seed${seed} \
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
    --CLIP_EPS=0.2 \
    --ENT_COEF=0.005 \
    --VF_COEF=2.0 \
    --MAX_GRAD_NORM=0.5 \
    --ACTIVATION=tanh \
    --CUDA_USE=0 \
    --ANNEAL_LR \
    --ANNEAL_ENT \
    --NAME=RESET_halfcheetah_raa_reset_goal_seed${seed} \
    --SEED=${seed} \
    --WANDB_GROUP=${wandb_group}

    python ./rraa_rl/src/rl/RAA-PPO.py \
    --EXP_NAME=HalfCheetahReachAlwaysAvoid \
    --DEC_INIT_TYPE=toinput_safegoal \
    --SAVE_MILESTONE \
    --DIR=RESET_halfcheetah_raa_reset_safegoal_seed${seed} \
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
    --CLIP_EPS=0.2 \
    --ENT_COEF=0.005 \
    --VF_COEF=2.0 \
    --MAX_GRAD_NORM=0.5 \
    --ACTIVATION=tanh \
    --CUDA_USE=0 \
    --ANNEAL_LR \
    --ANNEAL_ENT \
    --NAME=RESET_halfcheetah_raa_reset_safegoal_seed${seed} \
    --SEED=${seed} \
    --WANDB_GROUP=${wandb_group}

    python ./rraa_rl/src/rl/RAA-PPO.py \
    --EXP_NAME=HalfCheetahReachAlwaysAvoid \
    --DEC_INIT_TYPE=toinput_safegoal_nearcrash \
    --SAVE_MILESTONE \
    --DIR=RESET_halfcheetah_raa_reset_safegoal_nearcrash_seed${seed} \
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
    --CLIP_EPS=0.2 \
    --ENT_COEF=0.005 \
    --VF_COEF=2.0 \
    --MAX_GRAD_NORM=0.5 \
    --ACTIVATION=tanh \
    --CUDA_USE=0 \
    --ANNEAL_LR \
    --ANNEAL_ENT \
    --NAME=RESET_halfcheetah_raa_reset_safegoal_nearcrash_seed${seed} \
    --SEED=${seed} \
    --WANDB_GROUP=${wandb_group}
    
done