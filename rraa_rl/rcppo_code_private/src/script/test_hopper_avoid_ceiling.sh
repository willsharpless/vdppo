python ./src/rl/evaluation.py \
--EXP_NAME=HopperAvoidCeiling \
--DIR=hopper_avoid_ceiling_1 \
--DIR_MODEL=checkpoint_487 \
--NUM_ENVS=40 \
--NUM_STEPS=800 \
--ACTIVATION=tanh \
--TEST_MODE=True