python ./rraa_rl/EFPPO/src/rl/evaluation.py \
--EXP_NAME=PendulumConstraint \
--DIR=pendulum_constraint \
--DIR_MODEL=checkpoint_624 \
--NUM_ENVS=40 \
--NUM_STEPS=2000 \
--ACTIVATION=tanh \
--TEST_MODE=True