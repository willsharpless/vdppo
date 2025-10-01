
eval_dir="eval_halfcheetah_resetgoal_model"

# python ./rraa_rl/src/rl/eval/evaluation_all_RAA_hopper.py --EVAL_DIR $eval_dir
python ./rraa_rl/src/rl/eval/evaluation_all_RAA_halfcheetah.py --EVAL_DIR $eval_dir
# python ./rraa_rl/src/rl/eval/evaluation_all_RAA_F16.py --EVAL_DIR $eval_dir
# python ./rraa_rl/src/rl/eval/evaluation_all_RAA_safetygym.py --EVAL_DIR $eval_dir

# python ./rraa_rl/src/rl/eval/evaluation_all_RR_hopper.py --EVAL_DIR $eval_dir
# python ./rraa_rl/src/rl/eval/evaluation_all_RR_halfcheetah.py --EVAL_DIR $eval_dir
# python ./rraa_rl/src/rl/eval/evaluation_all_RR_F16.py --EVAL_DIR $eval_dir
# python ./rraa_rl/src/rl/eval/evaluation_all_RR_safetygym.py --EVAL_DIR $eval_dir