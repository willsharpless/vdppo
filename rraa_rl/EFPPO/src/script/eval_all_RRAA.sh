
$eval_dir="eval_all_080725"

python ./rraa_rl/EFPPO/src/rl/evaluation_all_RAA_hopper.py --EVAL_DIR $eval_dir
python ./rraa_rl/EFPPO/src/rl/evaluation_all_RAA_halfcheetah.py --EVAL_DIR $eval_dir
python ./rraa_rl/EFPPO/src/rl/evaluation_all_RAA_F16.py --EVAL_DIR $eval_dir
python ./rraa_rl/EFPPO/src/rl/evaluation_all_RAA_safetygym.py --EVAL_DIR $eval_dir

python ./rraa_rl/EFPPO/src/rl/evaluation_all_RR_hopper.py --EVAL_DIR $eval_dir
python ./rraa_rl/EFPPO/src/rl/evaluation_all_RR_halfcheetah.py --EVAL_DIR $eval_dir
python ./rraa_rl/EFPPO/src/rl/evaluation_all_RR_F16.py --EVAL_DIR $eval_dir
python ./rraa_rl/EFPPO/src/rl/evaluation_all_RR_safetygym.py --EVAL_DIR $eval_dir