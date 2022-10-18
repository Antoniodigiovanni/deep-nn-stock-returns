#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='ayo0bvzp'
export NNI_SYS_DIR='/home/antoniodg98/nni-experiments/ayo0bvzp/trials/UYe3B'
export NNI_TRIAL_JOB_ID='UYe3B'
export NNI_OUTPUT_DIR='/home/antoniodg98/nni-experiments/ayo0bvzp/trials/UYe3B'
export NNI_TRIAL_SEQ_ID='5'
export NNI_CODE_DIR='/home/antoniodg98/thesis/src/tuning'
cd $NNI_CODE_DIR
eval 'python hp_tuning.py --saveDirName NN9LRtuning --expandingTuning' 1>/home/antoniodg98/nni-experiments/ayo0bvzp/trials/UYe3B/stdout 2>/home/antoniodg98/nni-experiments/ayo0bvzp/trials/UYe3B/stderr
echo $? `date +%s%3N` >'/home/antoniodg98/nni-experiments/ayo0bvzp/trials/UYe3B/.nni/state'