#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='edco8i1p'
export NNI_SYS_DIR='/home/antonio/thesis/src/tuning/nni-experiments/edco8i1p/trials/EzI5f'
export NNI_TRIAL_JOB_ID='EzI5f'
export NNI_OUTPUT_DIR='/home/antonio/thesis/src/tuning/nni-experiments/edco8i1p/trials/EzI5f'
export NNI_TRIAL_SEQ_ID='0'
export NNI_CODE_DIR='/home/antonio/thesis/src'
cd $NNI_CODE_DIR
eval python3 tune.py 1>/home/antonio/thesis/src/tuning/nni-experiments/edco8i1p/trials/EzI5f/stdout 2>/home/antonio/thesis/src/tuning/nni-experiments/edco8i1p/trials/EzI5f/stderr
echo $? `date +%s%3N` >'/home/antonio/thesis/src/tuning/nni-experiments/edco8i1p/trials/EzI5f/.nni/state'