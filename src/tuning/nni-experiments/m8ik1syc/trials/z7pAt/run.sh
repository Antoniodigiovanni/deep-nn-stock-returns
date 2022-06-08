#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='m8ik1syc'
export NNI_SYS_DIR='/home/antonio/thesis/src/tuning/nni-experiments/m8ik1syc/trials/z7pAt'
export NNI_TRIAL_JOB_ID='z7pAt'
export NNI_OUTPUT_DIR='/home/antonio/thesis/src/tuning/nni-experiments/m8ik1syc/trials/z7pAt'
export NNI_TRIAL_SEQ_ID='1'
export NNI_CODE_DIR='/home/antonio/thesis/src'
cd $NNI_CODE_DIR
eval python3 tune.py 1>/home/antonio/thesis/src/tuning/nni-experiments/m8ik1syc/trials/z7pAt/stdout 2>/home/antonio/thesis/src/tuning/nni-experiments/m8ik1syc/trials/z7pAt/stderr
echo $? `date +%s%3N` >'/home/antonio/thesis/src/tuning/nni-experiments/m8ik1syc/trials/z7pAt/.nni/state'