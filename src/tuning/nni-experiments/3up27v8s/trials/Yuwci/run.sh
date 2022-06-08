#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='3up27v8s'
export NNI_SYS_DIR='/home/antonio/thesis/src/tuning/nni-experiments/3up27v8s/trials/Yuwci'
export NNI_TRIAL_JOB_ID='Yuwci'
export NNI_OUTPUT_DIR='/home/antonio/thesis/src/tuning/nni-experiments/3up27v8s/trials/Yuwci'
export NNI_TRIAL_SEQ_ID='6'
export NNI_CODE_DIR='/home/antonio/thesis/src'
cd $NNI_CODE_DIR
eval python3 tune.py 1>/home/antonio/thesis/src/tuning/nni-experiments/3up27v8s/trials/Yuwci/stdout 2>/home/antonio/thesis/src/tuning/nni-experiments/3up27v8s/trials/Yuwci/stderr
echo $? `date +%s%3N` >'/home/antonio/thesis/src/tuning/nni-experiments/3up27v8s/trials/Yuwci/.nni/state'