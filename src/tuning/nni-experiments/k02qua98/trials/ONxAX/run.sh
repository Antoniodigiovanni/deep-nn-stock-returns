#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='k02qua98'
export NNI_SYS_DIR='/home/antonio/thesis/src/tuning/nni-experiments/k02qua98/trials/ONxAX'
export NNI_TRIAL_JOB_ID='ONxAX'
export NNI_OUTPUT_DIR='/home/antonio/thesis/src/tuning/nni-experiments/k02qua98/trials/ONxAX'
export NNI_TRIAL_SEQ_ID='2'
export NNI_CODE_DIR='/home/antonio/thesis/src'
cd $NNI_CODE_DIR
eval python3 tune.py 1>/home/antonio/thesis/src/tuning/nni-experiments/k02qua98/trials/ONxAX/stdout 2>/home/antonio/thesis/src/tuning/nni-experiments/k02qua98/trials/ONxAX/stderr
echo $? `date +%s%3N` >'/home/antonio/thesis/src/tuning/nni-experiments/k02qua98/trials/ONxAX/.nni/state'