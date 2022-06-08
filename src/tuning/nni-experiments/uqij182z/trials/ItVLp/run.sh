#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='uqij182z'
export NNI_SYS_DIR='/home/antonio/thesis/src/tuning/nni-experiments/uqij182z/trials/ItVLp'
export NNI_TRIAL_JOB_ID='ItVLp'
export NNI_OUTPUT_DIR='/home/antonio/thesis/src/tuning/nni-experiments/uqij182z/trials/ItVLp'
export NNI_TRIAL_SEQ_ID='4'
export NNI_CODE_DIR='/home/antonio/thesis/src'
cd $NNI_CODE_DIR
eval python3 tune.py 1>/home/antonio/thesis/src/tuning/nni-experiments/uqij182z/trials/ItVLp/stdout 2>/home/antonio/thesis/src/tuning/nni-experiments/uqij182z/trials/ItVLp/stderr
echo $? `date +%s%3N` >'/home/antonio/thesis/src/tuning/nni-experiments/uqij182z/trials/ItVLp/.nni/state'