#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='xdmyp2sc'
export NNI_SYS_DIR='/home/antonio/thesis/src/tuning/nni-experiments/xdmyp2sc/trials/UD0DP'
export NNI_TRIAL_JOB_ID='UD0DP'
export NNI_OUTPUT_DIR='/home/antonio/thesis/src/tuning/nni-experiments/xdmyp2sc/trials/UD0DP'
export NNI_TRIAL_SEQ_ID='1'
export NNI_CODE_DIR='/home/antonio/thesis/src'
cd $NNI_CODE_DIR
eval python3 tune.py 1>/home/antonio/thesis/src/tuning/nni-experiments/xdmyp2sc/trials/UD0DP/stdout 2>/home/antonio/thesis/src/tuning/nni-experiments/xdmyp2sc/trials/UD0DP/stderr
echo $? `date +%s%3N` >'/home/antonio/thesis/src/tuning/nni-experiments/xdmyp2sc/trials/UD0DP/.nni/state'