#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='xdmyp2sc'
export NNI_SYS_DIR='/home/antonio/thesis/src/tuning/nni-experiments/xdmyp2sc/trials/i70lK'
export NNI_TRIAL_JOB_ID='i70lK'
export NNI_OUTPUT_DIR='/home/antonio/thesis/src/tuning/nni-experiments/xdmyp2sc/trials/i70lK'
export NNI_TRIAL_SEQ_ID='0'
export NNI_CODE_DIR='/home/antonio/thesis/src'
cd $NNI_CODE_DIR
eval python3 tune.py 1>/home/antonio/thesis/src/tuning/nni-experiments/xdmyp2sc/trials/i70lK/stdout 2>/home/antonio/thesis/src/tuning/nni-experiments/xdmyp2sc/trials/i70lK/stderr
echo $? `date +%s%3N` >'/home/antonio/thesis/src/tuning/nni-experiments/xdmyp2sc/trials/i70lK/.nni/state'