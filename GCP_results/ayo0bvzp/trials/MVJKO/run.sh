#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='ayo0bvzp'
export NNI_SYS_DIR='/home/antoniodg98/nni-experiments/ayo0bvzp/trials/MVJKO'
export NNI_TRIAL_JOB_ID='MVJKO'
export NNI_OUTPUT_DIR='/home/antoniodg98/nni-experiments/ayo0bvzp/trials/MVJKO'
export NNI_TRIAL_SEQ_ID='3'
export NNI_CODE_DIR='/home/antoniodg98/thesis/src/tuning'
cd $NNI_CODE_DIR
eval 'python hp_tuning.py --saveDirName NN9LRtuning --expandingTuning' 1>/home/antoniodg98/nni-experiments/ayo0bvzp/trials/MVJKO/stdout 2>/home/antoniodg98/nni-experiments/ayo0bvzp/trials/MVJKO/stderr
echo $? `date +%s%3N` >'/home/antoniodg98/nni-experiments/ayo0bvzp/trials/MVJKO/.nni/state'