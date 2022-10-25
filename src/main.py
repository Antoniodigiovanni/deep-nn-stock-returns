from config import config
import nni
import visualization.plots as plots
import os
import torch

if torch.cuda.is_available():
    print('CUDA is Available!!')

if os.path.exists(config.paths['resultsPath']) == False:
            os.makedirs(config.paths['resultsPath'])
        
with open(config.paths['resultsPath'] + '/../readme.txt', 'w') as f:
   f.write('readme')

if config.args.expandingLearningRateTuning or config.args.expandingTuning \
    or config.args.guNetworkTuning or config.args.guSimpleTuning \
        or config.args.batchExperiment or config.args.normalTuning \
            or config.args.guEnsemblePrediction or config.args.ensemblePrediction:
    import tuning.experiment_hpOptimization


# if config.args.resumeTuning:
#     from nni.experiment import Experiment

#     experiment = Experiment('local')
#     experiment.resume('ud1vlf93', 8081)

# ...