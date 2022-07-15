from config import config
import nni

if config.args.tuningExperiment or config.args.guNetworkTuning or config.args.guSimpleTuning or config.args.batchExperiment:
    import tuning.experiment_hpOptimization

# if config.args.resumeTuning:
#     from nni.experiment import Experiment

#     experiment = Experiment('local')
#     experiment.resume('ud1vlf93', 8081)

# ...
