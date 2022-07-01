from config import config

if config.args.tuningExperiment:
    import tuning.experiment_hpOptimization

if config.args.guNetworkTuning:
    import tuning.experiment_hpOptimization

# ...
