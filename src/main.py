from config import config
import nni
import visualization.plots as plots

if config.args.expandingTuning or config.args.guNetworkTuning or config.args.guSimpleTuning or config.args.batchExperiment or config.args.normalTuning:
    import tuning.experiment_hpOptimization
    # import tuning.aml_experiment


# if config.args.resumeTuning:
#     from nni.experiment import Experiment

#     experiment = Experiment('local')
#     experiment.resume('ud1vlf93', 8081)

# ...

# plots.plot_cumulative_returns(0)