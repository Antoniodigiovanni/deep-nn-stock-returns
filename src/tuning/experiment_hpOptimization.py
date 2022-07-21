import config
from nni.experiment import Experiment
import os

experiment = Experiment('local')


# Trying to use the same file for tuning Gu et al's network and the self experiment (add grid search on best results of
# self experiment afterwards)
if config.args.tuningExperiment:
    print('Tuning Experiment to discover an optimal architecture from scratch')
    print('Minimising validation loss')
    experiment.config.experiment_name = 'Hyperparameter_optimization'
    experiment.config.trial_command = 'python hp_tuning.py'
    experiment.config.search_space_file = (os.getcwd()+'/src/tuning/search_spaces/searchSpace_test.json')
    experiment.config.max_trial_number = 200

    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args['optimize_mode'] = 'minimize'
    # experiment.config.tuner.name = 'Evolution'
    # experiment.config.tuner.class_args = {
    #     'optimize_mode': 'maximize',
    #     'population_size': 100
    # }
    experiment.config.assessor.name = 'Medianstop'
    experiment.config.assessor.class_args['optimize_mode'] ='minimize'
    experiment.config.assessor.class_args['start_step'] = 10

elif config.args.batchExperiment:
    experiment.config.experiment_name = 'Batch Experiment'
    # Finish to implement
    experiment.config.trial_command = 'python gunetworkOptimization.py --ExpandingBatchTest'
    experiment.config.search_space_file = (os.getcwd()+'/src/tuning/search_spaces/batch_correlation_test.json')

    experiment.config.tuner.name = 'BatchTuner'


elif config.args.guNetworkTuning:
    experiment.config.experiment_name = "Gu et al.'s NN4 Optimization"
    experiment.config.trial_command = 'python gunetworkOptimization.py'
    experiment.config.search_space_file = (os.getcwd()+'/src/tuning/search_spaces/gu_grid_search_space_small.json')

    experiment.config.tuner.name = 'GridSearch'

    experiment.config.assessor.name = 'Medianstop'
    experiment.config.assessor.class_args['optimize_mode'] ='minimize'
    experiment.config.assessor.class_args['start_step'] = 10


elif config.args.guSimpleTuning:
    experiment.config.experiment_name = "Gu et al.'s NN4 - simple Optimization"
    experiment.config.trial_command = 'python gunetworkOptimization.py'
    experiment.config.search_space_file = (os.getcwd()+'/src/tuning/search_spaces/gu_grid_search_space_small.json')

    experiment.config.tuner.name = 'GridSearch'

    experiment.config.assessor.name = 'Medianstop'
    experiment.config.assessor.class_args['optimize_mode'] ='minimize'
    experiment.config.assessor.class_args['start_step'] = 10


experiment.config.trial_code_directory = './src/tuning'

# experiment.config.search_space = search_space  #Used when the search space is defined in the file




experiment.config.trial_concurrency = 2
experiment.config.max_experiment_duration = '12h' 

# Add logger for experiment id - in order to be able to view the experiment afterwards
print(f'Experiment ID: {experiment.id}')

experiment.run(8081)

# input() or signal.pause() can be used to block the web app from closing
# after the experiment is finished

input('Experiment completed, press enter to quit')
experiment.stop()

# After the experiment is stopped, you can run :meth:`nni.experiment.Experiment.view` 
# to restart web portal.

# experiment.view(check documentation to see what goes inside here.)
