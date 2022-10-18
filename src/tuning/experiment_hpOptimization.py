import config
from nni.experiment import Experiment
import os

print('In experiment declaration, save dir is:')
print(config.saveDir)

saveDir = config.saveDir.split('/')[-1]

experiment = Experiment('local')


# Trying to use the same file for tuning Gu et al's network and the self experiment (add grid search on best results of
# self experiment afterwards)
if config.args.expandingTuning:
    print('Tuning Experiment to discover an optimal architecture from scratch (expanding train)')
    print('Minimising validation loss')
    experiment.config.experiment_name = 'Hyperparameter_optimization'
    experiment.config.trial_command = 'python hp_tuning.py --saveDirName ' + saveDir + ' --expandingTuning'
    experiment.config.search_space_file = (os.getcwd()+'/src/tuning/search_spaces/searchSpace.json')
    experiment.config.max_trial_number = 2000
    experiment.config.trial_gpu_number = 2
   
    # experiment.config.tuner.name = 'Anneal'
    # experiment.config.tuner.class_args['optimize_mode'] = 'minimize'


    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args['optimize_mode'] = 'minimize'
    
    experiment.config.assessor.name = 'Medianstop'
    experiment.config.assessor.class_args['optimize_mode'] ='minimize'
    experiment.config.assessor.class_args['start_step'] = 50
    

if config.args.normalTuning:
    print('Tuning Experiment to discover an optimal architecture from scratch')
    print('Minimising validation loss')
    experiment.config.experiment_name = 'Hyperparameter_optimization'
    experiment.config.trial_command = 'python hp_tuning.py --saveDirName ' + saveDir + ' --normalTuning'
    experiment.config.search_space_file = (os.getcwd()+'/src/tuning/search_spaces/searchSpace.json')
    experiment.config.max_trial_number = 2000

    # experiment.config.tuner.name = 'TPE'
    # experiment.config.tuner.class_args['optimize_mode'] = 'minimize'
    experiment.config.tuner.name = 'Evolution'
    experiment.config.tuner.class_args = {
        'optimize_mode': 'maximize',
        'population_size': 100
    }
    experiment.config.assessor.name = 'Medianstop'
    experiment.config.assessor.class_args['optimize_mode'] ='minimize'
    experiment.config.assessor.class_args['start_step'] = 100


elif config.args.batchExperiment:
    experiment.config.experiment_name = 'Batch Experiment'
    # Finish to implement
    experiment.config.trial_command = 'python gunetworkOptimization.py --saveDirName' + saveDir + ' --ExpandingBatchTest'
    experiment.config.search_space_file = (os.getcwd()+'/src/tuning/search_spaces/batch_correlation_test.json')

    experiment.config.tuner.name = 'BatchTuner'


elif config.args.guNetworkTuning: #guExpandingGridSearch
    experiment.config.experiment_name = "Gu et al.'s NN4 Optimization"
    experiment.config.trial_command = 'python gunetworkOptimization.py --saveDirName ' + saveDir + ' --expandingTraining'
    experiment.config.search_space_file = (os.getcwd()+'/src/tuning/search_spaces/gu_grid_search_space_part2.json')

    experiment.config.tuner.name = 'GridSearch'

    experiment.config.assessor.name = 'Medianstop'
    experiment.config.assessor.class_args['optimize_mode'] ='minimize'
    experiment.config.assessor.class_args['start_step'] = 100


elif config.args.guSimpleTuning: #guSimpleGridSearch
    experiment.config.experiment_name = "Gu et al.'s NN4 - simple Optimization"
    experiment.config.trial_command = 'python gunetworkOptimization.py --saveDirName ' + saveDir + ' --normalTraining'
    experiment.config.search_space_file = (os.getcwd()+'/src/tuning/search_spaces/gu_grid_search_space_small.json')

    experiment.config.tuner.name = 'GridSearch'

    experiment.config.assessor.name = 'Medianstop'
    experiment.config.assessor.class_args['optimize_mode'] ='minimize'
    experiment.config.assessor.class_args['start_step'] = 10

elif config.args.expandingLearningRateTuning:
    experiment.config.experiment_name = "Final Tuning of best network"
    experiment.config.trial_command = 'python hp_tuning.py --saveDirName ' + saveDir + ' --expandingTuning'
    experiment.config.search_space_file = (os.getcwd()+'/src/tuning/search_spaces/searchSpace_gridTuningNN3.json')
    # experiment.config.search_space_file = (os.getcwd()+'/src/tuning/search_spaces/searchSpace_gridTuningNN9.json')

    experiment.config.tuner.name = 'GridSearch'

    # experiment.config.assessor.name = 'Medianstop'
    # experiment.config.assessor.class_args['optimize_mode'] ='minimize'
    # experiment.config.assessor.class_args['start_step'] = 100

elif config.args.guEnsemblePrediction: #guExpandingGridSearch
    experiment.config.experiment_name = "Gu et al.'s NN4 Ensemble Prediction"
    experiment.config.trial_command = 'python gu_ensemblePrediction.py --saveDirName ' + saveDir + ' --expandingTraining'
    experiment.config.search_space_file = (os.getcwd()+'/src/tuning/search_spaces/gu_ensemblePrediction.json')
    experiment.config.max_trial_number = 10
    #if config.device != 'cpu':
    #    experiment.config.trial_gpu_number = 1
    #    experiment.config.training_service.use_active_gpu = True
    experiment.config.tuner.name = 'Random'


experiment.config.trial_code_directory = './src/tuning'

# experiment.config.search_space = search_space  #Used when the search space is defined in the file




experiment.config.trial_concurrency = 1

# experiment.config.max_experiment_duration = '480h' 

# Add logger for experiment id - in order to be able to view the experiment afterwards
print(f'Experiment ID: {experiment.id}')

experiment.run(8080)

# Event Loop
# while True:
#     if experiment.get_status() == 'DONE':
#         search_data = experiment.export_data()
#         search_metrics = experiment.get_job_metrics()
#         input("Experiment is finished. Press any key to exit...")
#         break

# input() or signal.pause() can be used to block the web app from closing
# after the experiment is finished

input('Experiment completed, press enter to quit')
experiment.stop()

# After the experiment is stopped, you can run :meth:`nni.experiment.Experiment.view` 
# to restart web portal.

# experiment.view(check documentation to see what goes inside here.)
