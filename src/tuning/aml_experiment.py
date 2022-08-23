import config
from nni.experiment import Experiment
import os

experiment = Experiment('aml')
print('In AML experiment')
experiment.config.experiment_name = "Gu et al.'s NN4 Optimization"
experiment.config.trial_command = 'python gunetworkOptimization.py --expandingTraining'
experiment.config.search_space_file = (os.getcwd()+'/src/tuning/search_spaces/gu_grid_search_space.json')

experiment.config.tuner.name = 'GridSearch'

experiment.config.assessor.name = 'Medianstop'
experiment.config.assessor.class_args['optimize_mode'] ='minimize'
experiment.config.assessor.class_args['start_step'] = 10

experiment.config.trial_code_directory = './src/tuning'

# experiment.config.search_space = search_space  #Used when the search space is defined in the file

experiment.config.trial_concurrency = 2
experiment.config.max_experiment_duration = '480h' 

experiment.config.training_service.platform = 'aml'
experiment.config.training_service.docker_image = 'msranni/nni'
experiment.config.training_service.subscription_id = '68ac269d-2ad1-489a-aad4-f7134d95768c'
experiment.config.training_service.resource_group = 'Risorse1'
experiment.config.training_service.workspace_name = 'aml-test'
experiment.config.training_service.compute_target = 'antoniodg981'

experiment.run(8082)
experiment.stop()