from nni.experiment import Experiment
import os

experiment = Experiment('local')


experiment.config.experiment_name = 'Hyperparameter_optimization'

# Adjust folders and trial_code_directory #TODO

experiment.config.trial_command = 'python hp_tuning.py'
experiment.config.trial_code_directory = './src'

# experiment.config.search_space = search_space  #Used when the search space is defined in the file
experiment.config.search_space_file = (os.getcwd()+'/src/tuning/currently_not_used_searchSpace.json')


experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

experiment.config.assessor.name = 'Medianstop'
experiment.config.assessor.class_args['optimize_mode'] ='maximize'
experiment.config.assessor.class_args['start_step'] = 15


experiment.config.max_trial_number = 10
experiment.config.trial_concurrency = 2
#experiment.config.max_experiment_duration = 10h 

experiment.run(8080)

# input() or signal.pause() can be used to block the web app from closing
# after the experiment is finished

input('Experiment completed, press enter to quit')
experiment.stop()

# After the experiment is stopped, you can run :meth:`nni.experiment.Experiment.view` 
# to restart web portal.

# experiment.view(check documentation to see what goes inside here.)