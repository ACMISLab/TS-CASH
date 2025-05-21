search_space = {
    'features': {'_type': 'choice', '_value': [128, 256, 512, 1024]},
    'lr': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
    'momentum': {'_type': 'uniform', '_value': [0, 1]},
}
from nni.experiment import Experiment

experiment = Experiment('local')
experiment.config.experiment_name = "test1"
experiment.config.trial_command = 'python demo_model.py'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.max_trial_number = 10
experiment.config.trial_concurrency = 2
experiment.run(8080)
