from unittest import TestCase

from pylibs.utils.util_log import get_logger
from tuner.sample_base.dds_tuner import DDSTuner
from tuner.sample_base.halton_tuner import HaltonTuner
from tuner.sample_base.lhs_tuner import LHSTuner
from tuner.sample_base.rs_tuner import RandomTuner
from tuner.sample_base.scbol_tuner import SCBOLTuner

log = get_logger()


class TestSampling(TestCase):
    def test_sampling_method(self):
        cf = {'f1': {'_type': 'quniform', '_value': [2, 20, 1]},
              'f2': {'_type': 'choice', '_value': ['zhang', 'li']},
              'f3': {'_type': 'uniform', '_value': [1, 10]}}
        config = {
            "n_trails": 20
        }
        tuner = DDSTuner(**config)
        tuner.update_search_space(cf)
        for i in range(config['n_trails']):
            parameters = tuner.generate_parameters(i)
            tuner.receive_trial_result(i, parameters, 1)
        tuner = HaltonTuner(**config)
        tuner.update_search_space(cf)
        for i in range(config['n_trails']):
            parameters = tuner.generate_parameters(i)
            tuner.receive_trial_result(i, parameters, 1)
        tuner = LHSTuner(**config)
        tuner.update_search_space(cf)
        for i in range(config['n_trails']):
            parameters = tuner.generate_parameters(i)
            tuner.receive_trial_result(i, parameters, 1)
        tuner = RandomTuner(**config)
        tuner.update_search_space(cf)
        for i in range(config['n_trails']):
            parameters = tuner.generate_parameters(i)
            tuner.receive_trial_result(i, parameters, 1)
        tuner = SCBOLTuner(**config)
        tuner.update_search_space(cf)
        for i in range(config['n_trails']):
            parameters = tuner.generate_parameters(i)
            tuner.receive_trial_result(i, parameters, 1)
