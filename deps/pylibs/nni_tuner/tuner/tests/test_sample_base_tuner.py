from unittest import TestCase

from pylibs.utils.util_log import get_logger
from tuner.sampling.dds_sampling_pysample import DDSSamplingPySample
from tuner.sampling.halton_sampling_pysample import HaltonSamplingPySample

log = get_logger()


class TestLHSRun(TestCase):
    def test_trail1(self):
        cf = {'f1': {'_type': 'quniform', '_value': [2, 20, 1]},
              'f2': {'_type': 'choice', '_value': ['zhang', 'li']},
              'f3': {'_type': 'uniform', '_value': [1, 10]}}
        sampler = HaltonSamplingPySample(20, cf)
        init_samples = sampler.get_samples()

    def test_trail2(self):
        cf = {'f1': {'_type': 'quniform', '_value': [2, 20, 1]},
              'f2': {'_type': 'choice', '_value': ['zhang', 'li']},
              'f3': {'_type': 'uniform', '_value': [1, 10]}}
        sampler = DDSSamplingPySample(20, cf)
        init_samples = sampler.get_samples()

        sampler = DDSSamplingPySample(20, cf)
        init_samples = sampler.get_samples()
