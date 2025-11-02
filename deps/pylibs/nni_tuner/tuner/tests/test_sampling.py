from unittest import TestCase

from pylibs.utils.util_log import get_logger
from tuner.sampling.dds_sampling_pysample import DDSSamplingPySample
from tuner.sampling.halton_sampling_pysample import HaltonSamplingPySample
from tuner.sampling.lhs_sampling_pysample import LHSSamplingPySample
from tuner.sampling.random_sampling_pysample import RandomSamplingPySample
from tuner.sampling.scbol_sampling_pysample import ScbolSamplingPySample

log = get_logger()


class TestSampling(TestCase):
    def test_sampling_method(self):
        cf = {'f1': {'_type': 'quniform', '_value': [2, 20, 1]},
              'f2': {'_type': 'choice', '_value': ['zhang', 'li']},
              'f3': {'_type': 'uniform', '_value': [1, 10]}}
        sampler = HaltonSamplingPySample(20, cf)
        sampler.get_samples()
        sampler = DDSSamplingPySample(20, cf)
        sampler.get_samples()
        sampler = HaltonSamplingPySample(20, cf)
        sampler.get_samples()
        sampler = LHSSamplingPySample(20, cf)
        sampler.get_samples()
        sampler = RandomSamplingPySample(20, cf)
        sampler.get_samples()
        sampler = ScbolSamplingPySample(20, cf)
        sampler.get_samples()
