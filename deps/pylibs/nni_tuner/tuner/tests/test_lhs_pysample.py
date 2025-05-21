from unittest import TestCase

from tuner.sampling.common import TestCommon
from tuner.sampling.lhs_sampling_pysample import LHSSamplingPySample


class TestSampling(TestCase):
    def test_trail1(self):
        constraint = TestCommon.search_space_01
        sam = LHSSamplingPySample(10, constraint)
        samples_1 = sam.get_samples()
        assert len(samples_1) == 10
        print(samples_1)
