from matplotlib import pyplot as plt
from unittest import TestCase

from tuner.sampling.random_sampling import RandomSampling


class TestSampling(TestCase):
    def test_trail1(self):
        constraint = {
            'latent_dim': {'_type': 'quniform', '_value': [2, 20, 1]},
            'res': {'_type': 'quniform', '_value': [3, 8, 1]},
            "ho": {"_type": "choice", "_value": ["s", "r", "p"]}
        }
        sample_size = 20
        sam = RandomSampling(sample_size, constraint, seed=0)
        samples = sam.get_samples()
        assert len(samples) == sample_size
        print(samples)
