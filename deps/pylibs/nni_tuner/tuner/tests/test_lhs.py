from matplotlib import pyplot as plt
from unittest import TestCase

from pylibs.util_matplotlib import save_fig
from tuner.sampling.lhs_sampling import LHSSampling


class TestSampling(TestCase):
    def test_trail1(self):
        constraint = {
            'latent_dim': {'_type': 'quniform', '_value': [2, 20, 1]},
            'res': {'_type': 'quniform', '_value': [3, 8, 1]},
            "ho": {"_type": "choice", "_value": ["s", "r", "p"]}
        }
        sam = LHSSampling(10, constraint)
        samples_1 = sam.get_samples()
        assert len(samples_1) == 10
        print(samples_1)

    def test_trail2(self):
        constraint = {
            'latent_dim': {'_type': 'quniform', '_value': [0, 10, 1]},
            'res': {'_type': 'quniform', '_value': [0, 10, 1]},
        }
        sample_size = 20
        sam = LHSSampling(sample_size, constraint, seed=0)
        samples = sam.get_samples()
        assert len(samples) == sample_size

    def test_trail3(self):

        try:
            constraint = {
                'latent_dim': {'_type': 'quniform', '_value': [0, 10, 1]},
                'res': {'_type': 'quniform', '_value': [0, 10, 1]},
            }
            sample_size = 0
            sam = LHSSampling(sample_size, constraint, seed=0)
            samples = sam.get_samples()
            assert len(samples) == sample_size
            plt.scatter(samples.iloc[:, 0], samples.iloc[:, 1])
            plt.show()
            assert False
        except:
            assert True
