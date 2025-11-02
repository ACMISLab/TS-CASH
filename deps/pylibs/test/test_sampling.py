from unittest import TestCase
import matplotlib.pyplot as plt
from pysampling.sample import sample

from pylibs.util_matplotlib import save_fig


class TestKFoldMetricProcess(TestCase):

    def setUp(self) -> None:
        super().setUp()

    def test_show_lhs_samples(self):
        # Random (‘random’)
        #
        # Latin Hypercube Sampling (‘lhs’)
        #
        # Sobol (‘sobol’)
        #
        # Halton (‘halton’)
        X = sample("lhs", 10, 2, seed=2)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(X[:, 0], X[:, 1], s=30, facecolors='none', edgecolors='r')
        save_fig(fig)
