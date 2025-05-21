from unittest import TestCase

import numpy as np

from pylibs.util_matplotlib import plot_3d, save_fig


class TestMatplotLib(TestCase):

    def setUp(self) -> None:
        super().setUp()

    def testPlotLy(self):
        # NNI_OUTPUT_DIR
        x = np.linspace(1, 10, 10)
        y = np.linspace(1, 10, 10)
        z = np.linspace(1, 10, 10)
        fig = plot_3d(x, y, z)
        save_fig(fig)
