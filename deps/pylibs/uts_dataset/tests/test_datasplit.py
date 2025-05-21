import unittest

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.testing import assert_almost_equal

from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView
from pylibs.utils.util_numpy import enable_numpy_reproduce

enable_numpy_reproduce()
from pylibs.uts_dataset.dataset_loader import DatasetLoader


class TestDatasetLoader(unittest.TestCase):
    def test_data_split(self):
        dl = DatasetLoader("DEBUG", "period_test.out", test_rate=0.3, anomaly_window_type="coca", window_size=60)
        train_x, train_y, test_x, test_y = dl.get_sliding_windows_train_and_test()
        uv = UnivariateTimeSeriesView(
            name="period_test_train",
            is_save_fig=True)
        uv.plot_x_label(train_x[:, -1], train_y)
        uv = UnivariateTimeSeriesView(
            name="period_test_test",
            is_save_fig=True)
        uv.plot_x_label(test_x[:, -1], test_y)
        assert_almost_equal(np.argwhere(test_y > 0), [[166],
                                                      [167],
                                                      [168],
                                                      [246]])
