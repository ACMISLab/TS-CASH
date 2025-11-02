from unittest import TestCase
from pylibs.utils.util_matplotlib_1 import plot_time_series_for_ndarray, plot_time_series_for_merlion
from pylibs.utils.util_merlion import generate_univariate_timeseries_from_list, generate_timeseries_from_list
import numpy as np


class TestPlot(TestCase):
    def test_plot(self):
        n_points = 10
        value = np.random.random_sample((n_points,)) * 100
        predict_score = np.asarray([0, 0.2, 0.3, 0.99, 0.97, 0.2, 0.1, 0.2, 0.88, 0.83])
        ground_truth = np.asarray([0, 0, 0, 1, 1, 0, 0, 0, 1, 1])
        predict_label = np.asarray([0, 1, 1, 1, 1, 0, 0, 0, 1, 1])
        plot_time_series_for_ndarray(predict_score=predict_score, value=value, ground_truth=ground_truth)


if __name__ == '__main__':
    n_points = 10
    value = generate_timeseries_from_list(np.random.random_sample((n_points,)) * 100)
    predict_score = np.asarray([0, 0.2, 0.3, 0.99, 0.97, 0.2, 0.1, 0.2, 0.88, 0.83])
    ground_truth = np.asarray([0, 0, 0, 1, 1, 0, 0, 0, 1, 1])
    predict_label = np.asarray([0, 1, 1, 1, 1, 0, 0, 0, 1, 1])

    plot_time_series_for_ndarray(predict_score=predict_score, value=value, ground_truth=ground_truth)

    fig, ax = plot_time_series_for_merlion(title="My View",
                                           value=value,
                                           ground_truth=ground_truth,
                                           predict_score=predict_score)

    value = generate_timeseries_from_list(np.random.random_sample((n_points,)) * 100)
    predict_score = generate_timeseries_from_list([0, 0.2, 0.3, 0.99, 0.97, 0.2, 0.1, 0.2, 0.88, 0.83])
    ground_truth = generate_timeseries_from_list([0, 0, 0, 1, 1, 0, 0, 0, 1, 1])
    predict_label = generate_timeseries_from_list([0, 1, 1, 1, 1, 0, 0, 0, 1, 1])

    fig, ax = plot_time_series_for_merlion(title="My View",
                                           value=value,
                                           ground_truth=ground_truth,
                                           predict_score=predict_score)

    value = generate_univariate_timeseries_from_list(np.random.random_sample((n_points,)) * 100)
    predict_score = generate_univariate_timeseries_from_list([0, 0.2, 0.3, 0.99, 0.97, 0.2, 0.1, 0.2, 0.88, 0.83])
    ground_truth = generate_univariate_timeseries_from_list([0, 0, 0, 1, 1, 0, 0, 0, 1, 1])
    predict_label = generate_univariate_timeseries_from_list([0, 1, 1, 1, 1, 0, 0, 0, 1, 1])

    fig, ax = plot_time_series_for_merlion(title="My View",
                                           value=value,
                                           ground_truth=ground_truth,
                                           predict_score=predict_score)
