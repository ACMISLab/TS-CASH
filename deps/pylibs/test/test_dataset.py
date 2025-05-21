import os
from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal

from pylibs.util_dataset import DatasetName, load_data_sampling_train_two_split, load_odds_data_two_splits, \
    load_odds_data_two_splits_with_sampling, complete_timestamp


class TestDataset(TestCase):
    def test_load_dataset_two_split(self):
        data_name = DatasetName.ODDS_THYROID
        # Thyroid	3772	6
        x_train, y_train, x_test, y_test = load_odds_data_two_splits(data_name, test_size=0.2,
                                                                     include_train_anomaly=True)
        assert x_train.shape[0] + x_test.shape[0] == 3772

        x_train, y_train, x_test, y_test = load_odds_data_two_splits(data_name, test_size=0, include_train_anomaly=True)
        # Thyroid	3772	6	93 (2.5%)
        assert y_train.shape[0] == 3772
        assert x_test.shape[0] == 0

    def test_sample_training_data_02(self):
        data_sample_method = "RS"
        data_name = DatasetName.ODDS_THYROID
        test_size = 0.2
        data_sample_rate = 1
        include_train_anomaly = True
        x, y, x_t, y_t = load_odds_data_two_splits(
            data_name,
            include_train_anomaly=include_train_anomaly,
            test_size=test_size
        )
        x_train, y_train, X_test, y_test = load_data_sampling_train_two_split(
            data_name,
            data_sample_method=data_sample_method,
            data_sample_rate=data_sample_rate,
            include_train_anomaly=include_train_anomaly,
            test_size=test_size
        )
        assert x.shape[0] == x_train.shape[0]

    def test_sample_training_data_03(self):
        data_sample_method = "RS"
        data_sample_rate = 1. / 2
        data_name = DatasetName.ODDS_THYROID
        test_size = 0.2
        include_train_anomaly = True
        x, y, x_t, y_t = load_odds_data_two_splits(
            data_name,
            include_train_anomaly=include_train_anomaly,
            test_size=test_size
        )
        x_train, y_train, X_test, y_test = load_data_sampling_train_two_split(
            data_name,
            data_sample_method=data_sample_method,
            data_sample_rate=data_sample_rate,
            include_train_anomaly=include_train_anomaly,
            test_size=test_size
        )
        assert int(x.shape[0] * data_sample_rate) == x_train.shape[0]

    def test_load_data_two_split(self):
        # data=[1,2,...,100]

        data_x = np.linspace(1, 100, 100)  # type: np.ndarray
        y = np.zeros((100,))
        file = "test.npz"
        np.savez_compressed(file, X=data_x, y=y)
        loaded_data = np.load(file)
        assert loaded_data['X'].shape[0] == 100
        assert np.array_equal(data_x, loaded_data['X'])

        x_train, y_train, x_test, y_test = load_odds_data_two_splits(file, test_size=0)
        assert np.array_equal(x_train, data_x[:100])
        assert np.array_equal(x_test, data_x[100:])
        assert np.array_equal(y_train, y[:100])
        assert np.array_equal(y_test, y[100:])

        x_train, y_train, x_test, y_test = load_odds_data_two_splits(file, test_size=0.2)
        assert np.array_equal(x_train, data_x[:80])
        assert np.array_equal(x_test, data_x[80:])
        assert np.array_equal(y_train, y[:80])
        assert np.array_equal(y_test, y[80:])

        x_train, y_train, x_test, y_test = load_odds_data_two_splits(file, test_size=0.5)
        assert np.array_equal(x_train, data_x[:50])
        assert np.array_equal(x_test, data_x[50:])
        assert np.array_equal(y_train, y[:50])
        assert np.array_equal(y_test, y[50:])

        x_train, y_train, x_test, y_test = load_odds_data_two_splits(file, test_size=1)
        assert np.array_equal(x_train, data_x[:0])
        assert np.array_equal(x_test, data_x[0:])
        assert np.array_equal(y_train, y[:0])
        assert np.array_equal(y_test, y[0:])

    def test_load_data_two_split_error(self):
        # data=[1,2,...,100]

        data_x = np.linspace(1, 100, 100)  # type: np.ndarray
        y = np.ones((100,))
        y[3] = 0
        y[90] = 0
        file = "test.npz"
        np.savez_compressed(file, X=data_x, y=y)
        loaded_data = np.load(file)
        assert loaded_data['X'].shape[0] == 100
        assert np.array_equal(data_x, loaded_data['X'])

        x_train, y_train, x_test, y_test = load_odds_data_two_splits(file, test_size=0, include_train_anomaly=False)
        assert np.array_equal(x_train, [4, 91])
        assert np.array_equal(x_test, data_x[100:])
        assert np.array_equal(y_train, [0, 0])
        assert np.array_equal(y_test, y[100:])

        x_train, y_train, x_test, y_test = load_odds_data_two_splits(file, test_size=1, include_train_anomaly=False)
        assert np.array_equal(x_train, [])
        assert np.array_equal(x_test, data_x)
        assert np.array_equal(y_train, [])
        assert np.array_equal(y_test, y)

    def test_load_data_two_split_error_load_name(self):
        # data=[1,2,...,100]

        data_x = np.linspace(1, 100, 100)  # type: np.ndarray
        y = np.ones((100,))
        y[3] = 0
        y[90] = 0
        file = "all.npz"
        np.savez_compressed(file, X=data_x, y=y)
        loaded_data = np.load(file)
        assert loaded_data['X'].shape[0] == 100
        assert np.array_equal(data_x, loaded_data['X'])

    def test_load_data_two_split_test_size(self):
        # data=[1,2,...,100]

        data_x = np.linspace(1, 100, 100)  # type: np.ndarray
        y = np.concatenate([np.zeros((50,)), np.ones((50,))], axis=0)
        file = "all.npz"
        np.savez_compressed(file, X=data_x, y=y)
        loaded_data = np.load(file)
        assert loaded_data['X'].shape[0] == 100
        assert np.array_equal(data_x, loaded_data['X'])

        x_train, y_train, x_test, y_test = load_odds_data_two_splits(file, test_size=0.2, include_train_anomaly=False)
        assert np.array_equal(x_train, data_x[:50])
        assert np.array_equal(x_test, data_x[80:])
        assert np.array_equal(y_train, y[:50])
        assert np.array_equal(y_test, y[80:])

        x_train, y_train, x_test, y_test = load_odds_data_two_splits(file, test_size=0, include_train_anomaly=False)
        assert np.array_equal(x_train, data_x[:50])
        assert np.array_equal(x_test, data_x[100:])
        assert np.array_equal(y_train, y[:50])
        assert np.array_equal(y_test, y[100:])

        x_train, y_train, x_test, y_test = load_odds_data_two_splits(file, test_size=1)
        assert np.array_equal(x_train, [])
        assert np.array_equal(x_test, data_x)
        assert np.array_equal(y_train, [])
        assert np.array_equal(y_test, y)

    def test_size_str(self):
        data_x = np.linspace(1, 100, 100)  # type: np.ndarray
        y = np.concatenate([np.zeros((50,)), np.ones((50,))], axis=0)
        file = "all.npz"
        np.savez_compressed(file, X=data_x, y=y)
        loaded_data = np.load(file)
        assert loaded_data['X'].shape[0] == 100
        assert np.array_equal(data_x, loaded_data['X'])
        x_train, y_train, x_test, y_test = load_odds_data_two_splits(file, test_size=None)
        assert np.array_equal(x_train, data_x[:50])
        assert np.array_equal(x_test, data_x[60:])
        assert np.array_equal(y_train, y[:50])
        assert np.array_equal(y_test, y[60:])

        x_train, y_train, x_test, y_test = load_odds_data_two_splits(file, test_size='1')
        assert np.array_equal(x_test, data_x)

        x_train, y_train, x_test, y_test = load_odds_data_two_splits(file, test_size='1/2')
        assert np.array_equal(x_train, data_x[0:50])

    def test_size_split(self):
        data_x = np.linspace(1, 100, 100)  # type: np.ndarray
        y = np.concatenate([np.zeros((50,)), np.ones((50,))], axis=0)
        file = "all.npz"
        np.savez_compressed(file, X=data_x, y=y)
        loaded_data = np.load(file)
        assert loaded_data['X'].shape[0] == 100
        assert np.array_equal(data_x, loaded_data['X'])
        x_train, y_train, x_test, y_test = load_odds_data_two_splits_with_sampling(file, test_size=None)
        assert np.array_equal(x_train, data_x[:50])
        assert np.array_equal(x_test, data_x[60:])
        assert np.array_equal(y_train, y[:50])
        assert np.array_equal(y_test, y[60:])

        x_train, y_train, x_test, y_test = load_odds_data_two_splits_with_sampling(file, test_size=None,
                                                                                   include_train_anomaly=True,
                                                                                   train_data_sample_rate=0.2)
        assert x_train.shape[0] == 60 * 0.2

        x_train, y_train, x_test, y_test = load_odds_data_two_splits_with_sampling(file, test_size=0.4,
                                                                                   include_train_anomaly=True,
                                                                                   train_data_sample_rate=0.2,
                                                                                   train_data_sample_method="RS")
        assert x_train.shape[0] == 60 * 0.2

        try:
            load_odds_data_two_splits_with_sampling(file, test_size=0.4,
                                                    include_train_anomaly=True,
                                                    train_data_sample_rate=0.2,
                                                    train_data_sample_method="xxxxxxxx")
            assert False
        except Exception as e:
            assert str(e) == "Unsupported sample method xxxxxxxx"

        x_train, y_train, x_test, y_test = load_odds_data_two_splits_with_sampling(file, test_size=0.4,
                                                                                   include_train_anomaly=True,
                                                                                   train_data_sample_rate="1/2",
                                                                                   train_data_sample_method="RS")
        assert x_train.shape[0] == 30
        assert y_train.shape[0] == 30

    def load_time_series(self):
        x_train, y_train, x_test, y_test = load_odds_data_two_splits(
            dataset_name="AIOPS_KPI_01",
            test_size=None,
            include_train_anomaly=True,
        )

    def test_fill_missing(self):
        data = pd.read_csv("./data/test.csv")
        timestamp, missing = complete_timestamp(data['timestamp'].values)
        assert_almost_equal(missing, [0, 1, 1, 1, 0, 0, 0, 0, 0])
        assert_almost_equal(timestamp, [1500012900, 1500012960, 1500013020, 1500013080, 1500013140, 1500013200,
                                        1500013260, 1500013320, 1500013380])

        data = pd.read_csv("./data/test.csv")
        timestamp, missing, (value, label) = complete_timestamp(data['timestamp'].values,
                                                                (data['value'].values, data['label'].values))
        assert_almost_equal(missing, [0, 1, 1, 1, 0, 0, 0, 0, 0])
        assert_almost_equal(timestamp, [1500012900, 1500012960, 1500013020, 1500013080, 1500013140, 1500013200,
                                        1500013260, 1500013320, 1500013380])

        assert_almost_equal(value, [42.69, 0., 0., 0., 41.97, 42.86, 41.87, 40.91, 41.55])
