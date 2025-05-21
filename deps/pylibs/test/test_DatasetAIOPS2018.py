from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal

from pylibs.dataset.DatasetAIOPS2018 import DatasetAIOps2018
from pylibs.utils.util_log import get_logger

log = get_logger()


class TestDatasetAIOps2018(TestCase):

    def setUp(self) -> None:
        # 180 minutes
        self.kpi_id = "01"
        dr = pd.date_range(start="2022/1/1 00:00", end='2022/1/1 2:59', freq='1min')
        target_time_stamp = np.concatenate([dr.values[0:32], dr.values[100:]])
        value = np.arange(target_time_stamp.shape[0])
        train_test_data = np.r_[value, value]
        scale = train_test_data.mean()
        bias = train_test_data.std()
        normal_value = (train_test_data - scale) / bias
        self.scale_all = scale
        self.bias_all = bias

        UtilSys.is_debug_mode() and log.info(f"Setup scale:{scale}, bias:{bias}")
        data = pd.DataFrame({
            'timestamp': target_time_stamp,
            'value': value,
            'label': np.zeros((target_time_stamp.shape[0],))

        })
        # DatasetAIOps2018.save_dataset_to_db('01', data, data_type="train", fill_missing=True)
        # DatasetAIOps2018.save_dataset_to_db('01', data, data_type="test", fill_missing=True)
        self.source_train = pd.DataFrame({
            'timestamp': target_time_stamp,
            'value': value,
            'label': np.zeros((target_time_stamp.shape[0],))
        })
        self.source_test = self.source_train = pd.DataFrame({
            'timestamp': target_time_stamp,
            'value': value,
            'label': np.zeros((target_time_stamp.shape[0],))
        })

        self.source_train_normal = (value - scale) / bias
        self.source_test_normal = (value - scale) / bias
        DatasetAIOps2018.save_dataset_to_db_v2(self.kpi_id, data, data, fill_missing=True)
        self._normal_value = normal_value

    def test_normal(self):
        dr = pd.date_range(start="2022/1/1 00:00", end='2022/1/1 2:59', freq='1min')
        target_data = np.concatenate([dr.values[0:32], dr.values[100:]])

        data = np.arange(target_data.shape[0])
        normal_data = (data - data.min()) / (data.max() - data.min())
        data = pd.DataFrame({
            'timestamp': target_data,
            'value': normal_data,
            'label': np.zeros((target_data.shape[0],))

        })
        kpi_id = 'com_normal'
        DatasetAIOps2018.save_dataset_to_db_v2(kpi_id, data, data, fill_missing=False)

    def test_get_train_data_two_splits(self):
        di = DatasetAIOps2018(kpi_id="01", windows_size=1, is_include_anomaly_windows=True, valid_rate=0.2)
        train_x, train_label, test_x, test_label = di.get_sliding_windows_two_splits()
        assert train_x.shape[0] == 180
        assert test_x.shape[0] == 180
        assert train_label.shape[0] == 180
        assert test_label.shape[0] == 180

    def test_get_train_data_two_splits1(self):
        da = DatasetAIOps2018(kpi_id=self.kpi_id,
                              windows_size=1,
                              is_include_anomaly_windows=True,
                              sampling_method='random',
                              sampling_rate=0.5, valid_rate=0.2)
        train_x, train_y, valid_x, valid_y, test_x, test_y = da.get_sliding_windows_three_splits()
        assert train_x.shape[0] == 180 * .8 * .5
        assert train_y.shape[0] == 180 * .8 * .5
        assert valid_x.shape[0] == 180 * .2
        assert valid_y.shape[0] == 180 * .2
        assert test_x.shape[0] == 180
        assert test_y.shape[0] == 180

    def test_get_train_data_two_splits02(self):
        windows_size = 179
        di = DatasetAIOps2018(kpi_id="01", windows_size=windows_size, is_include_anomaly_windows=True, valid_rate=0.2)
        train_x, train_label, test_x, test_label = di.get_sliding_windows_two_splits()

        assert train_x.shape[1] == windows_size
        assert test_x.shape[1] == windows_size
        # There only contain two windows, so the number of label should be two
        assert train_label.shape[0] == 2
        assert test_label.shape[0] == 2

    def test_get_train_label(self):
        kpi_id = 'test01'
        # Start 1, stop 8, Missing 03,04
        n_total_points = 8
        data_time = ['2022-01-01 00:01:00', '2022-01-01 00:02:00', '2022-01-01 00:05:00',
                     '2022-01-01 00:06:00', '2022-01-01 00:07:00', '2022-01-01 00:08:00']
        value = [0, 0.1, 0.2, 0.3, 0.56, 1]
        label = [0, 0, 0, 0, 0, 1]

        accept_value = [0, 0.1, 0, 0, 0.2, 0.3, 0.56, 1]
        accept_label = [0, 0, 1, 1, 0, 0, 0, 1]
        data = pd.DataFrame({
            "timestamp": pd.to_datetime(data_time),
            "value": value,
            "label": label
        })
        DatasetAIOps2018.save_dataset_to_db_v2(kpi_id, data, data, fill_missing=True)

        da = DatasetAIOps2018(kpi_id=kpi_id, windows_size=7, is_include_anomaly_windows=True,
                              valid_rate=0.2)
        train_x, train_y, test_x, test_y = da.get_sliding_windows_two_splits()

        assert_almost_equal(train_x[0], [0, 0.1, 0, 0, 0.2, 0.3, 0.56])
        assert_almost_equal(train_y, [1, 1])

        da = DatasetAIOps2018(kpi_id=kpi_id, windows_size=1, is_include_anomaly_windows=True, valid_rate=0.1)
        train_x, train_y, valid_x, valid_y, test_x, test_y = da.get_sliding_windows_three_splits()

        assert train_x.shape[0] + valid_x.shape[0] == 8
        assert test_x.shape[0] == 8

        da = DatasetAIOps2018(kpi_id=kpi_id, windows_size=7, is_include_anomaly_windows=True, valid_rate=0.1)
        train_x, train_y, valid_x, valid_y, test_x, test_y = da.get_sliding_windows_three_splits()

        assert_almost_equal(train_x, [[0, 0.1, 0, 0, 0.2, 0.3, 0.56]])
        assert_almost_equal(valid_x, [[0.1, 0, 0, 0.2, 0.3, 0.56, 1]])
        assert_almost_equal(test_x, [[0, 0.1, 0, 0, 0.2, 0.3, 0.56], [0.1, 0, 0, 0.2, 0.3, 0.56, 1]])

    def test_get_test_label(self):
        kpi_id = 'test02'
        # Start 1, stop 10, no missing
        n_total_points = 10
        data_time = [
            '2022-01-01 00:01:00',
            '2022-01-01 00:02:00',
            '2022-01-01 00:03:00',
            '2022-01-01 00:04:00',
            '2022-01-01 00:05:00',
            '2022-01-01 00:06:00',
            '2022-01-01 00:07:00',
            '2022-01-01 00:08:00',
            '2022-01-01 00:09:00',
            '2022-01-01 00:10:00'
        ]
        value = [0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.9, 0.99, 1]
        label = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]

        data = pd.DataFrame({
            "timestamp": pd.to_datetime(data_time),
            "value": value,
            "label": label
        })
        # DatasetAIOps2018.save_dataset_to_db(kpi_id, data, data_type=DatasetAIOps2018.DATA_TYPE_TRAIN,
        #                                     fill_missing=False)
        # DatasetAIOps2018.save_dataset_to_db(kpi_id, data, data_type=DatasetAIOps2018.DATA_TYPE_TEST, fill_missing=False)

        DatasetAIOps2018.save_dataset_to_db_v2(kpi_id, data, data, fill_missing=False)

        da = DatasetAIOps2018(kpi_id=kpi_id,
                              windows_size=1,
                              is_include_anomaly_windows=False,
                              valid_rate=0.2)
        train_x, train_y, valid_x, valid_y, test_x, test_y = da.get_sliding_windows_three_splits()
        assert train_x.shape[0] == 8 - 2
        assert train_y.shape[0] == train_x.shape[0]
        assert valid_x.shape[0] == 2
        assert valid_y.shape[0] == 2
        assert test_x.shape[0] == 10
        assert test_y.shape[0] == 10
        da = DatasetAIOps2018(kpi_id=kpi_id,
                              windows_size=1,
                              is_include_anomaly_windows=True,
                              valid_rate=0.2)
        train_x, train_y, valid_x, valid_y, test_x, test_y = da.get_sliding_windows_three_splits()
        assert train_x.shape[0] == 8
        assert train_y.shape[0] == train_x.shape[0]
        assert valid_x.shape[0] == 2
        assert valid_y.shape[0] == 2
        assert test_x.shape[0] == 10
        assert test_y.shape[0] == 10

    def test_get_test_data(self):
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes
        import matplotlib.pyplot as plt
        # 创建图
        fig: Figure = plt.figure(figsize=(32, 8))

        # 添加子图
        # add_subplot(n_rows, n_cols, exp_index)
        ax1: Axes = fig.add_subplot(2, 1, 1)
        ax1.plot(self._normal_value)
        print("mean and std")
        print(self._normal_value.mean())
        print(self._normal_value.std())
        # fig.savefig()

    def test_load_data_two_splits(self):
        kpi_id = "00041"
        UtilSys.is_debug_mode() and log.info("================================")

        # Start 1, stop 10, no missing
        n_total_points = 10
        data_time = [
            '2022-01-01 00:01:00',
            '2022-01-01 00:02:00',
            '2022-01-01 00:03:00',
            '2022-01-01 00:04:00',
            '2022-01-01 00:05:00',
            '2022-01-01 00:06:00',
            '2022-01-01 00:07:00',
            '2022-01-01 00:08:00',
            '2022-01-01 00:09:00',
            '2022-01-01 00:10:00'
        ]
        value = [0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.9, 0.99, 1]
        label = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]

        target = np.r_[value, value]
        bias = target.std()
        scale = target.mean()
        UtilSys.is_debug_mode() and log.info(f"bias = {bias},scale:{scale}")
        train = pd.DataFrame({
            "timestamp": pd.to_datetime(data_time),
            "value": value,
            "label": label
        })

        n_insert = DatasetAIOps2018.save_dataset_to_db_v2(kpi_id, train, train, fill_missing=True,
                                                          preprocessing="normalize")
        assert n_insert == 20
        di = DatasetAIOps2018(kpi_id=kpi_id,
                              windows_size=9,
                              is_include_anomaly_windows=True,
                              valid_rate=0.2)
        train_x, train_y, valid_x, valid_y, test_x, test_y = di.get_sliding_windows_three_splits()
        assert train_x.shape[0] == 1
        assert_almost_equal(train_x[0], ((np.asarray(value) - scale) / bias)[:9])
        assert_almost_equal(valid_x[0], ((np.asarray(value) - scale) / bias)[1:10])
        assert test_x.shape[0] == 2
        assert test_y.shape[0] == 2

    def test_train_data_without_sliding_windows_1(self):
        di = DatasetAIOps2018(kpi_id=DatasetAIOps2018.KPI_IDS.D1,
                              sampling_rate=0,
                              is_include_anomaly_windows=True,
                              valid_rate=0.2)
        train_x, train_y, valid_x, valid_y, test_x, test_y = di.get_three_splits()
        print(train_x.shape[0])

    def test_train_data_without_sliding_windows(self):
        kpi_id = "00042"

        # Start 1, stop 10, no missing
        n_total_points = 10
        data_time = [
            '2022-01-01 00:01:00',
            '2022-01-01 00:02:00',
            '2022-01-01 00:03:00',
            '2022-01-01 00:04:00',
            '2022-01-01 00:05:00',
            '2022-01-01 00:06:00',
            '2022-01-01 00:07:00',
            '2022-01-01 00:08:00',
            '2022-01-01 00:09:00',
            '2022-01-01 00:10:00'
        ]
        value = [0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.9, 0.99, 1]
        label = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]

        target = np.r_[value, value]
        bias = target.std()
        scale = target.mean()
        UtilSys.is_debug_mode() and log.info(f"bias = {bias},scale:{scale}")
        train = pd.DataFrame({
            "timestamp": pd.to_datetime(data_time),
            "value": value,
            "label": label
        })

        n_insert = DatasetAIOps2018.save_dataset_to_db_v2(kpi_id, train, train, fill_missing=True,
                                                          preprocessing=None)
        assert n_insert == 20

        di = DatasetAIOps2018(kpi_id=kpi_id,
                              sampling_rate=1,
                              windows_size=7,
                              is_include_anomaly_windows=True,
                              valid_rate=0.2)
        train_x, train_y, train_origin_y, \
            val_x, val_y, val_origin_y, \
            test_x, test_y, test_origin_y = di.get_sliding_windows_three_splits_with_origin_label()
        assert_almost_equal(train_x, [[0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7], [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.9],
                                      [0.2, 0.3, 0.5, 0.6, 0.7, 0.9, 0.99]])
        assert_almost_equal(train_y, [1, 1, 1])
        assert_almost_equal(train_origin_y, [1, 0, 0])

        assert_almost_equal(val_x, [[0.3, 0.5, 0.6, 0.7, 0.9, 0.99, 1]])
        assert_almost_equal(val_y, [1])
        assert_almost_equal(val_origin_y, [0])

    def test_train_data_without_sliding_windows_v2(self):
        kpi_id = "00042"

        # Start 1, stop 10, no missing
        n_total_points = 10
        data_time = [
            '2022-01-01 00:01:00',
            '2022-01-01 00:02:00',
            '2022-01-01 00:03:00',
            '2022-01-01 00:04:00',
            '2022-01-01 00:05:00',
            '2022-01-01 00:06:00',
            '2022-01-01 00:07:00',
            '2022-01-01 00:08:00',
            '2022-01-01 00:09:00',
            '2022-01-01 00:10:00'
        ]
        value = [0, 0.8, 0.2, 0.3, 0.5, 0.6, 0.7, 0.9, 0.99, 1]
        label = [0, 1, 0, 0, 0, 1, 1, 0, 0, 0]

        target = np.r_[value, value]
        bias = target.std()
        scale = target.mean()
        UtilSys.is_debug_mode() and log.info(f"bias = {bias},scale:{scale}")
        train = pd.DataFrame({
            "timestamp": pd.to_datetime(data_time),
            "value": value,
            "label": label
        })

        n_insert = DatasetAIOps2018.save_dataset_to_db_v2(kpi_id, train, train, fill_missing=True,
                                                          preprocessing=None)
        assert n_insert == 20

        di = DatasetAIOps2018(kpi_id=kpi_id,
                              sampling_rate=1,
                              windows_size=4,
                              is_include_anomaly_windows=True,
                              valid_rate=0.5)
        train_x, train_y, train_origin_y, \
            val_x, val_y, val_origin_y, \
            test_x, test_y, test_origin_y = di.get_sliding_windows_three_splits_with_origin_label_v2()

        assert train_x.shape[0] == 2
        assert val_x.shape[0] == 2
        assert_almost_equal(train_x, [[0, 0.8, 0.2, 0.3], [0.8, 0.2, 0.3, 0.5]])
        assert_almost_equal(train_y, [1, 1])
        assert_almost_equal(train_origin_y, [0, 0])

    def test_train_data_without_sliding_windows_v3(self):
        kpi_id = "00042"

        # Start 1, stop 10, no missing
        n_total_points = 10
        data_time = [
            '2022-01-01 00:01:00',
            '2022-01-01 00:02:00',
            '2022-01-01 00:03:00',
            '2022-01-01 00:04:00',
            '2022-01-01 00:05:00',
            '2022-01-01 00:06:00',
            '2022-01-01 00:07:00',
            '2022-01-01 00:08:00',
            '2022-01-01 00:09:00',
            '2022-01-01 00:10:00'
        ]
        value = [0, 0.1, 0.2, 0.9, 0.5, 0.6, 0.7, 0.9, 0.99, 1]
        label = [0, 0, 0, 1, 0, 1, 1, 0, 0, 0]

        target = np.r_[value, value]
        bias = target.std()
        scale = target.mean()
        UtilSys.is_debug_mode() and log.info(f"bias = {bias},scale:{scale}")
        train = pd.DataFrame({
            "timestamp": pd.to_datetime(data_time),
            "value": value,
            "label": label
        })

        n_insert = DatasetAIOps2018.save_dataset_to_db_v2(kpi_id, train, train, fill_missing=True,
                                                          preprocessing=None)
        assert n_insert == 20

        di = DatasetAIOps2018(kpi_id=kpi_id,
                              sampling_rate=1,
                              windows_size=4,
                              is_include_anomaly_windows=True,
                              valid_rate=0.5)
        train_x, train_y, train_origin_y, \
            val_x, val_y, val_origin_y, \
            test_x, test_y, test_origin_y = di.get_sliding_windows_three_splits_with_origin_label_v2()

        assert train_x.shape[0] == 2
        assert val_x.shape[0] == 2
        assert_almost_equal(train_x, [[0, 0.1, 0.2, 0.9], [0.1, 0.2, 0.9, 0.5]])
        assert_almost_equal(train_y, [1, 1])
        assert_almost_equal(train_origin_y, [1, 0])

    def test_train_data_without_sliding_windows_v4(self):
        kpi_id = "00043"

        # Start 1, stop 10, no missing
        n_total_points = 10
        data_time = [
            '2022-01-01 00:01:00',
            '2022-01-01 00:02:00',
            '2022-01-01 00:03:00',
            '2022-01-01 00:04:00',
            '2022-01-01 00:05:00',
            '2022-01-01 00:06:00',
            '2022-01-01 00:07:00',
            '2022-01-01 00:08:00',
            '2022-01-01 00:09:00',
            '2022-01-01 00:10:00'
        ]
        value = [0, 0.1, 0.2, 0.9, 0.5, 0.6, 0.7, 0.9, 0.99, 1]
        label = [0, 0, 0, 1, 0, 1, 1, 0, 0, 0]

        target = np.r_[value, value]
        bias = target.std()
        scale = target.mean()
        UtilSys.is_debug_mode() and log.info(f"bias = {bias},scale:{scale}")
        train = pd.DataFrame({
            "timestamp": pd.to_datetime(data_time),
            "value": value,
            "label": label
        })

        n_insert = DatasetAIOps2018.save_dataset_to_db_v2(kpi_id, train, train, fill_missing=True,
                                                          preprocessing=None)
        assert n_insert == 20

        di = DatasetAIOps2018(kpi_id=kpi_id,
                              sampling_rate=2,
                              windows_size=1,
                              is_include_anomaly_windows=True,
                              valid_rate=0.1)
        train_x, train_y, train_origin_y, \
            val_x, val_y, val_origin_y, \
            test_x, test_y, test_origin_y = di.get_sliding_windows_three_splits_with_origin_label_v2()
        assert train_x.shape[0] == 2
