"""
this class works after execute datasets/2018-AIOPS/data_process.py
"""
import os
import pprint

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from pylibs.common import SampleType
from pylibs.dataset.DatasetBase import Dataset

from pylibs.utils.util_message import log_metric_msg, logw
from pylibs.utils.utils import convert_str_to_float
from pylibs.utils.util_log import get_logger

log = get_logger()


def subsequences(sequence, window_size, time_step):
    # An array of non-contiguous memory is converted to an array of contiguous memory
    sq = np.ascontiguousarray(sequence)
    a = (sq.shape[0] - window_size + time_step) % time_step
    # label array
    if sq.ndim == 1:
        shape = (int((sq.shape[0] - window_size + time_step) / time_step), window_size)
        stride = sq.itemsize * np.array([time_step * 1, 1])
        if a != 0:
            sq = sq[:sq.shape[0] - a]
    # data array
    elif sq.ndim == 2:
        shape = (int((sq.shape[0] - window_size + time_step) / time_step), window_size, sq.shape[1])
        stride = sq.itemsize * np.array([time_step * sq.shape[1], sq.shape[1], 1])
        if a != 0:
            sq = sq[:sq.shape[0] - a, :]
    else:
        raise RuntimeError('Array dimension error')
    sq = np.lib.stride_tricks.as_strided(sq, shape=shape, strides=stride)
    return sq


class AIOpsKPIID:
    TEST_01 = "fake_kpi_01"
    TEST_LINE = "fake_kpi_01"
    TEST_LINE_OBVIOUS = "fake_kpi_01"
    TEST_LINE_VAGUE = "fake_kpi_01_vague"
    # The anomalies are obvious

    TEST_PERIOD_OBVIOUS = "fake_period_obvious"
    # The anomalies are vague
    TEST_PERIOD_VAGUE = "fake_period_vague"
    # available KPI id:
    D1 = "05f10d3a-239c-3bef-9bdc-a2feeb0037aa"
    D2 = "0efb375b-b902-3661-ab23-9a0bb799f4e3"
    D3 = "1c6d7a26-1f1a-3321-bb4d-7a9d969ec8f0"
    D4 = "301c70d8-1630-35ac-8f96-bc1b6f4359ea"
    D5 = "42d6616d-c9c5-370a-a8ba-17ead74f3114"
    D6 = "43115f2a-baeb-3b01-96f7-4ea14188343c"
    D7 = "431a8542-c468-3988-a508-3afd06a218da"
    D8 = "4d2af31a-9916-3d9f-8a8e-8a268a48c095"
    D9 = "54350a12-7a9d-3ca8-b81f-f886b9d156fd"
    D10 = "55f8b8b8-b659-38df-b3df-e4a5a8a54bc9"
    D11 = "57051487-3a40-3828-9084-a12f7f23ee38"
    D12 = "6a757df4-95e5-3357-8406-165e2bd49360"
    D13 = "6d1114ae-be04-3c46-b5aa-be1a003a57cd"
    D14 = "6efa3a07-4544-34a0-b921-a155bd1a05e8"
    D15 = "7103fa0f-cac4-314f-addc-866190247439"
    D16 = "847e8ecc-f8d2-3a93-9107-f367a0aab37d"
    D17 = "8723f0fb-eaef-32e6-b372-6034c9c04b80"
    D18 = "9c639a46-34c8-39bc-aaf0-9144b37adfc8"
    D19 = "a07ac296-de40-3a7c-8df3-91f642cc14d0"
    D20 = "a8c06b47-cc41-3738-9110-12df0ee4c721"
    D21 = "ab216663-dcc2-3a24-b1ee-2c3e550e06c9"
    D22 = "adb2fde9-8589-3f5b-a410-5fe14386c7af"
    D23 = "ba5f3328-9f3f-3ff5-a683-84437d16d554"
    D24 = "c02607e8-7399-3dde-9d28-8a8da5e5d251"
    D25 = "c69a50cf-ee03-3bd7-831e-407d36c7ee91"
    D26 = "da10a69f-d836-3baa-ad40-3e548ecf1fbd"
    D27 = "e0747cad-8dc8-38a9-a9ab-855b61f5551d"
    D28 = "f0932edd-6400-3e63-9559-0a9860a1baa9"
    D29 = "ffb82d38-5f00-37db-abc0-5d2e4e4cb6aa"

    LARGE_01 = "6efa3a07-4544-34a0-b921-a155bd1a05e8"
    LARGE_02 = "42d6616d-c9c5-370a-a8ba-17ead74f3114"
    LARGE_03 = "6d1114ae-be04-3c46-b5aa-be1a003a57cd"


class DatasetAIOps2018(Dataset):
    DATA_TYPE_TRAIN = 'train'
    DATA_TYPE_TEST = 'test'
    KPI_IDS: AIOpsKPIID = AIOpsKPIID

    """
    The class for AIOPS 2018 m_dataset. One instance of this class is used to manage a specific KPI ID.
    
    If windows_size is greater than the number of sliding_windows_data points, raise ValueError.
    
    
    If is_include_anomaly_windows = True, all the rolling windows in training and test m_dataset will be returned.
    
    If is_include_anomaly_windows = False, *the normal rolling windows in training m_dataset (remove the windows including
    anomaly sliding_windows_data point)*  and all the rolling windows in test m_dataset and  will be returned; 
    
    
    """

    def __init__(self,
                 kpi_id=AIOpsKPIID.D1,
                 windows_size=16,
                 time_step=1,
                 is_include_anomaly_windows=False,
                 valid_rate=0.2,
                 sampling_method=SampleType.RANDOM,
                 sampling_rate=1.,
                 anomaly_window_type="all",
                 con=None
                 ):
        """
        Init m_dataset specific kpi_id.

        If windows_size is greater than the number of sliding_windows_data points, raise ValueError.


        Parameters
        ----------
        kpi_id : str
            The kpi id of the m_dataset of AIOPS 2018.
        windows_size : int
            The size of the sliding window or the rolling window.
        anomaly_window_type: str, default all
            "all" means an anomaly window contain at least one anomaly point.
            'coca' means  an anomaly window  if the first n time step point is anomaly.
            For more details seeing https://uniplore.feishu.cn/wiki/wikcnCDt1FT72TcwAkXOPTao8Ec?sheet=EAgI0Q

        sampling_rate: float, or int


            If sampling_rate = 0,  randomly generate a window with L = window_size. L means the size of the generated.

            if 0<sampling_rate<=1, sampling_rate sample_rate% (percentage) data.

            if sampling_rate >1, randomly sample sampling_rate windows (N windows, such as 1 windows, 2 windows, etc. ).
            A number of windows fixed.


        Returns
        -------

        """
        if con is None:
            self._con = create_pandas_sqlite_connection()
        self._anomaly_window_type = anomaly_window_type
        self._kpi_id = kpi_id
        self._windows_size = int(windows_size)
        self._is_include_anomaly_windows = is_include_anomaly_windows
        self._sampling_method = sampling_method
        self._sampling_rate = float(sampling_rate)
        self._time_step = time_step
        self._valid_rate = float(valid_rate)
        UtilSys.is_debug_mode() and log.info(f"Init by:\n{pprint.pformat(self.__dict__)}")

        # Load the data from the database.
        train_data, test_data = self._get_data_from_database()

        # Load all the data.
        self._source_train_data, self._source_train_label, self._source_train_missing, self._source_train_timestamp \
            = train_data.loc[:, "value"].values, \
            train_data.loc[:, "label"].values, \
            train_data.loc[:, "missing"].values, \
            train_data.loc[:, "timestamp"].values

        self._source_test_data, self._source_test_label, self._source_test_missing, self._source_test_timestamp \
            = test_data.loc[:, "value"].values, \
            test_data.loc[:, "label"].values, \
            test_data.loc[:, "missing"].values, \
            test_data.loc[:, "timestamp"].values

    def _get_data_from_database(self):
        """
        Get aiops sliding_windows_data by kpi_id from database

        Return  x, train_label, train_missing, test_x, test_label, test_missing

        """
        # train_prefix = "train"
        # test_prefix = "test"
        # train_data = pd.read_sql(sql=f"select timestamp,value,label,missing from `{train_prefix}_{self._kpi_id}`",
        #                          con=create_pandas_sqlite_connection())
        # test_data = pd.read_sql(sql=f"select timestamp,value,label,missing from `{test_prefix}_{self._kpi_id}`",
        #                         con=create_pandas_sqlite_connection())
        # return train_data, test_data
        return self.get_train_test_data_source(self._kpi_id)

    @classmethod
    def get_train_test_data_source(cls, kpi_id):
        """
        Get aiops sliding_windows_data by kpi_id from database

        Return  x, train_label, train_missing, test_x, test_label, test_missing

        """
        train_prefix = "train"
        test_prefix = "test"
        con = create_pandas_sqlite_connection()
        train_data = pd.read_sql(sql=f"select timestamp,value,label,missing from `{train_prefix}_{kpi_id}`",
                                 con=con)
        test_data = pd.read_sql(sql=f"select timestamp,value,label,missing from `{test_prefix}_{kpi_id}`",
                                con=con)
        return train_data, test_data

    def _get_normal_sliding_windows(self, sliding_windows_data, label, missing):
        """
        Get the normal sliding windows from all the sliding windows

        Parameters
        ----------
        sliding_windows_data : np.ndarray
            2 dim of data for all sliding windows.
        label : np.ndarray
            1D array of label indicating for each point
        missing :
            1D array of missing indicating for each point

        Returns
        -------

        """

        anomaly_index = np.logical_or(label, missing)
        anomaly_windows = sliding_window_view(anomaly_index, window_shape=self._windows_size)
        anomaly_windows_indicator = np.sum(anomaly_windows, axis=1)
        normal_index = np.where(anomaly_windows_indicator == 0, True, False)
        return sliding_windows_data[normal_index]

    def get_source_train_timestamp(self):
        # return pd.to_datetime(self._source_train_timestamp)
        return train_test_split(pd.to_datetime(self._source_train_timestamp), test_size=self._valid_rate,
                                shuffle=False)[0]

    def get_source_test_timestamp(self):
        return pd.to_datetime(self._source_test_timestamp)

    def get_source_train_missing(self):
        # return self._source_train_missing
        return train_test_split(self._source_train_missing, test_size=self._valid_rate,
                                shuffle=False)[0]

    def get_source_test_missing(self):
        # return self._source_test_missing
        return self._source_test_missing

    def get_source_valid_data(self):
        # return self._source_valid_data
        return train_test_split(self._source_train_data, test_size=self._valid_rate,
                                shuffle=False)[1]

    def get_source_valid_missing(self):
        # return self._source_valid_missing
        return train_test_split(self._source_train_missing, test_size=self._valid_rate,
                                shuffle=False)[1]

    def get_source_valid_timestamp(self):
        # return pd.to_datetime(self._source_valid_timestamp)
        return train_test_split(self._source_train_timestamp, test_size=self._valid_rate,
                                shuffle=False)[1]

    def get_source_train_data(self):
        return train_test_split(self._source_train_data, test_size=self._valid_rate,
                                shuffle=False)[0]

    def get_source_train_label(self):
        return np.asarray(train_test_split(self._source_train_label, test_size=self._valid_rate,
                                           shuffle=False)[0][self._windows_size - 1:], dtype=np.int32)

    def get_source_valid_label(self):
        return np.asarray(train_test_split(self._source_train_label, test_size=self._valid_rate,
                                           shuffle=False)[1][self._windows_size - 1:], dtype=np.float32)

    def get_source_test_label(self):
        return np.asarray(self._source_train_label[self._windows_size - 1:], dtype=np.int32)

    def get_source_test_data(self):
        return self._source_test_data

    def _get_sampling_index(self, n_samples):
        """

        Parameters
        ----------
        n_samples :
            The number of training data (samples)

        Returns
        -------

        """
        assert isinstance(self._sampling_method, str), "Data sample method must be RS"
        if self._sampling_method == SampleType.RANDOM:
            train_data_sample_rate = convert_str_to_float(self._sampling_rate)
            if train_data_sample_rate > 1:
                n_sample_train = int(train_data_sample_rate)
                if n_sample_train >= n_samples:
                    n_sample_train = int(n_samples)
                    logw(f"There only have {n_samples} sample(s), but received {n_sample_train} sample(s). "
                         f"So changing set the number samples to {n_sample_train}")

                n_selected_sample_index = np.sort(np.random.choice(
                    a=n_samples,
                    size=n_sample_train,
                    replace=False
                ))
                return n_selected_sample_index
            else:
                n_sample_train = int(train_data_sample_rate * n_samples)
                n_selected_sample_index = np.sort(np.random.choice(
                    a=n_samples,
                    size=n_sample_train,
                    replace=False
                ))
                return n_selected_sample_index
        else:
            raise RuntimeError(f"Unsupported sample method {self._sampling_method}")

    def _get_normal_windows_index(self, label, missing):
        anomaly_index = np.logical_or(label, missing)
        anomaly_windows = sliding_window_view(anomaly_index, window_shape=self._windows_size)
        anomaly_windows_indicator = np.sum(anomaly_windows, axis=1)
        normal_windows_index = np.argwhere(anomaly_windows_indicator == 0)
        return normal_windows_index.reshape(-1)

    def _get_test_sliding_window(self):
        """
        Get the sliding window of the test data, including all windows.

        Returns
        -------

        """
        return sliding_window_view(self._source_test_data, window_shape=self._windows_size)

    def get_sliding_windows_two_splits(self):
        """
        Return the train,test splits

        Examples
        --------
        x, train_y, test_x, test_y = da.get_sliding_windows_two_splits()
        Returns
        -------

        """
        train_x, train_y, valid_x, valid_y, test_x, test_y = self.get_sliding_windows_three_splits()
        if valid_x.shape[0] > 0:
            return np.r_[train_x, valid_x], np.r_[train_y, valid_y], test_x, test_y
        else:
            return train_x, train_y, test_x, test_y

    def get_sliding_windows_three_splits(self, log_data_info=True):
        """
        Return the sliding windows for train,valid,test splits

        Examples
        --------
        x, train_y, valid_x, valid_y, test_x, test_y = da.get_sliding_windows_three_splits()
        x: (n_samples, window_size)
        train_y: (n_samples,), a vector indicating a window is anomaly or not.

        Returns
        -------

        """
        train_x, train_y, train_origin_y, \
            val_x, val_y, val_origin_y, \
            test_x, test_y, test_origin_y = self.get_sliding_windows_three_splits_with_origin_label()
        return train_x, train_y, val_x, val_y, test_x, test_y

    def get_sliding_windows_three_splits_with_origin_label(self, log_data_info=True):
        """
        Return the sliding windows for train,valid,test splits

        Examples
        --------
           x, train_y, train_origin_y, \
            val_x, val_y, val_origin_y, \
            test_x, test_y, test_origin_y = di.get_sliding_windows_three_splits_with_origin_label()

            train_origin_y, val_origin_y, test_origin_y  are the origin roll windows (corresponding to each x windows)
             which label do not been  post-processed.

            train_y, val_y, test_y is post processed by self.post_process_windows_label().


        Returns
        -------

        """
        if self._windows_size > self._source_train_data.shape[0] \
                or self._windows_size > self._source_test_data.shape[0]:
            raise ValueError("windows size must be less or equal to the number of sliding_windows_data points")

        # The data generator is followed by the paper:
        # R. Wang et al., “Deep Contrastive One-Class Time Series Anomaly Detection.” arXiv, Oct. 08, 2022.
        # Accessed: Dec. 09, 2022. [Online]. Available: http://arxiv.org/abs/2207.01472

        train_x_sliding_windows_source = subsequences(self._source_train_data, self._windows_size, self._time_step)
        test_x_sliding_windows_source = subsequences(self._source_test_data, self._windows_size, self._time_step)

        train_y_sliding_windows_source = subsequences(self._source_train_label, self._windows_size, self._time_step)
        test_y_sliding_windows_source = subsequences(self._source_test_label, self._windows_size, self._time_step)

        test_y_sliding_windows_processed, train_y_sliding_windows_processed = self.post_process_windows_label(
            test_y_sliding_windows_source, train_y_sliding_windows_source)

        # split windows into training set and validation set
        train_split_x, val_split_x, train_split_y, val_split_y = train_test_split(
            train_x_sliding_windows_source,
            train_y_sliding_windows_processed,
            test_size=self._valid_rate,
            shuffle=False)

        train_split_origin_y, val_split_origin_y = train_test_split(train_y_sliding_windows_source,
                                                                    test_size=self._valid_rate,
                                                                    shuffle=False)

        train_selected_index = self.__get_training_dataset_sampling_index(train_split_x,
                                                                          train_split_y)

        # Get post process training set
        ret_train_x, ret_train_y = \
            train_split_x[train_selected_index], train_split_y[train_selected_index]

        ret_train_origin_y = train_split_origin_y[train_selected_index]

        #  If sampling_rate = 0,  randomly generate a window with L = window_size.
        #  L means the size of the generated window
        if self._sampling_rate == 0.0:
            ret_train_x = np.random.random((1, self._windows_size))
            ret_train_y = np.asarray([1])

        # convert to float 32 for pytorch

        self.log_data_describe(log_data_info, ret_train_x, ret_train_y, test_x_sliding_windows_source,
                               test_y_sliding_windows_processed, train_split_x, train_split_y,
                               train_x_sliding_windows_source,
                               train_y_sliding_windows_source)

        return ret_train_x.astype("float32"), ret_train_y.astype("float32"), \
            ret_train_origin_y.astype('float32')[:, -1], \
            val_split_x.astype("float32"), val_split_y.astype("float32"), val_split_origin_y.astype("float32")[:, -1], \
            test_x_sliding_windows_source.astype("float32"), test_y_sliding_windows_processed.astype(
            "float32"), test_y_sliding_windows_source.astype("float32")[:, -1]

    def get_pydl_windows_3splits_with_origin_label(self, batch_size=32):
        """
        Return the sliding windows for train,valid,test splits

        Examples
        --------
            x, train_y, train_origin_y, \
            val_x, val_y, val_origin_y, \
            test_x, test_y, test_origin_y = di.get_sliding_windows_three_splits_with_origin_label()


            - train_y indicates whether the corresponding sliding window is anomaly. If a sliding window contains an
             anomaly data point, it is considered as an anomaly windows.
            - train_origin_y indicates whether **the last datapoint** in the corresponding sliding window is anomaly.

            - train_y, val_y, test_y is post processed by self.post_process_windows_label().
            - train_origin_y, val_origin_y, test_origin_y  are the origin labels which do not been  post-processed.



        Returns
        -------

        """
        train_x, train_y, train_origin_y, \
            val_x, val_y, val_origin_y, \
            test_x, test_y, test_origin_y = self.get_sliding_windows_three_splits_with_origin_label_v2()
        from pylibs.dataset.AIOpsDataset import AIOpsDataset
        train_dataset = AIOpsDataset(train_x, train_origin_y)
        valid_dataset = AIOpsDataset(val_x, val_origin_y)
        test_dataset = AIOpsDataset(test_x, test_origin_y)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_dataloader, valid_dataloader, test_dataloader

    def get_pydl_windows_3splits_with_origin_label_coca(self, batch_size=32):
        """
        Return the sliding windows for train,valid,test splits

        Examples
        --------
            x, train_y, train_origin_y, \
            val_x, val_y, val_origin_y, \
            test_x, test_y, test_origin_y = di.get_sliding_windows_three_splits_with_origin_label()


            - train_y indicates whether the corresponding sliding window is anomaly. If a sliding window contains an
             anomaly data point, it is considered as an anomaly windows.
            - train_origin_y indicates whether **the last datapoint** in the corresponding sliding window is anomaly.

            - train_y, val_y, test_y is post processed by self.post_process_windows_label().
            - train_origin_y, val_origin_y, test_origin_y  are the origin labels which do not been  post-processed.



        Returns
        -------

        """
        train_x, train_y, train_origin_y, \
            val_x, val_y, val_origin_y, \
            test_x, test_y, test_origin_y = self.get_sliding_windows_three_splits_with_origin_label_v2()
        from pylibs.dataset.AIOpsDataset import AIOpsDataset
        train_dataset = AIOpsDataset(np.expand_dims(train_x, 1), train_origin_y)
        valid_dataset = AIOpsDataset(np.expand_dims(val_x, 1), val_origin_y)
        test_dataset = AIOpsDataset(np.expand_dims(test_x, 1), test_origin_y)
        drop_last = False
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last)
        return train_dataloader, valid_dataloader, test_dataloader

    def get_sliding_windows_three_splits_with_origin_label_v2(self, log_data_info=True):
        """
        Return the sliding windows for train,valid,test splits

        Examples
        --------
           train_x, train_y, train_origin_y, \
            val_x, val_y, val_origin_y, \
            test_x, test_y, test_origin_y = di.get_sliding_windows_three_splits_with_origin_label_v2()

            - train_y indicates whether the corresponding sliding window is anomaly. If a sliding window contains an
             anomaly data point, it is considered as an anomaly windows.
            - train_origin_y indicates whether **the last datapoint** in the corresponding sliding window is anomaly.

            - train_y, val_y, test_y is post processed by self.post_process_windows_label().
            - train_origin_y, val_origin_y, test_origin_y  are the origin labels which do not been  post-processed.


        Returns
        -------

        """
        if self._windows_size > self._source_train_data.shape[0] \
                or self._windows_size > self._source_test_data.shape[0]:
            raise ValueError("windows size must be less or equal to the number of sliding_windows_data points")

        # The data generator is followed by the paper:
        # R. Wang et al., “Deep Contrastive One-Class Time Series Anomaly Detection.” arXiv, Oct. 08, 2022.
        # Accessed: Dec. 09, 2022. [Online]. Available: http://arxiv.org/abs/2207.01472

        assert self._source_train_data.shape[0] == self._source_train_label.shape[0]
        assert self._source_test_data.shape[0] == self._source_test_label.shape[0]

        train_x, val_x, train_y, val_y = train_test_split(self._source_train_data,
                                                          self._source_train_label,
                                                          test_size=self._valid_rate,
                                                          shuffle=False)
        test_x, test_y = self._source_test_data, self._source_test_label

        ret_train_x, ret_train_y, ret_train_y_origin = self.__process_train_data_with_sampling(train_x, train_y)
        ret_val_x, ret_val_y, ret_val_y_origin = self.__process_data(val_x, val_y)
        ret_test_x, ret_test_y, ret_test_y_origin = self.__process_data(test_x, test_y)

        return ret_train_x, ret_train_y, ret_train_y_origin, \
            ret_val_x, ret_val_y, ret_val_y_origin, \
            ret_test_x, ret_test_y, ret_test_y_origin

    def log_data_describe(self, log_data_info, ret_train_x, ret_train_y, test_x_sliding_windows_source,
                          test_y_window_processed, train_split_x, train_split_y, train_x_sliding_windows_source,
                          train_y_sliding_windows_source):
        if log_data_info:
            pt = PrettyTable(field_names=["DataType", "x.shape", "y.shape"])
            pt.add_row(
                ["training set", str(train_x_sliding_windows_source.shape), str(train_y_sliding_windows_source.shape)])
            pt.add_row(["sampling set", str(ret_train_x.shape), str(ret_train_y.shape)])
            pt.add_row(["valid set", str(train_split_x.shape), str(train_split_y.shape)])
            pt.add_row(["test set", str(test_x_sliding_windows_source.shape), str(test_y_window_processed.shape)])
            log_metric_msg(f"Data info:\n{pt}")

    def __get_training_dataset_sampling_index(self, train_x_sliding_windows_source, train_y_window_processed):
        # Sampling by the methods
        sampling_windows_index = self._get_sampling_index(train_x_sliding_windows_source.shape[0])
        # normal_windows_index = self._get_normal_windows_index(self._source_train_label, self._source_train_missing)
        normal_windows_index = np.argwhere(train_y_window_processed == 0).reshape(-1)
        if self._is_include_anomaly_windows:
            target_selected_windows_index = sampling_windows_index
        else:
            target_selected_windows_index = np.intersect1d(sampling_windows_index, normal_windows_index)
        return target_selected_windows_index

    def post_process_windows_label(self, test_y_sliding_windows_source, train_y_sliding_windows_source):
        if self._anomaly_window_type == 'all':
            # finding the anomaly windows.  Anomaly windows if a window contains at least one abnormal point,
            # normal window otherwise. For more detail seeing
            # https://uniplore.feishu.cn/wiki/wikcnCDt1FT72TcwAkXOPTao8Ec?sheet=EAgI0Q
            train_y_window_processed, train_anomaly_window_num = self.get_sliding_windows_label(
                train_y_sliding_windows_source)
            test_y_window_processed, _ = self.get_sliding_windows_label(test_y_sliding_windows_source)

        elif self._anomaly_window_type == 'coca':
            # finding the anomaly windows according to  R. Wang et al., “Deep Contrastive One-Class Time Series Anomaly
            # Detection.” arXiv, Oct. 08, 2022. Accessed: Dec. 09, 2022. [Online]. Available:
            # http://arxiv.org/abs/2207.01472. For more details  seeing
            # https://uniplore.feishu.cn/wiki/wikcnCDt1FT72TcwAkXOPTao8Ec?sheet=EAgI0Q
            #
            train_y_window_processed, train_anomaly_window_num = self.get_sliding_windows_label_coca(
                train_y_sliding_windows_source)
            test_y_window_processed, _ = self.get_sliding_windows_label_coca(test_y_sliding_windows_source)
        else:
            raise TypeError(f"Unknown anomaly_window_type={self._anomaly_window_type}")
        return test_y_window_processed, train_y_window_processed

    def __post_process_windows_label(self, label_windows):
        if self._anomaly_window_type == 'all':
            # finding the anomaly windows.  Anomaly windows if a window contains at least one abnormal point,
            # normal window otherwise. For more detail seeing
            # https://uniplore.feishu.cn/wiki/wikcnCDt1FT72TcwAkXOPTao8Ec?sheet=EAgI0Q
            label_processed, train_anomaly_window_num = self.get_sliding_windows_label(
                label_windows)

        elif self._anomaly_window_type == 'coca':
            # finding the anomaly windows according to  R. Wang et al., “Deep Contrastive One-Class Time Series Anomaly
            # Detection.” arXiv, Oct. 08, 2022. Accessed: Dec. 09, 2022. [Online]. Available:
            # http://arxiv.org/abs/2207.01472. For more details  seeing
            # https://uniplore.feishu.cn/wiki/wikcnCDt1FT72TcwAkXOPTao8Ec?sheet=EAgI0Q
            #
            label_processed, train_anomaly_window_num = self.get_sliding_windows_label_coca(
                label_windows)
        else:
            raise TypeError(f"Unknown anomaly_window_type={self._anomaly_window_type}")
        return label_processed

    @DeprecationWarning
    def get_sliding_windows_three_splits_pytorch_dataloader(self, conf):
        """
        train_dataloader, valid_dataloader, test_dataloader=da.get_sliding_windows_three_splits_pytorch_dataloader(conf)

        where label is the sliding window label.

        Parameters
        ----------
        conf :

        Returns
        -------

        """
        train_x, train_y, valid_x, valid_y, test_x, test_y = self.get_sliding_windows_three_splits()
        from pylibs.dataset.AIOpsDataset import AIOpsDataset
        train_dataset = AIOpsDataset(train_x, train_y)
        valid_dataset = AIOpsDataset(valid_x, valid_y)
        test_dataset = AIOpsDataset(test_x, test_y)
        train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=False)
        valid_dataloader = DataLoader(valid_dataset, batch_size=conf.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False)
        return train_dataloader, valid_dataloader, test_dataloader

    def get_sliding_windows_three_splits_with_source_label_pydl(self, conf):
        """
        train_dataloader, valid_dataloader, test_dataloader=da.get_sliding_windows_three_splits_pytorch_dataloader(conf)

        where label is the origin data.

        Parameters
        ----------
        conf :

        Returns
        -------

        """
        train_x, train_y, valid_x, valid_y, test_x, test_y = self.get_sliding_windows_three_splits()
        from pylibs.dataset.AIOpsDataset import AIOpsDataset
        train_dataset = AIOpsDataset(train_x, self.get_source_train_label())
        valid_dataset = AIOpsDataset(valid_x, self.get_source_valid_label())
        test_dataset = AIOpsDataset(test_x, self.get_source_test_label())
        train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=False)
        valid_dataloader = DataLoader(valid_dataset, batch_size=conf.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False)
        return train_dataloader, valid_dataloader, test_dataloader

    def get_three_splits(self):
        """
        Return  train,valid,test splits without sliding window technique.

        Examples
        --------
        x, train_y, valid_x, valid_y, test_x, test_y = da.get_three_splits()
        x: (n_samples,1 feature)
        train_y: (n_samples,), a vector indicating a window is anomaly or not.

        Returns
        -------

        """

        train_x, val_x, train_y, val_y = train_test_split(self._source_train_data,
                                                          self._source_train_label,
                                                          test_size=self._valid_rate,
                                                          shuffle=False)

        return train_x.reshape((-1, 1)), train_y, \
            val_x.reshape((-1, 1)), val_y, \
            self._source_test_data.reshape((-1, 1)), self._source_test_label

    def get_sliding_windows_label_coca(self, train_y):
        """
        Get the anomaly windows according to R. Wang et al., “Deep Contrastive One-Class Time Series Anomaly
        Detection.” arXiv, Oct. 08, 2022. Accessed: Dec. 09, 2022. [Online]. Available: http://arxiv.org/abs/2207.01472

        If the $N$ (step size) data points in the front of the window contain any anomaly, the windows is anomaly.
        Normal otherwise.

        return window_labels, n_anomaly_windows

        Parameters
        ----------
        train_y :

        Returns
        -------

        """
        n_anomaly_windows = 0
        window_labels = np.zeros(train_y.shape[0])
        for i, item in enumerate(train_y[:]):
            if sum(item[:self._time_step]) >= 1:
                n_anomaly_windows += 1
                window_labels[i] = 1
            else:
                window_labels[i] = 0
        return window_labels, n_anomaly_windows

    def get_sliding_windows_label(self, train_y):
        """
        finding the anomaly windows.  Anomaly windows if a window contains at least one abnormal point, normal window
        # otherwise. For more detail seeing
        https://uniplore.feishu.cn/wiki/wikcnCDt1FT72TcwAkXOPTao8Ec?sheet=EAgI0Q[Online].
        Available: http://arxiv.org/abs/2207.01472

        return window_labels, n_anomaly_windows

        Parameters
        ----------
        train_y :

        Returns
        -------

        """
        train_anomaly_window_num = 0
        window_labels = np.zeros(train_y.shape[0])
        for i, item in enumerate(train_y[:]):
            if sum(item) >= 1:
                train_anomaly_window_num += 1
                window_labels[i] = 1
            else:
                window_labels[i] = 0
        return window_labels, train_anomaly_window_num

    @staticmethod
    def save_dataset_to_db_v2(kpi_id, train_data, test_data, fill_missing=False, preprocessing=None):
        """
        Save x,y to three file: all.npz,train.npz,test.npz, the split rate is 8:2 of training set and test set.

        Parameters
        ----------
        preprocessing : Union[None,str]
            Do nothing if None;
            Normalize the data if "normalize"(scale by mean and std);
            Standardize the data if "standardize"(scale to [0,1]).
        test_data : pd.DataFrame
            The test set of AIPs2018
        train_data :  pd.DataFrame
             The training set of AIPs2018
        fill_missing : bool
            Whether to fill missing data points. in many papers, it is not filled the missing data points. seeing details
            here https://uniplore.feishu.cn/wiki/wikcnHWUDOLubE7M0yaarfmrQSb?sheet=zGMCSD
        kpi_id : str,uuid.UUID
            The kpi_id of the m_dataset.

        Returns
        -------
        int
            The number data which inserts to the db.
        """
        # We normalize the sliding_windows_data to [0,1] instead of standardizing the sliding_windows_data since if
        # the sliding_windows_data does not follow normal distribution then this will make problems,
        # see Ali, Peshawa Jamal Muhammad, et al. "Data normalization and standardization: a
        # technical report." Mach Learn Tech Rep 1.1 (2014): 1-6.
        train_data = train_data.copy(deep=True)
        test_data = test_data.copy(deep=True)
        # Check sliding_windows_data
        assert isinstance(train_data, pd.DataFrame), "input train_data must be a DataFrame"
        assert isinstance(test_data, pd.DataFrame), "input test_data must be a DataFrame"
        target_columns = np.intersect1d(train_data.columns, ['timestamp', 'value', 'label'])
        np.testing.assert_equal(np.sort(['timestamp', 'value', 'label']), np.sort(target_columns),
                                err_msg="sliding_windows_data columns must contain ['timestamp', 'value', 'label']")
        target_columns = np.intersect1d(test_data.columns, ['timestamp', 'value', 'label'])
        np.testing.assert_equal(np.sort(['timestamp', 'value', 'label']), np.sort(target_columns),
                                err_msg="sliding_windows_data columns must contain ['timestamp', 'value', 'label']")

        if "KPI ID" in train_data.columns:
            train_data = train_data.drop("KPI ID", axis=1)
        if "KPI ID" in test_data.columns:
            test_data = test_data.drop("KPI ID", axis=1)
        if preprocessing == "normalize":
            # Normalize sliding_windows_data
            # the normalization procedure is following  R. Wang et al., “Deep Contrastive One-Class Time Series
            # Anomaly Detection.” arXiv, Oct. 08, 2022. Accessed: Dec. 09, 2022. [Online]. Available:
            # http://arxiv.org/abs/2207.01472
            all_data = np.r_[train_data['value'].values, test_data['value'].values]
            _scale = all_data.mean()
            _bias = all_data.std()
            UtilSys.is_debug_mode() and log.info(f"Normalize scale: {_scale}, bias: {_bias}")
            train_data['value'] = (train_data['value'] - _scale) / _bias
            test_data['value'] = (test_data['value'] - _scale) / _bias
        elif preprocessing == "standardize":
            # Normalize sliding_windows_data
            # the normalization procedure is following  R. Wang et al., “Deep Contrastive One-Class Time Series
            # Anomaly Detection.” arXiv, Oct. 08, 2022. Accessed: Dec. 09, 2022. [Online]. Available:
            # http://arxiv.org/abs/2207.01472
            all_data = np.r_[train_data['value'].values, test_data['value'].values]
            _max = all_data.max()
            _min = all_data.min()
            train_data['value'] = (train_data['value'] - _min) / (_max - _min)
            test_data['value'] = (test_data['value'] - _min) / (_max - _min)
        elif preprocessing is None:
            pass
        else:
            raise ValueError(f"Unsupported preprocessing method = {preprocessing}")
        # Fill the missing timestamp
        if fill_missing:
            train_data = DatasetAIOps2018.fill_missing_data(train_data)
            test_data = DatasetAIOps2018.fill_missing_data(test_data)
        else:
            train_data['missing'] = 0
            test_data['missing'] = 0
        con = create_pandas_sqlite_connection()
        n_train_ = train_data.to_sql(
            con=con,
            name=DatasetAIOps2018.get_dataset_db_name(kpi_id=kpi_id, data_type=DatasetAIOps2018.DATA_TYPE_TRAIN),
            if_exists="replace",
            index=False
        )
        UtilSys.is_debug_mode() and log.info(
            f"Saving {n_train_} to table {DatasetAIOps2018.get_dataset_db_name(kpi_id=kpi_id, data_type=DatasetAIOps2018.DATA_TYPE_TRAIN)}")

        n_test = test_data.to_sql(
            con=con,
            name=DatasetAIOps2018.get_dataset_db_name(kpi_id=kpi_id, data_type=DatasetAIOps2018.DATA_TYPE_TEST),
            if_exists="replace",
            index=False
        )
        UtilSys.is_debug_mode() and log.info(
            f"Saving {n_test} to table {DatasetAIOps2018.get_dataset_db_name(kpi_id=kpi_id, data_type=DatasetAIOps2018.DATA_TYPE_TEST)}")

        UtilSys.is_debug_mode() and log.info(f"Affect rows {n_train_ + n_test}")
        return n_train_ + n_test

    @staticmethod
    def fill_missing_data(data):
        source_timestamp = data['timestamp']
        assert source_timestamp.dtype.name == 'datetime64[ns]', "timestamp must be a datetime, converting by " \
                                                                "pd.to_datetime(group['timestamp'], unit='s') "
        interval_seconds = source_timestamp.diff().min().seconds
        if interval_seconds == 0:
            raise ValueError("The sliding_windows_data contains duplicated timestamps")
        datatime_index = pd.date_range(start=source_timestamp.min(),
                                       end=source_timestamp.max(),
                                       freq=f"{interval_seconds}S"
                                       )
        target_timeseries = pd.merge(datatime_index.to_frame(index=False, name='timestamp'),
                                     data,
                                     left_on="timestamp",
                                     right_on="timestamp",
                                     how='left')
        # Append the missing indicator
        target_timeseries['missing'] = (target_timeseries['value'].isna()).astype('int')

        # If the sliding_windows_data is missing, fill the labels = 1 and value =1
        #
        target_timeseries['value'] = target_timeseries['value'].fillna(0)
        target_timeseries['label'] = target_timeseries['label'].fillna(1)
        return target_timeseries

    @staticmethod
    def get_dataset_db_name(kpi_id, data_type):
        """
        Get the m_dataset name saved to database
        Parameters
        ----------
        kpi_id : str
            The kpi id of the timeseries
        data_type :
            One of the "train" or 'test'
        Returns
        -------

        """
        return f"{data_type}_{kpi_id}"

    def __process_train_data_with_sampling(self, train_x, train_y):
        """
        Returns sampled x (sliding windows), train_y (labels post processed), train_y_origin (labels origin)

        x,train_y,train_y_origin = self.__process_train_data_with_sampling(x, train_y)

        Parameters
        ----------
        train_x :
        train_y :

        Returns
        -------

        """
        train_x_sliding_windows_source = subsequences(train_x, self._windows_size, self._time_step)
        train_y_sliding_windows_source = subsequences(train_y, self._windows_size, self._time_step)
        train_y_post_processed = self.__post_process_windows_label(train_y_sliding_windows_source)
        train_selected_index = self.__get_training_dataset_sampling_index(train_x_sliding_windows_source,
                                                                          train_y_post_processed)
        return train_x_sliding_windows_source[train_selected_index].astype('float32'), \
            train_y_post_processed[train_selected_index].astype('float32'), \
            train_y_sliding_windows_source[train_selected_index].astype('float32')[:, -1]

    def __process_data(self, x, label):
        """
        Returns train_x (sliding windows), train_y (labels post processed), train_y_origin (labels origin)

        train_x,train_y,train_y_origin = self.__process_train_data_with_sampling(train_x, train_y)

        Parameters
        ----------
        x :
        label :

        Returns
        -------

        """
        train_x_sliding_windows_source = subsequences(x, self._windows_size, self._time_step)
        train_y_sliding_windows_source = subsequences(label, self._windows_size, self._time_step)
        train_y_post_processed = self.__post_process_windows_label(train_y_sliding_windows_source)

        return train_x_sliding_windows_source.astype("float32"), train_y_post_processed.astype("float32"), \
            train_y_sliding_windows_source.astype("float32")[:, -1]
