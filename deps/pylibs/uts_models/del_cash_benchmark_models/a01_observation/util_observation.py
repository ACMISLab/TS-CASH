#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/9/11 10:47
# @Author  : gsunwu@163.com
# @File    : util_observation.py
# @Description:

from pylibs.uts_dataset.dataset_loader import KFoldDatasetLoader


class ObservationsUtil:
    @staticmethod
    def load_observation_data_origin(max_length=10000,
                                     dataset_name="IOPS",
                                     data_id="KPI-a8c06b47-cc41-3738-9110-12df0ee4c721.train.out"):
        dl = KFoldDatasetLoader()
        data = dl.read_data(dataset_name, data_id)
        data = data.iloc[:max_length, :]
        values = data.iloc[:, 0]
        labels = data.iloc[:, 1]
        return values, labels

    @staticmethod
    def load_observation_data_splits(
            dataset_name="IOPS",
            data_id="KPI-a8c06b47-cc41-3738-9110-12df0ee4c721.train.out",
            window_size=100,
            original_labels=True,
            sample_rate=-1,
            max_length=None):
        """
        load train_x, train_y, test_x, test_y = xx

        Parameters
        ----------
        max_length :
        dataset_name :
        data_id :
        window_size :

        Returns
        -------

        """
        dl = KFoldDatasetLoader(dataset_name,
                                data_id=data_id,
                                window_size=window_size,
                                max_length=max_length,
                                sample_rate=sample_rate,
                                fold_index=-1,
                                anomaly_window_type="all")
        return dl.get_kfold_sliding_windows_train_and_test_from_fold_index(original_labels=original_labels)

    @staticmethod
    def load_observation_data_splits_v2(
            dataset_name="IOPS",
            data_id="KPI-a8c06b47-cc41-3738-9110-12df0ee4c721.train.out",
            window_size=100,
            sample_rate=-1,
            fold_index=-1,
            max_length=None):
        """
        load train_x, train_y, test_x, test_y = xx

        Parameters
        ----------
        sample_rate :
        max_length :
        dataset_name :
        data_id :
        window_size :

        Returns
        -------

        """
        dl = KFoldDatasetLoader(dataset_name,
                                data_id=data_id,
                                window_size=window_size,
                                max_length=max_length,
                                sample_rate=sample_rate,
                                fold_index=fold_index,
                                anomaly_window_type="all")
        return dl.get_kfold_sliding_windows_train_and_test_from_fold_index_with_original_label()
