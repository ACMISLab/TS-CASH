import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal
from pylibs.common import SampleType
from pylibs.experiments.exp_config import ExpConf
from pylibs.utils.util_log import get_logger
from pylibs.utils.util_numpy import enable_numpy_reproduce
from pylibs.utils.util_system import UtilSys
from pylibs.uts_dataset.dataset_loader import DatasetLoader, DatasetType, KFoldDatasetLoader, UTSDataset

log = get_logger()


class TestDatasetLoader(unittest.TestCase):

    def test_data_sampling(self):
        dl = DatasetLoader(DatasetType.IOPS, 1, sample_rate=1)
        ret_train_x, ret_train_y = dl.get_sliding_windows()
        assert ret_train_x.shape[0] == ret_train_y.shape[0]

    def test_data_sampling_half(self):
        dl = DatasetLoader(DatasetType.IOPS, 1, sample_rate=0.5)
        ret_train_x, ret_train_y = dl.get_sliding_windows()
        UtilSys.is_debug_mode() and log.info(
            f"ret_train_x.shape={ret_train_x.shape},ret_train_y.shape={ret_train_y.shape},")
        assert ret_train_x.shape[0] == ret_train_y.shape[0]

    def test_data_sampling_2(self):
        dl = DatasetLoader(DatasetType.IOPS, 1, sample_rate=2)
        ret_train_x, ret_train_y = dl.get_sliding_windows()
        UtilSys.is_debug_mode() and log.info(
            f"ret_train_x.shape={ret_train_x.shape},ret_train_y.shape={ret_train_y.shape},")
        assert ret_train_x.shape[0] == ret_train_y.shape[0]
        assert ret_train_x.shape[0] == 2

    def test_load_dodgers(self):
        dl = DatasetLoader(DatasetType.DODGER, 0, sample_rate=2)
        df = dl.get_source_data()
        assert df.shape[0] == 50400

    def test_show_kpi(self):
        dl = DatasetLoader(DatasetType.NAB, "NAB_data_art0_0.out", sample_rate=2)
        df = dl.get_source_data()
        print(df)

    def test_show_all(self):
        dl = DatasetLoader(DatasetType.NAB)
        ids = dl.get_data_ids()
        for dataset_name, data_id in ids:
            print(dataset_name)

    def test_data_loader_sample(self):
        data_id = "NAB_data_CloudWatch_5.out"
        dl = DatasetLoader("NAB", data_id, window_size=32, is_include_anomaly_window=False, sample_rate=-1)
        train_x, train_y, origin_train_x, origin_train_y = dl.get_sliding_windows(return_origin=True)
        assert train_y.sum() == 0

    def test_data_loader_sample1(self):
        enable_numpy_reproduce(0)
        data_id = "NAB_data_CloudWatch_5.out"
        dl = DatasetLoader("NAB", data_id, window_size=32, is_include_anomaly_window=False, sample_rate=2)
        train_x, train_y, origin_train_x, origin_train_y = dl.get_sliding_windows(return_origin=True)
        assert train_x.shape[0] == 2

        data_id = "NAB_data_CloudWatch_5.out"
        dl = DatasetLoader("NAB", data_id, window_size=32, is_include_anomaly_window=True, sample_rate=1)
        train_x, train_y, origin_train_x, origin_train_y = dl.get_sliding_windows(return_origin=True)
        assert train_x.shape[0] == 1

    def test_data_sample(self):
        # 0.001_1_1000.out
        window_size = 10
        dl = DatasetLoader("DEBUG",
                           "1_1000_1000.out",
                           window_size=window_size,
                           is_include_anomaly_window=False,
                           sample_rate=1,
                           processing=False)
        x, y = dl.get_sliding_windows()
        assert x.shape[0] == 1
        assert y.sum() == 0

    def test_data_sample2(self):
        # 0.001_1_1000.out
        window_size = 10
        dl = DatasetLoader("DEBUG",
                           "1_1000_1000.out",
                           window_size=window_size,
                           is_include_anomaly_window=False,
                           sample_rate=-1,
                           processing="normal")
        x, y, origin_train_x, origin_train_y = dl.get_sliding_windows(return_origin=True)

        assert x.max() == 1
        assert x.min() == 0
        assert origin_train_x.max() == 1
        assert origin_train_y.max() == 0
        assert y.max() == 0
        assert x.mean() == 0.5

    def test_data_sample3(self):
        # 0.001_1_1000.out
        window_size = 10
        dl = DatasetLoader("DEBUG",
                           "1_1000_1000.out",
                           window_size=window_size,
                           is_include_anomaly_window=False,
                           sample_rate=-1,
                           processing="standardization")
        x, y, origin_train_x, origin_train_y = dl.get_sliding_windows(return_origin=True)

        assert np.round(x.std()) == 1
        assert np.round(x.mean()) == 0
        assert np.round(origin_train_x.std()) == 1
        assert np.round(origin_train_x.mean()) == 0
        print(x.std(), x.mean())

    def test_data_sample4(self):
        # 0.001_1_1000.out
        window_size = 10
        dl = DatasetLoader("DEBUG",
                           "1_1000_1000.out",
                           window_size=window_size,
                           is_include_anomaly_window=False,
                           sample_rate=0.1,
                           processing="standardization")
        x, y, origin_train_x, origin_train_y = dl.get_sliding_windows(return_origin=True)

        assert x.shape[0] == 99

    def test_aa(self):
        dl = DatasetLoader(data_set="ECG", data_id=0)
        print(dl.get_anomaly_rate())
        assert dl.get_anomaly_rate() == 0.0345

    def test_sampling_method(self):
        dl = DatasetLoader("SVDB", "856.test.csv@2.out", data_sampling_method=SampleType.NORMAL_RANDOM, sample_rate=1)
        labels = np.asarray([0, 1, 0, 0, 1])
        assert_almost_equal(dl._get_sampling_index(5, labels), [0, 1, 2, 3, 4])

    def test_sampling_method2(self):
        enable_numpy_reproduce(0)
        labels = np.asarray([0, 1, 0, 0, 1])
        dl = DatasetLoader("SVDB", "856.test.csv@2.out", data_sampling_method=SampleType.RANDOM, sample_rate=0.2)
        assert_almost_equal(dl._get_sampling_index(5, labels), [2])

        dl = DatasetLoader("SVDB", "856.test.csv@2.out", data_sampling_method=SampleType.NORMAL_RANDOM, sample_rate=0.2)
        assert_almost_equal(dl._get_sampling_index(5, labels), [0, 1, 4])

    def test_sampling_debug(self):
        enable_numpy_reproduce(0)
        dl = DatasetLoader("DEBUG", "1_10_10.out", data_sampling_method=SampleType.RANDOM,
                           sample_rate=2,
                           window_size=2,
                           processing=False)

        train_x, train_y, origin_train_x, origin_train_y = dl.get_sliding_windows(return_origin=True)
        assert_almost_equal(train_x, [
            [8., 9.],
            [3., 4.],
        ])

        assert_almost_equal(origin_train_x, [[1., 2.], [2., 3.],
                                             [3., 4],
                                             [4., 5.],
                                             [5., 6.],
                                             [6., 7.],
                                             [7., 8.],
                                             [8., 9.],
                                             [9., 10.]]
                            )

    def test_sampling_debug1(self):
        enable_numpy_reproduce(0)
        dl = DatasetLoader("DEBUG", "1_10_10.out", data_sampling_method=SampleType.RANDOM,
                           sample_rate=2,
                           window_size=2,
                           is_include_anomaly_window=False,
                           processing=False)

        train_x, train_y, origin_train_x, origin_train_y = dl.get_sliding_windows(return_origin=True)
        assert_almost_equal(train_x, [[3., 4.]])

        assert_almost_equal(origin_train_x, [[1., 2.], [2., 3.],
                                             [3., 4],
                                             [4., 5.],
                                             [5., 6.],
                                             [6., 7.],
                                             [7., 8.],
                                             [8., 9.],
                                             [9., 10.]]
                            )

    def test_sampling_debug4(self):
        enable_numpy_reproduce(0)
        dl = DatasetLoader("DEBUG", "1_10_10.out",
                           data_sampling_method=SampleType.RANDOM,
                           sample_rate=0.2,
                           window_size=2,
                           processing=False)

        train_x, train_y, origin_train_x, origin_train_y = dl.get_sliding_windows(return_origin=True)
        assert_almost_equal(train_x, [[8., 9.]])

        assert_almost_equal(origin_train_x, [[1., 2.],
                                             [2., 3.],
                                             [3., 4],
                                             [4., 5.],
                                             [5., 6.],
                                             [6., 7.],
                                             [7., 8.],
                                             [8., 9.],
                                             [9., 10.]]
                            )

    def test_sampling_debug5(self):
        enable_numpy_reproduce(0)
        dl = DatasetLoader("DEBUG", "1_10_10.out",
                           data_sampling_method=SampleType.NORMAL_RANDOM,
                           sample_rate=0.1,
                           window_size=1,
                           processing=False)

        train_x, train_y, origin_train_x, origin_train_y = dl.get_sliding_windows(return_origin=True)
        assert_almost_equal(train_x, [
            [6],
            [7.],
            [8.],
            [9.],
            [10.]])

    def test_sampling_debug6(self):
        enable_numpy_reproduce(0)
        dl = DatasetLoader("DEBUG", "1_10_10.out",
                           data_sampling_method=SampleType.NORMAL_RANDOM,
                           sample_rate=0,
                           window_size=2,
                           processing=False)

        train_x, train_y, origin_train_x, origin_train_y = dl.get_sliding_windows(return_origin=True)
        assert_almost_equal(train_x, [[0, 0]])
        dl = DatasetLoader("DEBUG", "1_10_10.out",
                           data_sampling_method=SampleType.NORMAL_RANDOM,
                           sample_rate=0,
                           window_size=3,
                           processing=False)

        train_x, train_y, origin_train_x, origin_train_y = dl.get_sliding_windows(return_origin=True)
        assert_almost_equal(train_x, [[0, 0, 0]])

    def test_sampling(self):
        enable_numpy_reproduce()
        dl = DatasetLoader("DEBUG", "1_10_10.out",
                           data_sampling_method=SampleType.RANDOM,
                           sample_rate=0.2,
                           window_size=1,
                           processing=False)
        train_x, train_y, origin_train_x, origin_train_y = dl.get_sliding_windows(return_origin=True)
        assert_almost_equal(train_x, [[9], [2]])

    def test_sampling01(self):
        enable_numpy_reproduce()
        dl = DatasetLoader("DEBUG", "1_10_10.out",
                           data_sampling_method=SampleType.NORMAL_RANDOM,
                           sample_rate=0.2,
                           window_size=1,
                           processing=False)
        train_x, train_y, origin_train_x, origin_train_y = dl.get_sliding_windows(return_origin=True)
        assert_almost_equal(train_x, [[2], [6], [7], [8], [9], [10]])

    def test_sampling02(self):
        enable_numpy_reproduce()
        dl = DatasetLoader("DEBUG", "1_10_10.out",
                           data_sampling_method=SampleType.STRATIFIED,
                           sample_rate=0.2,
                           window_size=1,
                           processing=False)
        train_x, train_y, origin_train_x, origin_train_y = dl.get_sliding_windows(return_origin=True)
        assert_almost_equal(train_x, [[2], [9]])

    def test_sampling04(self):
        enable_numpy_reproduce()
        dl = DatasetLoader("DEBUG", "1_10_10.out",
                           data_sampling_method=SampleType.STRATIFIED,
                           sample_rate=-1,
                           window_size=1,
                           processing=False)
        train_x, train_y, origin_train_x, origin_train_y = dl.get_sliding_windows(return_origin=True)
        assert_almost_equal(train_x, origin_train_x)

    def test_sampling05(self):
        enable_numpy_reproduce()
        dl = DatasetLoader("DEBUG", "1_10_10.out",
                           data_sampling_method=SampleType.STRATIFIED,
                           sample_rate=2,
                           window_size=1,
                           processing=False)
        train_x, train_y, origin_train_x, origin_train_y = dl.get_sliding_windows(return_origin=True)
        assert_almost_equal(train_x.shape[0], 2)

    def test_sampling06(self):
        enable_numpy_reproduce()
        dl = DatasetLoader("DEBUG", "1_10_10.out",
                           data_sampling_method=SampleType.STRATIFIED,
                           sample_rate=0,
                           window_size=1,
                           processing=False)
        train_x, train_y, origin_train_x, origin_train_y = dl.get_sliding_windows(return_origin=True)
        assert_almost_equal(train_x, [[0]])

    def test_data_split(self):
        dl = DatasetLoader("DEBUG", "1_10_10.out",
                           data_sampling_method=SampleType.STRATIFIED,
                           sample_rate=-1,
                           window_size=1,
                           test_rate=0.3,
                           processing=False)

        train_x, train_y, test_x, test_y = dl.get_sliding_windows_train_and_test()
        assert_almost_equal(train_x, np.asarray([1, 2, 3, 4, 5, 6, 7]).reshape((-1, 1)))
        assert_almost_equal(test_x, np.asarray([8, 9, 10]).reshape((-1, 1)))
        assert_almost_equal(test_y, np.asarray([1, 1, 1]))
        assert_almost_equal(train_y, np.asarray([0, 0, 0, 0, 0, 1, 1]))

    def test_five_split_include_anomaly(self):

        df = pd.DataFrame({
            "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "label": [0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
        })

        dl = KFoldDatasetLoader(dataset_name="NAB",
                                data_id="NAB_data_art0_0.out",
                                test_rate=0.2,
                                is_include_anomaly_window=True,
                                anomaly_window_type="coca",
                                window_size=2,
                                k_fold=5,
                                df=df
                                )
        # [1, 2], [2, 3],[3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10]]
        for i, train_x, train_y, test_x, test_y in dl.get_kfold_sliding_windows_train_and_test():

            if i == 0:
                assert_almost_equal(train_x, [[3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10]])
                assert_almost_equal(test_x, [[1, 2], [2, 3]])

                assert_almost_equal(train_y, [0, 0, 0, 1, 0, 0, 1])
                assert_almost_equal(test_y, [0, 0])
            if i == 4:
                assert_almost_equal(train_x, [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]])
                assert_almost_equal(test_x, [[9, 10]])

                assert_almost_equal(train_y, [0, 0, 0, 0, 0, 1, 0, 0])
                assert_almost_equal(test_y, [1])
            print(i, train_x.sum(), train_y.sum(), test_x.sum(), test_y.sum())
            print(i, train_x.shape, train_y.shape, test_x.shape, test_y.shape)
            print()

    def test_five_exclude_anomaly(self):

        df = pd.DataFrame({
            "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "label": [0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
        })

        dl = KFoldDatasetLoader(dataset_name="NAB",
                                data_id="NAB_data_art0_0.out",
                                test_rate=0.2,
                                is_include_anomaly_window=False,
                                anomaly_window_type="coca",
                                window_size=2,
                                k_fold=5,
                                df=df
                                )
        # [1, 2], [2, 3],[3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10]]
        for i, train_x, train_y, test_x, test_y in dl.get_kfold_sliding_windows_train_and_test():

            if i == 0:
                assert_almost_equal(train_x, [[3, 4], [4, 5], [5, 6], [7, 8], [8, 9]])
                assert_almost_equal(test_x, [[1, 2], [2, 3]])

                assert_almost_equal(train_y, [0, 0, 0, 0, 0])
                assert_almost_equal(test_y, [0, 0])
            if i == 4:
                assert_almost_equal(train_x, [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [7, 8], [8, 9]])
                assert_almost_equal(test_x, [[9, 10]])

                assert_almost_equal(train_y, [0, 0, 0, 0, 0, 0, 0])
                assert_almost_equal(test_y, [1])
            print(i, train_x.sum(), train_y.sum(), test_x.sum(), test_y.sum())
            print(i, train_x.shape, train_y.shape, test_x.shape, test_y.shape)
            print()

    def test_sample_rate(self):

        enable_numpy_reproduce()
        df = pd.DataFrame({
            "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "label": [0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
        })

        dl = KFoldDatasetLoader(dataset_name="NAB",
                                data_id="NAB_data_art0_0.out",
                                test_rate=0.2,
                                sample_rate=0.5,
                                is_include_anomaly_window=False,
                                anomaly_window_type="coca",
                                window_size=2,
                                k_fold=5,
                                df=df
                                )
        # [1, 2], [2, 3],[3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10]]
        for i, train_x, train_y, test_x, test_y in dl.get_kfold_sliding_windows_train_and_test():

            if i == 0:
                assert_almost_equal(train_x, [[3, 4], [4, 5], [8, 9]])

            if i == 4:
                assert_almost_equal(train_x, [[3, 4], [2, 3], [7, 8]])

    def test_sample_method_lhs(self):

        enable_numpy_reproduce()
        df = pd.DataFrame({
            "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "label": [0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
        })

        dl = KFoldDatasetLoader(
            sample_rate=100,
            is_include_anomaly_window=True,
            anomaly_window_type="coca",
            window_size=2,
            k_fold=5,
            data_sampling_method="lhs",
            df=df
        )
        train_x, train_y, test_x, test_y = dl.get_kfold_sliding_windows_train_and_test_by_fold_index(0)
        print("train_x\n", train_x, "\ntest_x\n", test_x)

        enable_numpy_reproduce()
        df = pd.DataFrame({
            "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "label": [0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
        })

        dl = KFoldDatasetLoader(
            sample_rate=0,
            is_include_anomaly_window=True,
            anomaly_window_type="coca",
            window_size=2,
            k_fold=5,
            data_sampling_method="lhs",
            df=df
        )
        train_x, train_y, test_x, test_y = dl.get_kfold_sliding_windows_train_and_test_by_fold_index(0)
        print("train_x\n", train_x, "\ntest_x\n", test_x)

    def test_sample_method_lhs_v2(self):
        enable_numpy_reproduce()
        label = np.zeros((100,))
        label[50] = 1
        df = pd.DataFrame({
            "value": np.arange(1, 101),
            "label": label
        })

        dl = KFoldDatasetLoader(
            sample_rate=0.05,
            is_include_anomaly_window=True,
            anomaly_window_type="coca",
            window_size=1,
            k_fold=5,
            data_sampling_method="lhs",
            df=df
        )
        train_x, train_y, test_x, test_y = dl.get_kfold_sliding_windows_train_and_test_by_fold_index(-1)
        print("train_x\n", train_x, "\ntest_x\n", test_x)
        assert_almost_equal(train_x, [[7], [40], [55], [71
                                                        ]])

    def test_sample_dist1_v1(self):
        enable_numpy_reproduce()
        df = pd.DataFrame({
            "value": [-1, 4, 5, 6, 10, 20],
            "label": [1, 0, 0, 0, 0, 1]
        })

        dl = KFoldDatasetLoader(
            sample_rate=0.8,
            is_include_anomaly_window=True,
            anomaly_window_type="coca",
            window_size=1,
            k_fold=5,
            data_sampling_method="dist1",
            df=df
        )
        train_x, train_y, test_x, test_y = dl.get_kfold_sliding_windows_train_and_test_by_fold_index(-1)
        print("train_x\n", train_x, "\ntest_x\n", test_x)

    def test_sample_rete_v3(self):
        enable_numpy_reproduce()
        df = pd.DataFrame({
            "value": [-1, 4, 5, 6, 10, 20],
            "label": [1, 0, 0, 0, 0, 1]
        })

        dl = KFoldDatasetLoader(
            sample_rate=9999999999999,
            is_include_anomaly_window=True,
            anomaly_window_type="coca",
            window_size=1,
            k_fold=5,
            data_sampling_method="dist1",
            df=df
        )
        train_x, train_y, test_x, test_y = dl.get_kfold_sliding_windows_train_and_test_by_fold_index(-1)
        print("train_x\n", train_x, "\ntest_x\n", test_x)

    def test_sample_rete_v4(self):
        enable_numpy_reproduce(0)
        dl = KFoldDatasetLoader(is_include_anomaly_window=True, fold_index=1, sample_rate=0.5, dataset_name="ECG",
                                data_id="MBA_ECG820_data.out", window_size=1)
        original_train_x, original_train_y, dirty_sample_x, dirty_sample_y, clean_sample_x, clean_sample_y, test_x, test_y = dl._load_kfold_data()
        assert original_train_x.shape[0] + test_x.shape[0] == 230400
        ori_size = original_train_x.shape[0]
        assert dirty_sample_x.shape[0] * 2 in [ori_size - 1, ori_size, ori_size + 1]

    def test_sample_rete_v5(self):
        enable_numpy_reproduce(0)
        dl = KFoldDatasetLoader(is_include_anomaly_window=True, fold_index=1, sample_rate=0.5, dataset_name="ECG",
                                data_id="MBA_ECG820_data.out", window_size=1)
        dirty_train_sample_x, dirty_train_sample_y, clean_train_sample_x, clean_train_sample_y, test_x, test_y = dl.get_kfold_data_for_lstm_cnn()

        assert dirty_train_sample_y.sum() == dirty_train_sample_y.shape[0]
        assert clean_train_sample_y.sum() == 0
        assert dirty_train_sample_x.shape[0] == dirty_train_sample_y.shape[0]
        assert clean_train_sample_x.shape[0] == clean_train_sample_y.shape[0]
        assert test_x.shape[0] == test_x.shape[0]

    def test_sample_rete_v6(self):
        enable_numpy_reproduce(0)

        # semi-supervised
        ec = ExpConf(model_name="vae",
                     dataset_name='IOPS',
                     data_id="KPI-0efb375b-b902-3661-ab23-9a0bb799f4e3.test.out")
        train_x, train_y, test_x, test_y = ec.load_csv()
        assert train_y.sum() == 0

        # unsupervised
        ec = ExpConf(model_name="lof",
                     dataset_name='IOPS',
                     data_id="KPI-0efb375b-b902-3661-ab23-9a0bb799f4e3.test.out")
        train_x, train_y, test_x, test_y = ec.load_csv()
        assert train_y.sum() > 0
        
    def test_load_kfold_dataset(self):
        from pylibs.uts_dataset.dataset_loader import UTSDataset, KFoldDatasetLoader, DataProcessingType
        datasets = UTSDataset.select_datasets(top_n=None, dataset_names=None)
        test_data = datasets[0]
        kf = KFoldDatasetLoader(
            dataset_name=test_data[0],
            data_id=test_data[1],
            processing=DataProcessingType.DISABLE,
            test_rate=0.3)
        train_x, train_y, test_x, test_y = kf.get_sliding_windows_train_and_test()
        print(train_x)

    def test_kfold_dataset_select(self):
        res = UTSDataset.select_datasets(["IOPS"], top_n=2)
        assert len(res) == 2

    def test_kfold_dataset_select2(self):
        res = UTSDataset.select_datasets(["MGAB", "OPPORTUNITY", "Daphnet"], top_n=5)
        assert len(res) == 15

    def test_kfold_dataset_select3(self):
        try:
            res = UTSDataset.select_datasets(["A"], top_n=5)
            assert False
        except:
            assert True

    def test_load_consider_data(self):
        print(UTSDataset.get_considered_good_uts())