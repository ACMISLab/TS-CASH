import os
import time
import traceback
import warnings
from pathlib import Path
from typing import Union

import distributed
import numpy as np
import pandas as pd
import rich
from sklearn.model_selection import KFold
from tqdm import tqdm

from pylibs.utils.util_joblib import cache_
from pylibs.common import SampleType
from pylibs.exp_ana_common.ExcelOutKeys import EK
from pylibs.utils.util_file import FileUtils, FileUtil
from pylibs.utils.util_joblib import JLUtil
from pylibs.utils.util_message import logw
from pylibs.utils.util_numpy import enable_numpy_reproduce, feature_
from pylibs.utils.util_pandas import PDUtil
from pylibs.utils.util_sampling import latin_hypercube_sampling
from pylibs.utils.util_system import UtilSys
from pylibs.utils.util_univariate_time_series import subsequences
from pylibs.utils.utils import convert_str_to_float
import logging
from pylibs.uts_dataset.util_const import K
from pylibs.uts_metrics.vus.uts_metric_helper import UTSMetricHelper, UTSMetrics

log = logging.getLogger(__name__)
from dask.distributed import print



@cache_
def _select_data(dataset_names, test_ratio, top_n):
    da = get_all_available_datasets(max_size_mb=20)
    _df = pd.DataFrame(da, columns=["dataset_name", "data_id"])
    # 过滤所需的数据集
    _filter_df = _df[_df['dataset_name'].isin(dataset_names)]
    # 判断y中是否同时包括0和1
    tmp_datasets = []
    for dataset_name, data_id in tqdm(_filter_df.values, desc="selecting dataset"):
        kf = KFoldDatasetLoader(dataset_name=dataset_name,
                                data_id=data_id,
                                test_rate=test_ratio)
        train_x, train_y, test_x, test_y = kf.load_train_and_test()
        if KFDL.is_test_y_valid(test_y):
            tmp_datasets.append({
                K.DATASET_NAME: dataset_name,
                K.DATA_ID: data_id,
            })
    # 选取前n个数据集
    final_df = pd.DataFrame(tmp_datasets)
    ret_arr = []
    for _i, _d in final_df.groupby(by=K.DATASET_NAME):
        _d1 = _d[[K.DATASET_NAME, K.DATA_ID]]
        _d2 = _d1.values.tolist()
        np.random.shuffle(_d2)
        ret_arr += _d2[:top_n]
    return ret_arr


class UTSDataset:
    # 'Daphnet', 'ECG', 'GHL', 'IOPS', 'KDD21', 'MGAB', 'MITDB', 'NAB', 'NASA-MSL', 'NASA-SMAP','OPPORTUNITY', 'Occupancy', 'SMD', 'SVDB', 'SensorScope', 'YAHOO', "DEBUG"
    DATASET_YAHOO: str = "YAHOO"
    DATASET_NASA_MSL: str = "NASA-MSL"
    DATASET_NASA_SMAP: str = "NASA-SMAP"
    DATASET_SMD: str = "SMD"
    # YAHOO SMD NAB
    """
    | dataset | avg_anomaly_rate | avg_length |
    | ------- | ---------------- | ---------- |
    | YAHOO   | 0.45             | 1543       |
    | SMD     | 3.52             | 25365      |
    | NAB     | 9.82             | 7384       |
    """
    SELECTED_DATASETS = ["YAHOO",
                         "NAB",
                         "SMD",
                         "IOPS",
                         ]
    ALL_DATASETS = ['Daphnet', 'ECG', 'GHL', 'IOPS', 'KDD21', 'MGAB', 'MITDB', 'NAB', 'NASA-MSL', 'NASA-SMAP',
                    'OPPORTUNITY', 'Occupancy', 'SMD', 'SVDB', 'SensorScope', 'YAHOO', "DEBUG"]

    @staticmethod
    def select_datasets_split(*, dataset_names=None, top_n=None, seed=42, test_ratio=0.3):
        """
        返回测试集中同时包含0和1的数据集.


        Parameters
        ----------
        test_ratio :
        dataset_names :
            e.g. ["IOPS","NAB"]
        top_n :
            the number of UTS in each dataset
        seed :

        Returns
        -------

        """
        enable_numpy_reproduce(seed)
        assert top_n > 0, "Top N must be greater than 0"
        _ALL_DATASETS = [
            'Daphnet', 'ECG', 'GHL', 'IOPS',
            'KDD21', 'MGAB', 'MITDB', 'NAB',
            'NASA-MSL', 'NASA-SMAP', 'OPPORTUNITY',
            'Occupancy', 'SMD', 'SVDB', 'SensorScope',
            'YAHOO', "DEBUG", "DEBUG1", "DEBUG2"
        ]
        if dataset_names is None:
            dataset_names = _ALL_DATASETS

        # check if dataset_name is existing
        for _d_name in dataset_names:
            assert _d_name in _ALL_DATASETS, f"不支持的数据集名称: [{_d_name}]"
        da = get_all_available_datasets(max_size_mb=20)
        _df = pd.DataFrame(da, columns=["dataset_name", "data_id"])
        # 过滤所需的数据集
        _filter_df = _df[_df['dataset_name'].isin(dataset_names)]
        # 判断y中是否同时包括0和1
        tmp_datasets = []
        for _group_dataset_name, _item in _filter_df.groupby(K.DATASET_NAME):
            _one_datasets = _item.values.tolist()
            _t_selected_datasets = UTSDataset._select_datasets(_one_datasets, test_ratio, top_n)
            tmp_datasets.extend(_t_selected_datasets)
        return tmp_datasets

    @staticmethod
    def _select_datasets(_datasets, test_ratio, top_n):
        np.random.shuffle(_datasets)
        _t_selected_datasets = []
        for _dataset_name, _data_id in _datasets:
            kf = KFoldDatasetLoader(dataset_name=_dataset_name,
                                    data_id=_data_id,
                                    test_rate=test_ratio)
            train_x, train_y, test_x, test_y = kf.load_train_and_test()
            if KFDL.is_test_y_valid(test_y):
                # {
                #                     K.DATASET_NAME: _dataset_name,
                #                     K.DATA_ID: _data_id,
                #                 }
                _t_selected_datasets.append([_dataset_name, _data_id])
            if len(_t_selected_datasets) >= top_n:
                return _t_selected_datasets
        return _t_selected_datasets

    @staticmethod
    def get_best_sample_rate(dataset_name, data_id):
        best_sample_file = Path(KFoldDatasetLoader.get_dataset_home(), "best_sample_rate.csv")
        # dataset_name, data_id, repeat_run, best_sample_rate
        # NASA - MSL, D - 16.
        # test.out, 4.5, 0.15

        df = pd.read_csv(best_sample_file.absolute())
        best_sample_file_df = df[(df['dataset_name'] == dataset_name) & (df['data_id'] == data_id)]
        best_sample_rete = best_sample_file_df['best_sample_rate'].values[0]
        log.debug({
            "dataset_name": dataset_name,
            "data_id": data_id,
            "best_sample": best_sample_rete
        })
        assert best_sample_rete is not None, "best_sample_rate is not found"
        return float(best_sample_rete)


def get_data_describe(train_x, **kwargs):
    kwargs.update({
        "mean": np.mean(train_x),
        "std": np.std(train_x),
        "count": train_x.shape[0],
    })
    return kwargs


def get_sliding_windows_label_all(train_y):
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


def get_sliding_windows_label_coca(train_y, time_step):
    """
    Get the anomaly windows, inspired by R. Wang et al., “Deep Contrastive One-Class Time Series Anomaly
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
        if sum(item[-time_step:]) >= 1:
            n_anomaly_windows += 1
            window_labels[i] = 1
        else:
            window_labels[i] = 0
    return window_labels, n_anomaly_windows


# def _load_data(dataset_home, dataset_name, data_id, max_length):
#     file = f"{dataset_home}/{dataset_name}/{data_id}"
#     print(file)
#     assert os.path.isfile(file), f"Not a file: {file}"
#     data = pd.read_csv(file, header=None, names=["value", "label"])
#     if max_length != -1:
#         warnings.warn(
#             f"Max length is specified, only take the first {max_length} datapoints.")
#         data = data[:max_length]
#     return data


def post_process_sliding_window_label(label_sliding_window, anomaly_window_type="all", time_step=None):
    if anomaly_window_type == AnomalyWindowType.ALL:
        # finding the anomaly windows.  Anomaly windows if a window contains at least one abnormal point,
        # normal window otherwise. For more detail seeing
        # https://uniplore.feishu.cn/wiki/wikcnCDt1FT72TcwAkXOPTao8Ec?sheet=EAgI0Q
        UtilSys.is_debug_mode() and log.info("Anomaly windows type: all")
        train_y_window_processed, train_anomaly_window_num = get_sliding_windows_label_all(label_sliding_window)
        return train_y_window_processed
    elif anomaly_window_type == AnomalyWindowType.COCA:
        # finding the anomaly windows, inspired by R. Wang et al., “Deep Contrastive One-Class Time Series Anomaly
        # Detection.” arXiv, Oct. 08, 2022. Accessed: Dec. 09, 2022. [Online]. Available:
        # http://arxiv.org/abs/2207.01472. For more details  seeing
        # https://uniplore.feishu.cn/wiki/wikcnCDt1FT72TcwAkXOPTao8Ec?sheet=EAgI0Q
        #
        log.debug("Anomaly windows type: coca")
        assert time_step is not None, "Time step must not be None when anomaly_window_type is COCA"
        train_y_window_processed, train_anomaly_window_num = get_sliding_windows_label_coca(
            label_sliding_window, time_step)
        return train_y_window_processed
    else:
        raise TypeError(f"Unknown anomaly_window_type={anomaly_window_type}")


class BenchmarkBins:
    bins = {
        "<=5%": ["MGAB", "YAHOO", "IOPS", "SMD", "OPPORTUNITY"],
        "5%~10%": ["ECG", "NAB"],
        "10%~15%": ["MITDB", "Daphnet", "SVDB"],
        ">15%": ["NASA-MSL", "SensorScope", "NASA-SMAP", "Occupancy"]
    }


class DataItem:
    def __init__(self, dataset_name, data_id):
        self.dataset_name = dataset_name
        self.data_id = data_id

    def get_dataset_name(self):
        return self.dataset_name

    def get_data_id(self):
        return self.data_id

    def __str__(self):
        return str([self.dataset_name, self.data_id])


class DatasetAnomalyRate:
    _ar = None
    _HOME = os.path.dirname(__file__)
    _FILE_NAME = "benchmark_dataset_anomaly_rate.csv"

    @staticmethod
    def generate_anomaly_rate_file():
        fu = FileUtil()
        files = fu.get_all_files(os.path.join(DatasetAnomalyRate._HOME, "benchmark"), ext=".out")
        _output = []
        for file in files:
            file = file.split("benchmark/")[-1]
            dataset, data_id = file.split("/")
            dl = DatasetLoader(dataset, data_id)
            _output.append([dataset, data_id, dl.get_anomaly_rate()])

        pd.DataFrame(_output, columns=["dataset", "data_id", "anomaly_rate"]).to_csv(
            os.path.join(DatasetAnomalyRate._HOME, DatasetAnomalyRate._FILE_NAME),
            index=False,
        )

    @classmethod
    def get_anomaly_rate(cls, dataset: str, data_id: str) -> float:
        """
        Given the dataset name and data_id, return the anomaly rate.

        Parameters
        ----------
        dataset :
        data_id :

        Returns
        -------

        """
        if cls._ar is None:
            file_anomaly_rate = os.path.join(DatasetAnomalyRate._HOME, DatasetAnomalyRate._FILE_NAME)
            if not os.path.exists(file_anomaly_rate):
                cls.generate_anomaly_rate_file()

            cls._ar = pd.read_csv(file_anomaly_rate)

        res = cls._ar[cls._ar['dataset'] == dataset][cls._ar['data_id'] == data_id]
        return res['anomaly_rate'].values[0]


class DataDEMOKPI:
    DEMO_KPIS = [
        ["ECG", "MBA_ECG805_data.out"],
        ["IOPS", "KPI-54350a12-7a9d-3ca8-b81f-f886b9d156fd.test.out"]

    ]


class DatasetType:
    DODGER = "Dodgers"  # 1
    IOPS = "IOPS"  # 58
    MGAB = "MGAB"  # 10
    SensorScope = "SensorScope"  # 23
    MITDB = "MITDB"  # 32
    NAB = "NAB"
    ECG = "ECG"


from pylibs.utils.util_joblib import cache_


def get_all_available_datasets(max_size_mb=10):
    """
    去掉文件大小大于 max_size_mb 的单变量时间序列

    Parameters
    ----------
    max_size_mb :

    Returns
    -------

    """
    files = FileUtil().get_all_files(KFoldDatasetLoader.get_dataset_home())
    target_data = []
    for f in files:
        f_size = FileUtil.get_file_size_mb(f)
        if f_size > max_size_mb:  # >10m
            log.info(f"{f} is too large, file size ={f_size} mb")
            continue
        attr_arr = str(f).split("/")
        dataset_name = attr_arr[-2]
        dataset_id = attr_arr[-1]
        target_data.append([dataset_name, dataset_id])
    return target_data


def _get_selected_avaliable_data(dataset_names=None, top_n=15, seed=0):
    """

    Parameters
    ----------
    dataset_names : list
        ['Daphnet', 'ECG']
    top_n : int
    seed : int

    Returns
    -------

    """
    if dataset_names is None:
        dataset_names = ['Daphnet', 'ECG', 'GHL', 'IOPS', 'KDD21', 'MGAB', 'MITDB', 'NAB', 'NASA-MSL', 'NASA-SMAP',
                         'OPPORTUNITY', 'Occupancy', 'SMD', 'SVDB', 'SensorScope', 'YAHOO', "DEBUG"]
    da = get_all_available_datasets(max_size_mb=20)
    ret = []
    for dataset_name, data in pd.DataFrame(da).groupby(by=0):
        if not dataset_name in dataset_names:
            continue
        if data.shape[0] > 0:
            # 确保每个数据集获得的结果是一样的
            enable_numpy_reproduce(seed)
            _top_n = np.min([top_n, data.shape[0]])
            ret += data.sample(_top_n).values.tolist()
    return ret


def _random_selected_dataset_top_15(dataset_names, top_n, seed):
    """
    the default output result for _get_selected_avaliable_data()

    where _get_selected_avaliable_data is:

    def _get_selected_avaliable_data(dataset_names=None, top_n=15, seed=0):
    if dataset_names is None:
        dataset_names = ['Daphnet', 'ECG', 'GHL', 'IOPS', 'KDD21', 'MGAB', 'MITDB', 'NAB', 'NASA-MSL', 'NASA-SMAP',
                         'OPPORTUNITY', 'Occupancy', 'SMD', 'SVDB', 'SensorScope', 'YAHOO', "DEBUG"]
    da = get_all_available_datasets(max_size_mb=20)
    ret = []
    for dataset_name, data in pd.DataFrame(da).groupby(by=0):
        if not dataset_name in dataset_names:
            continue
        if data.shape[0] > 0:
            # 确保每个数据集获得的结果是一样的
            enable_numpy_reproduce(seed)
            _top_n = np.min([top_n, data.shape[0]])
            ret += data.sample(_top_n).values.tolist()
    return ret


    Returns
    -------

    """
    selected = [['DEBUG', 'period_with_noise.out'], ['DEBUG', '1_1000_1000.out'], ['DEBUG', '1_10_10.out'],
                ['DEBUG', 'period_without_noise.out'], ['DEBUG', '0.001_1_1000.out'],
                ['Daphnet', 'S02R01E0.test.csv@6.out'], ['Daphnet', 'S09R01E4.test.csv@3.out'],
                ['Daphnet', 'S03R01E1.test.csv@2.out'], ['Daphnet', 'S09R01E2.test.csv@8.out'],
                ['Daphnet', 'S03R02E0.test.csv@2.out'], ['Daphnet', 'S09R01E0.test.csv@1.out'],
                ['Daphnet', 'S03R01E0.test.csv@6.out'], ['Daphnet', 'S03R03E4.test.csv@1.out'],
                ['Daphnet', 'S09R01E4.test.csv@2.out'], ['Daphnet', 'S03R01E1.test.csv@1.out'],
                ['Daphnet', 'S09R01E4.test.csv@4.out'], ['Daphnet', 'S09R01E0.test.csv@7.out'],
                ['Daphnet', 'S03R02E0.test.csv@9.out'], ['Daphnet', 'S03R02E0.test.csv@4.out'],
                ['Daphnet', 'S09R01E4.test.csv@1.out'], ['ECG', 'MBA_ECG14046_data_9.out'],
                ['ECG', 'MBA_ECG14046_data_1.out'], ['ECG', 'MBA_ECG14046_data_35.out'],
                ['ECG', 'MBA_ECG14046_data_6.out'], ['ECG', 'MBA_ECG14046_data_32.out'],
                ['ECG', 'MBA_ECG14046_data_4.out'], ['ECG', 'MBA_ECG14046_data_13.out'],
                ['ECG', 'MBA_ECG14046_data_7.out'], ['ECG', 'MBA_ECG14046_data_41.out'],
                ['ECG', 'MBA_ECG14046_data_30.out'], ['ECG', 'MBA_ECG14046_data_31.out'],
                ['ECG', 'MBA_ECG14046_data_45.out'], ['ECG', 'MBA_ECG14046_data_5.out'],
                ['ECG', 'MBA_ECG14046_data_40.out'], ['ECG', 'MBA_ECG14046_data_39.out'],
                ['GHL', '06_Lev_fault_Temp_corr_seed_29_vars_23.test.csv@8.out'],
                ['GHL', '14_Lev_fault_Temp_corr_seed_47_vars_23.test.csv@8.out'],
                ['GHL', '11_Lev_fault_Temp_corr_seed_41_vars_23.test.csv@13.out'],
                ['GHL', '35_Lev_corr_Temp_fault_seed_153_vars_23.test.csv@13.out'],
                ['GHL', '12_Lev_fault_Temp_corr_seed_43_vars_23.test.csv@2.out'],
                ['GHL', '10_Lev_fault_Temp_corr_seed_39_vars_23.test.csv@10.out'],
                ['GHL', '08_Lev_fault_Temp_corr_seed_33_vars_23.test.csv@10.out'],
                ['GHL', '10_Lev_fault_Temp_corr_seed_39_vars_23.test.csv@13.out'],
                ['GHL', '08_Lev_fault_Temp_corr_seed_33_vars_23.test.csv@2.out'],
                ['GHL', '22_Lev_fault_Temp_corr_seed_777_vars_23.test.csv@13.out'],
                ['GHL', '12_Lev_fault_Temp_corr_seed_43_vars_23.test.csv@13.out'],
                ['GHL', '15_Lev_fault_Temp_corr_seed_49_vars_23.test.csv@10.out'],
                ['GHL', '20_Lev_fault_Temp_corr_seed_67_vars_23.test.csv@13.out'],
                ['GHL', '19_Lev_fault_Temp_corr_seed_62_vars_23.test.csv@13.out'],
                ['GHL', '09_Lev_fault_Temp_corr_seed_37_vars_23.test.csv@2.out'],
                ['IOPS', 'KPI-a07ac296-de40-3a7c-8df3-91f642cc14d0.train.out'],
                ['IOPS', 'KPI-ffb82d38-5f00-37db-abc0-5d2e4e4cb6aa.train.out'],
                ['IOPS', 'KPI-431a8542-c468-3988-a508-3afd06a218da.test.out'],
                ['IOPS', 'KPI-0efb375b-b902-3661-ab23-9a0bb799f4e3.train.out'],
                ['IOPS', 'KPI-e0747cad-8dc8-38a9-a9ab-855b61f5551d.test.out'],
                ['IOPS', 'KPI-6efa3a07-4544-34a0-b921-a155bd1a05e8.train.out'],
                ['IOPS', 'KPI-f0932edd-6400-3e63-9559-0a9860a1baa9.train.out'],
                ['IOPS', 'KPI-55f8b8b8-b659-38df-b3df-e4a5a8a54bc9.test.out'],
                ['IOPS', 'KPI-8723f0fb-eaef-32e6-b372-6034c9c04b80.test.out'],
                ['IOPS', 'KPI-6efa3a07-4544-34a0-b921-a155bd1a05e8.test.out'],
                ['IOPS', 'KPI-4d2af31a-9916-3d9f-8a8e-8a268a48c095.train.out'],
                ['IOPS', 'KPI-301c70d8-1630-35ac-8f96-bc1b6f4359ea.train.out'],
                ['IOPS', 'KPI-da10a69f-d836-3baa-ad40-3e548ecf1fbd.train.out'],
                ['IOPS', 'KPI-da10a69f-d836-3baa-ad40-3e548ecf1fbd.test.out'],
                ['IOPS', 'KPI-54350a12-7a9d-3ca8-b81f-f886b9d156fd.train.out'],
                ['KDD21', '029_UCR_Anomaly_DISTORTEDInternalBleeding18_2300_4485_4587.out'],
                ['KDD21', '008_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature4_4000_5549_5597.out'],
                ['KDD21', '053_UCR_Anomaly_DISTORTEDWalkingAceleration1_1500_2764_2995.out'],
                ['KDD21', '162_UCR_Anomaly_WalkingAceleration5_2700_5920_5979.out'],
                ['KDD21', '110_UCR_Anomaly_2sddb40_35000_56600_56900.out'],
                ['KDD21', '067_UCR_Anomaly_DISTORTEDinsectEPG3_5200_7000_7050.out'],
                ['KDD21', '169_UCR_Anomaly_gait3_24500_59900_60500.out'],
                ['KDD21', '210_UCR_Anomaly_Italianpowerdemand_36123_74900_74996.out'],
                ['KDD21', '247_UCR_Anomaly_tilt12755mtable_50211_121900_121980.out'],
                ['KDD21', '009_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature5_4000_4852_4900.out'],
                ['KDD21', '122_UCR_Anomaly_ECG3_8000_17000_17100.out'],
                ['KDD21', '117_UCR_Anomaly_CIMIS44AirTemperature5_4000_4852_4900.out'],
                ['KDD21', '231_UCR_Anomaly_mit14134longtermecg_8763_47530_47790.out'],
                ['KDD21', '083_UCR_Anomaly_DISTORTEDresperation9_38000_143411_143511.out'],
                ['KDD21', '184_UCR_Anomaly_resperation10_48000_130700_131880.out'], ['MGAB', '7.test.out'],
                ['MGAB', '5.test.out'], ['MGAB', '3.test.out'], ['MGAB', '4.test.out'], ['MGAB', '6.test.out'],
                ['MGAB', '8.test.out'], ['MGAB', '9.test.out'], ['MGAB', '2.test.out'], ['MGAB', '1.test.out'],
                ['MGAB', '10.test.out'], ['MITDB', '105.test.csv@2.out'], ['MITDB', '219.test.csv@2.out'],
                ['MITDB', '118.test.csv@2.out'], ['MITDB', '221.test.csv@2.out'], ['MITDB', '223.test.csv@1.out'],
                ['MITDB', '105.test.csv@1.out'], ['MITDB', '212.test.csv@2.out'], ['MITDB', '201.test.csv@1.out'],
                ['MITDB', '122.test.csv@1.out'], ['MITDB', '118.test.csv@1.out'], ['MITDB', '201.test.csv@2.out'],
                ['MITDB', '234.test.csv@1.out'], ['MITDB', '119.test.csv@1.out'], ['MITDB', '100.test.csv@2.out'],
                ['MITDB', '212.test.csv@1.out'], ['NAB', 'NAB_data_art1_5.out'], ['NAB', 'NAB_data_KnownCause_5.out'],
                ['NAB', 'NAB_data_art1_1.out'], ['NAB', 'NAB_data_CloudWatch_12.out'], ['NAB', 'NAB_data_tweets_1.out'],
                ['NAB', 'NAB_data_tweets_10.out'], ['NAB', 'NAB_data_tweets_8.out'],
                ['NAB', 'NAB_data_CloudWatch_8.out'], ['NAB', 'NAB_data_CloudWatch_11.out'],
                ['NAB', 'NAB_data_Exchange_3.out'], ['NAB', 'NAB_data_Traffic_5.out'],
                ['NAB', 'NAB_data_CloudWatch_17.out'], ['NAB', 'NAB_data_CloudWatch_1.out'],
                ['NAB', 'NAB_data_KnownCause_3.out'], ['NAB', 'NAB_data_tweets_9.out'], ['NASA-MSL', 'M-1.test.out'],
                ['NASA-MSL', 'F-8.test.out'], ['NASA-MSL', 'M-2.train.out'], ['NASA-MSL', 'T-4.train.out'],
                ['NASA-MSL', 'P-14.test.out'], ['NASA-MSL', 'M-7.train.out'], ['NASA-MSL', 'M-7.test.out'],
                ['NASA-MSL', 'T-13.test.out'], ['NASA-MSL', 'D-14.test.out'], ['NASA-MSL', 'T-9.train.out'],
                ['NASA-MSL', 'T-9.test.out'], ['NASA-MSL', 'D-15.train.out'], ['NASA-MSL', 'T-13.train.out'],
                ['NASA-MSL', 'F-7.test.out'], ['NASA-MSL', 'M-6.test.out'], ['NASA-SMAP', 'P-7.train.out'],
                ['NASA-SMAP', 'G-6.train.out'], ['NASA-SMAP', 'B-1.test.out'], ['NASA-SMAP', 'E-3.train.out'],
                ['NASA-SMAP', 'F-1.test.out'], ['NASA-SMAP', 'F-1.train.out'], ['NASA-SMAP', 'G-6.test.out'],
                ['NASA-SMAP', 'D-6.test.out'], ['NASA-SMAP', 'G-3.test.out'], ['NASA-SMAP', 'G-3.train.out'],
                ['NASA-SMAP', 'E-11.train.out'], ['NASA-SMAP', 'T-3.test.out'], ['NASA-SMAP', 'E-8.test.out'],
                ['NASA-SMAP', 'E-11.test.out'], ['NASA-SMAP', 'T-2.train.out'],
                ['OPPORTUNITY', 'S4-ADL2.test.csv@77.out'], ['OPPORTUNITY', 'S2-ADL1.test.csv@110.out'],
                ['OPPORTUNITY', 'S2-ADL1.test.csv@109.out'], ['OPPORTUNITY', 'S1-ADL1.test.csv@38.out'],
                ['OPPORTUNITY', 'S2-ADL5.test.csv@53.out'], ['OPPORTUNITY', 'S1-ADL5.test.csv@41.out'],
                ['OPPORTUNITY', 'S1-ADL3.test.csv@77.out'], ['OPPORTUNITY', 'S3-ADL2.test.csv@54.out'],
                ['OPPORTUNITY', 'S1-ADL1.test.csv@95.out'], ['OPPORTUNITY', 'S1-ADL3.test.csv@91.out'],
                ['OPPORTUNITY', 'S1-ADL5.test.csv@125.out'], ['OPPORTUNITY', 'S3-ADL3.test.csv@104.out'],
                ['OPPORTUNITY', 'S3-ADL4.test.csv@52.out'], ['OPPORTUNITY', 'S3-ADL5.test.csv@103.out'],
                ['OPPORTUNITY', 'S4-ADL2.test.csv@131.out'], ['Occupancy', 'room-occupancy-0.test.csv@4.out'],
                ['Occupancy', 'room-occupancy.train.csv@4.out'], ['Occupancy', 'room-occupancy-0.test.csv@1.out'],
                ['Occupancy', 'room-occupancy.train.csv@3.out'], ['Occupancy', 'room-occupancy-0.test.csv@5.out'],
                ['Occupancy', 'room-occupancy-0.test.csv@2.out'], ['Occupancy', 'room-occupancy.train.csv@5.out'],
                ['Occupancy', 'room-occupancy-1.test.csv@4.out'], ['Occupancy', 'room-occupancy-1.test.csv@3.out'],
                ['Occupancy', 'room-occupancy-0.test.csv@3.out'], ['SMD', 'machine-3-8.test.csv@35.out'],
                ['SMD', 'machine-1-2.test.csv@12.out'], ['SMD', 'machine-2-8.test.csv@25.out'],
                ['SMD', 'machine-2-4.test.csv@22.out'], ['SMD', 'machine-2-2.test.csv@6.out'],
                ['SMD', 'machine-3-11.test.csv@20.out'], ['SMD', 'machine-3-10.test.csv@21.out'],
                ['SMD', 'machine-2-8.test.csv@15.out'], ['SMD', 'machine-3-11.test.csv@31.out'],
                ['SMD', 'machine-2-4.test.csv@1.out'], ['SMD', 'machine-3-6.test.csv@10.out'],
                ['SMD', 'machine-1-2.test.csv@26.out'], ['SMD', 'machine-1-5.test.csv@1.out'],
                ['SMD', 'machine-3-10.test.csv@26.out'], ['SMD', 'machine-3-1.test.csv@20.out'],
                ['SVDB', '847.test.csv@2.out'], ['SVDB', '871.test.csv@1.out'], ['SVDB', '863.test.csv@1.out'],
                ['SVDB', '801.test.csv@1.out'], ['SVDB', '841.test.csv@1.out'], ['SVDB', '812.test.csv@2.out'],
                ['SVDB', '891.test.csv@1.out'], ['SVDB', '883.test.csv@2.out'], ['SVDB', '872.test.csv@2.out'],
                ['SVDB', '801.test.csv@2.out'], ['SVDB', '858.test.csv@2.out'], ['SVDB', '841.test.csv@2.out'],
                ['SVDB', '802.test.csv@2.out'], ['SVDB', '826.test.csv@2.out'], ['SVDB', '880.test.csv@2.out'],
                ['SensorScope', 'stb-3.test.out'], ['SensorScope', 'stb-32.test.out'],
                ['SensorScope', 'stb-4.test.out'], ['SensorScope', 'stb-11.test.out'],
                ['SensorScope', 'stb-17.test.out'], ['SensorScope', 'stb-18.test.out'],
                ['SensorScope', 'stb-10.test.out'], ['SensorScope', 'stb-5.test.out'],
                ['SensorScope', 'stb-9.test.out'], ['SensorScope', 'stb-7.test.out'],
                ['SensorScope', 'stb-14.test.out'], ['SensorScope', 'stb-8.test.out'],
                ['SensorScope', 'stb-31.test.out'], ['SensorScope', 'stb-13.test.out'],
                ['SensorScope', 'stb-20.test.out'], ['YAHOO', 'YahooA3Benchmark-TS98_data.out'],
                ['YAHOO', 'YahooA3Benchmark-TS50_data.out'], ['YAHOO', 'YahooA4Benchmark-TS3_data.out'],
                ['YAHOO', 'Yahoo_A2synthetic_89_data.out'], ['YAHOO', 'YahooA3Benchmark-TS94_data.out'],
                ['YAHOO', 'YahooA3Benchmark-TS9_data.out'], ['YAHOO', 'Yahoo_A2synthetic_94_data.out'],
                ['YAHOO', 'Yahoo_A1real_14_data.out'], ['YAHOO', 'Yahoo_A2synthetic_64_data.out'],
                ['YAHOO', 'Yahoo_A1real_59_data.out'], ['YAHOO', 'Yahoo_A2synthetic_77_data.out'],
                ['YAHOO', 'YahooA4Benchmark-TS35_data.out'], ['YAHOO', 'Yahoo_A2synthetic_26_data.out'],
                ['YAHOO', 'Yahoo_A1real_56_data.out'], ['YAHOO', 'YahooA3Benchmark-TS62_data.out']]
    if dataset_names is None:
        return selected
    res = []
    for _d, _di in selected:
        if _d in dataset_names:
            res.append([_d, _di])

    return res


class DatasetLoader:
    """
    """

    def __init__(self,
                 data_set: str = None,
                 data_id=None,
                 window_size=60,
                 time_step=1,
                 sample_rate=-1,
                 anomaly_window_type="coca",
                 data_sampling_method=SampleType.RANDOM,
                 is_include_anomaly_window=True,
                 test_rate: float = 0.,
                 data_home=None,
                 processing: Union[str, bool] = "normal",
                 fill_nan=True,
                 max_length=-1,
                 ):
        """

        Parameters
        ----------
        data_set :
        data_id : int or data_id(str)
        window_size :
        time_step :
        anomaly_window_type :
            `coca` or `all`
        sample_rate :
            "-1" for disabling sampling.
            0 means generate a rolling window filled with 0.

            sample_rate >= 1 for sampling n numbers samples.
            0< sample_rate < 1 for sampling ratio numbers


        data_sampling_method : str
            `random` for sampling from the whole dataset
            'normal_random` for only sampling from in the normal dataset
        is_include_anomaly_window :
        data_home :
        processing:
            `normal` for scaling to [0,1]
            False for not scaling
            `standardization` for scaling to mean=0 and std=1
        """
        self._processing = processing
        self._dataset_name = data_set
        self._data_id = data_id
        self._window_size = window_size
        self._time_step = time_step
        self._anomaly_window_type = anomaly_window_type
        self._sampling_rate = convert_str_to_float(sample_rate)
        self._data_sampling_method = data_sampling_method
        self._is_include_anomaly_windows = is_include_anomaly_window
        self._dataset_home = data_home
        self._all_data = None
        self._fill_nan = fill_nan
        self._test_rate = test_rate
        self._max_length = max_length
        if self._dataset_home is None:
            self._dataset_home = KFoldDatasetLoader.get_dataset_home()
            UtilSys.is_debug_mode() and log.info(f"Using the default dataset home {self._dataset_home}")
        # logs(f"Loading [{self._data_id}] from  {self._dataset_name}!")

    @staticmethod
    def get_all_available_datasets():
        """
        Return the datasets both the training and test set containing at least one anomaly.

        """
        return get_all_available_datasets()

    @staticmethod
    def select_top_data_ids(
            dataset_names=None,
            top_n=20,
            seed=0
    ):
        """

        Parameters
        ----------
        dataset_names : list
        top_n :
        seed :

        Returns
        -------

        """
        res = _random_selected_dataset_top_15(dataset_names, top_n, seed)
        return res
        # return _get_selected_avaliable_data(dataset_names, top_n, seed=seed)

    #
    def get_sliding_windows_with_origin_y(self):
        """
        Returns train_x, train_y_processed, train_y_origin
        train_x, train_y_processed, train_y_origin= dl.get_sliding_windows_with_origin_y()
            train_x: sliding windows
            train_y_processed:  Anomaly windows if a window contains at least one abnormal point,
        """
        data = self._load_data()
        if data is None:
            return None, None, None
        if self._max_length != -1:
            warnings.warn(f"Max length is specified, only take the first {self._max_length} datapoints.")
            data = data[:self._max_length]
        UtilSys.is_debug_mode() and log.info(f"[{self._dataset_name}:{self._data_id}] Data info:\n{data.describe()}")
        train_x = subsequences(data.iloc[:, 0], self._window_size, self._time_step)
        train_y_origin = subsequences(data.iloc[:, 1], self._window_size, self._time_step)
        train_y_processed = post_process_sliding_window_label(
            train_y_origin,
            anomaly_window_type=self._anomaly_window_type,
            time_step=self._time_step
        )

        def ct(arr):
            return arr.astype("float32")

        return ct(train_x), train_y_processed, train_y_origin

    def get_source_data(self) -> pd.DataFrame:
        return self._load_data()

    def get_sliding_windows(self, return_origin=False):

        """
        Returns the processed x and y.

        train_x, train_y = dl.get_sliding_windows(return_origin=False)
             return the processed data by self._is_include_anomaly_windows
                 train_x: train x, (n_samples, n_features), filtered by self._is_include_anomaly_windows
                 train_y: (n_samples,), filtered by self._is_include_anomaly_windows
                     indicates where a sliding window is anomaly.
                     Anomaly windows if a window contains at least one abnormal point,

        train_x, train_y, origin_train_x, origin_train_y = dl.get_sliding_windows(return_origin=True)
             return the processed data by self._is_include_anomaly_windows
             train_x: train x, (n_samples, n_features), filtered by self._is_include_anomaly_windows
             train_y: (n_samples,)  filtered by self._is_include_anomaly_windows
                indicates where a sliding window is anomaly.
                Anomaly windows if a window contains at least one abnormal point,

             origin_train_x: the origin train x, (n_samples, n_features)
             origin_train_y: the origin train y, (n_samples,)


        """
        assert self._data_id is not None, "data_id is required"

        train_x, train_y_processed, _ = self.get_sliding_windows_with_origin_y()
        train_x_sampled, train_y_sampled = self._data_sampling(train_x, train_y_processed)
        X, labels = self._remove_anomaly_windows(train_x_sampled, train_y_sampled)

        if convert_str_to_float(self._sampling_rate) == 0:
            X = np.zeros((1, X.shape[1]))
            labels = np.asarray([1])

        if return_origin:
            return X, labels, train_x, train_y_processed
        else:
            return X, labels

    def get_sliding_windows_train_and_test(self):

        """
        Returns the processed train_x, train_y, test_x, test_y according to self._test_rate

        test_x is a set not in train_x.

        train_x is used to train the model.
        test_x is used to test model (the model has been unseen so far)

        train_x, train_y, test_x, test_y = dl.get_sliding_windows_train_and_test(return_origin=False)

             return the processed data by self._is_include_anomaly_windows

             train_x: train x, (n_samples, n_features), filtered by self._is_include_anomaly_windows
             train_y: (n_samples,)  filtered by self._is_include_anomaly_windows
                indicates where a sliding window is anomaly.
                Anomaly windows if a window contains at least one abnormal point,

             test_x: (n_samples, n_features)
             test_y: (n_samples,)


        """

        train_x, train_y_processed, _ = self.get_sliding_windows_with_origin_y()
        if train_x is None:
            return None, None, None, None
        train_x, train_y, test_x, test_y = self._get_split_data(train_x, train_y_processed)

        train_x_sampled, train_y_sampled = self._data_sampling(train_x, train_y)

        X, labels = self._remove_anomaly_windows(train_x_sampled, train_y_sampled)

        if convert_str_to_float(self._sampling_rate) == 0:
            X = np.zeros((1, X.shape[1]))
            labels = np.asarray([1])

        return X, labels, test_x, test_y

    def _load_data(self):
        data = self.read_data(self._dataset_name, self._data_id)
        if data.shape[0] > 80 * 10000:
            warnings.warn(f"Data is too long, we do not process this file with length = {data.shape[0]}")
            return None
        if self._fill_nan:
            data = data.fillna(data.iloc[:, 0].median())
        if self._processing == "normal":
            values = data.iloc[:, 0]
            data.iloc[:, 0] = (values - values.min()) / (values.max() - values.min())
        elif self._processing is False or self._processing == "False":
            # do not process
            pass
        elif self._processing == "standardization":
            values = data.iloc[:, 0]
            data.iloc[:, 0] = (values - values.mean()) / (values.std())
        else:
            raise RuntimeError(f"Unsupported processing type: {self._processing}")
        return data

    def _get_sampling_index(self, n_samples: int, labels: np.ndarray = None):
        """

        Parameters
        ----------
        n_samples :
            The number of training data (samples)

        Returns
        -------

        """
        assert isinstance(self._data_sampling_method, str), "Data sample method must be RS"

        n_sample_train = int(self._sampling_rate)

        # if sample_rate==1, do not sampling
        if self._sampling_rate == 1:
            return np.arange(0, n_samples)

        if n_sample_train >= n_samples:
            n_sample_train = int(n_samples)
            logw(f"There only have {n_samples} sample(s), but received {n_sample_train} sample(s). "
                 f"So changing set the number samples to {n_sample_train}")

        if self._data_sampling_method == SampleType.RANDOM:
            if self._sampling_rate > 1:
                selected_sample_index = np.sort(np.random.choice(
                    a=n_samples,
                    size=n_sample_train,
                    replace=False
                ))
            else:
                n_sample_train = int(self._sampling_rate * n_samples)
                selected_sample_index = np.sort(np.random.choice(
                    a=n_samples,
                    size=n_sample_train,
                    replace=False
                ))
        elif self._data_sampling_method == SampleType.NORMAL_RANDOM:
            if self._sampling_rate > 1:
                selected_sample_index = np.sort(np.random.choice(
                    a=n_samples,
                    size=n_sample_train,
                    replace=False
                ))
            else:
                n_sample_train = int(self._sampling_rate * n_samples)
                selected_sample_index = np.sort(np.random.choice(
                    a=n_samples,
                    size=n_sample_train,
                    replace=False
                ))

            # Only sampling in the normal set
            normal_rolling_window_index = np.argwhere(labels == 1).reshape(-1)
            selected_sample_index = np.union1d(selected_sample_index, normal_rolling_window_index)

        else:
            raise RuntimeError(f"Unsupported sample method {self._data_sampling_method}")
        return selected_sample_index

    def __get_sampling_training_data(self, train_x, labels):
        """
        Return the selected training dataset.


        Parameters
        ----------
        train_x :
        labels :

        Returns
        -------

        """
        sampling_windows_index = self._get_sampling_index(train_x.shape[0], labels)
        return train_x[sampling_windows_index], labels[sampling_windows_index]

    def get_all_dataset_from_dir(self) -> pd.DataFrame:
        """
        Get all dataset in the directory benchmark, filtering by ext

        returns all dataset contain the dataset_name and data_id
        [
            ['IOPS', 'KPI-0efb375b-b902-3661-ab23-9a0bb799f4e3.test.out'],
            ["NAB", 'NAB_data_art0_0.out'],
            ....
        ]
        Returns
        -------

        """
        if self._all_data is not None:
            return self._all_data
        names = ["dataset_name", "data_id"]
        catch_file = ".cache_dataset_names"

        if os.path.exists(catch_file) and os.path.getsize(catch_file) > 1024:
            UtilSys.is_debug_mode() and log.info(f"Load from cache file: {os.path.abspath(catch_file)}")
            self._all_data = pd.read_csv(catch_file, names=names)
            return self._all_data

        _arr = []
        _dataset_names = [
            "Daphnet",
            "Dodgers",
            "ECG",
            "Genesis",
            "GHL",
            "IOPS",
            "KDD21",
            "MGAB",
            "MITDB",
            "NAB",
            "NASA-MSL",
            "NASA-SMAP",
            "Occupancy",
            "OPPORTUNITY",
            "SensorScope",
            "SMD",
            "SVDB",
            "YAHOO",
        ]
        for dataset_name in _dataset_names:
            home = os.path.join(self._dataset_home, dataset_name)
            fu = FileUtils()
            all_files = fu.get_all_files(home, ext=".out")
            for f in all_files:
                _arr.append([dataset_name, os.path.basename(f)])
        data = pd.DataFrame(_arr, columns=names)
        data.to_csv(catch_file, index=False)
        return data

    def get_data_ids(self, return_list=True) -> Union[list, pd.DataFrame]:
        """
        Return all data id of the specified dataset name.

        if return_list is True:
           [
            ['IOPS', 'KPI-0efb375b-b902-3661-ab23-9a0bb799f4e3.test.out'],
            ["NAB", 'NAB_data_art0_0.out'],
            ....
        ]

        if return_list is False:
        Return pd.DataFrame

        Parameters
        ----------
        return_list :

        Returns
        -------

        """
        all_data = self.get_all_dataset_from_dir()
        if return_list:
            return all_data[all_data["dataset_name"] == self._dataset_name].reset_index(drop=True).values.tolist()
        else:
            return all_data[all_data["dataset_name"] == self._dataset_name].reset_index(drop=True)

    def get_data_id(self, data_index) -> DataItem:
        all_data = self.get_data_ids(return_list=False)
        assert data_index < all_data.shape[
            0], f"The dataset exp_index = {data_index} is out of range. except [0 ~ {all_data.shape[0] - 1}]"
        curdata = all_data[all_data["dataset_name"] == self._dataset_name].reset_index(drop=True).iloc[data_index, :]
        return DataItem(curdata[0], curdata[1])

    def read_data(self, dataset_name, data_id):
        self._dataset_name = dataset_name
        self._data_id = data_id
        assert data_id is not None, "data_id is required"
        if isinstance(data_id, str):
            file = f"{self._dataset_home}/{dataset_name}/{data_id}"
        else:
            # todo
            file = f"{self._dataset_home}/{dataset_name}/{self.get_data_id(data_id).get_data_id()}"
        try:
            return pd.read_csv(file, header=None, names=["value", "label"])
        except Exception as e:
            raise RuntimeError(f"File is not find: {file}")

    @classmethod
    def debug_dataset(cls):
        """
        Generate the debug dataset.
        return:

         [
            [  1  11  21  31  41  51  61  71  81  91]
            [101 111 121 131 141 151 161 171 181 191]
            [201 211 221 231 241 251 261 271 281 291]
            [301 311 321 331 341 351 361 371 381 391]
            [401 411 421 431 441 451 461 471 481 491]
            [501 511 521 531 541 551 561 571 581 591]
            [601 611 621 631 641 651 661 671 681 691]
            [701 711 721 731 741 751 761 771 781 791]
            [801 811 821 831 841 851 861 871 881 891]
            [901 911 921 931 941 951 961 971 981 991]
         ]
        Returns
        -------

        """
        data = np.arange(1, 1001, 10)
        data = data.reshape((10, 10))

        return data

    def _remove_anomaly_windows(self, data, labels):
        """
        Remove anomaly sliding windows.

        X, labels = self._remove_anomaly_windows_by_flag(data, labels):

        Parameters
        ----------
        data :
        labels :

        Returns
        -------

        """
        if self._is_include_anomaly_windows is False:
            normal_windows_index = np.argwhere(labels == 0).reshape(-1)
            return data[normal_windows_index], labels[normal_windows_index]
        else:
            return data, labels

    def get_anomaly_rate(self):
        data = self._load_data()
        return np.round(data['label'].sum() / data.shape[0], 4)

    def get_length(self):
        data = self._load_data()
        return data.shape[0]

    def _data_sampling(self, train_x, labels):
        if self._sampling_rate == -1:
            return train_x, labels

        source_data = np.hstack((train_x, np.expand_dims(labels, -1)))
        source_data_df = pd.DataFrame(source_data)

        # convert to n_samples

        label_name = source_data_df.columns[-1]
        if self._data_sampling_method == SampleType.RANDOM:
            n_samples = self._get_sample_number(train_x.shape[0])
            sampled_data = source_data_df.sample(n_samples)

        elif self._data_sampling_method == SampleType.NORMAL_RANDOM:
            _all = []
            for label, group_data in source_data_df.groupby(by=label_name):
                if label == 1:
                    _all.append(group_data)
                else:
                    n_samples = self._get_sample_number(group_data.shape[0])
                    _all.append(group_data.sample(n_samples))
            sampled_data = pd.concat(_all)

        elif self._data_sampling_method == SampleType.STRATIFIED:
            _all = []
            for label, group_data in source_data_df.groupby(by=label_name):
                n_samples = self._get_sample_number(group_data.shape[0])
                _all.append(group_data.sample(n_samples))
            sampled_data = pd.concat(_all)
        else:
            raise ValueError("Unsupported sample method")

        return sampled_data.iloc[:, :-1].values, sampled_data.iloc[:, -1].values

    def _get_sample_number(self, n_train):
        """
        Convert the sample_rate to integer: how many data to sample.

        Parameters
        ----------
        n_train :

        Returns
        -------

        """
        sample_rate_float = convert_str_to_float(self._sampling_rate)

        if sample_rate_float >= 1:
            n_samples = sample_rate_float
            if self._data_sampling_method == SampleType.STRATIFIED:
                n_samples = n_samples / 2
        else:
            n_samples = sample_rate_float * n_train
        n_samples = np.min([n_train, n_samples])
        n_samples = int(n_samples)
        return n_samples

    def _get_split_data(self, train_x, train_y):
        """
        Return the splited data according to self._test_rate.
        """
        assert self._test_rate > 0, "test_rate must be larger than 0."
        n_train = int(len(train_y) * (1 - self._test_rate))
        return train_x[:n_train], train_y[:n_train], train_x[n_train:], train_y[n_train:]

    @classmethod
    def init_filter_dataset(cls):
        files = FileUtil().get_all_files(os.path.join(os.path.dirname(__file__), "./benchmark"))
        target_data = []
        for f in files:
            attr_arr = str(f).split("/")
            dataset_name = attr_arr[-2]
            dataset_id = attr_arr[-1]
            dl = DatasetLoader(dataset_name, dataset_id,
                               data_sampling_method=SampleType.STRATIFIED,
                               sample_rate=-1,
                               window_size=1,
                               test_rate=0.2,
                               processing=False)

            train_x, train_y, test_x, test_y = dl.get_sliding_windows_train_and_test()

            if train_x is None:
                continue

            if train_x.shape[0] > 80 * 10000:
                continue

            flag1 = np.sum(test_y)
            flat2 = np.sum(train_y)
            if flag1 > 0 and flat2 > 0 and train_x.shape[0] > 100:
                target_data.append([dataset_name, dataset_id])

        data = pd.DataFrame(target_data)
        data.to_csv(
            os.path.join(os.path.dirname(__file__), "has_anomaly_in_test_set.csv"),
            index=False)
        return data


class AnomalyWindowType:
    # A sliding window is anomaly if the latest datapoints is anomaly points.
    COCA = "coca"
    # A sliding window is anomaly if it contains at least one anomaly point.
    ALL = "all"


class DataProcessingType:
    """
      `normal` for scaling to [0,1]
            False for not scaling
            `standardization` for scaling to mean=0 and std=1
    """
    # scale to [0,1]
    NORMAL = "normal"

    # disable scaling
    DISABLE = False

    # scale to mean=0 and std=1
    STANDARDIZATION = "standardization"


from pylibs.utils.util_joblib import cache_


@cache_
def _read_csv_from_file(file_path, prepocessing="standardization"):
    _data = pd.read_csv(file_path, header=None, names=["value", "label"])
    if prepocessing == "normal":
        values = _data.iloc[:, 0]
        _data.iloc[:, 0] = (values - values.min()) / (values.max() - values.min())
    elif prepocessing == "standardization":
        values = _data.iloc[:, 0]
        _data.iloc[:, 0] = (values - values.mean()) / (values.std())
    else:
        pass
    return _data


class KFoldDatasetLoader:
    """
    Examples:

    1.按照 7:3 的比例加载训练
    from pylibs.uts_dataset.dataset_loader import UTSDataset, KFoldDatasetLoader
    from pylibs.uts_dataset.dataset_loader import KFoldDatasetLoader, DataProcessingType
    datasets = UTSDataset.select_datasets(top_n=None, dataset_names=None)
    test_data = datasets[0]
    kf = KFoldDatasetLoader(
        dataset_name=test_data[0],
        data_id=test_data[1],
        processing=DataProcessingType.DISABLE,
        test_rate=0.3)
    train_x, train_y, test_x, test_y = kf.get_sliding_windows_train_and_test()

    """

    def __init__(self,
                 dataset_name: str = "ECG",
                 data_id="MBA_ECG820_data.out",
                 window_size=64,
                 time_step=1,
                 sample_rate=-1,
                 is_include_anomaly_window=True,
                 anomaly_window_type=AnomalyWindowType.COCA,
                 data_sampling_method=SampleType.RANDOM,
                 processing: Union[str, bool] = DataProcessingType.STANDARDIZATION,
                 test_rate: float = 0.2,
                 data_home=None,
                 max_length=-1,
                 k_fold=5,
                 df=None,
                 data_scale_beta=None,
                 fold_index=None
                 ):
        """

        Parameters
        ----------
        data_id : int or data_id(str)
        anomaly_window_type :
            `coca` or `all`
        sample_rate :
            sample_rate= -1 :  disabling sampling.
            sample_rate=  0 :  means generate two rolling window filled with 0, one label is 0 and other label is 1
            sample_rate= [2, +inf]:  for sampling n numbers samples.
                e.g., sample_rate=2 means sampling 2 samples from the data.
            sample_rate= (0, 1]: sampling ratio numbers

        data_sampling_method : str
            `random` for sampling from the whole dataset
            `normal_random` for only sampling from in the normal dataset

        is_include_anomaly_window :
        data_home :
        processing:
            `normal` for scaling to [0,1]
            False for not scaling
            `standardization` for scaling to mean=0 and std=1
        df: pd.DataFrame
            To support testing
        """
        # 原始训练数据：指原始单变量转为滑动窗口后的数据
        self._origin_train_x = None
        self._origin_train_y = None
        # --------------------------------

        self._data_processing_time = -1
        self._processing = processing
        self._dataset_name = dataset_name
        self._data_id = data_id
        self._window_size = window_size
        self._time_step = time_step
        self._anomaly_window_type = anomaly_window_type
        self._sampling_rate = convert_str_to_float(sample_rate)
        self._data_sampling_method = str(data_sampling_method).lower()
        self._is_include_anomaly_windows = is_include_anomaly_window
        self._dataset_home = data_home
        self._all_data = None
        self._test_rate = test_rate
        self._k_fold = k_fold
        self._max_length = max_length
        self._df = df
        self._fold_index = fold_index
        # 抽样时用来调节的比例
        self._data_scale_beta = data_scale_beta
        self._dataset_home = KFoldDatasetLoader.get_dataset_home()

        # assert self._dataset_home is not None and os.path.isdir(
        #     self._dataset_home), "Home of dataset must be specified by Environment Variable DATASET_HOME, such as export DATASET_HOME=/Users/sunwu/Documents/uts_benchmark_dataset/benchmark/ \n Under the benchmark directory, it contains " \
        # "multiple datasets such as:  \nbenchmark/Daphnet \nbenchmark/SMD \nbenchmark/..."
        # if self._dataset_home is None:
        #     self._dataset_home = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark")
        # UtilSys.is_debug_mode() and log.info(f"Using the default dataset home {self._dataset_home}")

    @staticmethod
    def get_dataset_home():

        default_home = Path("/remote-home/cs_acmis_sunwu/uts_benchmark_dataset")
        if default_home.exists():
            return default_home
        else:
            home = UtilSys.get_environ_by_key("DATASET_HOME")
            assert home is not None and os.path.isdir(
                home), f"{default_home} is not existed. So you has to set environment variable DATASET_HOME, such as export DATASET_HOME=/Users/sunwu/Documents/uts_benchmark_dataset/benchmark/ \n Under the benchmark directory, it contains " \
                       "multiple datasets such as:  \nbenchmark/Daphnet \nbenchmark/SMD \nbenchmark/..."
            return home

    def get_sample_rate(self):
        return self._sampling_rate

    def get_dataset_name(self):
        return self._dataset_name

    @DeprecationWarning
    @staticmethod
    def select_top_data_ids(
            dataset_names=None,
            top_n=20,
            seed=0,
            **kwargs
    ):
        res = _random_selected_dataset_top_15(dataset_names, top_n, seed)
        return res

    @DeprecationWarning
    @staticmethod
    def get_all_available_datasets():
        """
        Return the datasets both the training and test set containing at least one anomaly.

        """
        file = os.path.join(os.path.dirname(__file__), "has_anomaly_in_test_set.csv")
        if not os.path.exists(file):
            DatasetLoader.init_filter_dataset()
        source_data = pd.read_csv(file, header=1)
        return source_data.values.tolist()

    # def _load_data(self):
    #     return _load_data(self._dataset_home, self._dataset_name, self._data_id, max_length=self._max_length)
    def get_origin_sliding_windows(self):
        """
        return train_x_windows, train_y_windows =


        Returns
        -------

        """
        if self._origin_train_x is None:
            data = self._load_processed_data()
            self._origin_train_x = subsequences(data.iloc[:, 0], self._window_size, self._time_step)
            self._origin_train_y = subsequences(data.iloc[:, 1], self._window_size, self._time_step)
        return self._origin_train_x, self._origin_train_y

    def __get_sliding_windows_with_origin_y(self):
        """
        Returns train_x, train_y_processed, train_y_origin
        train_x, train_y_processed, train_y_origin= dl.get_sliding_windows_with_origin_y()
            train_x: sliding windows
            train_y_processed:  Anomaly windows if a window contains at least one abnormal point,
            train_y_origin： the original label for univariate time series
        """
        data = self._load_processed_data()

        if data.shape[0] <= self._window_size:
            return None, None, None
        else:
            train_x, train_y_origin = self.get_origin_sliding_windows()

            train_y_processed = post_process_sliding_window_label(
                train_y_origin,
                anomaly_window_type=self._anomaly_window_type,
                time_step=self._time_step
            )

            return train_x.astype("float32"), train_y_processed, train_y_origin

    def load_train_and_test(self):
        """
        train_x, train_y, test_x, test_y=kf.get_sliding_windows_train_and_test()

        Returns
        -------

        """
        assert self._data_id is not None, "data_id is required"
        assert self._test_rate > 0, "test rate must be larger than 0."
        log.debug(f"Test rate: {self._test_rate}")

        train_x, train_y_processed, _ = self.__get_sliding_windows_with_origin_y()
        if train_x is None:
            return None, None, None, None

        train_x, train_y, test_x, test_y = self._get_split_data(train_x, train_y_processed)

        train_x_sampled, train_y_sampled = self._data_sampling(train_x, train_y)

        assert train_x.shape[0] + test_x.shape[0] == self.get_origin_sliding_windows()[0].shape[0]

        X, labels = self._remove_anomaly_windows_by_flag(train_x_sampled, train_y_sampled)
        if convert_str_to_float(self._sampling_rate) == 0:
            X = np.zeros((2, X.shape[1]))
            labels = np.asarray([0, 1])
        _data_info = {
            "dataset_name": self._dataset_name,
            "data_id": self._data_id,
            "ori_train_x": train_x.shape,
            "sampled_train_x": X.shape,
            "test_x": test_x.shape,
            "x_mean": train_x.mean(),
            "x_std": train_x.std()
        }
        log.debug(f'data info: {_data_info}')

        # 检查返回的数据

        return X, labels, test_x, test_y

    def _check_data(self, data: np.ndarray):

        if self._sampling_rate > 1:
            # 如果大于1，表示取固定条数的数据
            train_x_windows, train_y_windows = self.get_origin_sliding_windows()
            assert data.shape[0] == self._sampling_rate, \
                f"Data length must be equal to sampling rate. Excepted {self._sampling_rate}," \
                f"but received {data.shape[0]}"

    @DeprecationWarning
    def get_sliding_windows_train_and_test(self):

        """
        Returns the processed train_x, train_y, test_x, test_y according to self._test_rate

        test_x is a set not in train_x.

        train_x is used to train the model.
        test_x is used to test model (the model has been unseen so far)

        train_x, train_y, test_x, test_y = dl.get_sliding_windows_train_and_test(return_origin=False)

             return the processed data by self._is_include_anomaly_windows

             train_x: train x, (n_samples, n_features), filtered by self._is_include_anomaly_windows
             train_y: (n_samples,)  filtered by self._is_include_anomaly_windows
                indicates where a sliding window is anomaly.
                Anomaly windows if a window contains at least one abnormal point,

             test_x: (n_samples, n_features)
             test_y: (n_samples,)


        """
        assert self._data_id is not None, "data_id is required"
        assert self._test_rate > 0, "test rate must be larger than 0."
        UtilSys.is_debug_mode() and log.info(f"Test rate: {self._test_rate}")

        train_x, train_y_processed, _ = self.get_sliding_windows_with_origin_y()
        if train_x is None:
            return None, None, None, None

        train_x, train_y, test_x, test_y = self._get_split_data(train_x, train_y_processed)

        train_x_sampled, train_y_sampled = self._data_sampling(train_x, train_y)

        X, labels = self._remove_anomaly_windows_by_flag(train_x_sampled, train_y_sampled)

        if convert_str_to_float(self._sampling_rate) == 0:
            X = np.zeros((1, X.shape[1]))
            labels = np.asarray([1])
        log.info('data info:', {
            "dataset_name": self._dataset_name,
            "data_id": self._data_id,
            "ori_train_x": train_x.shape,
            "sampled_train_x": X.shape,
            "test_x": test_x.shape,
            "x_mean": train_x.mean(),
            "x_std": train_x.std()
        })
        return X, labels, test_x, test_y

    @DeprecationWarning
    def get_kfold_sliding_windows_train_and_test(self):
        """
        for _fold_index, train_x, train_y, test_x, test_y in dl.get_kfold_sliding_windows_train_and_test():

        Returns
        -------

        """
        assert self._data_id is not None, "data_id is required"
        assert self._test_rate > 0, "test rate must be larger than 0."

        train_x, train_y_processed, _ = self.get_sliding_windows_with_origin_y()
        UtilSys.is_debug_mode() and log.info(f"Train_x mean: {train_x.mean()}, train x std: {train_x.std()},")
        kf = KFold(n_splits=self._k_fold)
        kf.split(train_y_processed)
        for i, (_train_index, _test_index) in enumerate(kf.split(train_x)):
            _train_x = train_x[_train_index]
            _train_y = train_y_processed[_train_index]

            _test_x = train_x[_test_index]
            _test_y = train_y_processed[_test_index]
            train_x_sampled, train_y_sampled = self._data_sampling(_train_x, _train_y)
            X, labels = self._remove_anomaly_windows_by_flag(train_x_sampled, train_y_sampled)
            if self._sampling_rate == 0:
                # Generate a unit vector for initialing the model
                X, labels = self._generate_init_data(X, labels)

            yield i, X, labels, _test_x, _test_y

    @DeprecationWarning
    def get_data_processing_time(self):
        return self._data_processing_time

    @DeprecationWarning
    def get_kfold_sliding_windows_train_and_test_by_fold_index(self, fold_index=-1):
        """
        train_x, train_y, test_x, test_y=dl.get_kfold_sliding_windows_train_and_test():

        Returns
        -------

        """
        # UtilSys.is_debug_mode() and log.info(f"Dataloader Config: \n{pprint.pformat(self.__dict__)}")
        if fold_index < 0:
            fold_index = self._k_fold + fold_index

        train_x, train_y_processed, _ = self.get_sliding_windows_with_origin_y()
        kf = KFold(n_splits=self._k_fold)
        for i, (_train_index, _test_index) in enumerate(kf.split(train_x)):
            if fold_index == i:
                _train_x = train_x[_train_index]
                _train_y = train_y_processed[_train_index]

                _test_x = train_x[_test_index]
                _test_y = train_y_processed[_test_index]
                _start_data_process_time = time.time_ns()
                train_x_sampled, train_y_sampled = self._data_sampling(_train_x, _train_y)
                X, labels = self._remove_anomaly_windows_by_flag(train_x_sampled, train_y_sampled)
                _end_data_process_time = time.time_ns()
                self._data_processing_time = 1e-9 * (_end_data_process_time - _start_data_process_time)
                if self._sampling_rate == 0:
                    # Generate a unit vector for initialing the model
                    X, labels = self._generate_init_data(X, labels)

                if self._is_include_anomaly_windows is False:
                    assert labels.sum() == 0

                #
                # UtilSys.is_debug_mode() and log.info(
                #     f"Data pre-processing: {self._processing}, Train_x mean: {X.mean():.2f} {X.shape}, train x std: {train_x.std():.2f} for [{self._dataset_name}:{self._data_id}]")

                return X, labels, _test_x, _test_y

        # If data is not loaded in the iteration.
        raise RuntimeError("Could load data")
        # return None, None, None, None

    @DeprecationWarning
    def get_split_data(self):
        """
        train_x, train_y, test_x, test_y=dl.get_split_data()
        Returns
        -------

        """

        train_x, train_y, test_x, test_y, data_processing_time = self.get_kfold_sliding_windows_train_and_test_from_fold_index()
        return train_x, train_y, test_x, test_y

    @DeprecationWarning
    def get_kfold_sliding_windows_train_and_test_from_fold_index(self):
        """
        train_x, train_y, test_x, test_y, data_processing_time=dl.get_kfold_sliding_windows_train_and_test():

        Parameters
        ----------
        original_labels:bool
            False 返回处理过的 label
            True 返回原始latel

        Returns
        -------

        """
        # UtilSys.is_debug_mode() and log.info(f"Dataloader Config: \n{pprint.pformat(self.__dict__)}")
        # assert self._fold_index is not None, "k-fold exp_index cannot be None."
        # if self._fold_index < 0:
        #     self._fold_index = self._k_fold + self._fold_index
        # if original_labels is False:
        #     # 返回处理过的 labels
        #     train_x, train_y_processed, train_y_original = self.get_sliding_windows_with_origin_y()
        # else:
        #     # 返回原始 labels
        #     train_x, _, original_y = self.get_sliding_windows_with_origin_y()
        #     train_y_processed = original_y[:, -1]
        #
        # kf = KFold(n_splits=self._k_fold)
        # for i, (_train_index, _test_index) in enumerate(kf.split(train_x)):
        #     if self._fold_index == i:
        #         _train_x = train_x[_train_index]
        #         _train_y = train_y_processed[_train_index]
        #
        #         _test_x = train_x[_test_index]
        #         _test_y = train_y_processed[_test_index]
        #
        #         _start_data_process_time = time.time_ns()
        #         train_x_sampled, train_y_sampled = self._data_sampling(_train_x, _train_y)
        #         X, labels = self._remove_anomaly_windows_by_flag(train_x_sampled, train_y_sampled)
        #         _end_data_process_time = time.time_ns()
        #         self._data_processing_time = 1e-9 * (_end_data_process_time - _start_data_process_time)
        #         if self._sampling_rate == 0:
        #             # Generate a unit vector for initialing the model
        #             X, labels = self._generate_init_data(X, labels)
        #
        #         if self._is_include_anomaly_windows is False:
        #             assert labels.sum() == 0
        #
        #         if UtilSys.is_debug_mode():
        #             try:
        #                 data_disc = [
        #                     get_data_describe(X, data_type=f"train_x", processing=self._processing),
        #                     get_data_describe(labels, data_type="label", processing=self._processing),
        #                 ]
        #                 df = pd.DataFrame(data_disc, columns=data_disc[0].keys())
        #                 PDUtil.print_pretty_table_with_header(df)
        #             except:
        #                 pass
        #         return X, labels, _test_x, _test_y, self._data_processing_time
        #
        # # If data is not loaded in the iteration.
        # raise RuntimeError("Could load data")
        # return None, None, None, None
        original_train_x, original_train_y, \
            original_train_sample_x, original_train_sample_y, \
            dirty_train_sample_x, dirty_train_sample_y, \
            clean_train_sample_x, clean_train_sample_y, \
            test_x, test_y = self._load_kfold_data()
        train_x, train_y = self._remove_anomaly_windows_by_flag(original_train_sample_x,
                                                                original_train_sample_y)
        return train_x, train_y, test_x, test_y, self._data_processing_time

    @DeprecationWarning
    def get_kfold_data_for_lstm_cnn(self):
        """
        get the training data for lstm and cnn.
        Returns
        -------

        """
        original_train_x, original_train_y, \
            original_train_sample_x, original_train_sample_y, \
            dirty_train_sample_x, dirty_train_sample_y, \
            clean_train_sample_x, clean_train_sample_y, \
            test_x, test_y = self._load_kfold_data()
        return dirty_train_sample_x, dirty_train_sample_y, clean_train_sample_x, clean_train_sample_y, test_x, test_y

    @DeprecationWarning
    def _load_kfold_data(self):
        """
        original_train_x, original_train_y, \
         original_train_sample_x, original_train_sample_y,  \
         dirty_train_sample_x, dirty_train_sample_y, \
         clean_train_sample_x, clean_train_sample_y, \
          test_x, test_y = self._load_kfold_data()

        # 训练数据
        original_train_x, original_train_y: 原始未抽样的数据
        original_train_sample_x, original_train_sample_y: 抽样后的原始数据
        dirty_train_sample_x, dirty_train_sample_y: 只包含异常窗口
        clean_train_sample_x, clean_train_sample_y: 只包含正常窗口

        # 测试数据
        test_x, test_y: 测试集



        Returns
        -------

        """
        # UtilSys.is_debug_mode() and log.info(f"Dataloader Config: \n{pprint.pformat(self.__dict__)}")
        assert self._fold_index is not None, "k-fold exp_index cannot be None."
        if self._fold_index < 0:
            self._fold_index = self._k_fold + self._fold_index
        train_x, train_y_processed, train_y_original = self.get_sliding_windows_with_origin_y()

        kf = KFold(n_splits=self._k_fold)
        for i, (_train_index, _test_index) in enumerate(kf.split(train_x)):
            if self._fold_index == i:
                original_train_x = train_x[_train_index]
                original_train_y = train_y_processed[_train_index]

                test_x = train_x[_test_index]
                test_y = train_y_processed[_test_index]

                _start_data_process_time = time.time_ns()
                # 抽样的原始样本
                original_train_sample_x, original_train_sample_y = self._data_sampling(original_train_x,
                                                                                       original_train_y)

                # 根据目标算法的类型,决定是否移除相关样本
                clean_train_sample_x, clean_train_sample_y = self._get_normal_windows(original_train_sample_x,
                                                                                      original_train_sample_y)

                dirty_train_sample_x, dirty_train_sample_y = self._get_anomaly_windows(original_train_sample_x,
                                                                                       original_train_sample_y)
                _end_data_process_time = time.time_ns()
                self._data_processing_time = 1e-9 * (_end_data_process_time - _start_data_process_time)

                if self._sampling_rate == 0:
                    # Generate a unit vector for initialing the model
                    clean_train_sample_x, clean_train_sample_y = self._generate_init_data(clean_train_sample_x,
                                                                                          clean_train_sample_y)

                if self._is_include_anomaly_windows is False:
                    assert clean_train_sample_y.sum() == 0

                if UtilSys.is_debug_mode():
                    try:
                        data_disc = [
                            get_data_describe(clean_train_sample_x, data_type=f"train_x", processing=self._processing),
                            get_data_describe(clean_train_sample_y, data_type="label", processing=self._processing),
                        ]
                        df = pd.DataFrame(data_disc, columns=data_disc[0].keys())
                        PDUtil.print_pretty_table_with_header(df)
                    except:
                        pass
                return original_train_x, original_train_y, original_train_sample_x, original_train_sample_y, dirty_train_sample_x, dirty_train_sample_y, clean_train_sample_x, clean_train_sample_y, test_x, test_y

        # If data is not loaded in the iteration.
        raise RuntimeError(f"Could load data {self._dataset_name}:{self._data_id}")
        # return None, None, None, None

    @DeprecationWarning
    def get_kfold_sliding_windows_train_and_test_from_fold_index_with_original_label(self):
        """
        train_x, train_y, test_x, test_y, data_processing_time=dl.get_kfold_sliding_windows_train_and_test():

        Parameters
        ----------
        original_labels:bool
            False 返回处理过的 label
            True 返回原始latel

        Returns
        -------

        """
        # UtilSys.is_debug_mode() and log.info(f"Dataloader Config: \n{pprint.pformat(self.__dict__)}")
        assert self._fold_index is not None, "k-fold exp_index cannot be None."
        if self._fold_index < 0:
            self._fold_index = self._k_fold + self._fold_index
        train_x, train_y_processed, train_y_original = self.get_sliding_windows_with_origin_y()
        kf = KFold(n_splits=self._k_fold)
        for i, (_train_index, _test_index) in enumerate(kf.split(train_x)):
            if self._fold_index == i:
                _train_x = train_x[_train_index]
                _train_y = train_y_processed[_train_index]

                _test_x = train_x[_test_index]
                _test_y = train_y_processed[_test_index]
                _text_y_original = train_y_original[:, -1][_test_index]

                _start_data_process_time = time.time_ns()
                train_x_sampled, train_y_sampled = self._data_sampling(_train_x, _train_y)
                X, labels = self._remove_anomaly_windows_by_flag(train_x_sampled, train_y_sampled)
                _end_data_process_time = time.time_ns()
                self._data_processing_time = 1e-9 * (_end_data_process_time - _start_data_process_time)
                if self._sampling_rate == 0:
                    # Generate a unit vector for initialing the model
                    X, labels = self._generate_init_data(X, labels)

                if self._is_include_anomaly_windows is False:
                    assert labels.sum() == 0

                # UtilSys.is_debug_mode() and log.info(
                #     f"Data pre-processing: {self._processing}, Train_x mean: {X.mean():.2f} {X.shape}, train x std: {train_x.std():.2f} for [{self._dataset_name}:{self._data_id}]")

                if UtilSys.is_debug_mode():
                    try:
                        data_disc = [
                            get_data_describe(X, data_type=f"train_x", processing=self._processing),
                            get_data_describe(labels, data_type="label", processing=self._processing),
                        ]
                        df = pd.DataFrame(data_disc, columns=data_disc[0].keys())
                        PDUtil.print_pretty_table_with_header(df)
                    except:
                        pass
                return X, labels, _test_x, _test_y, _text_y_original

        # If data is not loaded in the iteration.
        raise RuntimeError("Could load data")
        # return None, None, None, None

    @DeprecationWarning
    def _generate_init_data(self, X, labels=None):
        # Generate a vector for initialing the model
        _n_samples = 10
        X = np.ones((_n_samples, X.shape[1]))
        labels = np.zeros(_n_samples)
        return X, labels

    def _load_processed_data(self):
        """
        Load file to pd.DataFrame
        Returns
        -------

        """
        if self._df is None:
            self.__preprocess_data()
        return self._df

    def _source_df(self):
        return self._load_processed_data()

    def __preprocess_data(self):
        """
        Fill NaN, then normalize or standardization the data
        Returns
        -------

        """
        data = self.__read_data_csv(self._dataset_name, self._data_id)
        assert np.isnan(
            data['value'].values).any() == False, "Data contains NaN values, you need to further process the data"

        if self._processing == "normal":
            # Normalize the data
            log.info("Normalize data...")
            data['value'] = (data['value'] - data['value'].min()) / (data['value'].max() - data['value'].min())
        elif self._processing is False or self._processing == "False":
            log.info("Skip precessing data...")
            pass
        elif self._processing == "standardization":
            log.debug("Standardize data...")
            data['value'] = (data['value'] - data['value'].mean()) / data['value'].std()
        else:
            raise RuntimeError(f"Unsupported processing type: {self._processing}")
        self._df = data

    @DeprecationWarning
    def load_data_by_file(self, file_path):
        self.parse_dataset_name_and_id_from_file(file_path)
        _data = _read_csv_from_file(file_path, self._processing)
        return _data

    @DeprecationWarning
    def statis_anomaly_ratio_for_file(self, file_path):
        data = self.load_data_by_file(file_path)

        return {
            EK.DATASET_NAME: self._dataset_name,
            EK.DATA_ID: self._data_id,
            EK.DATA_ANOMALY_RATE: data[EK.LABEL].sum() / data.shape[0],
            EK.LENGTH: data.shape[0],
            EK.FILE_SIZE: os.path.getsize(file_path)
        }

    @DeprecationWarning
    def _get_sampling_index(self, n_samples: int, labels: np.ndarray = None):
        """

        Parameters
        ----------
        n_samples :
            The number of training data (samples)

        Returns
        -------

        """
        assert isinstance(self._data_sampling_method, str), "Data sample method must be RS"

        n_sample_train = int(self._sampling_rate)

        # if sample_rate==1, do not sampling
        if self._sampling_rate == 1:
            return np.arange(0, n_samples)

        if n_sample_train >= n_samples:
            n_sample_train = int(n_samples)
            logw(f"There only have {n_samples} sample(s), but received {n_sample_train} sample(s). "
                 f"So changing set the number samples to {n_sample_train}")

        if self._data_sampling_method == SampleType.RANDOM:
            if self._sampling_rate > 1:
                selected_sample_index = np.sort(np.random.choice(
                    a=n_samples,
                    size=n_sample_train,
                    replace=False
                ))
            else:
                n_sample_train = int(self._sampling_rate * n_samples)
                selected_sample_index = np.sort(np.random.choice(
                    a=n_samples,
                    size=n_sample_train,
                    replace=False
                ))
        elif self._data_sampling_method == SampleType.NORMAL_RANDOM:
            if self._sampling_rate > 1:
                selected_sample_index = np.sort(np.random.choice(
                    a=n_samples,
                    size=n_sample_train,
                    replace=False
                ))
            else:
                n_sample_train = int(self._sampling_rate * n_samples)
                selected_sample_index = np.sort(np.random.choice(
                    a=n_samples,
                    size=n_sample_train,
                    replace=False
                ))

            # Only sampling in the normal set
            normal_rolling_window_index = np.argwhere(labels == 1).reshape(-1)
            selected_sample_index = np.union1d(selected_sample_index, normal_rolling_window_index)

        else:
            raise RuntimeError(f"Unsupported sample method {self._data_sampling_method}")
        return selected_sample_index

    @DeprecationWarning
    def __get_sampling_training_data(self, train_x, labels):
        """
        Return the selected training dataset.


        Parameters
        ----------
        train_x :
        labels :

        Returns
        -------

        """
        sampling_windows_index = self._get_sampling_index(train_x.shape[0], labels)
        return train_x[sampling_windows_index], labels[sampling_windows_index]

    @DeprecationWarning
    def get_all_dataset_from_dir(self) -> pd.DataFrame:
        """
        Get all dataset in the directory benchmark, filtering by ext

        returns all dataset contain the dataset_name and data_id
        [
            ['IOPS', 'KPI-0efb375b-b902-3661-ab23-9a0bb799f4e3.test.out'],
            ["NAB", 'NAB_data_art0_0.out'],
            ....
        ]
        Returns
        -------

        """
        if self._all_data is not None:
            return self._all_data
        names = ["dataset_name", "data_id"]
        # catch_file = ".cache_dataset_names"

        # if os.path.exists(catch_file) and os.path.getsize(catch_file) > 1024:
        #     UtilSys.is_debug_mode() and log.info(f"Load from cache file: {os.path.abspath(catch_file)}")
        #     self._all_data = pd.read_csv(catch_file, names=names)
        #     return self._all_data

        _arr = []
        _dataset_names = [
            "Daphnet",
            "Dodgers",
            "ECG",
            "Genesis",
            "GHL",
            "IOPS",
            "KDD21",
            "MGAB",
            "MITDB",
            "NAB",
            "NASA-MSL",
            "NASA-SMAP",
            "Occupancy",
            "OPPORTUNITY",
            "SensorScope",
            "SMD",
            "SVDB",
            "YAHOO",
        ]
        for dataset_name in _dataset_names:
            home = os.path.join(self._dataset_home, dataset_name)
            fu = FileUtils()
            all_files = fu.get_all_files(home, ext=".out")
            for f in all_files:
                _arr.append([dataset_name, os.path.basename(f)])
        data = pd.DataFrame(_arr, columns=names)
        # data.to_csv(catch_file, exp_index=False)
        return data

    @DeprecationWarning
    def get_data_ids(self, return_list=True) -> Union[list, pd.DataFrame]:
        """
        Return all data id of the specified dataset name.

        if return_list is True:
           [
            ['IOPS', 'KPI-0efb375b-b902-3661-ab23-9a0bb799f4e3.test.out'],
            ["NAB", 'NAB_data_art0_0.out'],
            ....
        ]

        if return_list is False:
        Return pd.DataFrame

        Parameters
        ----------
        return_list :

        Returns
        -------

        """
        all_data = self.get_all_dataset_from_dir()
        if return_list:
            return all_data[all_data["dataset_name"] == self._dataset_name].reset_index(drop=True).values.tolist()
        else:
            return all_data[all_data["dataset_name"] == self._dataset_name].reset_index(drop=True)

    @DeprecationWarning
    def get_data_id(self, data_index) -> DataItem:
        all_data = self.get_data_ids(return_list=False)
        assert data_index < all_data.shape[
            0], f"The dataset exp_index = {data_index} is out of range. except [0 ~ {all_data.shape[0] - 1}]"
        curdata = all_data[all_data["dataset_name"] == self._dataset_name].reset_index(drop=True).iloc[data_index, :]
        return DataItem(curdata[0], curdata[1])

    def __read_data_csv(self, dataset_name, data_id):
        assert data_id is not None, "data_id is required"
        file = f"{self._dataset_home}/{dataset_name}/{data_id}"
        try:
            df = pd.read_csv(file, header=None, names=["value", "label"])
            df = df.fillna(0)
            return df
        except Exception as e:
            raise RuntimeError(f"File is not find: {file}")

    @DeprecationWarning
    @classmethod
    def debug_dataset(cls):
        """
        Generate the debug dataset.
        return:

         [
            [  1  11  21  31  41  51  61  71  81  91]
            [101 111 121 131 141 151 161 171 181 191]
            [201 211 221 231 241 251 261 271 281 291]
            [301 311 321 331 341 351 361 371 381 391]
            [401 411 421 431 441 451 461 471 481 491]
            [501 511 521 531 541 551 561 571 581 591]
            [601 611 621 631 641 651 661 671 681 691]
            [701 711 721 731 741 751 761 771 781 791]
            [801 811 821 831 841 851 861 871 881 891]
            [901 911 921 931 941 951 961 971 981 991]
         ]
        Returns
        -------

        """
        data = np.arange(1, 1001, 10)
        data = data.reshape((10, 10))

        return data

    def _remove_anomaly_windows_by_flag(self, data, labels):
        """
        为了训练半监督学习方法，需要移除包含异常的滑动窗口，

        Remove anomaly sliding windows.
        X, labels = self._remove_anomaly_windows_by_flag(data, labels):

        Parameters
        ----------
        data :
        labels :

        Returns
        -------

        """
        if self._is_include_anomaly_windows is False:
            return self._remove_anomaly_windows(data, labels)
        else:
            return data, labels

    def _remove_anomaly_windows(self, data, labels):
        """
        Remove anomaly sliding windows and return the clean data.
        The clean data means the training data without anomaly windows.

        X, labels = self._remove_anomaly_windows_by_flag(data, labels):

        Parameters
        ----------
        data :
        labels :

        Returns
        -------

        """
        normal_windows_index = np.argwhere(labels == 0).reshape(-1)
        return data[normal_windows_index], labels[normal_windows_index]

    def _get_normal_windows(self, data, labels):
        """
        Remove anomaly sliding windows and return the clean data.
        The clean data means the training data without anomaly windows.

        X, labels = self._remove_anomaly_windows_by_flag(data, labels):

        Parameters
        ----------
        data :
        labels :

        Returns
        -------

        """
        anomaly_windows_index = np.argwhere(labels == 0).reshape(-1)
        return data[anomaly_windows_index], labels[anomaly_windows_index]

    def _get_anomaly_windows(self, data, labels):
        """
        Remove normal sliding windows and return the clean data.
        The clean data means the training data without anomaly windows.

        X, labels = self._remove_anomaly_windows_by_flag(data, labels):

        Parameters
        ----------
        data :
        labels :

        Returns
        -------

        """
        anomaly_windows_index = np.argwhere(labels == 1).reshape(-1)
        return data[anomaly_windows_index], labels[anomaly_windows_index]

    def _data_sampling(self, train_x, labels):

        if self._sampling_rate == -1 or self._sampling_rate == 1:
            log.debug(
                f"Skip sampling training data since sampling rate == {self._sampling_rate}")
            return train_x, labels

        source_data = np.hstack((train_x, np.expand_dims(labels, -1)))
        source_data_df = pd.DataFrame(source_data)

        # 根据数据量, 自动调节sample rates
        if self._data_scale_beta is not None and self._data_scale_beta > 0:
            UtilSys.is_debug_mode() and log.info("Data scale is enabled")
            _n_all_samples = train_x.shape[0]
            _beta = self._data_scale_beta / _n_all_samples
            _ori_sr = self._sampling_rate
            self._sampling_rate = self._sampling_rate * _beta
            if self._sampling_rate >= 1:
                self._sampling_rate = 0.99
            UtilSys.is_debug_mode() and log.info(
                f"Scale sr from [{_ori_sr}] to [{self._sampling_rate}] by [{_beta}], scale beta {self._data_scale_beta}, original sample x: {_n_all_samples}")
        else:
            UtilSys.is_debug_mode() and log.info("Data size auto scale is disabled")

        label_name = source_data_df.columns[-1]
        if self._data_sampling_method == SampleType.RANDOM:
            n_samples = self._get_sample_number(train_x.shape[0])
            sampled_data = source_data_df.sample(n_samples)

        elif self._data_sampling_method == SampleType.NORMAL_RANDOM:
            _all = []
            for label, group_data in source_data_df.groupby(by=label_name):
                if label == 1:
                    _all.append(group_data)
                else:
                    n_samples = self._get_sample_number(group_data.shape[0])
                    _all.append(group_data.sample(n_samples))
            sampled_data = pd.concat(_all)

        elif self._data_sampling_method == SampleType.STRATIFIED:
            _all = []
            for label, group_data in source_data_df.groupby(by=label_name):
                n_samples = self._get_sample_number(group_data.shape[0])
                _all.append(group_data.sample(n_samples))
            sampled_data = pd.concat(_all)

        elif self._data_sampling_method == SampleType.LHS:
            n_all = source_data_df.shape[0]
            n_samples = self._get_sample_number(train_x.shape[0])
            if n_all == n_samples:
                # all data is needed, skip sampling.
                sampled_data = source_data_df
            else:
                sampled_index = latin_hypercube_sampling(n_all, n_samples)
                # Only sampling in the normal set
                sampled_data = source_data_df.iloc[sampled_index, :]
                UtilSys.is_debug_mode() and log.info(
                    f"Sampling method is LHS with sampling exp_index shape: {len(sampled_index)}, first 10:\n{sampled_index[:10]}")
        elif self._data_sampling_method == SampleType.DIST1:
            n_samples = self._get_sample_number(train_x.shape[0])
            if n_samples <= 0:
                sampled_data = source_data_df.iloc[:1]
            else:

                n_each_samples = n_samples // 3
                dist = feature_(train_x)
                _index_small = np.where(dist <= 0.33)[0]
                _index_middle = np.where((dist > 0.33) & (dist < 0.67))[0]
                _index_large = np.where(dist >= 0.67)[0]

                _index_sampled_small = _index_small[latin_hypercube_sampling(_index_small.shape[0], n_each_samples)]
                _index_sampled_mid = _index_middle[latin_hypercube_sampling(_index_middle.shape[0], n_each_samples)]
                _index_sampled_large = _index_large[latin_hypercube_sampling(_index_large.shape[0], n_each_samples)]

                _all_sampled_index = np.concatenate([_index_sampled_small, _index_sampled_mid, _index_sampled_large])

                sampled_data = source_data_df.iloc[_all_sampled_index, :]
                UtilSys.is_debug_mode() and log.info(
                    f"Sampling method is {self._data_sampling_method} with sampling exp_index shape: {len(_all_sampled_index)}, first 10:\n{_all_sampled_index[:10]}")

        else:
            raise ValueError(f"Unsupported sample method [{self._data_sampling_method}]")

        sampled_x, sampled_label = sampled_data.iloc[:, :-1].values, sampled_data.iloc[:, -1].values
        log.debug("Data sampled info: ", dict(origin_x_shape=train_x.shape,
                                              origin_label_shape=labels.shape,
                                              sampled_x_shape=sampled_x.shape,
                                              sampled_label_shape=sampled_label.shape
                                              ))
        return sampled_x, sampled_label

    def _get_sample_number(self, n_train):
        """
        Convert the sample_rate to integer: how many data to sample.

        Parameters
        ----------
        n_train :

        Returns
        -------

        """
        sample_rate_float = convert_str_to_float(self._sampling_rate)

        if sample_rate_float >= 1:
            n_samples = sample_rate_float
            if self._data_sampling_method == SampleType.STRATIFIED:
                n_samples = n_samples / 2
        else:
            n_samples = sample_rate_float * n_train
        n_samples = np.min([n_train, n_samples])
        n_samples = int(n_samples)
        return n_samples

    def _get_split_data(self, train_x, train_y):
        """
        Return the splited data according to self._test_rate.
        """
        assert self._test_rate > 0, "test_rate must be larger than 0."
        n_train = int(len(train_y) * (1 - self._test_rate))
        return train_x[:n_train], train_y[:n_train], train_x[n_train:], train_y[n_train:]

    def _get_split_data_with_fold(self, train_x, train_y, fold_index):
        """
        Return the splited data according to self._test_rate.
        """
        assert self._test_rate > 0, "test_rate must be larger than 0."
        n_train = int(len(train_y) * (1 - self._test_rate))
        return train_x[:n_train], train_y[:n_train], train_x[n_train:], train_y[n_train:]

    @DeprecationWarning
    @classmethod
    def init_filter_dataset(cls):
        files = FileUtil().get_all_files(os.path.join(os.path.dirname(__file__), "./benchmark"))
        target_data = []
        for f in files:
            attr_arr = str(f).split("/")
            dataset_name = attr_arr[-2]
            dataset_id = attr_arr[-1]
            dl = DatasetLoader(dataset_name, dataset_id,
                               data_sampling_method=SampleType.STRATIFIED,
                               sample_rate=-1,
                               window_size=1,
                               test_rate=0.3,
                               processing=False)

            train_x, train_y, test_x, test_y = dl.get_sliding_windows_train_and_test()

            flag1 = np.sum(test_y)
            flat2 = np.sum(train_y)
            if flag1 > 0 and flat2 > 0 and train_x.shape[0] > 100:
                print("👍👍👍", dataset_name, dataset_id)
                target_data.append([dataset_name, dataset_id])

        data = pd.DataFrame(target_data)
        data.to_csv(
            os.path.join(os.path.dirname(__file__), "has_anomaly_in_test_set.csv"),
            index=False)
        return data

    @DeprecationWarning
    @staticmethod
    def get_debug_dataset(window_size=64, is_include_anomaly_window=False):
        """
        生成10组训练数据量不一样的数据。

        Parameters
        ----------
        window_size :

        Returns
        -------

        """
        aarr = []
        for i in np.linspace(1, 1000, 10):
            dataset, data_id = ["ECG", "MBA_ECG805_data.out"]
            dl = KFoldDatasetLoader(dataset, data_id,
                                    window_size=window_size,
                                    is_include_anomaly_window=False,
                                    max_length=10000,
                                    sample_rate=i)
            train_x, train_y, test_x, test_y = dl.get_kfold_sliding_windows_train_and_test_by_fold_index(1)
            aarr.append([train_x, train_y, test_x, test_y])

        return aarr

    @DeprecationWarning
    def parse_dataset_name_and_id_from_file(self, file_path):
        fs = str(file_path).split(os.sep)
        self._dataset_name = fs[-2]
        self._data_id = fs[-1]
        return self._dataset_name, self._data_id

    @staticmethod
    def is_test_y_valid(data: np.ndarray) -> bool:
        """
        如果 data 同时包含0和1, 返回 true

        否则返回false

        Parameters
        ----------
        data :

        Returns
        -------

        """
        if np.min(data) == 0 and np.max(data) == 1:
            return True
        else:
            return False


class KFDL(KFoldDatasetLoader):
    pass


mem = JLUtil.get_memory()


@mem.cache
def get_debug_datasize(window_size, is_include_anomaly_window):
    return KFoldDatasetLoader.get_debug_dataset(window_size, is_include_anomaly_window)


if __name__ == '__main__':
    # data = _get_selected_avaliable_data()
    # print(data)
    dts = UTSDataset.select_datasets_split(dataset_names=[UTSDataset.DATASET_NASA_SMAP, "SMD"], top_n=9999)
    rich.print(dts)
