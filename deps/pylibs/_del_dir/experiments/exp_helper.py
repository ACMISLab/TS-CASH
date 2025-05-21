import copy
import multiprocessing
import pprint
import sys
import threading
import time
from multiprocessing import Manager

from joblib import Parallel, delayed, hashing
from tqdm import tqdm

from pylibs._del_dir.experiments.exp_config import load_model, ExpConf
from pylibs.exp_ana_common.ExcelOutKeys import EK
from pylibs.utils.util_bash import CMD, UtilBash
from pylibs.utils.util_common import UtilComm
from pylibs.utils.util_gpu import UGPU
from pylibs.utils.util_joblib import JLUtil
from pylibs.utils.util_log import get_logger, logconf
from pylibs.utils.util_number import str_to_float
from pylibs.utils.util_sys import get_gpu_count
from pylibs.utils.util_system import UtilSys
from pylibs.utils.util_tqdm import set_percent
from pylibs.uts_dataset.dataset_loader import KFoldDatasetLoader, DatasetLoader, DataProcessingType

log = get_logger()
memory = JLUtil.get_memory()


def get_models():
    if is_ml_jobs():
        _models = ['lof', 'hbos', 'ocsvm', 'iforest']
    else:
        _models = ['lstm-ad', 'cnn', 'ae', 'vae', 'dagmm']

    return _models


# mp.set_start_method("fork")
def set_debug_mode():
    os.environ["LOG_LEVEL"] = "DEBUG"


def is_debug_mode():
    return os.environ.get("LOG_LEVEL") is not None


def _print_end_job(exp_name):
    print(f"✅✅✅ All tasks for [{exp_name}] has mark_as_finished!")


def get_joblib_arg_hash(job: ExpConf):
    """
    这个函数用来定位 joblib 的缓存存储路径, 用来备份的.

    模型训练的缓存路径: /remote-home/acmis_fdd/joblib_cache/joblib/pylibs/experiments/exp_helper/_train_model_kfold/

    Parameters
    ----------
    job :

    Returns
    -------

    """
    _conf = _reprocess_config(job)
    _input = {
        'params': _conf.encode_key_params_bs64()
    }
    arg_hash = hashing.hash(_input, coerce_mmap=False)
    return arg_hash


def fsnotify(fun):
    def wrap(*args, **kwargs):
        jobs = args[0]
        if len(jobs) > 0:
            exp_name = args[0][0].exp_name
            FSUtil.send_msg_to_feishu(f"🚀 Experiment [{exp_name}] has started with [{len(jobs)}] tasks.")
            fun(*args, **kwargs)
            FSUtil.send_msg_to_feishu(f"✅All experiments for [{exp_name}] is mark_as_finished")
            _print_end_job(exp_name)

    return wrap


def main_fast_uts(job: ExpConf):
    try:
        FastUTS(job).find()
    except KeyboardInterrupt:
        pass
    except:
        FSUtil.report_error_trace_to_feishu(job.encode_key_params_bs64())
        traceback.print_exc()


class ExpConfiSet:
    @staticmethod
    def get_optimal_perf():
        target_perf = UtilSys.get_environ_by_key("OPT_TARGET_PERF")
        if target_perf is None:
            target_perf = "VUS_ROC"
        print(f"当前优化目标: {target_perf}", end="\r")
        return target_perf

    @staticmethod
    def get_stop_criteria():
        target_perf = UtilSys.get_environ_by_key("OPT_STOP_CRITERIA")
        if target_perf is None:
            target_perf = "VUS_ROC"
        print(f"当前优化目标: {target_perf}", end="\r")
        return target_perf


def get_optimal_perf():
    return ExpConfiSet.get_optimal_perf()


@DeprecationWarning
def main_fast_uts_gpu(job, exp_name):
    try:
        FastUTS().find(job)
    except KeyboardInterrupt:
        pass
    except Exception:
        FSUtil.report_error_trace_to_feishu(job.get_parameters())
        traceback.print_exc()


@fsnotify
def debug_sk_jobs(original_jobs):
    log.info("Debug mode on MacOS!")
    debug_jobs = get_debug_jobs(original_jobs)
    for job in debug_jobs:
        train_models_kfold_v2(job)


def debug_dl_jobs(original_jobs):
    debug_sk_jobs(original_jobs)


@fsnotify
def start_ml_jobs(original_jobs):
    Parallel(n_jobs=int(os.cpu_count() * 0.98), verbose=0, batch_size=1)(
        delayed(train_models_kfold)(job) for job in tqdm(original_jobs, position=0))


@fsnotify
def start_sk_fastuts_jobs(original_jobs):
    Parallel(n_jobs=int(os.cpu_count() * 0.98), verbose=0, batch_size=1)(
        delayed(main_fast_uts)(job) for job in tqdm(original_jobs, position=0))


@fsnotify
def start_tf_fastuts_jobs(original_jobs):
    gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    gma = ParallelFastUTS(gpus, parallel_for_each_gpu=8)
    gma.run(original_jobs)


@fsnotify
def debug_tf_fastuts_jobs(original_jobs):
    original_jobs = get_debug_jobs(original_jobs)
    for job in original_jobs:
        FastUTS(job, gpu_index=1).find()


@fsnotify
def debug_fastuts_jobs(original_jobs):
    original_jobs = get_debug_jobs(original_jobs)
    for job in original_jobs:
        FastUTS(job, gpu_index=1).find()


@fsnotify
def debug_sk_fastuts_jobs(original_jobs):
    assert len(original_jobs) > 0, "❌❌❌  Jobs array cant be empty!"
    log.info("Debug mode on MacOS!")
    exp_name = original_jobs[0].exp_name
    for job in original_jobs:
        main_fast_uts(job)


@fsnotify
def start_dl_jobs(original_jobs, parallel_for_each_gpu=8):
    assert len(original_jobs) > 0, "❌❌❌  Jobs array cant be empty!"
    gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    gma = ParallelJobs(gpus, parallel_for_each_gpu=parallel_for_each_gpu)
    gma.run(original_jobs)


@fsnotify
def start_fastuts_jobs(original_jobs, parallel_on_each_device: int = 8, devices_index: list = None):
    """
    AutoRun Jobs

    Parameters
    ----------
    devices_index :
    original_jobs :
    parallel_on_each_device :

    Returns
    -------

    """
    if UtilSys.is_macos():
        original_jobs = original_jobs[:5]

    assert len(original_jobs) > 0, "❌❌❌  Jobs array cant be empty!"

    if devices_index is None:
        # 8 GPU cards in a device
        devices_index = [0, 1, 2, 3, 4, 5, 6, 7]
    if is_ml_jobs():
        devices_index = [i for i in range(int(os.cpu_count() * 0.9))]
        parallel_on_each_device = 1
    else:
        if not UtilSys.is_macos():
            assert get_gpu_count() > 0, "Deep jobs are runing on GPU devices, but not find GPU devices"
        parallel_on_each_device = parallel_on_each_device

    if UtilSys.is_macos():
        devices_index = [0, 1, 2, 3]
        parallel_on_each_device = 1

    gma = ParallelFastUTS(devices_index, parallel_on_each_device=parallel_on_each_device)
    gma.run(original_jobs)


@fsnotify
def start_jobs(original_jobs, parallel_on_each_device: int = 3, devices_index: list = None,
               debug_on_macos: bool = True):
    """
    AutoRun Jobs

    Parameters
    ----------
    devices_index :
    original_jobs :
    parallel_on_each_device :

    Returns
    -------

    """
    if UtilSys.is_macos() and debug_on_macos:
        original_jobs = original_jobs[:20]
        for conf in original_jobs:
            train_models_kfold_v2(conf)
    else:
        assert len(original_jobs) > 0, "❌❌❌  Jobs array cant be empty!"
        if devices_index is None:
            # 8 GPU cards in a device
            devices_index = [0, 1, 2, 3, 4, 5, 6, 7]
        if is_ml_jobs():
            devices_index = [i for i in range(int(os.cpu_count() * 0.80))]
            parallel_on_each_device = 1
        else:
            if not UtilSys.is_macos():
                assert get_gpu_count() > 0, "Deep jobs are runing on GPU devices, but not find GPU devices"
            parallel_on_each_device = parallel_on_each_device

        gma = ParallelJobs(devices_index, parallel_on_each_device=parallel_on_each_device)
        gma.run(original_jobs)
    print("✅ Job down!")


class ExpHelper:
    DATA_SAMPLE_RATE = ["1/64", "1/32", "1/8", "1/4", "1/1"]
    AIOPS_KPI_E074 = ["e0747cad-8dc8-38a9-a9ab-855b61f5551d"]

    TEST_AIOPS_DATA_KPI = [""]
    TEST_SAMPLE_RATE = ["1/128"]

    DATA_NUMBERS = ["2", "8", "16", "32", "64", "128", "256", "512", "1024", "2048",
                    "4096", "8192", "16384"]


import argparse
import os.path
import traceback

import numpy as np
import pandas as pd

from pylibs.utils.util_feishu import FSUtil
from pylibs.utils.util_joblib import JLUtil
from pylibs.utils.util_log import get_logger
from pylibs.utils.util_hash import get_str_hash
from pylibs.utils.util_numpy import enable_numpy_reproduce

memory = JLUtil.get_memory()
log = get_logger()


def run_cmd(sc, cmd, end=""):
    t = threading.Thread(target=sc.exec, args=(cmd, end))
    t.start()  # start_or_restart thread

    return t


def parallel_run_cmd(sc, cmd, end="\r\n"):
    return run_cmd(sc, cmd, end)


class JobConfV1:
    """
    实验中配置
    """

    EXP_VERSION = "V09".lower()

    DATASET_ALL = ['Daphnet', 'ECG', 'GHL', 'IOPS', 'KDD21', 'MGAB', 'MITDB', 'NAB', 'NASA-MSL', 'NASA-SMAP',
                   'OPPORTUNITY', 'Occupancy', 'SMD', 'SVDB', 'SensorScope', 'YAHOO']
    DATASET_SELECTED_ALL = ["SVDB", "Daphnet", "ECG", "OPPORTUNITY", "IOPS", "SMD", "YAHOO", "MGAB"]
    ANOMALY_WINDOW_TYPE = "all"

    SAMPLE_RATES_V402 = ["0.01", "0.05", "0.10", "0.15", "0.19", "0.30", "0.50", "0.75", "0.95"]
    SAMPLE_RATES_V403 = ["1/10000", "1/1000", "0.01", "0.05", "0.10", "0.15", "0.19", "0.30", "0.50", "0.75", "0.95"]
    SAMPLE_RATES_V404 = ["1/1000", "0.01", "0.05", "0.10", "0.15", "0.19", "0.30", "0.50", "0.75", "0.95"]
    SAMPLE_RATES_V405 = ["0", "0.01", "0.05", "0.10", "0.15", "0.19", "0.30", "0.50", "0.75", "0.95"]

    SAMPLE_RATES_V406 = ["0.01", "0.05", "0.10", "0.15", "0.20", "0.30", "0.40", "0.50", "0.60", "0.70", "0.80", "0.90"]
    # ['0', '256', '512', '768', '1024', '1280', '1536', '1792',...]

    # 抽样间隔64
    SAMPLE_RATES_V920 = [str(64 * i) for i in range(200)]
    # 抽样间隔128
    SAMPLE_RATES_V921 = [str(128 * i) for i in range(200)]

    # 抽样间隔256
    SAMPLE_RATES_V922 = [str(256 * i) for i in range(200)]

    # 抽样间隔512
    SAMPLE_RATES_V923 = [str(512 * i) for i in range(200)]

    # 抽样间隔1024
    SAMPLE_RATES_V924 = [str(1024 * i) for i in range(200)]

    SAMPLE_RATES_OBSERVATION_SMALL = ['0', '20/1000', '40/1000', '60/1000',
                                      '80/1000', '100/1000', '120/1000',
                                      '140/1000', '160/1000', '180/1000', '200/1000', '220/1000', '240/1000',
                                      '260/1000',
                                      '280/1000', '300/1000', '320/1000', '340/1000', '360/1000', '380/1000',
                                      '400/1000',
                                      '420/1000', '440/1000', '460/1000', '480/1000', '500/1000', '520/1000',
                                      '540/1000',
                                      '560/1000', '580/1000', '600/1000', '620/1000', '640/1000', '660/1000',
                                      '680/1000',
                                      '700/1000', '720/1000', '740/1000', '760/1000', '780/1000', '800/1000',
                                      '820/1000',
                                      '840/1000', '860/1000', '880/1000', '900/1000', '920/1000', '940/1000',
                                      '960/1000',
                                      '980/1000', "-1"]
    # [f"{i}/1000" for i in range(1000)][::40]
    SAMPLE_RATES_OBSERVATION = ['0',
                                '40/1000', '80/1000', '120/1000', '160/1000', '200/1000', '240/1000',
                                '280/1000', '320/1000', '360/1000', '400/1000', '440/1000', '480/1000', '520/1000',
                                '560/1000', '600/1000', '640/1000', '680/1000', '720/1000', '760/1000', '800/1000',
                                '840/1000', '880/1000', '920/1000', '960/1000', "-1"]
    SAMPLE_RATES_OBSERVATION_V2 = ['1', '2', '4', '8', '16', '32', '64', '128', '256', '512', '1024', '2048', '4096',
                                   '8192', '16384', '32768', '65536', '131072', '262144', '524288', "-1"]
    SAMPLE_RATES_OBSERVATION_V3 = ['1', '2', '4', '8', '16', '32', '64', '128', '256', '512', '1024', '2048']
    SAMPLE_RATES_OBSERVATION_V4 = ["8", '16', '32', '64', '128', '256', '512',
                                   '1024', '2048', '3072', '4096', '5120',
                                   '6144', '7168', '8192', '9216', '10240',
                                   '11264', '12288', '13312', '14336', '15360',
                                   '16384', '17408', '18432', '19456',
                                   '20480', '21504', '22528', '23552', '24576',
                                   '25600', '26624', '27648', '28672',
                                   '29696', '30720', '31744', '32768', '33792',
                                   '34816', '35840', '36864', '37888',
                                   '38912', '39936', '40960', '41984', '43008',
                                   '44032', '45056', '46080', '47104',
                                   '48128', '49152', '50176']
    SAMPLE_RATES_OBSERVATION_V5 = ["8", '16', '32', '64', '128', '256', '512',
                                   '1024', '2048', '3072', '4096', '5120',
                                   '6144', '7168', '8192', '9216', '10240',
                                   '11264', '12288', '13312', '14336', '15360',
                                   '16384', "-1"]
    SAMPLE_RATES_OBSERVATION_V6 = ["8", '16', '32', '64', '128', '256', '512',
                                   '1024', '2048', '3072', '4096', '5120',
                                   '6144', '7168', '8192', '9216', '10240', "-1"]
    SAMPLE_RATES_OBSERVATION_V7 = ["2", "4", "8", '16', '32', '64', '128', '256', '512',
                                   '1024', '2048', '3072', '4096', '5120',
                                   '6144', '7168', '8192', '9216', '10240', "-1"]
    SAMPLE_RATES_OBSERVATION_V8 = ["2", "4", "8", '16', '32', '64', '128', '256', '512',
                                   '1024', '2048', '3072', '4096', '5120',
                                   '6144', '7168', '8192', '9216', '10240']

    SAMPLE_RATES_OBSERVATION_SINGLE_DATASET = [str(i / 100) for i in range(30)][1:] + [str(-1)]

    SAMPLE_RATES_DEBUGS = ["2", "3", "4", "5", "6", "7", "8", '16', '32', '64', '128', '256', '512',
                           '1024', '4096']

    @staticmethod
    def get_exp_version():
        exp_ver = UtilSys.get_environ_by_key("EXP_VERSION")
        if exp_ver is not None:
            return exp_ver
        else:
            return JobConfV1.EXP_VERSION
            # raise RuntimeError("Env EXP_VERSION is not found")

    @staticmethod
    def get_sample_rates():
        if JobConfV1.get_exp_version() == "V402":
            return JobConfV1.SAMPLE_RATES_V402
        if JobConfV1.get_exp_version() == "V403":
            return JobConfV1.SAMPLE_RATES_V403
        if JobConfV1.get_exp_version() == "V404":
            return JobConfV1.SAMPLE_RATES_V404
        if JobConfV1.get_exp_version() == "V405":
            return JobConfV1.SAMPLE_RATES_V405

        return JobConfV1.SAMPLE_RATES_V404

    SAMPLE_METHODS = [
        "random",
        "dist1",
        "lhs",
    ]

    OPT_TARGET_DEFAULT = "VUS_ROC"
    OPT_TARGET_SUP1 = "VUS_PR"
    OPT_TARGET_SUP2 = "RF"
    OPT_TARGETS = [OPT_TARGET_SUP2, OPT_TARGET_DEFAULT, OPT_TARGET_SUP1]
    DATA_SCALE_BETA = "-1"
    # DATASET_SEG_SMALL = ["Daphnet", "OPPORTUNITY", "SMD"]
    # DATASET_SEG_SMALL = ["ECG", "MGAB", "SMD"]
    DATASET_DAP = ["Daphnet"]
    DATASET_ECG = ["ECG"]
    DATASET_NAB = ["NAB"]
    DATASET_TEST = ["YAHOO"]
    DATASET_OBSERVATION = ["Daphnet", "SMD", "YAHOO"]
    DATASET_SEG_SMALL = ["Daphnet", "OPPORTUNITY", "MGAB"]
    DATASET_SEG_LARGE = ["SVDB", "ECG", "IOPS", "SMD"]
    DATASET = DATASET_SEG_SMALL + DATASET_SEG_LARGE + DATASET_OBSERVATION
    DATASET_LARGE = ["MITDB"],
    DATASET_V41 = ["SMD", "Daphnet", "IOPS"]
    # "decision_tree"
    MODEL_SKLEARN = ["hbos", "pca", "lof", "iforest", 'random_forest', 'ocsvm']
    # MODEL_OBSERVATION_ML = ["hbos", "knn", "lof", "iforest", 'random_forest', 'ocsvm']
    MODEL_OBSERVATION_ML = ["hbos", "iforest", "lof", 'ocsvm']

    # all:  ['ae', 'vae', 'dagmm', 'lstm', 'tadgan', 'coca']
    # MODEL_OBSERVATION_DL = ['ae', 'vae', 'dagmm', 'lstm', 'coca', 'tadgan']
    # MODEL_OBSERVATION_DL = ['ae', 'dagmm', 'coca']
    MODEL_OBSERVATION_DL = ['ae', 'vae', 'lstm-ad', 'cnn', 'dagmm']

    MODEL_TORCH = ['ae', 'vae', 'dagmm', 'lstm', 'tadgan', 'coca']
    MODEL_SKLEARN_SLOW = ["ocsvm"]
    MODEL_TF = ["lstm", "vae", "ae", "cnn", 'dagmm']
    MODEL_TF_AE_CNN = ["cnn", 'ae']

    # 选择的单变量时间序了数据的数量
    SELECTED_N_UNIVARIATE = "15"
    TEST_DATA_V60 = ["Daphnet&S09R01E4.test.csv@3.out",
                     "Daphnet&S09R01E4.test.csv@2.out",
                     "SMD&machine-2-8.test.csv@16.out",
                     "SMD&machine-3-11.test.csv@6.out"]

    # 第一次尝试用的抽样算法, 默认抽样算法
    DEFAULT_DATA_SAMPLE_METHOD = "random"
    RANDOM_SAMPLE = "random"
    # 第二次尝试用的抽样算法
    DEFAULT_DATA_SAMPLE_METHOD_V2 = "lhs"
    DEFAULT_DATA_SAMPLE_METHOD_V3 = "dist1"
    # 交叉验证的次数
    K_FOLD = "5"

    # 模型训练的次数
    EPOCH = "1" if UtilSys.is_macos() else "50"
    STOP_ALPHA = "0.001"
    STOP_ALPHA_DEFAULT = "0.001"
    STOP_ALPHA_SUP2 = ["0.001", "0.01", "0.1", "0.5"]

    BASELINE_SAMPLE_RATE = "-1"

    BATCH_SIZE = "128"

    SAMPLE_GAP_VERSIONS = {
        "v4010_64": SAMPLE_RATES_V920,
        "v4010_128": SAMPLE_RATES_V921,
        "v4010_256": SAMPLE_RATES_V922,
        "v4010_512": SAMPLE_RATES_V923,
        "v4010_1024": SAMPLE_RATES_V924
    }
    SAMPLE_GAP_VERSIONS_OLD = {
        "v2000_64": SAMPLE_RATES_V920,
        "v2001_128": SAMPLE_RATES_V921,
        "v2002_256": SAMPLE_RATES_V922,
        "v2003_512": SAMPLE_RATES_V923,
        "v2004_1024": SAMPLE_RATES_V924
    }

    @classmethod
    def get_sample_gap(cls, _v):
        sample_gat = JobConfV1.SAMPLE_GAP_VERSIONS.get(_v)
        return int(sample_gat[1]) - int(sample_gat[0])


def is_debug_enabled(args) -> bool:
    flag = args.prod is False
    return flag


def post_process_metrics(conf: ExpConf, dth, score, test_y, **args):
    metrics = UTSMetricHelper.get_metrics_all_cache(labels=test_y, score=score, window_size=conf.window_size)
    if metrics is None:
        return None
    dth.end()
    # post processing metrics data
    metrics.update(args)
    metrics.update(dth.collect_metrics())
    metrics.update(conf.get_dict())
    metrics.update({"exp_id": conf.get_exp_id(), 'job_id': conf.job_id})
    return metrics


def get_metric_file_name(conf):
    try:
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "results",
            conf.exp_name,
            conf.model_name,
            conf.dataset_name,
            f"{conf.get_exp_id()}_metrics.csv"
        )
    except:
        traceback.print_exc()
        pass


def load_dataset(conf: ExpConf):
    is_include_anomaly_window = True
    if conf.model_name == "ocsvm":
        is_include_anomaly_window = False

    dl = DatasetLoader(
        conf.dataset_name,
        conf.data_id,
        sample_rate=conf.data_sample_rate,
        window_size=conf.window_size,
        is_include_anomaly_window=is_include_anomaly_window,
        processing="standardization",
        anomaly_window_type=conf.anomaly_window_type,
        fill_nan=True
    )
    train_x, train_y, test_x, test_y = dl.get_sliding_windows(return_origin=True)
    return train_x, train_y, test_x, test_y


def load_dataset_v2(conf: ExpConf):
    is_include_anomaly_window = True
    if conf.model_name == "ocsvm":
        is_include_anomaly_window = False

    dl = DatasetLoader(
        conf.dataset_name,
        conf.data_id,
        sample_rate=conf.data_sample_rate,
        window_size=conf.window_size,
        is_include_anomaly_window=is_include_anomaly_window,
        processing="False",
        anomaly_window_type=conf.anomaly_window_type,
        test_rate=conf.test_rate,
        fill_nan=True
    )
    res = dl.get_sliding_windows_train_and_test()
    if res is None:
        return None, None, None, None
    train_x, train_y, test_x, test_y = res
    return train_x, train_y, test_x, test_y


m = JLUtil.get_memory()


# @m.cache
def load_dataset_kfold(conf: ExpConf):
    is_include_anomaly_window = True
    if conf.is_semi_supervised_model():
        is_include_anomaly_window = False

    dl = KFoldDatasetLoader(
        conf.dataset_name,
        conf.data_id,
        sample_rate=conf.data_sample_rate,
        window_size=conf.window_size,
        is_include_anomaly_window=is_include_anomaly_window,
        processing=DataProcessingType.STANDARDIZATION,
        anomaly_window_type=conf.anomaly_window_type,
        test_rate=conf.test_rate,
        fill_nan=True,
    )

    return dl.get_kfold_sliding_windows_train_and_test_by_fold_index(conf.fold_index)


"""
Need too much disk, disable it"""


@memory.cache
@DeprecationWarning
def load_dataset_v2_cache(conf: ExpConf):
    is_include_anomaly_window = True
    if conf.model_name == "ocsvm":
        is_include_anomaly_window = False

    dl = DatasetLoader(
        conf.dataset_name,
        conf.data_id,
        sample_rate=conf.data_sample_rate,
        window_size=conf.window_size,
        is_include_anomaly_window=is_include_anomaly_window,
        processing="normal",
        anomaly_window_type=conf.anomaly_window_type,
        test_rate=conf.test_rate,
        fill_nan=True
    )
    train_x, train_y, test_x, test_y = dl.get_sliding_windows_train_and_test()
    return train_x, train_y, test_x, test_y


def get_all_selected_kpis(selected_dataset, n_kpis=None):
    """
    Contains 103 KPI with different length
    Returns

    [
        ['NASA-MSL', 'C-2.train.out'],
        .....
    ]

    -------

    Parameters:
    -----------
    selected_dataset: list
        A list of the selected datasets:
        selected_dataset = [
            "Genesis",
            "NASA-MSL",
            "NASA-SMAP",
            "Occupancy",
            "SensorScope",
            "MGAB",
            "IOPS",
            "NAB",
            "Daphnet"
            # "ECG", # data is two large
        ]
    n_kpis: int
        How many you want to choose
    """

    _all_kid = []
    for dataset in selected_dataset:
        _kpis = DatasetLoader(dataset).get_data_ids()
        if n_kpis is None:
            _all_kid = _all_kid + _kpis
        else:
            _all_kid = _all_kid + _kpis[0:np.min([n_kpis, len(_kpis)])]
    UtilSys.is_debug_mode() and log.info(f"KPI length: {len(selected_dataset)}")

    return _all_kid


def get_all_selected_kpis_v2(selected_dataset, n_kpis=10):
    """
    Contains 103 KPI with different length
    Returns

    [
        ['NASA-MSL', 'C-2.train.out'],
        .....
    ]

    -------

    Parameters:
    -----------
    selected_dataset: list
        A list of the selected datasets:
        selected_dataset = [
            "Genesis",
            "NASA-MSL",
            "NASA-SMAP",
            "Occupancy",
            "SensorScope",
            "MGAB",
            "IOPS",
            "NAB",
            "Daphnet"
            # "ECG", # data is two large
        ]
    n_kpis: int
        How many you want to choose
    """
    all_available_datasets = [['Daphnet', 'S01R02E0.test.csv@6.out'], ['Daphnet', 'S03R01E1.test.csv@8.out'],
                              ['Daphnet', 'S01R02E0.test.csv@4.out'], ['Daphnet', 'S03R01E0.test.csv@8.out'],
                              ['Daphnet', 'S01R02E0.test.csv@1.out'], ['Daphnet', 'S09R01E0.test.csv@8.out'],
                              ['Daphnet', 'S09R01E0.test.csv@9.out'], ['Daphnet', 'S09R01E0.test.csv@7.out'],
                              ['Daphnet', 'S09R01E0.test.csv@6.out'], ['Daphnet', 'S09R01E0.test.csv@4.out'],
                              ['ECG', 'MBA_ECG14046_data_27.out'], ['ECG', 'MBA_ECG14046_data_33.out'],
                              ['ECG', 'MBA_ECG14046_data_32.out'], ['ECG', 'MBA_ECG14046_data_26.out'],
                              ['ECG', 'MBA_ECG14046_data_30.out'], ['ECG', 'MBA_ECG14046_data_24.out'],
                              ['ECG', 'MBA_ECG820_data.out'], ['ECG', 'MBA_ECG14046_data_18.out'],
                              ['ECG', 'MBA_ECG14046_data_19.out'], ['ECG', 'MBA_ECG14046_data_25.out'],
                              ['IOPS', 'KPI-a8c06b47-cc41-3738-9110-12df0ee4c721.train.out'],
                              ['IOPS', 'KPI-6d1114ae-be04-3c46-b5aa-be1a003a57cd.train.out'],
                              ['IOPS', 'KPI-6efa3a07-4544-34a0-b921-a155bd1a05e8.train.out'],
                              ['IOPS', 'KPI-847e8ecc-f8d2-3a93-9107-f367a0aab37d.test.out'],
                              ['IOPS', 'KPI-4d2af31a-9916-3d9f-8a8e-8a268a48c095.train.out'],
                              ['IOPS', 'KPI-57051487-3a40-3828-9084-a12f7f23ee38.train.out'],
                              ['IOPS', 'KPI-f0932edd-6400-3e63-9559-0a9860a1baa9.test.out'],
                              ['IOPS', 'KPI-0efb375b-b902-3661-ab23-9a0bb799f4e3.test.out'],
                              ['IOPS', 'KPI-ffb82d38-5f00-37db-abc0-5d2e4e4cb6aa.test.out'],
                              ['IOPS', 'KPI-a07ac296-de40-3a7c-8df3-91f642cc14d0.test.out'], ['MGAB', '1.test.out'],
                              ['MGAB', '6.test.out'], ['MGAB', '7.test.out'], ['MGAB', '2.test.out'],
                              ['MGAB', '3.test.out'], ['MGAB', '10.test.out'], ['MGAB', '8.test.out'],
                              ['MGAB', '9.test.out'], ['MGAB', '5.test.out'], ['MGAB', '4.test.out'],
                              ['MITDB', '109.test.csv@1.out'], ['MITDB', '234.test.csv@2.out'],
                              ['MITDB', '221.test.csv@2.out'], ['MITDB', '233.test.csv@1.out'],
                              ['MITDB', '109.test.csv@2.out'], ['MITDB', '234.test.csv@1.out'],
                              ['MITDB', '221.test.csv@1.out'], ['MITDB', '233.test.csv@2.out'],
                              ['MITDB', '100.test.csv@2.out'], ['MITDB', '100.test.csv@1.out'],
                              ['NAB', 'NAB_data_Traffic_3.out'], ['NAB', 'NAB_data_Traffic_6.out'],
                              ['NAB', 'NAB_data_CloudWatch_14.out'], ['NAB', 'NAB_data_tweets_2.out'],
                              ['NAB', 'NAB_data_CloudWatch_3.out'], ['NAB', 'NAB_data_CloudWatch_10.out'],
                              ['NAB', 'NAB_data_tweets_7.out'], ['NAB', 'NAB_data_CloudWatch_13.out'],
                              ['NAB', 'NAB_data_CloudWatch_12.out'], ['NAB', 'NAB_data_tweets_4.out'],
                              ['NASA-MSL', 'C-1.test.out'], ['NASA-MSL', 'T-8.test.out'], ['NASA-MSL', 'M-4.test.out'],
                              ['NASA-MSL', 'M-2.test.out'], ['NASA-MSL', 'M-3.test.out'], ['NASA-MSL', 'C-2.test.out'],
                              ['NASA-MSL', 'T-13.test.out'], ['NASA-MSL', 'D-14.test.out'],
                              ['NASA-MSL', 'D-15.test.out'], ['NASA-MSL', 'M-1.test.out'],
                              ['NASA-SMAP', 'E-5.test.out'], ['NASA-SMAP', 'E-4.test.out'],
                              ['NASA-SMAP', 'D-7.test.out'], ['NASA-SMAP', 'E-12.test.out'],
                              ['NASA-SMAP', 'E-13.test.out'], ['NASA-SMAP', 'E-2.test.out'],
                              ['NASA-SMAP', 'E-3.test.out'], ['NASA-SMAP', 'D-1.test.out'],
                              ['NASA-SMAP', 'E-8.test.out'], ['NASA-SMAP', 'E-9.test.out'],
                              ['OPPORTUNITY', 'S1-ADL5.test.csv@114.out'], ['OPPORTUNITY', 'S4-ADL2.test.csv@79.out'],
                              ['OPPORTUNITY', 'S4-ADL5.test.csv@55.out'], ['OPPORTUNITY', 'S4-ADL5.test.csv@96.out'],
                              ['OPPORTUNITY', 'S2-ADL3.test.csv@38.out'], ['OPPORTUNITY', 'S3-ADL1.test.csv@50.out'],
                              ['OPPORTUNITY', 'S1-ADL5.test.csv@40.out'], ['OPPORTUNITY', 'S1-ADL2.test.csv@78.out'],
                              ['OPPORTUNITY', 'S1-ADL5.test.csv@54.out'], ['OPPORTUNITY', 'S1-ADL2.test.csv@44.out'],
                              ['Occupancy', 'room-occupancy-1.test.csv@3.out'],
                              ['Occupancy', 'room-occupancy-0.test.csv@5.out'],
                              ['Occupancy', 'room-occupancy-0.test.csv@4.out'],
                              ['Occupancy', 'room-occupancy-1.test.csv@4.out'],
                              ['Occupancy', 'room-occupancy-0.test.csv@1.out'],
                              ['Occupancy', 'room-occupancy-0.test.csv@3.out'],
                              ['Occupancy', 'room-occupancy-0.test.csv@2.out'],
                              ['Occupancy', 'room-occupancy.train.csv@5.out'],
                              ['Occupancy', 'room-occupancy.train.csv@4.out'],
                              ['Occupancy', 'room-occupancy.train.csv@3.out'], ['SMD', 'machine-3-6.test.csv@31.out'],
                              ['SMD', 'machine-3-1.test.csv@35.out'], ['SMD', 'machine-1-2.test.csv@13.out'],
                              ['SMD', 'machine-1-2.test.csv@9.out'], ['SMD', 'machine-2-3.test.csv@7.out'],
                              ['SMD', 'machine-3-10.test.csv@3.out'], ['SMD', 'machine-3-10.test.csv@2.out'],
                              ['SMD', 'machine-2-3.test.csv@6.out'], ['SMD', 'machine-2-5.test.csv@1.out'],
                              ['SMD', 'machine-1-2.test.csv@12.out'], ['SVDB', '849.test.csv@2.out'],
                              ['SVDB', '825.test.csv@2.out'], ['SVDB', '841.test.csv@2.out'],
                              ['SVDB', '854.test.csv@2.out'], ['SVDB', '853.test.csv@1.out'],
                              ['SVDB', '849.test.csv@1.out'], ['SVDB', '825.test.csv@1.out'],
                              ['SVDB', '841.test.csv@1.out'], ['SVDB', '853.test.csv@2.out'],
                              ['SVDB', '824.test.csv@1.out'], ['SensorScope', 'stb-19.test.out'],
                              ['SensorScope', 'stb-18.test.out'], ['SensorScope', 'stb-13.test.out'],
                              ['SensorScope', 'stb-12.test.out'], ['SensorScope', 'stb-31.test.out'],
                              ['SensorScope', 'stb-20.test.out'], ['SensorScope', 'stb-14.test.out'],
                              ['SensorScope', 'stb-15.test.out'], ['SensorScope', 'stb-7.test.out'],
                              ['SensorScope', 'stb-6.test.out'], ['YAHOO', 'YahooA4Benchmark-TS8_data.out'],
                              ['YAHOO', 'YahooA4Benchmark-TS58_data.out'], ['YAHOO', 'YahooA3Benchmark-TS88_data.out'],
                              ['YAHOO', 'YahooA3Benchmark-TS89_data.out'], ['YAHOO', 'YahooA4Benchmark-TS11_data.out'],
                              ['YAHOO', 'YahooA4Benchmark-TS82_data.out'], ['YAHOO', 'YahooA3Benchmark-TS53_data.out'],
                              ['YAHOO', 'YahooA4Benchmark-TS67_data.out'], ['YAHOO', 'YahooA3Benchmark-TS24_data.out'],
                              ['YAHOO', 'YahooA3Benchmark-TS25_data.out']]
    all_available_data = pd.DataFrame(all_available_datasets, columns=["dataset_name", "dataid"])
    _all_kid = []

    for dataset_name, _data in all_available_data.groupby(by="dataset_name"):
        if dataset_name in selected_dataset:
            if n_kpis is None:
                _all_kid = _all_kid + _data.values.tolist()
            else:
                _all_kid = _all_kid + _data[0:np.min([n_kpis, len(_data)])].values.tolist()
    UtilSys.is_debug_mode() and log.info(f"KPI length: {len(selected_dataset)}")

    return _all_kid


def get_debug_jobs(original_jobs):
    """
    每个模型+一个数据

    Returns
    -------

    """
    debug_jobs = []
    _cached = []
    for job in original_jobs:
        if job.model_name in _cached:
            continue
        _cached.append(job.model_name)
        job.dataset_name = "IOPS"
        job.kfold = 2
        job.data_id = "KPI-301c70d8-1630-35ac-8f96-bc1b6f4359ea.train.out"
        log.info(
            f"Debug mode, set data to {job.dataset_name}:{job.data_id}, k_fold={job.kfold} "
            f"for [{job.model_name}]")
        debug_jobs.append(job)
    return debug_jobs


mem = JLUtil.get_memory()


def _write_progress(conf: ExpConf):
    with open(os.path.join(os.path.expanduser('~'), "fast_progress.txt"), "w") as f:
        f.write(f"{pprint.pformat(conf.__dict__)}")


def _reprocess_config(conf: ExpConf) -> ExpConf:
    _conf = copy.deepcopy(conf)
    # 移除与模型性能不相关得参数(控制FastUTS的参数),保证已训练的模型的可重用性

    # 这3个参数与模型的性能无关. 他是用来决定FastUTS的停止条件的, 设置成什么, 对模型的性能都无所谓无所谓
    # 这里固定的目的是: 使用joblib 缓存同一个模型即可
    _conf.stop_alpha = 0.001
    _conf.fastuts_sample_rate = None
    _conf.optimal_target = "VUS_ROC"
    return _conf


def train_models_kfold(conf: ExpConf):
    try:
        enable_numpy_reproduce(conf.seed)
        _write_progress(conf)
        if conf.is_computed_score():
            UtilSys.is_debug_mode() and log.info(f"OptMetricsType is existed at \n{conf.get_metrics_file_name()}\n")
            return None
        _conf = _reprocess_config(conf)
        metrics = _train_model_kfold_v2(_conf.encode_key_params_bs64())
        metrics = _post_process_metrics(conf, metrics)
        _print_model_metrics(conf, metrics)
        return metrics
    except Exception as e:
        log.error(traceback.format_exc())
        FSUtil.report_error_trace_to_feishu(conf.encode_key_params_bs64())
        UtilSys.is_debug_mode() and log.info(f"❌❌❌ Experiment [{conf.get_exp_id()}] is error")
        return False


def train_models_kfold_v2(conf: ExpConf):
    try:
        enable_numpy_reproduce(conf.seed)
        # _write_progress(conf)
        if conf.is_metrics_file_exists():
            UtilSys.is_debug_mode() and logconf(f"Exp has computed at {conf.get_metrics_file_name()}")
            return None

        _conf = _reprocess_config(conf)
        UtilSys.is_debug_mode() and logconf(f"Exp conf: \n{pprint.pformat(_conf.key_dicts())}")

        score, dth = _train_model_kfold_v2(_conf.encode_key_params_bs64())
        if score is None:
            metrics = {}
            UtilSys.is_debug_mode() and log.warn(f"Score is {score}.")
        else:
            metrics = conf.calculate_metrcs_by_conf_score_dth(conf, score, dth)

        # conf.save_score_dth_exp_conf(score, dth)
        # metrics = conf.get_metrics()
        metrics = _post_process_metrics(conf, metrics)

        conf.save_metrics_to_file(metrics)


    except Exception as e:
        log.error(traceback.format_exc())
        FSUtil.report_error_trace_to_feishu(conf.encode_key_params_bs64() + "\n\n" + " ".join(sys.argv))
        UtilSys.is_debug_mode() and log.info(f"❌❌❌ Experiment [{conf.get_exp_id()}] is error")
        return False


def _post_update_metric(conf, metrics):
    # 确保所有信息是最新的,这是信息不是性能信息,不重要
    metrics.update(conf.__dict__)
    metrics.update({
        "data_processing_time": conf.get_data_processing_time(),
        "config": conf.encode_configs_bs64(),
        "debug": UtilSys.is_debug_mode()
    })
    return metrics


def train_models_kfold_online(conf: ExpConf) -> dict:
    """
    训练并返回多折交叉验证的结果. 返回当前五折交叉验证的模型性能结果

    Parameters
    ----------
    conf :

    Returns
    -------

    """
    try:
        # load dataset by conf
        out = []
        for _fold_index in range(conf.kfold):
            # assign fold exp_index
            conf.fold_index = _fold_index

            # ---
            if conf.is_compute_metrics():
                tqdm.write(f"OptMetricsType {conf.get_metrics_file_name()} is computed!")
                metrics = conf.load_metrics_as_dict()
            else:
                tqdm.write(f"🚕Computing {conf.get_metrics_file_name()}...")
                score, dth = _train_model_kfold_v2(conf.encode_key_params_bs64())
                metrics = conf.calculate_metrcs_by_conf_score_dth(conf, score, dth)
            metrics = _post_process_metrics(conf, metrics)
            # ---

            # metrics = conf.get_metrics()
            # metrics = _post_process_metrics(conf, metrics)

            # metrics is a dict
            # remove the failed metrics
            if metrics is not None and metrics[EK.VUS_ROC] >= 0:
                out.append(metrics)

            if not conf.is_metrics_file_exists():
                conf.save_metrics_to_file(metrics)

        metrics = pd.DataFrame(out)
        if metrics.shape[0] == 0:
            return None
        else:
            metric_mean = metrics.mean(numeric_only=True)
            return metric_mean.to_dict()
    except Exception as e:
        log.error(traceback.format_exc())
        FSUtil.report_error_trace_to_feishu(conf.encode_key_params_bs64())
        traceback.print_exc()
        UtilSys.is_debug_mode() and log.info(
            f"❌❌❌ Experiment [{conf.get_exp_id()}] is error. config: {conf.get_dump_file_path()}")
        return None


def _post_process_metrics(conf, metrics):
    if metrics is not None:
        metrics = _post_update_metric(conf, metrics)

    # conf.save_metrics_to_file(metrics)
    return metrics


def save_detect_picture(conf, metrics, score, test_x, test_y):
    from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView
    uv = UnivariateTimeSeriesView(
        name=conf.get_exp_id(),
        dataset_name=conf.dataset_name,
        dataset_id=conf.data_id,
        is_save_fig=True,
        home=conf.get_image_save_home()
    )
    uv.plot_x_label_score_metrics_row2(test_x[:, -1], test_y, score, metrics)


def parse_args(command=None):
    if command is not None:
        log.info(f"Params: \n python  {sys.argv[0]}  {' '.join(command)}\n")
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--exp_name",
                        help="The experiment name, used to save results. recommend: version+expname, e.g. V700_baseline",
                        default=f"debug_experiment")
    parser.add_argument("-t", "--anomaly_window_type", help="How to decide a anomaly sliding windows.", default="all")
    parser.add_argument("--n_jobs", help="The number of jobs to parallel", default=-1, type=int)  # all cores=64
    parser.add_argument("--data_sample_methods", default=["random"], help="The experiment name, used to save results.",
                        nargs="+")
    parser.add_argument("--datasets",
                        default=["IOPS"],
                        nargs="+")
    parser.add_argument("--models",
                        default=["iforest"],
                        nargs="+")
    parser.add_argument("--test_rate", default=0.3, type=float)
    parser.add_argument("--window_size", default=64, type=int, help="The size of the rolling window.")
    parser.add_argument("--sample_rates", default=[-1], nargs="+")
    parser.add_argument("--kfold", default=5, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--epoch", default=50, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--dataset_top", default=15, type=int, help="How many data to select in a given dataset."
                                                                    "Default 10, which means select 10 dataset to represent"
                                                                    "the dataset.")
    parser.add_argument("--datasets_and_ids", default=[], nargs="+",
                        required=False,
                        help="Specify the dataset and data id, split by &, such as "
                             "IOPS&KPI-54350a12-7a9d-3ca8-b81f-f886b9d156fd.test.out"
                             "If this parameter is existed, the --datasets will be ignored")
    parser.add_argument("--fastuts_sample_rate", default=[], nargs="+",
                        required=False)
    parser.add_argument("--stop_alpha", default=0.001, type=float,
                        required=False)
    parser.add_argument("--opt_target", default="VUS_ROC", type=str,
                        required=False)
    parser.add_argument("--data_scale_beta", default=None, type=float,
                        required=False)

    args = parser.parse_args(command)

    return args


def _update_exp_name(args):
    _sample_methods = "_".join(args.data_sample_methods)
    _f_name = str(os.path.basename(sys.argv[0])).split(".")[0]
    return f"{JobConfV1.get_exp_version()}_{_f_name}_{args.opt_target}_{args.stop_alpha}_{_sample_methods}"


def get_job_type():
    """
    获取是机器学习任务还是深度学习任务

    Returns
    -------

    """
    model_type = str(UtilSys.get_environ_by_key("JOBS_TYPE")).lower()
    if model_type not in ["dl", "ml"]:
        sys.exit("Environment JOBS_TYPE is not specified. can only ml or dl")
    return model_type


def is_ml_jobs():
    """
    是否是 ml 类型的任务.

    Returns
    -------

    """
    return get_job_type() == "ml"


def clear_cache():
    # 	rm -rf /Users/sunwu/joblib_cache
    # 	rm -rf /Users/sunwu/SW-Research/runtime
    UtilBash.run_command_print_progress("rm -rf /Users/sunwu/joblib_cache")
    UtilBash.run_command_print_progress("rm -rf /Users/sunwu/SW-Research/runtime")


def generate_jobs_baseline(args, auto_update=False):
    """
    根据配置生成任务

    名字是自动生的
    Parameters
    ----------
    args :

    Returns
    -------

    """
    enable_numpy_reproduce(args.seed)
    # assert UtilSys.get_environ_by_key("EXP_NAME"), "Env EXP_NAME cant be empty"
    # args.exp_name = UtilSys.get_environ_by_key("EXP_NAME")
    args.exp_name = _update_exp_name(args)
    if len(args.datasets_and_ids) > 0:
        kpis = set(args.datasets_and_ids)
        selected_kpis = [str(kpi).split("&") for kpi in kpis]
    else:
        selected_kpis = KFoldDatasetLoader.select_top_data_ids(dataset_names=args.datasets,
                                                               top_n=args.dataset_top,
                                                               seed=args.seed)
    UtilSys.is_debug_mode() and log.info(f"Selected dataset length: {len(selected_kpis)}. ")

    jobs_array = []
    _index = 0

    sample_rates = [str_to_float(sr) for sr in args.sample_rates]
    job_id = get_str_hash(str(args))
    for _sr in sample_rates:
        for _model_name in args.models:
            for _dataset_name, _data_id in selected_kpis:
                for _data_sample_method in args.data_sample_methods:
                    for _fold_index in range(args.kfold):
                        _index += 1
                        _conf = ExpConf(
                            data_sample_method=_data_sample_method,
                            data_sample_rate=_sr,
                            model_name=_model_name,
                            dataset_name=_dataset_name,
                            data_id=_data_id,
                            exp_index=_index,
                            exp_name=args.exp_name,
                            anomaly_window_type=args.anomaly_window_type,
                            test_rate=args.test_rate,
                            window_size=args.window_size,
                            seed=args.seed,
                            metrics_save_home=UtilComm.get_runtime_directory(),
                            job_id=job_id,
                            fold_index=_fold_index,
                            epoch=args.epoch,
                            batch_size=args.batch_size,
                            data_scale_beta=args.data_scale_beta,
                            optimal_target=args.opt_target
                        )
                        if not _conf.is_compute_metrics():
                            jobs_array.append(_conf)
                        else:
                            log.info(f"✅ OptMetricsType file is exist at {_conf.get_metrics_file_name()}")

    n_experiments = len(jobs_array)
    print(f"Exp. name: {args.exp_name}. Jobs Count: {n_experiments}")
    for job in jobs_array:
        job.update_exp_total(n_experiments)
    return jobs_array


def generate_jobs_fast_uts(args):
    """
    FastUTS 任务生成器.

    这里的 FoldIndex 是交叉验证的次数.

    Parameters
    ----------
    args :

    Returns
    -------

    """

    enable_numpy_reproduce(args.seed)
    args.exp_name = _update_exp_name(args)
    UtilSys.is_debug_mode() and log.info(f"Experiment parameters: \n{args}\n")
    if len(args.datasets_and_ids) > 0:
        kpis = set(args.datasets_and_ids)
        selected_kpis = [str(kpi).split("&") for kpi in kpis]
        args.datasets = None
        args.dataset_top = None

    else:
        selected_kpis = DatasetLoader.select_top_data_ids(args.datasets, args.dataset_top, args.seed)
    UtilSys.is_debug_mode() and log.info(f"Selected dataset length: {len(selected_kpis)}. ")

    jobs_array = []
    _index = 0

    fastuts_sample_rates = [str_to_float(sr) for sr in args.fastuts_sample_rate]
    job_id = get_str_hash(str(args))
    for _model_name in args.models:
        for _dataset_name, _data_id in selected_kpis:
            for _data_sample_method in args.data_sample_methods:
                _index += 1
                _conf = ExpConf(
                    data_sample_method=_data_sample_method,
                    fastuts_sample_rate=fastuts_sample_rates,
                    model_name=_model_name,
                    dataset_name=_dataset_name,
                    data_id=_data_id,
                    stop_alpha=args.stop_alpha,
                    exp_index=_index,
                    exp_name=args.exp_name,
                    anomaly_window_type=args.anomaly_window_type,
                    test_rate=args.test_rate,
                    window_size=args.window_size,
                    seed=args.seed,
                    metrics_save_home=UtilComm.get_runtime_directory(),
                    job_id=job_id,
                    kfold=args.kfold,
                    epoch=args.epoch,
                    data_scale_beta=args.data_scale_beta,
                    batch_size=args.batch_size,
                    optimal_target=args.opt_target
                )
                jobs_array.append(_conf)

    n_experiments = len(jobs_array)
    for job in jobs_array:
        job.update_exp_total(n_experiments)

    # if UtilSys.is_debug_mode() or UtilSys.is_macos():
    #     jobs_array = get_debug_jobs(jobs_array)
    # print(f"Job count: {len(jobs_array)}, Exp. name: {args.exp_name}, config\n"
    #       f"{pprint.pformat(args.__dict__)}")
    return jobs_array


def main_cpu_v5(args):
    jobs_array = generate_jobs_baseline(args)
    Parallel(n_jobs=args.n_jobs, verbose=0)(
        delayed(train_models_kfold)(job) for job in tqdm(jobs_array, position=0))
    UtilSys.is_debug_mode() and log.info("✅✅✅ Tasks mark_as_finished!")
    FSUtil.send_msg_to_feishu(f"✅All experiments for [{args.exp_name}] is mark_as_finished")


def run_job_on_gpu(array, gpu_index):
    while len(array) > 0:
        conf: ExpConf = array.pop(0)

        if conf.is_computed_score():
            log.info(f"✅ Job with exp_index [{conf.exp_index}] is finished")
        else:
            UtilSys.is_debug_mode() and \
            log.info(f"🚕 Get job ({conf.exp_index}/{conf.exp_total}) at pid= {os.getpid()}, gpu_index={gpu_index}")

            conf.gpu_index = gpu_index
            # encode all parameters, not key parameters.
            params = conf.encode_configs_bs64()

            exe_cmd = f"export CUDA_VISIBLE_DEVICES={conf.gpu_index}; python baseline_model_executor.py --params {params}"
            CMD.exe_cmd(exe_cmd)


def _get_script_home():
    home = UtilSys.get_environ_by_key("SCRIPT_HOME")
    assert home is not None and os.path.isdir(
        home), "SCRIPT_HOME  must be specified by Environment Variable SCRIPT_HOME, such as export \nSCRIPT_HOME=/Users/sunwu/SW-Research/sw-research-code/A01_paper_exp"
    return home


def parallel_run_jobs(array, gpu_index):
    while len(array) > 0:
        conf: ExpConf = array.pop(0)
        if conf.is_compute_metrics():
            log.info(f"✅ Job with exp_index [{conf.exp_index}] is finished")
        else:
            UtilSys.is_debug_mode() and \
            log.info(f"🚕 Get job ({conf.exp_index}/{conf.exp_total}) at pid= {os.getpid()}, gpu_index={gpu_index}")

            conf.gpu_index = gpu_index
            # encode all parameters, not key parameters.
            params = conf.encode_configs_bs64()

            entry = os.path.join(_get_script_home(), "baseline_model_executor.py")
            exe_cmd = f"export CUDA_VISIBLE_DEVICES={conf.gpu_index}; python  {entry} --params {params}"
            CMD.exe_cmd(exe_cmd)


def run_job_gpu_tf_v0(array, gpu_index):
    while len(array) > 0:
        conf: ExpConf = array.pop(0)
        if UGPU.get_available_memory(gpu_index) < 4096 and not UtilSys.is_macos():
            UtilSys.is_debug_mode() and log.info(
                f"⚠️⚠️⚠️⚠️⚠️ No enough memory to execute experiment, waiting for 10 seconds.")
            # 如果显存不够，那就放回去，等待10秒
            array.append(conf)
            time.sleep(10)
            continue
        else:
            UtilSys.is_debug_mode() and log.info(
                f"🚕🚕🚕🚕🚕 Get job ({conf.exp_index}/{conf.exp_total}) at pid= {os.getpid()}, gpu_index={gpu_index}")
            conf.gpu_index = gpu_index
            assert conf.gpu_index is not None
            exe_cmd = f"export CUDA_VISIBLE_DEVICES={conf.gpu_index}; python baseline_model_executor.py --params {conf.encode_key_params_bs64()}"
            CMD.exe_cmd(exe_cmd)


def _print_model_metrics(conf, metrics):
    if UtilSys.is_debug_mode() or UtilSys.is_macos():
        log.info(
            f"✅✅✅ Model [{conf.model_name}]({conf.exp_name}) trained mark_as_finished. OptMetricsType:\n{metrics}\n")


def _train_model(conf, dth, train_x, train_y, test_x, test_y):
    # train models
    clf = load_model(conf)
    # UtilSys.is_debug_mode() and conf.print_data_info()
    dth.train_start()
    clf.fit(train_x, train_y)
    dth.train_end()

    # evaluate model by whole dataset
    score = clf.score(test_x)

    # calculate metrics
    metrics = post_process_metrics(conf, dth, score, test_y,
                                   train_len=train_x.shape[0],
                                   test_len=test_x.shape[0])

    _print_model_metrics(conf, metrics)

    return metrics


def _train_model_v2(conf: ExpConf, dth, train_x, train_y, test_x, test_y):
    # train models
    clf = load_model(conf)
    UtilSys.is_debug_mode() and conf.print_data_info()
    dth.train_start()
    clf.fit(train_x, train_y)
    dth.train_end()

    # evaluate model by whole dataset
    dth.evaluate_start()
    score = clf.score(test_x)
    dth.evaluate_end()
    return score, dth


from pylibs.utils.util_joblib import cache_


@DeprecationWarning
@cache_
def _train_model_kfold(params):
    conf = ExpConf.decode_configs_bs64(params)

    dth = DateTimeHelper()
    dth.start_or_restart()
    train_x, train_y, test_x, test_y = conf.load_dataset_at_fold_k()

    UtilSys.is_debug_mode() and log.info("Starting training on fold {}".format(conf.fold_index))
    if train_x is None or train_x.shape[0] == 0:
        UtilSys.is_debug_mode() and log.warning("Skipped. Training data is None, skipping training!")
        return None
    if test_y.sum() == 0 or train_x.shape[0] == 0:
        UtilSys.is_debug_mode() and log.warning("Skipped. Test label must contain one anomaly. ")
        return None
    metrics = _train_model(conf, dth, train_x, train_y, test_x, test_y)
    return metrics


@cache_
def _train_model_kfold_v2(params):
    # load conf
    conf = ExpConf.decode_configs_bs64(params)
    dth = DateTimeHelper()
    dth.start_or_restart()

    # load data and check
    train_x, train_y, test_x, test_y = conf.load_csv()
    UtilSys.is_debug_mode() and log.info("Starting training on fold {}".format(conf.fold_index))
    if train_x is None or train_x.shape[0] == 0:
        UtilSys.is_debug_mode() and log.warning("Skipped. Training data is None, skipping training!")
        return None, None

    if test_y.sum() == 0:
        UtilSys.is_debug_mode() and log.warning("Skipped. Test label is all normal!")
        return None, None
    # train
    clf = load_model(conf)
    UtilSys.is_debug_mode() and conf.print_data_info()
    dth.train_start()
    clf.fit(train_x, train_y)
    dth.train_end()

    # evaluate
    dth.evaluate_start()
    score = clf.score(test_x)
    from sklearn.preprocessing import MinMaxScaler
    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    dth.evaluate_end()
    return score, dth


class ParallelJobs:
    """
    Parallel run jobs on multi-GPUs and multi-CPUs
    """

    def __init__(self, devices_index, parallel_on_each_device=8):
        """

        Parameters
        ----------
        devices_index : the gpu_index or cpu_index for use
        parallel_on_each_device : parallel on each cpu or gpu
        """
        self._devices = devices_index
        assert np.min(self._devices) >= 0, f"GPUS must be greater than or equal to 0, but received {devices_index}"
        assert parallel_on_each_device >= 1, "parallel_for_each_gpu must be greater than 0,but received {" \
                                             "parallel_for_each_gpu}"
        self._parallel_on_each_device = parallel_on_each_device

    def run(self, original_jobs):
        n_jobs = len(original_jobs)
        with Manager() as manager:
            target_jobs_array = manager.list()
            for _job in original_jobs:
                if not _job.is_compute_metrics():
                    target_jobs_array.append(_job)

            processes = []
            # 支持多显卡
            for _device_index in self._devices:
                # 每个显卡上多并发
                for _ in range(self._parallel_on_each_device):
                    mp = multiprocessing.Process(target=parallel_run_jobs, args=(target_jobs_array, _device_index))
                    mp.start()
                    processes.append(mp)
            # process bar
            progress_bar = multiprocessing.Process(target=_progress_bar, args=(target_jobs_array, n_jobs))
            progress_bar.start()

            processes.append(progress_bar)
            for mp in processes:
                mp.join()


# @mem.cache
def clean_main_fast_uts_jobs(original_jobs):
    """
    只保留fold exp_index =0 的任务。因为这里不要写。
    Parameters
    ----------
    original_jobs :

    Returns
    -------

    """
    ret = []
    for job in original_jobs:
        if job.fold_index == 0:
            ret.append(job)
    return ret


class FastUTS:

    def __init__(self, conf: ExpConf, gpu_index=None,
                 _post_see=3):

        self._conf: ExpConf = conf
        self._stop_criteria = conf.stop_alpha
        self._target_perf = conf.optimal_target
        self._post_see = _post_see
        self._gpu_index = gpu_index
        # 找最优过程中的所有性能数据
        self._found_metrics = []

        # 找到的最好的数据
        self._best_found = None
        self._sample_rates = np.sort(self._init_sample_rates(self._conf))

    def find(self):
        UtilSys.is_debug_mode() and logconf(f"Opti sample rates: {self._sample_rates}")
        # _write_progress(self._conf)
        for _cur_sr in self._sample_rates:
            UtilSys.is_macos() and log.info(
                f"Try model on sr=[{_cur_sr}] for {self._conf.get_small_identifier()}")
            # 从最小的开始训练
            self._conf.data_sample_rate = _cur_sr

            # 返回当前五折交叉验证的模型性能结果. 性能考虑 VUS_ROC, PRECISION, F1
            _metrics = train_models_kfold_online(self._conf)

            # 过滤无效数据
            if _metrics is None:
                continue

            UtilSys.is_debug_mode() and logconf(f"Optimizing target [{self._target_perf}].")
            self._found_metrics.append(_metrics[self._target_perf])

            # 决定是否继续训练
            if not self._is_stopped():
                # 性能还在上升,继续训练
                continue
            else:
                # 性能已经不再上升了,停止训练
                break

        # 导出结果
        # self._conf.report_message_to_feishu()

    def _is_stopped(self):
        return self._is_stop_training_v2()

    def get_best_found(self):
        _best_found = pd.concat(self._found_metrics).sort_values(by=self._target_perf, ascending=False).iloc[
                      0:1]
        return _best_found

    @staticmethod
    def _init_sample_rates(conf):
        # train_len = conf.get_training_len()
        # return _get_sample_rates_v1(train_len)
        # return _get_sample_rates_v3()
        assert len(conf.fastuts_sample_rate) > 1, "fastuts_sample_rate is not specified"
        return conf.fastuts_sample_rate

    def _is_stop_training_v2(self):
        """
           相对于当前的性能，如果增加数据集两次后(post_see)，
           模型的性能没有提高0.001 (alpha)，那么我们就停止训练。

           例如：
           self.assertTrue(is_stopped_training([0, 0, 0]))
           self.assertFalse(is_stopped_training([0, 0.1, 0.01]))
           self.assertFalse(is_stopped_training([0, 0.001, 0.001]))
           """
        if len(self._found_metrics) < 3:
            UtilSys.is_debug_mode() and logconf(
                f"Need for more metrics to decide whether to stop training, current : {self._found_metrics}")
            return False
        metrics = np.asarray(self._found_metrics[-3:], dtype=np.float32)
        current = metrics[0]
        values = metrics - current
        # 只要有增加,就继续训练.
        increment = np.max(values[1:self._post_see + 1])
        is_stop = increment < self._stop_criteria
        msg = f"Model: {self._conf.model_name}, Dataset: {self._conf.dataset_name}, data id= {self._conf.data_id}\n" \
              f"===========> Stop find? = [{is_stop}]\nCur increment =[{increment}]," \
              f"post see={self._post_see}, stop alpha={self._stop_criteria}, decide boundary: {metrics}, all found so far: {self._found_metrics}\n"
        UtilSys.is_debug_mode() and logconf(msg)

        return is_stop


@mem.cache
def _get_sample_rates_v1(train_len):
    """
    第一个版本的增量递增策略,效果不好.
    见https://uniplore.feishu.cn/wiki/BsKKwfTHHiM7kikGQUNc7dFynvc

    根据输入数据的大小,自动返回合适的sample rate.

    如输入是 17025, 那么返回:
    [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 17025]

    Parameters
    ----------
    train_len :

    Returns
    -------

    """
    power = int(np.floor(np.log2(train_len)))
    srs = [2 ** i for i in range(power + 1)]
    srs.append(train_len)
    # 最少取16条数据
    srs = srs[4:]
    return srs


@mem.cache
def _get_sample_rates_v2():
    """
    第一个版本的增量递增策略,效果不好.
    见https://uniplore.feishu.cn/wiki/BsKKwfTHHiM7kikGQUNc7dFynvc



    Parameters
    ----------
    train_len :

    Returns
    -------

    """
    # 322% 最坏情况下,需要训练3倍的时间
    srs = [0.01, 0.05, 0.10, 0.15, 0.19, 0.30, 0.50, 0.75, 0.95]
    return srs


def _get_sample_rates_v3():
    """
    第一个版本的增量递增策略,效果不好.
    见https://uniplore.feishu.cn/wiki/BsKKwfTHHiM7kikGQUNc7dFynvc

    根据输入数据的大小,自动返回合适的sample rate.

    如输入是 17025, 那么返回:
    [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 17025]

    Parameters
    ----------
    train_len :

    Returns
    -------

    """
    # 322% 最坏情况下,需要训练3倍的时间
    srs = [0.01, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 0.95]
    return srs


def run_job_gpu_fastuts_tf(array, gpu_index=-1):
    while len(array) > 0:
        try:
            conf: ExpConf = array.pop(0)
            if not UtilSys.is_macos() and UGPU.get_available_memory(gpu_index) < 4048:
                log.warning(
                    f"⚠️⚠️⚠️⚠️⚠️ No enough memory to execute experiment, waiting for 10 seconds.")
                # 如果显存不够，那就放回去，等待10秒
                array.append(conf)
                time.sleep(10)
                continue
            else:
                UtilSys.is_debug_mode() and log.info(
                    f"🚕🚕🚕🚕🚕 Get job ({conf.exp_index}/{conf.exp_total}) at pid= {os.getpid()}, gpu_index={gpu_index}")
                conf.gpu_index = gpu_index
                assert conf.gpu_index is not None
                assert conf.fold_index is not None

                CMD.run_command_print_progress(
                    f"export CUDA_VISIBLE_DEVICES={conf.gpu_index}; python fastuts_model_executor.py --params {conf.encode_configs_bs64()}")
        except:
            traceback.print_exc()


def run_job_fastuts(array, gpu_index=-1):
    while len(array) > 0:
        try:
            conf: ExpConf = array.pop(0)
            UtilSys.is_debug_mode() and log.info(
                f"🚕🚕🚕🚕🚕 Get job ({conf.exp_index}/{conf.exp_total}) at pid= {os.getpid()}, gpu_index={gpu_index}")
            conf.gpu_index = gpu_index
            assert conf.gpu_index is not None
            assert conf.fold_index is not None

            entry = os.path.join(_get_script_home(), "fastuts_model_executor.py")
            CMD.run_command_print_progress(
                f"export CUDA_VISIBLE_DEVICES={conf.gpu_index}; python {entry} --params {conf.encode_configs_bs64()}")
        except:
            traceback.print_exc()


def _progress_bar(jobs, num_jobs):
    with tqdm(total=num_jobs) as pbar:
        _pre = num_jobs - len(jobs)
        set_percent(pbar, _pre, num_jobs)
        while len(jobs) > 0:
            _cur = num_jobs - len(jobs)
            if _cur == _pre:
                time.sleep(3)
                continue
            else:
                _pre = _cur
                conf: ExpConf = jobs[0]
                # update the progress
                pbar.set_postfix_str(
                    f"{conf.model_name}|{conf.data_sample_rate}|{conf.anomaly_window_type}|{conf.dataset_name}|{conf.exp_name}")
                set_percent(pbar, _cur, num_jobs)
                time.sleep(3)

        # While is break, so update the progress to full (all)
        set_percent(pbar, num_jobs)


class ParallelFastUTS:
    def __init__(self, gpus, parallel_on_each_device=8):
        self._gpus = gpus
        assert np.min(self._gpus) >= 0, f"GPUS must be greater than or equal to 0, but received {gpus}"
        assert parallel_on_each_device >= 1, "parallel_for_each_gpu must be greater than 0,but received {" \
                                             "parallel_for_each_gpu}"
        self._parallel_on_each_device = parallel_on_each_device
        self._pbar = None

    def run(self, original_jobs):
        j_jobs = len(original_jobs)

        with Manager() as manager:
            target_jobs_array = manager.list()
            for _job in original_jobs:
                if not _job.is_compute_metrics():
                    target_jobs_array.append(_job)

            processes = []
            # 支持多显卡
            for gpu_index in self._gpus:
                # 每个显卡上多并发
                for _ in range(self._parallel_on_each_device):
                    mp = multiprocessing.Process(target=run_job_fastuts, args=(target_jobs_array, gpu_index))
                    mp.start()
                    processes.append(mp)
            # process bar
            progress_bar = multiprocessing.Process(target=_progress_bar, args=(target_jobs_array, j_jobs))
            progress_bar.start()

            processes.append(progress_bar)
            for mp in processes:
                mp.join()


def get_observation_args():
    if len(sys.argv) > 2:
        args = parse_args()
    else:
        params = [
                     "--datasets", ] + JobConfV1.DATASET_OBSERVATION + [
                     "--kfold", JobConfV1.K_FOLD,
                     "--models", ] + get_models() + [
                     "--epoch", JobConfV1.EPOCH,
                     "--batch_size", JobConfV1.BATCH_SIZE,
                     "--data_sample_method", JobConfV1.RANDOM_SAMPLE,
                     "--anomaly_window_type", JobConfV1.ANOMALY_WINDOW_TYPE,
                     "--sample_rates"] + JobConfV1.SAMPLE_RATES_OBSERVATION_V8 + [
                     "--dataset_top", JobConfV1.SELECTED_N_UNIVARIATE,
                 ]
        args = parse_args(params)
        return args


if __name__ == '__main__':
    pass
    # print(train_models([['IOPS', 'KPI-0efb375b-b902-3661-ab23-9a0bb799f4e3.test.out'], 1]))
    # ExpConf().get_exp_id()

    # selected_dataset = [
    #     "Genesis",
    #     "NASA-MSL",
    #     "NASA-SMAP",
    #     "Occupancy",
    #     "SensorScope",
    #     "MGAB",
    #     "IOPS",
    #     "NAB",
    #     "Daphnet"
    #     # "ECG", # data is two large
    # ]
    # all_kpis = get_all_selected_kpis(selected_dataset, n_kpis=1)
    # assert len(all_kpis) == 9
    # all_kpis = get_all_selected_kpis(selected_dataset)
    # print(len(all_kpis))
    # assert len(all_kpis) == 372
    # _write_progress(ExpConf())
    #
    # # start_ml_jobs([[1], [2]], "fff")
