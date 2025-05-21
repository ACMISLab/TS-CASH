import os.path
import re
import sys

import numpy as np

from pylibs.utils.util_joblib import JLUtil

mem = JLUtil.get_memory()


def _get_diff1d(perf_keys, all_columns):
    return np.setdiff1d(all_columns, perf_keys)


class ExcelMetricsKeys:
    EMPTY = ""
    VERBOSE = "verbose"
    ORI_TRAIN_TIME = "$T_{ori}$"
    FAST_PERF = "$ACC_{fast}$"
    FOLD_INDEX = "fold_index"
    FILE_SIZE = "file_size"
    LENGTH = "length"
    LABEL = "label"
    VALUE = 'value'
    MEAN = "mean"
    STD = "std"
    MEDIA = "median"
    MEDIAN = "median"
    MODEL_TYPE = "model type"
    STOP_ALPHA = "Stop Alpha"
    TARGET_PERF = "opt_target"
    DATA_ANOMALY_RATE = "data_anomaly_rate"
    SORT_KEY = "sort_key"
    SPEED_UP = "speedup"
    ORI_PERF_MEAN = "ori. perf. mean"
    ORI_PERF_STD = "ori. perf. std"
    FAST_PERF_MEAN = "fast. perf. mean"
    FAST_PERF_STD = "fast. perf. std"
    ORI_TRAIN_TIME_MEAN = "ori. train time mean"
    ORI_TRAIN_TIME_STD = "ori. train time std"
    FAST_TRAIN_TIME_MEAN = "fast. train time mean"
    FAST_TRAIN_TIME_STD = "fast. train time std"
    FAST_BEST_SR = "fast. best sr"
    FAST_BEST_SR_MEAN = "fast. best sr mean"
    FAST_BEST_SR_STD = "fast. best sr std"
    P_VALUE = "p-value"
    FAST_BEST_TRAIN_LEN = "fast. best train len"
    FAST_DATA_PROCESSING_TIME_MEAN = "fast. data proc. time mean"
    FAST_DATA_PROCESSING_TIME_STD = "fast. data proc. time std"

    DATA_PROCESSING_TIME = "data_processing_time"
    TRAIN_LEN = 'train_len'
    TEST_LEN = 'test_len'
    PRECISION = "Precision"
    RECALL = "Recall"
    F = "F"
    AUC_ROC = "AUC_ROC"
    AUC_PR = "AUC_PR"
    PRECISION_AT_K = "Precision_at_k"
    RPRECISION = "Rprecision"
    RRECALL = "Rrecall"
    RF = "RF"
    R_AUC_ROC = "R_AUC_ROC"
    R_AUC_PR = "R_AUC_PR"
    VUS_ROC = "VUS_ROC"
    VUS_PR = "VUS_PR"

    TIME_APP_START = "time_app_start"
    TIME_APP_END = "time_app_end"
    TIME_TRAIN_START = "time_train_start"
    TIME_TRAIN_END = "time_train_end"
    TIME_EVAL_START = "time_eval_start"
    TIME_EVAL_END = "time_eval_end"
    ELAPSED_TRAIN = "elapsed_train"
    ANOMALY_WINDOW_TYPE = "anomaly_window_type"
    DATA_ID = "data_id"
    DATA_SAMPLE_METHOD = "data_sample_method"
    DATA_SAMPLE_RATE = "data_sample_rate"
    DATA_SAMPLE_RATE_PERCENTAGE = "data_sample_rate_percentage"
    DATASET_NAME = "dataset_name"
    DEBUG = "debug"
    EXP_INDEX = "exp_index"
    EXP_NAME = "exp_name"
    EXP_TOTAL = "exp_total"
    JOB_ID = "job_id"
    METRICS_SAVE_HOME = "metrics_save_home"
    MODEL_NAME = "model_name"
    SEED = "seed"
    TEST_RATE = "test_rate"
    WINDOW_SIZE = "window_size"
    EXP_ID = "exp_id"
    ELAPSED_BETWEEN_TRAIN_START_AND_EVAL_END = 'elapsed_between_train_start_and_eval_end'

    PERFORMENCE_KEYS = [
        PRECISION,
        RECALL,
        F,
        AUC_ROC,
        AUC_PR,
        PRECISION_AT_K,
        RPRECISION,
        RRECALL,
        RF,
        R_AUC_ROC,
        R_AUC_PR,
        VUS_ROC,
        VUS_PR,
        ELAPSED_TRAIN,
        TRAIN_LEN,
        TEST_LEN,
        DATA_PROCESSING_TIME,
        ELAPSED_BETWEEN_TRAIN_START_AND_EVAL_END
    ]

    @staticmethod
    def get_none_performance_keys(all_columns):
        """
        获取所有非性能指标的Keys

        Returns
        -------

        """
        # out_attr = []
        # # 获取ExcelMetricsKeys类的所有属性
        # attributes = [attr for attr in dir(ExcelMetricsKeys) if
        #               not callable(getattr(ExcelMetricsKeys, attr)) and not attr.startswith("__")]
        #
        # # 打印属性
        # for attribute in attributes:
        #     if attribute in ExcelMetricsKeys.PERFORMENCE_KEYS:
        #         pass
        #     else:
        #         out_attr.append(attribute)
        # return out_attr
        return list(_get_diff1d(ExcelMetricsKeys.PERFORMENCE_KEYS, all_columns))

    @staticmethod
    def get_performance_keys():
        return ExcelMetricsKeys.PERFORMENCE_KEYS

    @classmethod
    def post_process_latex(cls, latex_table):
        latex_table = latex_table.replace('model_name', 'Models')
        latex_table = latex_table.replace('dataset_name', 'Dataset Name')
        latex_table = latex_table.replace('_sr', '\_sr')
        return latex_table

    @staticmethod
    def convert_dataset_name(dataset_name):
        maps = {
            'OPPORTUNITY': "OPP"
        }
        return maps.get(dataset_name) if maps.get(dataset_name) is not None else dataset_name

    @classmethod
    def convert_sr(cls, param):
        if param > 999:
            return 1
        return np.round(param, 2)


class PaperFormat:
    RF = "RF"
    TIME_SCALE = 1 / 3600
    DATA_SAMPLE_RATE = "s_r (%)"
    STOP_ALPHA = "$\\alpha$"
    HOME_LATEX_HOME = "/Users/sunwu/SW-Research/sw-research-code/A01-papers/01_less_is_nore_pvldb/data_new/"
    HOME_LATEX_HOME_IET = "/Users/sunwu/SW-Research/sw-research-code/A01-papers/02_less_is_more_final/figs_and_tables/"
    VUS_ROC = "VUS ROC"
    VUS_PR = "VUS PR"

    DATA_SAMPLE_METHOD = "Sample Method"
    EXCLUDE_METHODS = [
        "decision_tree",  # 不需要这个方法,因为在论文中没有找到引用. 好像是我拍脑袋拍出来的
    ]
    model_name_maps = {
        "random_forest": "RF",
        "iforest": "IForest",
        "lof": "LOF",
        "pca": "PCA",
        "hbos": "HBOS",
        "decision_tree": "DT",
        "vae": "VAE",
        "ae": "AE",
        "cnn": "CNN",
        "OPPORTUNITY": "OPP",
        "Daphnet": "DAP",
        "lhs": "LHS",
        "dist1": "Dist",
        "dist": "Dist",
        "random": "Random",
        'tadgan': "TadGAN",
        "coca": "COCA",
        "ocsvm": "OCSVM",
        "knn": "KNN",
        'dagmm': "DAGMM",
        'lstm-ad': "LSTM-AD",
    }

    P_VALUE = "p-value"
    BEST_SR = r"$s_r^*$ (\%)"
    MODEL = "Model"
    ORI_PERF = "Ori. Acc."
    FAST_PERF = "Fast. Acc."
    ORI_TIME = "Ori. Time"
    FAST_TIME = "Fast. Time"
    DATA_PROC_TIME = "Proc. Time"
    SPEED_UP = "Speedup"
    EMPTY = ""
    @staticmethod
    def get_label_name():
        _name = os.path.basename(sys.argv[0]).lower()
        # remove version. 比如移除 v4_effective 中的 v4_, 有下划线
        _name = re.sub("v\d+[_-]", "", _name)
        # remove .py
        _name = re.sub(".py", "", _name)
        return _name

    @staticmethod
    def maps(key):
        if PaperFormat.model_name_maps.get(key) is not None:
            return PaperFormat.model_name_maps.get(key)
        return key

    @staticmethod
    def format_value(val):
        return PaperFormat.maps(val)


class EK(ExcelMetricsKeys):

    def __init__(self):
        pass


class PF(PaperFormat):
    def __init__(self):
        pass


if __name__ == '__main__':
    print(ExcelMetricsKeys.get_performance_keys())
    print(ExcelMetricsKeys.get_none_performance_keys())
