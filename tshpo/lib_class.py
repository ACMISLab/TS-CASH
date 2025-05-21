#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/10/22 10:27
# @Author  : gsunwu@163.com
# @File    : class_lib.py
# @Description:
import copy
import itertools
import os
import re
import shutil
import sys
import typing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace
from tqdm import tqdm

from pylibs.utils.util_md5 import get_str_md5
from pylibs.utils.util_rsync import Rsync
from pylibs.utils.util_servers import Servers
# 创建一个缓存目录
from tshpo.lib_func import get_memory
from tshpo.result_files import ResultFiles

DIR_CRT = os.path.dirname(__file__)
MODEL_SELECT_INFO_CACHE_FILE = os.path.join(DIR_CRT, "model_select_info.pkl")
memory = get_memory()
PROJECT_HOME = os.environ["TSHPO_HOME"]
DATA_BACK_HOME = f"{PROJECT_HOME}/fig_and_tables/data"


@dataclass
class ExpConf:
    """ ExpConf(dataset='kc1', folds=5, fold_index=0, random_state=42, config_file_name=None, metric='roc_auc', wall_time_limit_in_s=N..., hpo_opt_method=None, max_samples=None, n_exploration=None, data_sample_method=' stratified', data_sample_rate=None, hpc=None)
    """
    dataset: str = 'kc1'
    folds: int = 5
    fold_index: int = 0
    random_state: int = 42
    config_file_name: str = None
    # 精度指标，如f1,acc,recall
    metric: str = "roc_auc"
    wall_time_limit_in_s: typing.Union[int, None] = None  # 最长运行时间，None表示不限制

    debug: bool = False
    # bo参数
    # 初始化样本数
    bo_n_warm_start: int = None
    n_warm_start: int = None

    # 根据名字觉得是降维还是特征选择，RF特征选择，PCA降维，其他方法直接加即可
    feature_selec_method: typing.Union[str, float] = "RF",
    feature_selec_rate: typing.Union[None, float] = 0.3,
    # 我自己方法所用的参数
    # 初试数据大小
    init_data_size: int = None
    # 初试训练用的迭代次数，默认500
    warm_start_selection_iteration: int = None

    # 入口文件夹的名称极其参数
    entry_file_name = os.path.basename(sys.argv[0])  # 入口文件
    # entry_args = str(sys.argv[1:])  # 入口文件的参数

    # 启发式算法的参数，用于自动觉得训练数据量
    sample_step: int = None
    stop_threshold: typing.Union[int, float] = None
    max_iteration: typing.Union[int, None] = None

    # 选中的模型名称
    model_name: typing.Union[None, str] = None

    # 是否修剪超参数空间
    is_trim_sp: bool = False

    # 维度降维方法
    # dim_redu_method: str = None
    # important feature ratio, None 表示选择全部特征，0-1之前的数表示选择百分比特征
    # dim_redu_rate: typing.Union[float, None] = None
    # 转用feature_selec_method 和 feature_selec_rate

    # 剪枝超参数空间时，保留的算法的数量
    n_high_performing_model: typing.Union[int, None] = None
    n_samples_for_each_model: typing.Union[int, None] = None

    # 超参数优化算法，例如贝叶斯优化，随机优化，Hyperband等
    # opt_method: typing.Union[str, None] = None
    # 超参数优化算法，例如贝叶斯优化，随机优化，Hyperband等
    hpo_opt_method: typing.Union[str, None] = None

    # 最多选择多少样本（行）,None 使用全部行，否者随机选择max_samples行
    max_samples: typing.Union[None, int] = None

    # HPO之前，需要探索每个算法在少量数据集上的进度，这个是探索的次数
    n_exploration: typing.Union[int, None] = None

    # "stratified","random"

    data_sample_method: typing.Union[str, None] = " stratified"
    data_sample_rate: typing.Union[float, None] = None

    # 随机优化时，配置可以单独到每个实验，目的是为了加快速度
    hpc: Configuration = None

    def get_id(self):
        return get_str_md5(str(self.__dict__))


@dataclass
class RunJob:
    """
    定义一个待运行的任务
    """
    X_train: np.ndarray
    y_train: np.ndarray

    X_test: np.ndarray
    y_test: np.ndarray
    # 精度指标，如f1，acc，recall
    metric: str

    # 模型的超参数配置
    config: Configuration

    # 超参数搜索空间
    cs: ConfigurationSpace

    debug: bool

    seed: int

    exp_conf: ExpConf = None

    alg_db_cash_file: str = os.path.join(os.environ["TSHPO_HOME"], "tshpo/tshpo_alg_perf.sqlite")

    def __post_init__(self):
        assert self.X_train.shape[0] == self.y_train.shape[0]
        assert self.X_test.shape[0] == self.y_test.shape[0]

    def is_cache_avaiable(self):
        """如果不对数据进行抽样，就缓存"""

        return self.exp_conf.feature_selec_rate == 1 and self.exp_conf.data_sample_rate == 1


@dataclass
class RunValue:
    """
    定义一个待运行的任务, 返回的值是原始值，不处理最大化还是最小的问题
    """
    default: float = -1  # 执行超参数优化时，只会用这个值，不会考虑其他值
    roc_auc: float = -1
    f1: float = -1
    accuracy: float = -1
    recall: float = -1
    log_loss: float = -1
    precision: float = -1

    elapsed_seconds: float = -1

    error_msg: str = ""
    run_job: dict = None

    exp_conf: dict = None

    # 弃用，但是保留字段，因为分析的时候要用结构
    is_error: str = None

    def get_dict(self):
        _t = {
            "default": round(self.default, 4),
            "roc_auc": round(self.roc_auc, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "recall": round(self.recall, 4),
            "log_loss": round(self.log_loss, 4),
            "precision": round(self.precision, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            # "error_msg": self.error_msg,
        }
        return _t


class AnaHelper:
    """
    roc_auc: float = None
    f1: float = None
    accuracy: float = None
    recall: float = None
    log_loss: float = None
    precision: float = None
    """
    # 决定是最大化任务还是最小化任务
    df_ = None
    important_feature_ratio = "important_feature_ratio"
    max_samples = "max_samples"
    dataset = "dataset"
    max_iterations = "max_iterations"
    metric = "metric"
    opt_method = "hpo_opt_method"
    n_high_performing_model = "n_high_performing_model"
    is_trim_sp = "is_trim_sp",
    N_ALGS: int = 14
    mean_without_trim_cs = "mean_without_trim_cs"
    mean_with_trim_cs = "mean_with_trim_cs"

    df_baseline_: pd.DataFrame = None
    df_dim_redu_: pd.DataFrame = None
    DIM_REDU_METHOD = f"{PROJECT_HOME}/results/08-effect-of-dim-redu-method/08_effect_of_optimize_round_and_dimension_size_v3_original_20240923_1823.csv"
    METRIC_FILE_BASELINE = f"{PROJECT_HOME}/results/13_ablation_test/13_test_ablation_study_rs_baseline_original_20241009_0948.csv"

    MAXIMIZE = "maximize"

    # 迭代次数
    ITERATIONS = "iterations"
    IMPORTANT_FEATUER_RADIO = "important_feature_ratio"
    METRIC_FILE_ME = f"{PROJECT_HOME}/results/00-observation01/07_effect_of_optimize_round_and_dimension_size_original_20240919_0103.csv"
    METRIC = "metric"  # 进度指标的类型，如roc_auc, f1, precision 等
    DATASET = "dataset"  # 数据集名称
    METRIC_ROC_AUC = "roc_auc"
    METRIC_F1 = "f1"
    METRIC_ACCURACY = "accuracy"
    METRIC_RECALL = "recall"
    METRIC_LOG_LOSS = "log_loss"
    METRIC_PRECISION = "precision"
    # 小中大三个数据集
    SELECTED_DATASET = ["dresses-sales", "cylinder-bands", "climate-model-simulation-crashes", "ilpd",
                        "credit-approval", "breast-w", "tic-tac-toe", "diabetes", "credit-g", "qsar-biodeg", "pc1",
                        "pc4", "pc3", "kc1", "ozone-level-8hr", "madelon", "kr-vs-kp", "Bioresponse", "sick",
                        "spambase", "wilt", "churn", "phoneme", "jm1", "PhishingWebsites", "nomao", "bank-marketing",
                        "electricity"]
    SMALL_DATASET = ["dresses-sales",
                     "climate-model-simulation-crashes",
                     "cylinder-bands",
                     "ilpd",
                     "credit-approval",
                     "breast-w",
                     "diabetes",
                     "tic-tac-toe",
                     "credit-g",
                     "qsar-biodeg",
                     "pc1"]
    MIDDLE_DATASET = ["pc4", "pc3", "kc1", "ozone-level-8hr", "madelon", "kr-vs-kp", "Bioresponse", "sick",
                      "spambase", "wilt", "churn", 'phoneme']
    LARGE_DAGASET = ["jm1", "PhishingWebsites", "nomao", "bank-marketing",
                     "electricity"]

    @staticmethod
    @memory.cache
    def load_csv_file(file_csv):
        """
        加载实验文件，并解析文本配置为df.
        所有的文件都通过这个读取

        Parameters
        ----------
        file_csv :

        Returns
        -------

        """
        # 下载并加载文件
        df = AnaHelper.prepare_csv(file_csv)
        # df = df.iloc[:3, :]
        # 解析配置
        # 使用 ProcessPoolExecutor 进行并行处理
        print("Parse configs")
        with ProcessPoolExecutor(max_workers=int(os.cpu_count() * 0.6)) as executor:
            # 将每一行传递给解析函数
            parsed_configs = list(
                executor.map(AnaHelper.parse_configs_of_series_exploration, [row for _, row in df.iterrows()]))

        # df["parsed_configs_and_metrics"] = df.apply(AnaHelper.parse_configs_of_series_exploration, axis=1)
        df["parsed_configs_and_metrics"] = parsed_configs
        return df

    @classmethod
    # @memory.cache
    def prepare_csv(cls, name, return_df=True):
        """
        远程下载csv文件到本地
        Parameters
        ----------
        name :

        Returns
        -------

        """
        server = Servers.BENCHMARK_SERVER
        remote_file = os.path.join(server.data_home, name)
        local_file = os.path.join(PROJECT_HOME, "exp_results/tshpo", name)
        if not os.path.exists(local_file):
            Rsync.download_file(server, remote_file=remote_file, local_file=local_file)

        # 将数据备份到数据目录下
        # shutil.copyfile(local_file, os.path.join(DATA_BACK_HOME, os.path.basename(local_file)))
        if return_df is True:
            return pd.read_csv(local_file)
        else:
            return local_file

    @staticmethod
    def get_data_size_type(name):
        """
        返回数据大小对应的类型
        输入：数据名称
        输出：数据类型，[small,large,middle]
        Parameters
        ----------
        name :

        Returns
        -------

        """

        if name in AnaHelper.SMALL_DATASET:
            return "small"
        elif name in AnaHelper.LARGE_DAGASET:
            return "large"
        elif name in AnaHelper.MIDDLE_DATASET:
            return "middle"
        else:
            raise RuntimeError("Unsupported dataset name ")

    @staticmethod
    def get_dataset_task_id_by_name(name):
        maps = {
            "3": "kr-vs-kp",
            "15": "breast-w",
            "29": "credit-approval",
            "31": "credit-g",
            "37": "diabetes",
            "3021": "sick",
            "43": "spambase",
            "49": "tic-tac-toe",
            "219": "electricity",
            "3902": "pc4",
            "3903": "pc3",
            "3904": "jm1",
            "3913": "kc2",
            "3917": "kc1",
            "3918": "pc1",
            "14965": "bank-marketing",
            "10093": "banknote-authentication",
            "10101": "blood-transfusion-service-center",
            "9971": "ilpd",
            "9976": "madelon",
            "9977": "nomao",
            "9978": "ozone-level-8hr",
            "9952": "phoneme",
            "9957": "qsar-biodeg",
            "9946": "wdbc",
            "9910": "Bioresponse",
            "14952": "PhishingWebsites",
            "14954": "cylinder-bands",
            "125920": "dresses-sales",
            "167120": "numerai28.6",
            "167141": "churn",
            "167125": "Internet-Advertisements",
            "146820": "wilt",
            "146819": "climate-model-simulation-crashes"
        }
        for k in maps:
            v = maps[k]
            if v == str(name).strip():
                return k
        raise ValueError(f"Unknown dataset name: {name}")

    @staticmethod
    def convert_num_to_alp(i):
        """
        数字转字符，用来画图
        Parameters
        ----------
        i :

        Returns
        -------

        """
        assert i >= 1
        return chr(96 + i)

    @DeprecationWarning
    @staticmethod
    def get_all_configs_and_values_of_series(series: pd.Series, metric: str = "roc_auc",
                                             iterations: int = None) -> pd.DataFrame:
        """
        解析一条数据中configs_and_metrics的字段，将其处理为 pd.DataFrame 数据集结构
        Parameters
        ----------
        iterations :
        metric :  str
            value of AnaHelper.METRIC_*
        series :

        Returns
        -------

        """
        assert isinstance(series, pd.Series), "input must be a pd.Series"
        datas = re.findall(re.compile(r"\(Configuration\(values=(.*?precision=.*?\)\))", re.DOTALL),
                           series['configs_and_metrics'])
        output = []
        for index, d in enumerate(datas):
            if iterations is not None and index >= iterations:
                break
            mystr = d.replace("})", "}").replace("))", ")").replace("\n", "").strip()
            model_conf, ret_value = eval(mystr)
            run_value = ret_value  # type:RunValue

            output.append({
                "id": series['id'],
                "loss": run_value.get_dict()[metric],
                "model": model_conf["__choice__"],
                "metric": metric
            })
        model_acc = pd.DataFrame(output)
        return model_acc

    @staticmethod
    def parse_all_config_and_value_from_series(series: pd.Series, metric: str = "roc_auc") -> pd.DataFrame:
        """
        解析一条数据中configs_and_metrics的字段，将其处理为 pd.DataFrame 数据集结构
        Parameters
        ----------
        iterations :
        metric :  str
            value of AnaHelper.METRIC_*
        series :

        Returns
        -------

        """
        assert isinstance(series, pd.Series), "input must be a pd.Series"
        config_and_values = re.findall(re.compile(r"\(Configuration\(values=(.*?precision=.*?\)\))", re.DOTALL),
                                       series['configs_and_metrics'])
        assert len(config_and_values) == series["max_iterations"], "配置数不对，实验可能有问题"
        output = []
        for index, d in enumerate(config_and_values):
            mystr = d.replace("})", "}").replace("))", ")").replace("\n", "").strip()
            model_conf, ret_value = eval(mystr)
            run_value = ret_value  # type:RunValue

            output.append({
                "config_index": index,
                "loss": run_value.default,
                "model": model_conf["__choice__"],
                AnaHelper.metric: metric,
                "elapsed_seconds": run_value.elapsed_seconds
            })
        model_acc = pd.DataFrame(output)
        return model_acc

    @staticmethod
    def parse_config_and_value_from_series(series: pd.Series, metric: str = "roc_auc",
                                           max_iterations: int = 10000) -> pd.DataFrame:
        """
        解析一条数据中configs_and_metrics的字段，将其处理为 pd.DataFrame 数据集结构
        Parameters
        ----------
        max_iterations :
        metric :  str
            value of AnaHelper.METRIC_*
        series :

        Returns
        -------

        """
        assert isinstance(series, pd.Series), "input must be a pd.Series"
        # config_and_values = re.findall(re.compile(r"\(Configuration\(values=(.*?precision=.*?\)\))", re.DOTALL),
        #                                series['configs_and_metrics'])
        config_and_values = series['parsed_configs_and_metrics']
        # assert len(config_and_values) <= series["max_iteration"], "配置数不对，实验可能有问题"
        output = []
        for index, (k, v) in enumerate(config_and_values.iterrows()):
            if max_iterations is not None and index >= max_iterations:
                break

            output.append({
                "config_index": index,
                "loss": v[metric],
                "model": v["model"],
                AnaHelper.metric: metric,
                "default": v['default'],
                "elapsed_seconds": v['elapsed_seconds']
            })
        model_acc = pd.DataFrame(output)
        return model_acc

    @staticmethod
    def parse_configs_of_series_exploration(series: pd.Series) -> pd.DataFrame:
        """
        解析DataFrame中的每一行配置configs_and_metrics为DataFrame

        输入: DataFrame 中的每一行
        输出：configs_and_metrics 的DataFrame 表示
        Parameters
        ----------
        iterations :
        series :

        Returns
        -------

        """

        # 1）解析所有配置，2）取每个模型中最前面的max_exploration配置
        assert isinstance(series, pd.Series), "input must be a pd.Series"
        config_and_values = re.findall(re.compile(r"\(Configuration\(values=(.*?precision=.*?\)\))", re.DOTALL),
                                       series['configs_and_metrics'])
        # assert len(config_and_values) <= series["max_iteration"], "配置数不对，实验可能有问题"
        output = []
        for index, d in enumerate(config_and_values):
            mystr = d.replace("})", "}").replace("))", ")").replace("\n", "").strip()
            model_conf, ret_value = eval(mystr)
            run_value = ret_value  # type:RunValue
            _t = {
                "config_index": index,
                "model": model_conf["__choice__"],
                "default": run_value.default,
                "elapsed_seconds": run_value.elapsed_seconds
            }
            _t.update(run_value.__dict__)
            output.append(_t)
        configs = pd.DataFrame(output)
        return configs

    @classmethod
    def drop_column_none(cls, df):
        # 删除全是None的列
        return df.dropna(axis=1, how='all')

    @classmethod
    def pre_process(cls, df):
        # 删除全是None的列
        df = df.fillna("None")
        return df

    @staticmethod
    def append_task_id(df: pd.DataFrame):
        """
        在给定的df中加上task_id 列，task_id 是从名字对应到id的。
        给定的df中必须包括dataset列
        Parameters
        ----------
        df :

        Returns
        -------

        """
        assert 'dataset' in df.columns, "column must contain 'dataset'"

        def _append_task_id(ser):
            dataset_name = ser['dataset']
            ser['task_id'] = AnaHelper.get_dataset_task_id_by_name(dataset_name)
            return ser

        return df.apply(_append_task_id, axis=1)

    @staticmethod
    def append_normal_model_name(df: pd.DataFrame):
        """
        在给定的df中加上task_id 列，task_id 是从名字对应到id的。
        给定的df中必须包括dataset列
        Parameters
        ----------
        df :

        Returns
        -------

        """

        return AnaHelper.normal_model_name(df)

    @staticmethod
    def get_models_accuracy_of_series(series: pd.Series, metric="roc_auc", iterations=None):
        """

        Parameters
        ----------
        series :
        maximize : bool
            最大化还是最小化

        Returns
        -------

        """
        _all_configs = AnaHelper.parse_config_and_value_from_series(series, metric=metric, max_iterations=iterations)
        # assert _all_configs.shape[0] == series['max_iterations'], "数据解析不正确，请检查"
        if AnaHelper.is_maximize(metric):
            # 越大越好
            _t = _all_configs.groupby(by='model', as_index=False).max().sort_values(by='loss', ascending=False)
        else:
            # 越小越好
            _t = _all_configs.groupby(by='model', as_index=False).min().sort_values(by='loss', ascending=True)
        _t['metric'] = metric
        _t['maximize'] = AnaHelper.is_maximize(metric)
        return _t

    @staticmethod
    def parse_diff_iterations_of_series(series: pd.Series, metric="roc_auc", iterations=None):
        """

        Parameters
        ----------
        series :
        maximize : bool
            最大化还是最小化

        Returns
        -------

        """
        _all_configs = AnaHelper.get_all_configs_and_values_of_series(series, metric=metric, iterations=iterations)
        # assert _all_configs.shape[0] == series['max_iterations'], "数据解析不正确，请检查"
        if AnaHelper.is_maximize(metric):
            # 越大越好
            _t = _all_configs.groupby(by='model').max().sort_values(by='loss', ascending=False)
        else:
            # 越小越好
            _t = _all_configs.groupby(by='model').min().sort_values(by='loss', ascending=True)
        _t['metric'] = metric
        _t['maximize'] = AnaHelper.is_maximize(metric)
        # assert _t.shape[0] == AnaHelper.N_ALGS, f"算法数量解析不对，得到：{_t.shape[0]}, 期望: {AnaHelper.N_ALGS}"
        return _t

    @staticmethod
    def get_model_names(df: pd.DataFrame):
        return df['model_name'].drop_duplicates().tolist()

    @staticmethod
    @memory.cache
    def get_accuracy_of_models_on_dataset_tshpo(dataset: str = "kr-vs-kp", important_feature_ratio: float = 0.8,
                                                iterations=100, maximize=True, metric="roc_auc"):
        """
        获取个模型在数据上的表现
        Parameters
        ----------
        dataset :

        Returns
        -------

        """
        df = pd.read_csv(AnaHelper.METRIC_FILE_ME)
        df = AnaHelper.append_task_id(df)

        filter_df = df[(df['dataset'] == dataset) & (df['important_feature_ratio'] == important_feature_ratio)]
        # filter_df 应该是一个数据上的10折验证结果
        assert filter_df.shape[0] <= 10, "数据数量不正确，filter_df 应该是一个数据上的10折验证结果"
        assert filter_df.shape[0] >= 6, "数据数量不正确，filter_df 应该是一个数据上的10折验证结果"

        # 找出每个算法在该数据集上的平均表现：一折上找平均，十折上找均值
        output = []
        for i, v in filter_df.iterrows():
            _t = AnaHelper.get_models_accuracy_of_series(v,
                                                         metric=metric,
                                                         iterations=iterations)
            output.append(_t.reset_index())
        _df = pd.concat(output, axis=0)
        final_df = pd.pivot_table(_df, index=['model'], values=['loss'], aggfunc=["mean", "std"]).reset_index()
        final_df.columns = ['model_name', "mean", 'std']
        # groupby(by="model", as_index=False).mean().sort_values(by=['loss'],
        #                                                        ascending=False)
        final_df[AnaHelper.DATASET] = dataset
        final_df[AnaHelper.IMPORTANT_FEATUER_RADIO] = important_feature_ratio
        final_df[AnaHelper.ITERATIONS] = iterations
        final_df[AnaHelper.METRIC] = metric
        final_df[AnaHelper.MAXIMIZE] = maximize
        final_df.sort_values(by=['mean'], ascending=False, inplace=True)
        return final_df

    @staticmethod
    @memory.cache
    def get_runinig_time_seconds_of_models_on_dataset_tshpo_min(df, dataset: str = "kr-vs-kp",
                                                                important_feature_ratio: float = 0.2,
                                                                iterations=100, maximize=True, metric="roc_auc",
                                                                dim_redu_method="PCA"):
        """
        获取个模型在数据上的表现
        Parameters
        ----------
        dataset :

        Returns
        -------
        """
        # df = append_task_id(df)

        filter_df = df[(df['dataset'] == dataset) & (df['important_feature_ratio'] == important_feature_ratio) & (
                df['max_iterations'] == iterations) & (df['dim_redu_method'] == dim_redu_method)]
        # filter_df 应该是一个数据上的10折验证结果
        assert filter_df.shape[0] <= 10, "数据数量不正确，filter_df 应该是一个数据上的10折验证结果"
        assert filter_df.shape[0] >= 6, "数据数量不正确，filter_df 应该是一个数据上的10折验证结果"

        # 找出每个算法在该数据集上的平均表现：一折上找平均，十折上找均值
        output = []
        for i, v in filter_df.iterrows():
            output.append(v['t__stopwatch_'])
        # 返回时间转为分钟
        return np.mean(output)

    @staticmethod
    # @memory.cache
    def get_accuracy_of_models_on_dataset_tshpo_for_different_dim_redu_method(dataset: str = "kr-vs-kp",
                                                                              important_feature_ratio: float = 0.8,
                                                                              iterations=100, maximize=True,
                                                                              metric="roc_auc", dim_redu_method="PCA"):
        """
        获取个模型在数据上的表现
        Parameters
        ----------
        dataset :

        Returns
        -------

        """
        df = pd.read_csv(AnaHelper.DIM_REDU_METHOD)
        df = AnaHelper.append_task_id(df)

        filter_df = df[(df['dataset'] == dataset) & (df['important_feature_ratio'] == important_feature_ratio) & (
                df['dim_redu_method'] == dim_redu_method)]
        # filter_df 应该是一个数据上的10折验证结果
        assert filter_df.shape[0] <= 10, "数据数量不正确，filter_df 应该是一个数据上的10折验证结果"
        assert filter_df.shape[0] >= 6, "数据数量不正确，filter_df 应该是一个数据上的10折验证结果"

        # 找出每个算法在该数据集上的平均表现：一折上找平均，十折上找均值
        output = []
        for i, v in filter_df.iterrows():
            _t = AnaHelper.get_models_accuracy_of_series(v,
                                                         metric=metric,
                                                         iterations=iterations)
            output.append(_t.reset_index())
        _df = pd.concat(output, axis=0)
        final_df = pd.pivot_table(_df, index=['model'], values=['loss'], aggfunc=["mean", "std"]).reset_index()
        final_df.columns = ['model_name', "mean", 'std']
        # groupby(by="model", as_index=False).mean().sort_values(by=['loss'],
        #                                                        ascending=False)
        final_df[AnaHelper.DATASET] = dataset
        final_df[AnaHelper.IMPORTANT_FEATUER_RADIO] = important_feature_ratio
        final_df[AnaHelper.ITERATIONS] = iterations
        final_df[AnaHelper.METRIC] = metric
        final_df[AnaHelper.MAXIMIZE] = maximize
        final_df.sort_values(by=['mean'], ascending=False, inplace=True)
        return final_df

    @staticmethod
    # @memory.cache
    def get_best_top_n_models_on_dataset_tshpo(dataset: str = "kr-vs-kp", important_feature_ratio: float = 0.8,
                                               iterations=100, maximize=True, metric="roc_auc", top_n=1):
        """
        获取个模型在数据上的表现
        Parameters
        ----------
        dataset :

        Returns
        -------

        """
        df = pd.read_csv(AnaHelper.METRIC_FILE_ME)
        df = AnaHelper.append_task_id(df)

        filter_df = df[(df['dataset'] == dataset) & (df['important_feature_ratio'] == important_feature_ratio)]
        # filter_df 应该是一个数据上的10折验证结果
        assert filter_df.shape[0] <= 10, "数据数量不正确，filter_df 应该是一个数据上的10折验证结果"
        assert filter_df.shape[0] >= 6, "数据数量不正确，filter_df 应该是一个数据上的10折验证结果"

        # 找出每个算法在该数据集上的平均表现：一折上找平均，十折上找均值
        output = []
        for i, v in filter_df.iterrows():
            _t = AnaHelper.get_models_accuracy_of_series(v,
                                                         metric=metric,
                                                         iterations=iterations)
            output.append(_t.reset_index())
        _df = pd.concat(output, axis=0)
        final_df = pd.pivot_table(_df, index=['model'], values=['loss'], aggfunc=["mean", "std"]).reset_index()
        final_df.columns = ['model_name', "mean", 'std']
        return final_df.nlargest(top_n, "mean")['model_name'].to_list()

    @staticmethod
    # @memory.cache
    def get_best_top_n_models_on_dataset_tshpo_with_dim_redu(dataset: str = "kr-vs-kp",
                                                             important_feature_ratio: float = 0.8,
                                                             iterations=100, metric="roc_auc", top_n=None,
                                                             dim_redu_method=None):
        """
        获取个模型在数据上的表现
        Parameters
        ----------
        dataset :

        Returns
        -------

        """
        if AnaHelper.df_dim_redu_ is None:
            AnaHelper.df_dim_redu_ = pd.read_csv(AnaHelper.DIM_REDU_METHOD)

        df = AnaHelper.df_dim_redu_
        # df = append_task_id(df)

        filter_df = df[(df['dataset'] == dataset) & (df['important_feature_ratio'] == important_feature_ratio) & (
                df['dim_redu_method'] == dim_redu_method) & (df['max_iterations'] == iterations)]

        if filter_df.shape[0] < 5:
            pass
        # filter_df 应该是一个数据上的10折验证结果
        assert filter_df.shape[0] <= 10, "数据数量不正确，filter_df 应该是一个数据上的10折验证结果"
        # assert filter_df.shape[0] >= 6, f"数据数量不正确，filter_df 应该是一个数据上的10折验证结果，{dim_redu_method}"

        # 找出每个算法在该数据集上的平均表现：一折上找平均，十折上找均值
        output = []
        for i, v in filter_df.iterrows():
            _t = AnaHelper.get_models_accuracy_of_series(v,
                                                         metric=metric,
                                                         iterations=iterations)
            output.append(_t.reset_index())
        _df = pd.concat(output, axis=0)
        final_df = pd.pivot_table(_df, index=['model'], values=['loss'], aggfunc=["mean", "std"]).reset_index()
        final_df.columns = ['model_name', "mean", 'std']
        if top_n is None:
            return final_df.sort_values(by="mean", ascending=False)['model_name'].tolist()
        else:
            return final_df.nlargest(top_n, "mean")['model_name'].to_list()

    @staticmethod
    def get_best_top_n_models_on_dataset_tshpo_with_dim_redu_v2(df, dataset: str = "kr-vs-kp",
                                                                important_feature_ratio: float = 0.8,
                                                                iterations=100, metric="roc_auc", top_n=None,
                                                                dim_redu_method=None, n_fold=5):
        """
        获取个模型在数据上的表现
        Parameters
        ----------
        dataset :

        Returns
        -------

        """

        filter_df = df[(df['dataset'] == dataset) & (df['important_feature_ratio'] == important_feature_ratio) & (
                df['dim_redu_method'] == dim_redu_method)]

        if filter_df.shape[0] < 5:
            pass
        # filter_df 应该是一个数据上的10折验证结果
        assert filter_df.shape[0] == n_fold, f"数据数量不正确，filter_df 应该是一个数据上的 {n_fold} 折验证结果"
        # assert filter_df.shape[0] >= 6, f"数据数量不正确，filter_df 应该是一个数据上的10折验证结果，{dim_redu_method}"

        # 找出每个算法在该数据集上的平均表现：一折上找平均，十折上找均值
        output = []
        for i, v in filter_df.iterrows():
            _t = AnaHelper.get_models_accuracy_of_series(v,
                                                         metric=metric,
                                                         iterations=iterations)
            output.append(_t.reset_index())
        _df = pd.concat(output, axis=0)
        final_df = pd.pivot_table(_df, index=['model'], values=['loss'], aggfunc=["mean", "std"]).reset_index()
        final_df.columns = ['model_name', "mean", 'std']
        if top_n is None:
            return final_df.sort_values(by="mean", ascending=False)['model_name'].tolist()
        else:
            return final_df.nlargest(top_n, "mean")['model_name'].to_list()

    @staticmethod
    def rank_model_of_baseline(df,
                               dataset: str = "kr-vs-kp",
                               metric="roc_auc",
                               top_n=None,
                               n_fold=5
                               ):
        """
        获取个模型在数据上的表现， 用的是中位数，不是平均值
        Parameters
        ----------
        dataset :

        Returns
        -------

        """

        filter_df = df[(df['dataset'] == dataset)]

        if filter_df.shape[0] < 5:
            pass
        # filter_df 应该是一个数据上的10折验证结果
        assert filter_df.shape[0] == n_fold, f"数据数量不正确，filter_df 应该是一个数据上的 {n_fold} 折验证结果"
        # assert filter_df.shape[0] >= 6, f"数据数量不正确，filter_df 应该是一个数据上的10折验证结果，{dim_redu_method}"

        # 找出每个算法在该数据集上的平均表现：一折上找平均，十折上找均值
        output = []
        for i, v in filter_df.iterrows():
            _t = AnaHelper.get_models_accuracy_of_series(v, metric=metric)
            output.append(_t)
        _df = pd.concat(output, axis=0)
        final_df = pd.pivot_table(_df, index=['model'], values=['loss'], aggfunc=["mean", "std"]).reset_index()
        final_df.columns = ['model_name', "mean", 'std']
        if top_n is None:
            return final_df.sort_values(by="mean", ascending=False).reset_index(drop=True)
        else:
            return final_df.nlargest(top_n, "mean").reset_index(drop=True)

    @staticmethod
    def rank_model_of_baseline_tshpo(df,
                                     dataset: str = "kr-vs-kp",
                                     important_feature_ratio: float = None,
                                     iterations=None,
                                     metric="roc_auc",
                                     top_n=None,
                                     dim_redu_method=None,
                                     n_fold=5
                                     ):
        """
        获取个模型在数据上的表现
        Parameters
        ----------
        dataset :

        Returns
        -------

        """

        filter_df = df[(df['dataset'] == dataset) & (df['important_feature_ratio'] == important_feature_ratio) & (
                df['dim_redu_method'] == dim_redu_method)]

        if filter_df.shape[0] < 5:
            pass
        # filter_df 应该是一个数据上的10折验证结果
        assert filter_df.shape[0] == n_fold, f"数据数量不正确，filter_df 应该是一个数据上的 {n_fold} 折验证结果"
        # assert filter_df.shape[0] >= 6, f"数据数量不正确，filter_df 应该是一个数据上的10折验证结果，{dim_redu_method}"

        # 找出每个算法在该数据集上的平均表现：一折上找平均，十折上找均值
        output = []
        for i, v in filter_df.iterrows():
            _t = AnaHelper.get_models_accuracy_of_series(v,
                                                         metric=metric,
                                                         iterations=iterations)
            output.append(_t.reset_index())
        _df = pd.concat(output, axis=0)
        final_df = pd.pivot_table(_df, index=['model'], values=['loss'], aggfunc=["mean", "std"]).reset_index()
        final_df.columns = ['model_name', "mean", 'std']
        if top_n is None:
            return final_df.sort_values(by="mean", ascending=False)['model_name'].tolist()
        else:
            return final_df.nlargest(top_n, "mean")['model_name'].to_list()

    @staticmethod
    @memory.cache
    def get_accuracy_of_models_on_dataset_baseline(dataset: str = "kr-vs-kp", important_feature_ratio: float = 0.8,
                                                   iterations=100, maximize=True, metric="roc_auc"):
        """
        在总体上执行1000次随机优化
        Parameters
        ----------
        dataset :

        Returns
        -------

        """
        df = pd.read_csv(AnaHelper.METRIC_FILE_BASELINE)
        df = AnaHelper.append_task_id(df)

        filter_df = df[(df['dataset'] == dataset)]
        # filter_df = df[(df['dataset'] == dataset) & (df['important_feature_ratio'] == important_feature_ratio)]
        # filter_df 应该是一个数据上的10折验证结果
        assert filter_df.shape[0] <= 10, "数据数量不正确，filter_df 应该是一个数据上的10折验证结果"
        assert filter_df.shape[0] >= 6, "数据数量不正确，filter_df 应该是一个数据上的10折验证结果"

        # 找出每个算法在该数据集上的平均表现：一折上找平均，十折上找均值
        output = []
        for i, v in filter_df.iterrows():
            _t = AnaHelper.get_models_accuracy_of_series(v,
                                                         metric=metric,
                                                         maximize=maximize,
                                                         iterations=iterations)
            output.append(_t.reset_index())
        _df = pd.concat(output, axis=0)
        final_df = pd.pivot_table(_df, index=['model'], values=['loss'], aggfunc=["mean", "std"]).reset_index()
        final_df.columns = ['model_name', "mean", 'std']
        # groupby(by="model", as_index=False).mean().sort_values(by=['loss'],
        #                                                        ascending=False)
        final_df[AnaHelper.DATASET] = dataset
        final_df[AnaHelper.IMPORTANT_FEATUER_RADIO] = important_feature_ratio
        final_df[AnaHelper.ITERATIONS] = iterations
        final_df[AnaHelper.METRIC] = metric
        final_df[AnaHelper.MAXIMIZE] = maximize
        final_df.sort_values(by=['mean'], ascending=False, inplace=True)
        return final_df

    @staticmethod
    @memory.cache
    def get_best_top_n_models_on_dataset_baseline(dataset: str = "kr-vs-kp", important_feature_ratio: float = 0.8,
                                                  iterations=100, maximize=True, metric="roc_auc", top=2):
        """
        找到指定数据集下表现最好的前n个模型，10折交叉上
        Parameters
        ----------
        dataset :

        Returns
        -------

        """
        if AnaHelper.df_baseline_ is None:
            AnaHelper.df_baseline_ = pd.read_csv(AnaHelper.METRIC_FILE_BASELINE)
            # df = append_task_id(df)
        df = AnaHelper.df_baseline_
        filter_df = df[(df['dataset'] == dataset)]
        # filter_df = df[(df['dataset'] == dataset) & (df['important_feature_ratio'] == important_feature_ratio)]
        # filter_df 应该是一个数据上的10折验证结果
        assert filter_df.shape[0] <= 10, "数据数量不正确，filter_df 应该是一个数据上的10折验证结果"
        assert filter_df.shape[0] >= 6, "数据数量不正确，filter_df 应该是一个数据上的10折验证结果"

        # 找出每个算法在该数据集上的平均表现：一折上找平均，十折上找均值
        output = []
        for i, v in filter_df.iterrows():
            _t = AnaHelper.get_models_accuracy_of_series(v,
                                                         metric=metric,
                                                         iterations=iterations)
            output.append(_t.reset_index())
        _df = pd.concat(output, axis=0)
        final_df = pd.pivot_table(_df, index=['model'], values=['loss'], aggfunc=["mean", "std"]).reset_index()
        final_df.columns = ['model_name', "mean", 'std']
        return final_df.nlargest(top, "mean")['model_name'].to_list()

    @staticmethod
    @memory.cache
    def get_accuracy_of_models_on_dataset_baseline_each(dataset: str = "kr-vs-kp", metric: str = "vus_roc"):
        """
        每个模型单独执行50次优化的baseline
        Parameters
        ----------
        dataset :
        metric :

        Returns
        -------

        """
        df = pd.read_csv(AnaHelper.METRIC_FILE_BASELINE)
        df = AnaHelper.append_task_id(df)

        # 分析每个数据集上每个算法的表现
        outputs = []
        for _dataset, df_v in df.groupby(by=AnaHelper.DATASET):
            _postprocess_metric_name = metric + "_max"
            for _model_name in AnaHelper.get_model_names(df_v):
                filter_df = df[(df['dataset'] == _dataset) & (df['model_name'] == _model_name)]
                outputs.append({
                    "dataset": _dataset,
                    "model_name": _model_name,
                    "mean": filter_df[_postprocess_metric_name].mean(),
                    "std": filter_df[_postprocess_metric_name].std(),
                    "metric": metric
                })
                # break
            # break
        ana_df = pd.DataFrame(outputs)
        return ana_df[(ana_df[AnaHelper.DATASET] == dataset) & (
                ana_df[AnaHelper.METRIC] == metric)].sort_values(by=['mean'], ascending=False)

    @classmethod
    def get_datasets(cls):

        """
        返回数据集的名称，按数据的大小排序，小的在前

        Returns
        -------

        """
        # 数据来源 openml/dataset_statics.py
        all_datasets = {
            'task_id': {28: '125920', 12: '3913', 33: '146819', 27: '14954', 24: '9946', 18: '9971', 2: '29',
                        1: '15',
                        17: '10101', 4: '37', 7: '49', 3: '31', 23: '9957', 14: '3918', 16: '10093', 9: '3902',
                        10: '3903',
                        13: '3917', 21: '9978', 19: '9976', 0: '3', 31: '167125', 25: '9910', 5: '3021', 6: '43',
                        32: '146820', 30: '167141', 22: '9952', 11: '3904', 26: '14952', 20: '9977', 15: '14965',
                        8: '219',
                        29: '167120'},
            'dataset_name': {28: 'dresses-sales', 12: 'kc2', 33: 'climate-model-simulation-crashes',
                             27: 'cylinder-bands',
                             24: 'wdbc', 18: 'ilpd', 2: 'credit-approval', 1: 'breast-w',
                             17: 'blood-transfusion-service-center', 4: 'diabetes', 7: 'tic-tac-toe', 3: 'credit-g',
                             23: 'qsar-biodeg', 14: 'pc1', 16: 'banknote-authentication', 9: 'pc4', 10: 'pc3',
                             13: 'kc1',
                             21: 'ozone-level-8hr', 19: 'madelon', 0: 'kr-vs-kp', 31: 'Internet-Advertisements',
                             25: 'Bioresponse', 5: 'sick', 6: 'spambase', 32: 'wilt', 30: 'churn', 22: 'phoneme',
                             11: 'jm1',
                             26: 'PhishingWebsites', 20: 'nomao', 15: 'bank-marketing', 8: 'electricity',
                             29: 'numerai28.6'},
            '#instances': {28: 500, 12: 522, 33: 540, 27: 540, 24: 569, 18: 583, 2: 690, 1: 699, 17: 748, 4: 768,
                           7: 958,
                           3: 1000, 23: 1055, 14: 1109, 16: 1372, 9: 1458, 10: 1563, 13: 2109, 21: 2534, 19: 2600,
                           0: 3196,
                           31: 3279, 25: 3751, 5: 3772, 6: 4601, 32: 4839, 30: 5000, 22: 5404, 11: 10885, 26: 11055,
                           20: 34465, 15: 45211, 8: 45312, 29: 96320}}
        return list(all_datasets['dataset_name'].values())

    @staticmethod
    def normal_model_name(df):
        _maps = {
            'random_forest': "RF",
            'extra_trees': "ET",
            'adaboost': "Ada",
            'lda': "LDA",
            'sgd': "SGD",
            'mlp': "MLP",
            'k_nearest_neighbors': "KNN",
            'decision_tree': "DT",
            'qda': "QDA",
            'multinomial_nb': "MNB",
            'liblinear_svc': "SVC",
            'bernoulli_nb': "BNB",
            'libsvm_svc': "SVM",
            'gaussian_nb': "GNB",
        }
        df['model_name'] = df.apply(lambda x: _maps.get(x['model_name']), axis=1)
        return df

    @classmethod
    def normal_metric_name(cls, metric_key):
        maps = {
            "precision": "Precision",
            "precision_max": "Precision",
            "recall_max": "Recall",
            "recall": "Recall",
            "roc_auc_max": "ROC AUC",
            "roc_auc": "ROC AUC",
            "f1_max": "F1 Score",
            "f1": "F1 Score",
            "log_loss": "Log Loss",
            "accuracy": "Accuracy"
        }
        return maps.get(metric_key)

    @staticmethod
    @memory.cache
    def is_find_best_models(dataset, top_n=3):
        t1 = AnaHelper.get_best_top_n_models_on_dataset_baseline(dataset=dataset, metric=AnaHelper.METRIC_ROC_AUC)
        t2 = AnaHelper.get_best_top_n_models_on_dataset_tshpo(dataset=dataset, metric=AnaHelper.METRIC_ROC_AUC,
                                                              top_n=top_n)
        intersection = np.intersect1d(np.array(t1), np.array(t2))
        if len(intersection) > 0:
            return 1
        else:
            return 0

    @staticmethod
    @memory.cache
    def is_find_best_models_with_dim_redu_method(dataset, top_n=3, dim_redu_method=None, iterations=100, metric=None):
        assert metric is not None, "metric is required"
        assert dim_redu_method != None
        t1 = AnaHelper.get_best_top_n_models_on_dataset_baseline(dataset=dataset, metric=metric)
        # 这里我们只关心important_feature_ratio=0.2
        t2 = AnaHelper.get_best_top_n_models_on_dataset_tshpo_with_dim_redu(dataset=dataset,
                                                                            metric=metric,
                                                                            top_n=top_n,
                                                                            dim_redu_method=dim_redu_method,
                                                                            iterations=iterations,
                                                                            important_feature_ratio=0.2)

        intersection = np.intersect1d(np.array(t1), np.array(t2))
        if len(intersection) > 0:
            return 1
        else:
            return 0

    @staticmethod
    # @memory.cache
    def get_dim_redu_running_time():
        """
        每个降维方法所有的时间
        Parameters
        ----------
        dataset :
        top_n :
        dim_redu_method :
        iterations :
        metric :

        Returns
        -------

        """
        baseline_df = AnaHelper.load_baseline()
        my_df = AnaHelper.load_my_exp()

        _r_runtime = my_df.groupby(by=['max_iterations', 'dim_redu_method'], as_index=False)[
            't__stopwatch_'].agg(['mean', 'std', 'count']).reset_index()
        # _r_runtime = my_df.groupby(by=['dataset', 'dim_redu_method', 'max_iterations'], as_index=False)[
        #     't__stopwatch_'].agg(['mean', 'std', 'count']).reset_index()
        return baseline_df['t__stopwatch_'].agg(['mean', 'std']), _r_runtime

    @classmethod
    def get_all_metrics(cls):
        #  AnaHelper.METRIC_F1:
        return [AnaHelper.METRIC_ROC_AUC,
                AnaHelper.METRIC_F1,
                AnaHelper.METRIC_ACCURACY,
                AnaHelper.METRIC_PRECISION,
                AnaHelper.METRIC_LOG_LOSS,
                AnaHelper.METRIC_RECALL
                ]

    @classmethod
    def get_all_opt_methods(cls):
        return []

    @staticmethod
    def is_maximize(metric):
        if metric == AnaHelper.METRIC_LOG_LOSS:
            return False
        else:
            return True

    @classmethod
    def load_baseline(cls):
        if AnaHelper.df_baseline_ is None:
            AnaHelper.df_baseline_ = pd.read_csv(AnaHelper.METRIC_FILE_BASELINE)
            # df = append_task_id(df)

        return AnaHelper.df_baseline_

    @classmethod
    def load_my_exp(cls):
        if AnaHelper.df_dim_redu_ is None:
            AnaHelper.df_dim_redu_ = pd.read_csv(AnaHelper.DIM_REDU_METHOD)
        return AnaHelper.df_dim_redu_

    @classmethod
    def get_accuracy_of_iterations(cls, series: pd.Series, iterations: int):
        """
        获取每一个超参数优化实验中前N个实验项目的精度值，config_and_values_per_row 是每一行的值

        Parameters
        ----------
        config_and_values_per_row :
        iterations :

        Returns
        -------

        """
        metric = series['metric']
        config_and_values_per_row = \
            AnaHelper.parse_config_and_value_from_series(series, metric=metric, max_iterations=iterations)
        loss_df = config_and_values_per_row['loss']

        if AnaHelper.is_maximize(config_and_values_per_row.loc[0, "metric"]):
            return loss_df.max()
        else:
            return loss_df.min()

    @classmethod
    def get_accuracy_of_explorations(cls, series: pd.Series, exporation, metric):
        """
        获取每一个超参数优化实验中前N个实验项目的精度值，config_and_values_per_row 是每一行的值

        Parameters
        ----------
        config_and_values_per_row :
        iterations :

        Returns
        -------

        """
        config_and_values_per_row = \
            AnaHelper.parse_configs_of_series(series, explorations=[exporation])

        config_and_values_per_row.groupby("model")[metric].max().sort_values(ascending=False)

    @classmethod
    def get_accuracy_of_iterations_by_metric(cls, series: pd.Series, iterations: int, metric: str = "roc_auc"):
        """
        获取每一个超参数优化实验中前N个实验项目的精度值，config_and_values_per_row 是每一行的值

        Parameters
        ----------
        config_and_values_per_row :
        iterations :

        Returns
        -------

        """
        config_and_values_per_row = \
            AnaHelper.parse_config_and_value_from_series(series, metric=metric, max_iterations=iterations)
        loss_df = config_and_values_per_row['loss']

        if AnaHelper.is_maximize(metric):
            return loss_df.max()
        else:
            return loss_df.min()

    @staticmethod
    # @memory.cache
    @DeprecationWarning
    def load_exp_results(file_name: str, drop_colums: list = None):
        """
        see load_exp_from_csv_baseline
        Parameters
        ----------
        file_name :
        drop_colums :

        Returns
        -------

        """
        df = pd.read_csv(file_name)
        # 10,20,30,50,100,150,..., n+50
        iterations = np.append([10, 20, 30], np.arange(50, df['#instances'].drop_duplicates().values[0] + 1, 50))
        process_df = AnaHelper.append_accuracy_of_diff_iterations(df, iterations=iterations)
        if drop_colums is None:
            drop_colums = ['configs_and_metrics']
        df_drop = process_df.drop(columns=drop_colums, axis=1)
        return df_drop

    @staticmethod
    @memory.cache
    def load_exp_from_csv_baseline(file_name: str, metric: str = "roc_auc", drop_colums: list = None):
        """
        加载baseline文件。baseline中使用随机优化，所以要分精度指标讨论
        Parameters
        ----------
        file_name :
        metric :
        drop_colums :

        Returns
        -------

        """
        df = pd.read_csv(file_name)
        # 10,20,30,50,100,150,..., n+50
        iterations = list(np.append([1, 20, 30], np.arange(50, df['#instances'].drop_duplicates().values[0] + 1, 50)))
        process_df = AnaHelper.append_accuracy_of_diff_iterations_by_metric(df, metric=metric, iterations=iterations)
        if drop_colums is None:
            drop_colums = ['configs_and_metrics']
        df_drop = process_df.drop(columns=drop_colums, axis=1)
        df_drop['ana_metric_type'] = metric
        return df_drop

    @staticmethod
    @memory.cache
    def load_exp_from_csv_tshpo(file_name: str, drop_colums: list = None, iterations=None):
        """
        加载 tshpo 的精度。这里不用像baseline中一样分精度指标讨论，因为在优化的时候已经考虑了精度指标。
        因此，这里直接使用default即可。
        Parameters
        ----------
        file_name :
        metric :
        drop_colums :

        Returns
        -------

        """
        df = AnaHelper.load_csv_file(file_name)
        # df = pd.read_csv(file_name)
        # 10,20,30,50,100,150,..., n+50
        if iterations is None:
            iterations = sorted(list(
                np.append([1, 20, 30, df['#instances'].max()],
                          np.arange(50, df['#instances'].drop_duplicates().values[0] + 1, 50))))
        process_df = AnaHelper.append_accuracy_of_diff_iterations(df, iterations=iterations)
        if drop_colums is None:
            drop_colums = ['configs_and_metrics']
        df_drop = process_df.drop(columns=drop_colums, axis=1)
        return df_drop

    @staticmethod
    def load_exp_n_exploration_from_csv(file_name: str, drop_colums: list = None, explorations=None):
        """
        加载 tshpo 的精度, 注意这里的iterations要根据不能取最前面的几个了，因为是每个算法探索N次了，不是在整个超参数空间中探索n次了

        这里不用像baseline中一样分精度指标讨论，因为在优化的时候已经考虑了精度指标。
        因此，这里直接使用default即可。
        Parameters
        ----------
        file_name :
        metric :
        drop_colums :

        Returns
        -------

        """
        df = pd.read_csv(file_name)
        # 10,20,30,50,100,150,..., n+50
        if explorations is None:
            explorations = list(
                np.append([1], np.arange(10, df['n_exploration'].drop_duplicates().values[0] + 1, 10)))
        # process_df = AnaHelper.append_accuracy_of_diff_iterations(df, iterations=iterations)
        process_df = AnaHelper.append_accuracy_of_diff_n_explorations(df, explorations=explorations)
        if drop_colums is None:
            drop_colums = ['configs_and_metrics']
        df_drop = process_df.drop(columns=drop_colums, axis=1)
        return df_drop

    @staticmethod
    def get_rank_model_and_acc(df: pd.DataFrame, exporation=None, metric=None, return_time=False):
        """
        输入：df
        输出：在exporation和metric条件下 的模型名称排名及其算法排名

        Parameters
        ----------
        file_name :
        metric :
        drop_colums :

        Returns
        -------

        """
        # 平均计算同一个实验的多折交叉数据，意思是在配置中，自由fold不同，其他都相同
        assert df.shape[0] <= df['folds'].iloc[0], "除了fold不同，其他配置都应该相同"
        assert exporation is not None, "exporation cant be  None"
        outputs = []
        elapsed_second = []
        for k, v in df.iterrows():
            # configs = AnaHelper.parse_configs_of_series_exploration(v)
            configs = v['parsed_configs_and_metrics']
            filter_df = configs.groupby(by=['model'], group_keys=False).apply(
                lambda x: x.nsmallest(exporation, 'config_index')).reset_index(drop=True)
            filter_df['n_exploration'] = exporation
            elapsed_second.append(filter_df['elapsed_seconds'].sum())
            _t = filter_df.groupby("model")[metric].max().sort_values(ascending=False).reset_index()
            outputs.append(_t)
        r_df = pd.concat(outputs)
        if AnaHelper.is_maximize(metric):
            s = r_df.groupby(["model"]).mean().sort_values(by=[metric], ascending=False).reset_index()
            # 确保是降序排列
            assert s.iloc[0, 1] >= s.iloc[1, 1]
        else:
            s = r_df.groupby(["model"]).mean().sort_values(by=[metric], ascending=True).reset_index()
            # 确保是升序排列
            assert s.iloc[0, 1] <= s.iloc[1, 1]

        if return_time:
            return list(s.loc[:, "model"]), list(s.loc[:, metric]), np.median(elapsed_second)
        else:
            return list(s.loc[:, "model"]), list(s.loc[:, metric]),

    @staticmethod
    def get_rank_model_and_acc_bak(df: pd.DataFrame, exporation=None, metric=None, return_time=False):
        """
        输入：df
        输出：在exporation和metric条件下 的模型名称排名及其算法排名

        Parameters
        ----------
        file_name :
        metric :
        drop_colums :

        Returns
        -------

        """
        # 平均计算同一个实验的多折交叉数据，意思是在配置中，自由fold不同，其他都相同
        assert df.shape[0] <= df['folds'].iloc[0], "除了fold不同，其他配置都应该相同"
        assert exporation is not None, "exporation cant be  None"
        outputs = []
        elapsed_second = []
        for k, v in df.iterrows():
            configs = AnaHelper.parse_configs_of_series_exploration(v)
            filter_df = configs.groupby(by=['model'], group_keys=False).apply(
                lambda x: x.nsmallest(exporation, 'config_index')).reset_index(drop=True)
            filter_df['n_exploration'] = exporation
            elapsed_second.append(filter_df['elapsed_seconds'].sum())
            _t = filter_df.groupby("model")[metric].max().sort_values(ascending=False).reset_index()
            outputs.append(_t)
        r_df = pd.concat(outputs)
        s = r_df.groupby(["model"]).mean().sort_values(by=[metric], ascending=False).reset_index()
        # 确保是降序排列
        assert s.iloc[0, 1] >= s.iloc[1, 1]
        if return_time:
            return list(s.loc[:, "model"]), list(s.loc[:, metric]), np.median(elapsed_second)
        else:
            return list(s.loc[:, "model"]), list(s.loc[:, metric]),

    @classmethod
    # @DeprecationWarning
    def append_accuracy_of_diff_iterations(cls, df, iterations: list = None):
        """
        添加上不同 iteration \in iterations 对应的精度， 目的是不同 iteration 只用跑一次。 iteration 是超参数优化方法
        的优化次数，例如贝叶斯， 随机优化的迭代次数
        Parameters
        ----------
        df :
        iterations :

        Returns
        -------

        """
        if iterations is None:
            iterations = [10, 30, 50, 100, 150, 200]
        for _iter in iterations:
            _metric = df.apply(AnaHelper.get_accuracy_of_iterations, axis=1, iterations=_iter)
            df[f"metric_{_iter}"] = _metric
        return df

    @classmethod
    def append_accuracy_of_diff_iterations_by_metric(cls, df, iterations: list = None, metric: str = "roc_auc"):
        """
        添加上不同 iteration \in iterations 对应精度（由metric指定）， 目的是不同 iteration 只用跑一次。 iteration 是超参数优化方法
        的优化次数，例如贝叶斯， 随机优化的迭代次数
        Parameters
        ----------
        df :
        iterations :

        Returns
        -------

        """
        if iterations is None:
            iterations = [10, 30, 50, 100, 150, 200]
        for _iter in iterations:
            _metric = df.apply(AnaHelper.get_accuracy_of_iterations_by_metric, axis=1, iterations=_iter, metric=metric)
            df[f"metric_{_iter}"] = _metric
        return df

    @classmethod
    def normal_opt_method_name(cls, opt_method):
        _maps = {
            'smac': "SMAC3",
            'soar': "SOAR",
            'hyperband': "HB",
            'bo': "BO",
            'rs': "RS",
            'RS': "RS",
            'BO': "BO",
            'HB': "HB",
        }
        assert opt_method in _maps.keys(), f"Opt_method {opt_method} is not existed"
        return _maps.get(opt_method)

    @classmethod
    @memory.cache
    def find_top_model_from_baseline(cls, df, dataset, metric):
        """
        正在用的模型，排序模型
        Parameters
        ----------
        df :
        dataset :
        metric :

        Returns
        -------

        """
        alg_rank = AnaHelper.rank_model_of_baseline(
            df,
            dataset=dataset,
            metric=metric,
        )
        return alg_rank

    @classmethod
    def find_top_model_of_one_dataset(cls, df, metric, n_fold=5, top_n=3, is_return_model_names=True,
                                      strategy="min_max"):
        filter_df = df[df['metric'] == metric]
        assert len(filter_df['dataset'].drop_duplicates().tolist()) == 1, "只能计算一个数据"
        # filter_df 应该是一个数据上的10折验证结果
        assert filter_df.shape[0] == n_fold, f"数据数量不正确，filter_df 应该是一个数据上的 {n_fold} 折验证结果"
        # assert filter_df.shape[0] >= 6, f"数据数量不正确，filter_df 应该是一个数据上的10折验证结果，{dim_redu_method}"

        # 找出每个算法在该数据集上的平均表现：一折上找平均，十折上找均值
        output = []
        for i, v in filter_df.iterrows():
            _t = AnaHelper.get_models_accuracy_of_series(v, metric=metric)
            output.append(_t.reset_index())
        _df = pd.concat(output, axis=0)

        if strategy == "min_max":

            # 计算极差
            def rank_difference(group):
                return group['loss'].max() - group['loss'].min()

            final_df = _df.groupby(by=['model', 'metric'], as_index=False).apply(rank_difference)
            final_df.columns = ['model_name', 'metric', 'mean']

            if top_n is None:
                top_n = final_df.shape[0]

            return_df = final_df.sort_values(by="mean", ascending=False).reset_index(drop=True).iloc[:top_n]

            if is_return_model_names:
                return return_df['model_name'].tolist()
            else:
                return_df
        else:
            final_df = pd.pivot_table(_df, index=['model'], values=['loss'], aggfunc=["mean", "std"]).reset_index()
            final_df.columns = ['model_name', "mean", 'std']

            if top_n is None:
                top_n = final_df.shape[0]

            return_df = final_df.sort_values(by="mean", ascending=False).reset_index(drop=True).iloc[:top_n]

            if is_return_model_names:
                return return_df['model_name'].tolist()
            else:
                return_df

    @classmethod
    # @memory.cache
    def find_top_models_tshpo(cls, df, metric, alpha=0.005):
        """
        找出均值相差0.005的算法

        如果alpha为None，就返回每个算法的名称及其精度排名
        Parameters
        ----------
        df :
        metric :
        n_fold :

        Returns
        -------

        """
        # filter_df = df[df['metric'] == metric]
        filter_df = df
        if filter_df.shape[0] < 1:
            print("没有数据")
            pass
        assert filter_df['dataset'].drop_duplicates().shape[
                   0] == 1, f"只能计算一个数据集, but get {filter_df.shape[0]} "
        # filter_df 应该是一个数据上的10折验证结果
        assert filter_df.shape[0] <= df['folds'].iloc[0], f"交叉数据不对出错：{filter_df.shape[0]}"
        # assert filter_df.shape[0] >= 6, f"数据数量不正确，filter_df 应该是一个数据上的10折验证结果，{dim_redu_method}"

        output = []
        for i, v in filter_df.iterrows():
            _t = AnaHelper.get_models_accuracy_of_series(v, metric=metric)
            output.append(_t.reset_index())
        _df = pd.concat(output, axis=0)

        final_df = pd.pivot_table(_df, index=['model'], values=['loss'], aggfunc=["mean", "std"]).reset_index()
        final_df.columns = ['model_name', "mean", 'std']

        if alpha is None:
            d = final_df.sort_values(by="mean", ascending=False).reset_index(drop=True)
            return list(d.iloc[:, 0]), list(d.iloc[:, 1])
        else:
            _top1_acc = final_df['mean'].max()

            outputs = []
            for i, v in final_df.iterrows():
                if alpha >= 0.01:
                    _acc_diff = round(_top1_acc - v['mean'], 2)
                else:
                    _acc_diff = round(_top1_acc - v['mean'], 3)

                if _acc_diff <= alpha:
                    outputs.append([v['model_name'], v['mean']])

            d = np.asarray(outputs)
            return list(d[:, 0]), list(d[:, 1])

    @classmethod
    # @memory.cache
    def get_models_and_acc_baseline(cls, df, metric):
        """
        获取在每个数据集(df)在精度指标(metric)上面的算法名称及其精度, 按算法名称排序

        返回:
        0 = {list: 14} ['sgd', 'random_forest', 'qda', 'multinomial_nb', 'mlp', 'libsvm_svc', 'liblinear_svc', 'lda', 'k_nearest_neighbors', 'gaussian_nb', 'extra_trees', 'decision_tree', 'bernoulli_nb', 'adaboost']
        1 = {list: 14} [0.79532, 0.83158, 0.75586, 0.0, 0.8032999999999999, 0.64978, 0.74998, 0.78894, 0.57144, 0.60344, 0.82412, 0.7777000000000001, 0.68336, 0.81556]
        """
        filter_df = df
        if filter_df.shape[0] < 1:
            print("没有数据")
            pass
        assert filter_df['dataset'].drop_duplicates().shape[
                   0] == 1, f"只能计算一个数据集, but get {filter_df.shape[0]} "
        # filter_df 应该是一个数据上的10折验证结果
        assert filter_df.shape[0] <= df['folds'].iloc[0], f"交叉数据不对出错：{filter_df.shape[0]}"
        # assert filter_df.shape[0] >= 6, f"数据数量不正确，filter_df 应该是一个数据上的10折验证结果，{dim_redu_method}"

        output = []
        for i, v in filter_df.iterrows():
            _t = AnaHelper.get_models_accuracy_of_series(v, metric=metric)
            output.append(_t.reset_index())
        _df = pd.concat(output, axis=0)

        # final_df = pd.pivot_table(_df, index=['model'], values=['loss'], aggfunc=["mean", "std"]).reset_index()
        final_df = _df.groupby(by='model')['loss'].agg(['mean', 'std']).reset_index()
        final_df = final_df.fillna(0)
        final_df.columns = ['model_name', "mean", 'std']

        d = final_df.sort_values(by="model_name", ascending=False).reset_index(drop=True)
        return list(d.iloc[:, 0]), list(d.iloc[:, 1])

    @classmethod
    # @memory.cache
    def rank_models_baseline(cls, dataset, metric):
        """
        找出baseline中最好的模型
        Parameters
        ----------
        dataset :
        metric :

        Returns
        -------

        """
        if cls.df_ is None:
            cls.df_ = pd.read_csv(AnaHelper.METRIC_FILE_BASELINE)
        alg_rank = AnaHelper.rank_model_of_baseline(
            cls.df_,
            dataset=dataset,
            metric=metric,
        )
        return alg_rank

    @classmethod
    # @memory.cache
    def get_topn_model_name_acc_baseline(cls, dataset, metric, top_index=0):
        """
        获取baseline上给定数据集在给定metric上的排名top_index的算法名称，0表示排名第一的，1表示排名第二的

        返回：模型名称，该模型对应的精度

        Parameters
        ----------
        dataset :
        metric :
        top_index :

        Returns
        -------

        """
        top_best_model = AnaHelper.rank_models_baseline(dataset=dataset, metric=metric)
        return top_best_model.iloc[top_index, 0], top_best_model.iloc[top_index, 1]

    @classmethod
    @memory.cache
    def find_top_models_baseline(cls, dataset, metric, alpha=0.005):
        """
        获取baseline中与最佳精度相差alpha=0.01的算法名称。
        我找出baseline里面精度相差1%以类的算法，作为基线计算
        返回：模型名称，该模型对应的精度

        Parameters
        ----------
        dataset :
        metric :
        top_index :

        Returns
        -------

        """
        if alpha is None:
            top_best_model = AnaHelper.rank_models_baseline(dataset=dataset, metric=metric)
            return list(top_best_model.iloc[:, 0]), list(top_best_model.iloc[:, 1])
        else:
            outptus = []
            for i in range(14):
                _name, _acc = AnaHelper.get_topn_model_name_acc_baseline(dataset=dataset, metric=metric, top_index=i)

                # 初始化
                if i == 0:
                    outptus.append([_name, _acc])
                if alpha >= 0.01:
                    _acc_diff = round(outptus[0][1] - _acc, 2)
                else:
                    _acc_diff = round(outptus[0][1] - _acc, 3)

                if i != 0 and _acc_diff <= alpha:
                    outptus.append([_name, _acc])

            # 如果只找到一个，那么我们就把排名第二的算法也加进去，为了容错
            if len(outptus) == 1:
                _name, _acc = AnaHelper.get_topn_model_name_acc_baseline(dataset=dataset, metric=metric, top_index=1)
                outptus.append([_name, _acc])

            data = np.asarray(outptus)
            return list(data[:, 0]), list(data[:, 1])

    @classmethod
    # @memory.cache
    def load_acc_select(cls, file, alpha=0.005, n_exploration=None):
        df = AnaHelper.load_csv_file(file)
        metrics = df['metric'].drop_duplicates().tolist()
        datasets = df['dataset'].drop_duplicates().tolist()
        feature_selec_method = df['feature_selec_method'].drop_duplicates().tolist()
        feature_selec_rate = df['feature_selec_rate'].drop_duplicates().tolist()
        data_sample_rate = df['data_sample_rate'].drop_duplicates().tolist()
        data_sample_method = df['data_sample_method'].drop_duplicates().tolist()
        if n_exploration is None:
            n_exploration = df['n_exploration'].drop_duplicates().tolist()

        hpo_opt_method = df['hpo_opt_method'].drop_duplicates().tolist()
        items = list(itertools.product(
            datasets,
            metrics,
            feature_selec_method,
            feature_selec_rate,
            data_sample_method,
            data_sample_rate,
            n_exploration,
            hpo_opt_method
        ))
        outputs = []
        for _dataset, _metric, _fsm, _fsr, _dsm, _dsr, _n_exploration, _hpo_opt_method in tqdm(items):

            top_models_baseline, acc_baseline = BaselineHelper.get_models_and_acc_with_alpha(_dataset, _metric,
                                                                                             alpha=alpha,
                                                                                             # 与最佳算法的精度相差千分之1
                                                                                             include_top_2=False)

            top_models_tshpo, acc_tshpo = TSHPOHelper.get_models_and_acc(df, _dataset, _metric, _fsm, _fsr, _dsm, _dsr,
                                                                         _n_exploration, _hpo_opt_method)
            for _index in range(8):
                top_n = _index + 1
                acc = int(len(np.intersect1d(top_models_baseline, top_models_tshpo[:top_n])) > 0)
                _t = {
                    "top_n": top_n,
                    "acc": acc,
                    "dataset": _dataset,
                    "metric": _metric,
                    "fsm": _fsm,
                    "fsr": _fsr,
                    "dsm": _dsm,
                    "dsr": _dsr,
                    "n_exploration": _n_exploration,
                    "hpo_opt_method": _hpo_opt_method,
                    "top_models_tshpo": top_models_tshpo[:top_n]
                }
                outputs.append(_t)
                # print(outputs[-1])

        return pd.DataFrame(outputs)

    @classmethod
    @memory.cache
    def load_acc_select_random(cls, file, alpha=0.001, n_exploration=None, metrics: list = None):
        """
        专门用于分析HPO的优化方法是RS的情况，其他情况请使用 AnaHelper.load_acc_select
        Parameters
        ----------
        file :
        alpha :
        n_exploration :
        metric :

        Returns
        -------

        """
        df = AnaHelper.load_csv_file(file)

        df['hpo_opt_method'].drop_duplicates().tolist()

        assert df['hpo_opt_method'].drop_duplicates().tolist() == [
            'RS'], "超参数优化算法智能是HPO，如果不是，请使用AnaHelper.load_acc_select方法"
        if metrics is None:
            metrics = df['metric'].drop_duplicates().tolist()

        datasets = df['dataset'].drop_duplicates().tolist()
        feature_selec_method = df['feature_selec_method'].drop_duplicates().tolist()
        feature_selec_rate = df['feature_selec_rate'].drop_duplicates().tolist()
        data_sample_rate = df['data_sample_rate'].drop_duplicates().tolist()
        data_sample_method = df['data_sample_method'].drop_duplicates().tolist()
        if n_exploration is None:
            n_exploration = df['n_exploration'].drop_duplicates().tolist()

        hpo_opt_method = df['hpo_opt_method'].drop_duplicates().tolist()
        items = list(itertools.product(
            datasets,
            metrics,
            feature_selec_method,
            feature_selec_rate,
            data_sample_method,
            data_sample_rate,
            n_exploration,
            hpo_opt_method
        ))
        outputs = []
        for _dataset, _metric, _fsm, _fsr, _dsm, _dsr, _n_exploration, _hpo_opt_method in tqdm(items):

            top_models_baseline, acc_baseline = BaselineHelper.get_models_and_acc_with_alpha(_dataset, _metric,
                                                                                             alpha=alpha,
                                                                                             # 与最佳算法的精度相差千分之1
                                                                                             include_top_2=False)
            # 随机抽样事，metric是什么不重要
            df['metric'] = _metric
            top_models_tshpo, acc_tshpo = TSHPOHelper.get_models_and_acc(df, _dataset, _metric, _fsm, _fsr, _dsm, _dsr,
                                                                         _n_exploration, _hpo_opt_method)
            for _index in range(6):
                top_n = _index + 1
                acc = int(len(np.intersect1d(top_models_baseline, top_models_tshpo[:top_n])) > 0)
                _t = {
                    "top_n": top_n,
                    "acc": acc,
                    "dataset": _dataset,
                    "metric": _metric,
                    "fsm": _fsm,
                    "fsr": _fsr,
                    "dsm": _dsm,
                    "dsr": _dsr,
                    "n_exploration": _n_exploration,
                    "hpo_opt_method": _hpo_opt_method
                }
                outputs.append(_t)

        return pd.DataFrame(outputs)

    @classmethod
    @memory.cache
    def load_acc_select_back(cls, file_name, alpha=0.001, n_exploration=None):
        df = pd.read_csv(file_name)
        metrics = df['metric'].drop_duplicates().tolist()
        datasets = df['dataset'].drop_duplicates().tolist()
        feature_selec_method = df['feature_selec_method'].drop_duplicates().tolist()
        feature_selec_rate = df['feature_selec_rate'].drop_duplicates().tolist()
        data_sample_rate = df['data_sample_rate'].drop_duplicates().tolist()
        data_sample_method = df['data_sample_method'].drop_duplicates().tolist()
        if n_exploration is None:
            n_exploration = df['n_exploration'].drop_duplicates().tolist()

        hpo_opt_method = df['hpo_opt_method'].drop_duplicates().tolist()
        items = list(itertools.product(
            datasets,
            metrics,
            feature_selec_method,
            feature_selec_rate,
            data_sample_method,
            data_sample_rate,
            n_exploration,
            hpo_opt_method
        ))
        outputs = []
        for _dataset, _metric, _fsm, _fsr, _dsm, _dsr, _n_exploration, _hpo_opt_method in tqdm(items):

            top_models_baseline, acc_baseline = BaselineHelper.get_models_and_acc_with_alpha(_dataset, _metric,
                                                                                             alpha=alpha,
                                                                                             # 与最佳算法的精度相差千分之1
                                                                                             include_top_2=False)

            top_models_tshpo, acc_tshpo = TSHPOHelper.get_models_and_acc(df, _dataset, _metric, _fsm, _fsr, _dsm, _dsr,
                                                                         _n_exploration, _hpo_opt_method)
            for _index in range(6):
                top_n = _index + 1
                acc = int(len(np.intersect1d(top_models_baseline, top_models_tshpo[:top_n])) > 0)
                _t = {
                    "top_n": top_n,
                    "acc": acc,
                    "dataset": _dataset,
                    "metric": _metric,
                    "fsm": _fsm,
                    "fsr": _fsr,
                    "dsm": _dsm,
                    "dsr": _dsr,
                    "n_exploration": _n_exploration,
                    "hpo_opt_method": _hpo_opt_method
                }
                outputs.append(_t)
                # print(outputs[-1])

        return pd.DataFrame(outputs)

    @classmethod
    def append_data_size_type(cls, df):
        """
        在df上增加数据的类型，列名是data_type

        Parameters
        ----------
        df :

        Returns
        -------

        """
        df['data_size_type'] = df.apply(lambda x: AnaHelper.get_data_size_type(x['dataset']), axis=1)
        return df

    @classmethod
    def append_normal_hpo_name(cls, df):
        df['hpo_opt_method'] = df.apply(lambda x: AnaHelper.normal_opt_method_name(x['hpo_opt_method']), axis=1)
        return df

    @classmethod
    def append_normal_metric_name(cls, df):
        df['metric'] = df.apply(lambda x: AnaHelper.normal_metric_name(x['metric'][0]), axis=1)
        return df

    @classmethod
    @memory.cache
    def to_other_metric(cls, file, metric):
        """
        将文件中的 metric 指标改为其他指标，能够这样做的原因是随机优化时指标无所谓
        Parameters
        ----------
        file :
        metric :

        Returns
        -------

        """
        file = AnaHelper.prepare_csv(file, return_df=False)
        f = Path(file)
        output_file = os.path.join(f.parent, f"{f.stem}_{metric}{f.suffix}")
        df = pd.read_csv(file)
        df['metric'] = metric
        df.to_csv(output_file)
        return os.path.basename(output_file)


class HPTrimHelper:
    """
    修剪超参数的工具类
    """

    @staticmethod
    def generate_outputs(file_name):
        df_array = []
        # ori_df = AnaHelper.load_csv_file("c09_select_optimal_alg_v2_original_20241031_1352.csv.gz")
        ori_df = AnaHelper.load_csv_file(file_name)
        for _metric in AnaHelper.get_all_metrics():
            _df = copy.deepcopy(ori_df)
            _df["metric"] = _metric
            df_array.append(_df)
        df = pd.concat(df_array, ignore_index=True)
        outputs = []
        for _dataset in df['dataset'].drop_duplicates().tolist():
            for _metric in df['metric'].drop_duplicates().tolist():
                models, accs = TSHPOHelper.get_models_and_acc(df, _dataset, _metric,
                                                              _fsm="RF", _fsr=0.3,
                                                              _dsm='RS', _dsr=0.3,
                                                              _n_exploration=50,
                                                              _hpo_method="RS")
                outputs.append({
                    "metric": _metric,
                    "dataset": _dataset,
                    "models": models,
                    "accs": accs
                })
        pd.DataFrame(outputs).to_pickle(MODEL_SELECT_INFO_CACHE_FILE)

    @staticmethod
    def get_high_performing_models(dataset, metric, top_n):
        assert metric in AnaHelper.get_all_metrics(), f"unsupported metric={metric}, expected {AnaHelper.get_all_metrics()}"
        df = pd.read_pickle(MODEL_SELECT_INFO_CACHE_FILE)
        select_rows = df[(df['metric'] == metric) & (df['dataset'] == dataset)]
        if select_rows.shape[0] != 1:
            a = 1
            pass
        assert select_rows.shape[0] == 1
        select_rows = select_rows.iloc[0, 2]
        return select_rows[:top_n]


@dataclass
class TSHPOHelper:
    metric_file: str = "/Users/sunwu/Downloads/c03_method_select_v2_original_20241022_0402.csv.gz"
    df_: pd.DataFrame = None

    @classmethod
    @memory.cache
    def get_df(cls):
        df = pd.read_csv(cls.metric_file)
        # 加载数据并缓存
        # df = AnaHelper.pre_process(df)
        return df

    @classmethod
    def get_models_and_acc(cls, df, _dataset, _metric, _fsm, _fsr, _dsm, _dsr, _n_exploration, _hpo_method):
        """
        返回基线中算法名称和精度
        Parameters
        ----------
        dataset :
        metric :

        Returns
        -------

        """
        _tshpo = df[
            (df['dataset'] == _dataset)
            & (df['metric'] == _metric)
            & (df['feature_selec_method'] == _fsm) & (df['feature_selec_rate'] == _fsr)
            & (df['data_sample_method'] == _dsm) & (df['data_sample_rate'] == _dsr)
            & (df['hpo_opt_method'] == _hpo_method)
            ]
        assert _tshpo.shape[0] > 0 and _tshpo.shape[0] <= 5, f"数据量不对，必须是交叉验证的数量:{_tshpo.shape}"
        top_models_tshpo, acc_tshpo = AnaHelper.get_rank_model_and_acc(_tshpo, metric=_metric,
                                                                       exporation=_n_exploration)
        return top_models_tshpo, acc_tshpo


@dataclass
class BaselineHelper:
    df_: pd.DataFrame = None

    @DeprecationWarning
    @staticmethod
    @memory.cache
    def load_baseline(csv_file=None):
        # 每个算法探索100次，数据不抽样、不降维

        if csv_file is None:
            csv_file = "/Users/sunwu/Documents/baseline_original_20241019_0148.csv.gz"
        df = pd.read_csv(csv_file)
        metrics = AnaHelper.get_all_metrics()
        datasets = df['dataset'].drop_duplicates().tolist()
        feature_selec_method = df['feature_selec_method'].drop_duplicates().tolist()
        date_sample_method = df['data_sample_method'].drop_duplicates().tolist()
        feature_selec_rate = [1]
        date_sample_rate = [1]
        n_explorations = df['n_exploration'].drop_duplicates().tolist()
        _baseline_exploration = np.max(n_explorations)

        outputs = []
        for _feature_selec_method, _feature_selec_rate, _data_sample_method, _data_sample_rate, _dataset, _metric in itertools.product(
                feature_selec_method,
                feature_selec_rate,
                date_sample_method,
                date_sample_rate,
                datasets,
                metrics,
        ):
            _baseline_df = df[(df['feature_selec_method'] == _feature_selec_method) &
                              (df['feature_selec_rate'] == 1) &
                              (df['dataset'] == _dataset) &
                              (df['data_sample_rate'] == 1) &
                              (df['data_sample_method'] == _data_sample_method)
                              ]

            baseline_models, baseline_accs = AnaHelper.get_rank_model_and_acc(_baseline_df,
                                                                              _baseline_exploration,
                                                                              metric=_metric)

            _t = {
                "feature_selec_method": _feature_selec_method,
                "feature_selec_rate": _feature_selec_rate,
                "baseline_explorations": _baseline_exploration,
                "data_sample_method": _data_sample_method,
                "data_sample_rate": _data_sample_rate,
                "dataset": _dataset,
                "metric": _metric,
                "models": baseline_models,
                "accs": baseline_accs,
                "model_training_time_sum": _baseline_df['model_training_time'].sum(),
                "model_training_time_mean": _baseline_df['model_training_time'].mean(),
                "data_processing_time_sum": _baseline_df['data_processing_time'].mean(),
                "data_processing_time_mean": _baseline_df['data_processing_time'].mean(),

            }

            outputs.append(_t)
        return pd.DataFrame(outputs)

    @staticmethod
    @memory.cache
    def load_baseline_v2():
        # 每个算法探索100次，数据不抽样、不降维
        df = AnaHelper.load_csv_file(ResultFiles.get_baseline_file())
        metrics = AnaHelper.get_all_metrics()
        datasets = df['dataset'].drop_duplicates().tolist()
        feature_selec_method = df['feature_selec_method'].drop_duplicates().tolist()
        date_sample_method = df['data_sample_method'].drop_duplicates().tolist()
        feature_selec_rate = [1]
        date_sample_rate = [1]
        n_explorations = df['n_exploration'].drop_duplicates().tolist()
        _baseline_exploration = np.max(n_explorations)

        outputs = []
        for _feature_selec_method, _feature_selec_rate, _data_sample_method, _data_sample_rate, _dataset, _metric in itertools.product(
                feature_selec_method,
                feature_selec_rate,
                date_sample_method,
                date_sample_rate,
                datasets,
                metrics,
        ):
            _baseline_df = df[(df['feature_selec_method'] == _feature_selec_method) &
                              (df['feature_selec_rate'] == 1) &
                              (df['dataset'] == _dataset) &
                              (df['data_sample_rate'] == 1) &
                              (df['data_sample_method'] == _data_sample_method)
                              ]

            baseline_models, baseline_accs = AnaHelper.get_rank_model_and_acc(_baseline_df,
                                                                              _baseline_exploration,
                                                                              metric=_metric)

            _t = {
                "feature_selec_method": _feature_selec_method,
                "feature_selec_rate": _feature_selec_rate,
                "baseline_explorations": _baseline_exploration,
                "data_sample_method": _data_sample_method,
                "data_sample_rate": _data_sample_rate,
                "dataset": _dataset,
                "metric": _metric,
                "models": baseline_models,
                "accs": baseline_accs,
                "model_training_time_sum": _baseline_df['model_training_time'].sum(),
                "model_training_time_mean": _baseline_df['model_training_time'].mean(),
                "data_processing_time_sum": _baseline_df['data_processing_time'].mean(),
                "data_processing_time_mean": _baseline_df['data_processing_time'].mean(),

            }

            outputs.append(_t)
        return pd.DataFrame(outputs)

    @staticmethod
    @memory.cache
    def load_baseline_v2_bak(csv_file=None):
        # 每个算法探索100次，数据不抽样、不降维
        csv_file = AnaHelper.prepare_csv(ResultFiles.get_baseline_file())
        df = pd.read_csv(csv_file)
        metrics = AnaHelper.get_all_metrics()
        datasets = df['dataset'].drop_duplicates().tolist()
        feature_selec_method = df['feature_selec_method'].drop_duplicates().tolist()
        date_sample_method = df['data_sample_method'].drop_duplicates().tolist()
        feature_selec_rate = [1]
        date_sample_rate = [1]
        n_explorations = df['n_exploration'].drop_duplicates().tolist()
        _baseline_exploration = np.max(n_explorations)

        outputs = []
        for _feature_selec_method, _feature_selec_rate, _data_sample_method, _data_sample_rate, _dataset, _metric in itertools.product(
                feature_selec_method,
                feature_selec_rate,
                date_sample_method,
                date_sample_rate,
                datasets,
                metrics,
        ):
            _baseline_df = df[(df['feature_selec_method'] == _feature_selec_method) &
                              (df['feature_selec_rate'] == 1) &
                              (df['dataset'] == _dataset) &
                              (df['data_sample_rate'] == 1) &
                              (df['data_sample_method'] == _data_sample_method)
                              ]

            baseline_models, baseline_accs = AnaHelper.get_rank_model_and_acc(_baseline_df,
                                                                              _baseline_exploration,
                                                                              metric=_metric)

            _t = {
                "feature_selec_method": _feature_selec_method,
                "feature_selec_rate": _feature_selec_rate,
                "baseline_explorations": _baseline_exploration,
                "data_sample_method": _data_sample_method,
                "data_sample_rate": _data_sample_rate,
                "dataset": _dataset,
                "metric": _metric,
                "models": baseline_models,
                "accs": baseline_accs,
                "model_training_time_sum": _baseline_df['model_training_time'].sum(),
                "model_training_time_mean": _baseline_df['model_training_time'].mean(),
                "data_processing_time_sum": _baseline_df['data_processing_time'].mean(),
                "data_processing_time_mean": _baseline_df['data_processing_time'].mean(),

            }

            outputs.append(_t)
        return pd.DataFrame(outputs)

    @classmethod
    # @memory.cache
    def get_models_and_acc(cls, dataset, metric):
        """
        返回基线中算法名称和精度

        Parameters
        ----------
        dataset :
        metric :

        Returns
        -------

        """
        if cls.df_ is None:
            cls.df_ = BaselineHelper.load_baseline_v2()

        df = cls.df_
        _baseline_df = df[
            (df['feature_selec_rate'] == 1) &
            (df['dataset'] == dataset) &
            (df['data_sample_rate'] == 1) &
            (df['metric'] == metric)
            ]
        return _baseline_df['models'].iloc[0], _baseline_df['accs'].iloc[0]

    @classmethod
    @memory.cache
    def get_models_and_acc_with_alpha(cls, dataset, metric, alpha=0.01, include_top_2=True):
        """
        返回基线中相差千分之5的算法(alpha)名称和精度.include_top_2 = True 时最少包括前两个算法

        Parameters
        ----------
        dataset :
        metric :
        alpha :

        Returns
        -------

        """
        models, accs = cls.get_models_and_acc(dataset, metric)
        if AnaHelper.is_maximize(metric):
            assert accs[0] >= accs[1], "必须是降序排序"
        else:
            assert accs[0] <= accs[1], "必须是升序排序"
        outputs = []
        for _model, _acc in zip(models, accs):
            if accs[0] - _acc <= float(alpha):
                outputs.append([_model, _acc])

        # 最少包括前两个算法

        if len(outputs) < 2 and include_top_2:
            outputs.append([models[1], accs[1]])

        df = pd.DataFrame(outputs, columns=['model', 'acc'])
        return df["model"].tolist(), df["acc"].tolist()


@dataclass
class AccuracySel:
    pass


from distributed import LocalCluster
from distributed import Client


@dataclass
class DaskHelper:
    _features: list = None
    _client = None

    def __post_init__(self):
        self._features = []

    def submit(self, fun, info):
        """

        Parameters
        ----------
        train_model_smac :  要运行的函数
        jobs : 函数的参数

        Returns
        -------

        """
        _client = self.get_client()
        self._features.append([info, self._client.submit(fun, info=info)])

    def get_client(self):
        if DaskHelper._client is None:
            cluster = LocalCluster(n_workers=int(os.cpu_count() * 0.7), threads_per_worker=1, dashboard_address=":5901",
                                   ip="your_server_ip")
            DaskHelper._client = Client(cluster)
        return DaskHelper._client

    def gather(self):
        return self._client.gather(self._features)


class FileHelper:
    def __init__(self):
        self._result_home = os.path.abspath(os.path.join(DIR_CRT, "runtime", "exp_outputs"))
        os.makedirs(self._result_home, exist_ok=True)

    def get_file_name(self, file_name):
        return os.path.join(self._result_home, file_name)

    def to_pdf(self, fig: plt.Figure, file_name):
        file_name = self.get_file_name(file_name)
        print(f"Saved to {file_name}")
        fig.savefig(file_name, bbox_inches='tight', dpi=300)

    def to_latex(self, ttest_results: pd.DataFrame, file_name, index=True):
        file_name = self.get_file_name(file_name)
        print(f"Saved to {os.path.abspath(file_name)}")
        ttest_results.to_latex(file_name, index=index)

    def to_excel(self, df: pd.DataFrame, file_name, index=False):
        file_name = self.get_file_name(file_name)
        df.to_excel(file_name, index=index)


if __name__ == '__main__':
    # HPTrimHelper.generate_outputs()
    df = BaselineHelper.load_baseline_v2()
    df.to_excel("baseline_v2.xlsx", index=False)
    # print(df)
