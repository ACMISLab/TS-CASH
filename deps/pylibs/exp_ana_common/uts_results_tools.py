import abc
import json
import logging
import os
import pprint
import sys
import traceback
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pygnuplot import gnuplot
from tqdm import tqdm
from typeguard import typechecked
import swifter
from pylibs.config import ExpServerConf
from pylibs.exp_ana_common.ExcelOutKeys import ExcelMetricsKeys, EK, PaperFormat
from pylibs.exp_ana_common.gather_metrics import FileGathers
from pylibs.experiments.exp_helper import JobConfV1
from pylibs.utils.util_bash import CMD
from pylibs.utils.util_common import UtilComm
from pylibs.utils.util_directory import make_dirs
from pylibs.utils.util_exp_result import ER
from pylibs.utils.util_file import FileUtil, FileUtils
from pylibs.utils.util_gnuplot import _generate_xlabel, UTSViewGnuplot
from pylibs.utils.util_joblib import JLUtil
from pylibs.utils.util_log import get_logger
from pylibs.utils.util_number import round_dict
from pylibs.utils.util_pandas import PDUtil
from pylibs.utils.util_redis import RedisUtil
from pylibs.utils.util_statistical import UtilHypo
from pylibs.utils.util_sys import get_num_cpus
from pylibs.utils.util_system import UtilSys

log = get_logger(logging.INFO)
memory = JLUtil.get_memory()


def merge_kfold_data(df):
    output = []
    for _group_keys_item, _data in df.groupby(by=ExcelMetricAnalysis.GROUP_BY_KEYS):
        # 计算多折交叉验证的均值
        all_key = list(df.columns)
        comon_key = list(np.intersect1d(all_key, ExcelMetricsKeys.get_performance_keys()))

        itemkey = list(np.setxor1d(comon_key, all_key))

        _mean = _data[comon_key].mean()
        _median = _data[comon_key].median()
        _std = _data[comon_key].std()
        _other_info = _data[itemkey]

        _columns_1 = comon_key * 3
        len_perf_keys = len(comon_key)
        _columns_2 = [EK.MEAN] * len_perf_keys + [EK.STD] * len_perf_keys + [EK.MEDIAN] * len_perf_keys
        _columns_values = _mean.to_list() + _std.to_list() + _median.to_list()

        temp_res = pd.DataFrame([_columns_values], columns=list(zip(_columns_1, _columns_2)))

        for _key in _other_info.columns.tolist():
            _item_value = _other_info[_key].unique()
            if _item_value.shape[0] == 1:
                _insert_val = _item_value[0]
            else:
                _insert_val = ";".join(_item_value.astype("str"))
            temp_res.insert(0, _key, _insert_val)
        output.append(temp_res)

    # df: original data, not merag kfold
    # out: merged data. Merges the different k fold for each model

    avg_perf = pd.concat(output)

    # 计算时间*5, 因为上面计算时间只是一折交叉验证的结果
    avg_perf[(EK.MEAN, EK.ELAPSED_TRAIN)] = avg_perf[(EK.MEAN, EK.ELAPSED_TRAIN)] * 5
    avg_perf[(EK.ELAPSED_TRAIN, EK.MEDIAN)] = avg_perf[(EK.ELAPSED_TRAIN, EK.MEDIAN)] * 5

    # todo: data processing time
    # avg_perf[(EK.MEAN,EK.DATA_PROCESSING_TIME)] = avg_perf[(EK.MEAN,EK.DATA_PROCESSING_TIME)] * 5

    return df, avg_perf


def get_gnuplot_command_for_sr_on_perf():
    # 0 = {str} 'data_sample_method' 1
    # 1 = {str} 'model_name' 2
    # 2 = {str} 'sr*' 3
    # 3 = {str} 'VUS_ROC'  4
    # 4 = {str} 'VUS_PR'  5
    # 5 = {str} 'Rprecision' 6
    # 6 = {str} 'Recall' 7
    # 7 = {str} 'RF' 8
    # 8 = {str} 'elapsed_train' 9
    # 9 = {str} 'R_AUC_ROC' 10

    plot = """

set term pdfcairo lw 2 font "Times New Roman,20" enhanced size 16,9
set output "pic_data_size_and_accuracy.pdf"
set multiplot layout 2,2

#set style data histogram
#set style histogram errorbars gap 1 lw 1
set ylabel "Performance Metric"
set y2label "Train. Time"
set xlabel "Training Data Ratio"
set key right
# set key reverse
set key spacing 1
set key horizontal
set key top left outside
set key at screen 0.5,0.99 center top horizontal

set xtics rotate by -45
set grid
set y2tics
list=system('ls -1B *.dat')
print list


do for [filename in list] {
    set title  filename[1:(strlen(filename)-4)][8:strlen(filename)] noenhanced
    plot filename u 0:4:xtic(3) with lp lt 2 title "ROC" noenhanced,"" u 0:8:xtic(3) with lp lt 12 title "F1" noenhanced, "" u 0:6:xtic(3) with lp lt 18 title "Precision" noenhanced, "" u 0:8:xtic(3) axis x1y2 with line dt 2 lt 24 title "Train. Time" noenhanced
}


unset multiplot
set output

            """
    return plot


def get_gnuplot_command_for_sr_on_perf_for_every_model_and_dataset():
    # 0 = {str} 'data_sample_method' 1
    # 1 = {str} 'model_name' 2
    # 2 = {str} 'sr*' 3
    # 3 = {str} 'VUS_ROC'  4
    # 4 = {str} 'VUS_PR'  5
    # 5 = {str} 'Rprecision' 6
    # 6 = {str} 'Recall' 7
    # 7 = {str} 'RF' 8
    # 8 = {str} 'elapsed_train' 9
    # 9 = {str} 'R_AUC_ROC' 10

    plot = """

set term pdfcairo lw 2 font "Times New Roman,20" enhanced size 16,20
set output "pic_data_size_and_accuracy.pdf"
set multiplot layout 4,2

#set style data histogram
#set style histogram errorbars gap 1 lw 1
set ylabel "Performance Metric"
set y2label "Train. Time"
set xlabel "Training Data Ratio"
set key right
# set key reverse
set key spacing 1.5
set key horizontal
set key top left outside
set key at screen 0.5,1 center top horizontal

set xtics rotate by -45
set grid
set y2tics
list=system('ls -1B *.dat')
print list


do for [filename in list] {
    set title  filename[1:(strlen(filename)-4)][8:strlen(filename)] noenhanced
    plot filename u 0:4:xtic(3) with lp lt 2 title "ROC" noenhanced,"" u 0:8:xtic(3) with lp lt 12 title "F1" noenhanced, "" u 0:6:xtic(3) with lp lt 18 title "Precision" noenhanced, "" u 0:8:xtic(3) axis x1y2 with line dt 2 lt 24 title "Train. Time" noenhanced
}


unset multiplot
set output

            """
    return plot


from pylibs.utils.util_joblib import cache_


def is_stopped_training_v1(metrics, post_see, alpha=0.001):
    """
    相对于当前的性能，如果增加数据集两次后(post_see)，
    模型的性能没有提高0.001 (alpha)，那么我们就停止训练。

    例如：
    self.assertTrue(is_stopped_training([0, 0, 0]))
    self.assertFalse(is_stopped_training([0, 0.1, 0.01]))
    self.assertFalse(is_stopped_training([0, 0.001, 0.001]))
    """
    metrics = np.asarray(metrics, dtype=np.float32)
    current = metrics[0]
    values = metrics - current
    # 只要有增加,就继续训练.

    indexer = np.max(values[1:post_see + 1])
    if indexer <= alpha:
        return True
    else:
        return False


@cache_
def is_stopped_training_v2(metrics, post_see, alpha=0.001):
    """
    相对于当前的性能，如果增加数据集两次后(post_see)，
    模型的性能没有提高0.001 (alpha)，那么我们就停止训练。

    例如：
    self.assertTrue(is_stopped_training([0, 0, 0]))
    self.assertFalse(is_stopped_training([0, 0.1, 0.01]))
    self.assertFalse(is_stopped_training([0, 0.001, 0.001]))
    """
    metrics = np.asarray(metrics, dtype=np.float32)
    current = metrics[0]
    values = metrics - current
    # 只要有增加,就继续训练.
    indexer = np.max(values[1:post_see + 1])
    UtilSys.is_debug_mode() and log.info(
        f"Decide whether to stop. Cur value: {indexer}, is stop: {indexer < alpha}, post see={post_see}, stop alpha={alpha}, OptMetricsType: \n{metrics}")
    if indexer < alpha:
        return True
    else:
        return False


def is_stopped_training_v3(metrics, post_see, alpha=0.01):
    """
    相对于当前的性能，如果增加数据集两次后(post_see)，
    模型的性能没有提高0.001 (alpha)，那么我们就停止训练。

    例如：
    self.assertTrue(is_stopped_training([0, 0, 0]))
    self.assertFalse(is_stopped_training([0, 0.1, 0.01]))
    self.assertFalse(is_stopped_training([0, 0.001, 0.001]))
    """
    metrics = np.asarray(metrics, dtype=np.float32)
    current = metrics[0]
    values = metrics - current
    # 只要有增加,就继续训练.
    increased_perf = np.max(values[1:post_see + 1])
    if increased_perf <= current * alpha:
        return True
    else:
        return False


class OptKeys:
    KEY_DATA_ID = "data_id"
    KEY_TIME_ELAPSED = "Average of elapsed_train"
    KEY_VUS_PR = "Average of VUS_PR"
    KEY_VUS_ROC = "Average of VUS_ROC"
    KEY_DATA_SAMPLE_RATE = "data_sample_rate"
    KEY_ORIGINAL_METRIC = "original_metric"
    KEY_FOUND_METRIC = "found_metric"
    KEY_ORIGINAL_TRAIN_TIME = "original_train_time(s)"
    KEY_REAL_TRAIN_TIME = "real_train_time(s)"
    KEY_FOUND_BEST_SAMPLE_RATE = "found_best_sample_rate"
    KEY_BEST_METRIC = "best_metric"


from pylibs.utils.util_joblib import cache_


@cache_
def _load_all_metrics(metric_dir,
                      is_save_all=True,
                      baseline_sr=-1.,
                      target_metric="VUS_ROC",
                      out_home=UtilComm.get_runtime_directory(),
                      endswith="_metrics.csv",
                      job_id=None,
                      headers=None):
    return FileGathers(metric_dir,
                       is_save_all,
                       baseline_sr,
                       target_metric,
                       out_home,
                       endswith,
                       job_id,
                       headers).load_all_metrics()


class ExcelMetricAnalysis:
    BASE_SAMPLE_RATE = 9999999
    VUS_ROC = "VUS_ROC"
    VUS_PR = "VUS_PR"
    KEY_SR_START = "sr*"
    GROUP_BY_KEYS = [
        ExcelMetricsKeys.ANOMALY_WINDOW_TYPE,
        ExcelMetricsKeys.WINDOW_SIZE,
        ExcelMetricsKeys.MODEL_NAME,
        ExcelMetricsKeys.DATASET_NAME,
        ExcelMetricsKeys.DATA_ID,
        ExcelMetricsKeys.DATA_SAMPLE_RATE
    ]

    def __init__(self, analysis_key_metric="VUS_ROC", baseline_same_rate='-1', post_see=2, alpha=0.01,
                 is_cache=False,
                 job_id=None,
                 out_home=UtilComm.get_runtime_directory(),
                 endswith="_metrics.csv", header=None, save_to_file=True):
        """
        filename: iforest_original_metrics.xlsx or pd.DataFrame
        analysis_key_metric: VUS_ROC
        baseline_same_rate: -1
        job_id: 如果同一个文件夹中有多个jobid， 这个jobid可以用来过滤文件
        """
        self._save_to_file = save_to_file
        self._header = header
        self._alpha = alpha
        self._post_see = post_see
        self._ana_metric = analysis_key_metric
        self._baseline_sample_rate = float(baseline_same_rate)
        self._out_home = out_home
        self._is_cache = is_cache
        self._job_id = job_id
        self._endswith = endswith

    @staticmethod
    def merge_baseline_and_founds(baseline_pd, target_pd):
        """
        融合 baseline 和 found 的指标. baseline_pd 和 target_pd 是五折交叉合并和的结果

        Parameters
        ----------
        baseline_pd :
        target_pd :

        Returns
        -------

        """
        baseline = ExcelMetricAnalysis.merge_kfold_df(baseline_pd)
        found = ExcelMetricAnalysis.merge_kfold_df(target_pd)
        _comm_headers = found.columns.intersection(baseline.columns)
        return pd.concat([baseline[_comm_headers], found[_comm_headers]])

    def merge_seeds(self):
        """
        求不同 seed 的平均值

        Returns
        -------

        """
        df = self.load_all_metrics_from_dirs()

        by_keys = [
            ExcelMetricsKeys.JOB_ID,
            ExcelMetricsKeys.EXP_NAME,
            ExcelMetricsKeys.ANOMALY_WINDOW_TYPE,
            ExcelMetricsKeys.WINDOW_SIZE,
            ExcelMetricsKeys.DATA_SAMPLE_METHOD,
            ExcelMetricsKeys.MODEL_NAME,
            ExcelMetricsKeys.DATASET_NAME,
            ExcelMetricsKeys.TEST_RATE,
            ExcelMetricsKeys.DATA_ID]
        output = []
        for item, _data in df.groupby(by=by_keys):

            # 计算不同seed 的平均值
            _g_data = _data.groupby(by=ExcelMetricsKeys.DATA_SAMPLE_RATE, as_index=False)
            _means = _g_data[PDUtil.get_number_columns(_data)].agg([np.mean, np.std])

            for i in range(len(by_keys)):
                _means[by_keys[i]] = item[i]

            _means = _means.reset_index()
            output.append(_means)
        out = pd.concat(output)

        job_id = pd.unique(out[ExcelMetricsKeys.JOB_ID])[0]
        exp_name = pd.unique(out[ExcelMetricsKeys.EXP_NAME])[0]
        if self._out_home.find(job_id) == -1:
            self._out_home = os.path.join(self._out_home, exp_name, job_id)

        PDUtil.save_to_excel(out, "02_merged_metric", home=self._out_home)
        return out

    def load_all_metrics_from_dirs(self, metric_dir):
        assert metric_dir is not None, "metric_dir cannot be None"
        _df = _load_all_metrics(metric_dir=metric_dir,
                                baseline_sr=self._baseline_sample_rate,
                                target_metric=self._ana_metric,
                                out_home=self._out_home,
                                is_save_all=self._is_cache,
                                endswith=self._endswith,
                                headers=self._header,
                                job_id=self._job_id)

        metrics_keys = ExcelMetricsKeys.get_performance_keys()

        try:
            for key in metrics_keys:
                _df[key] = _df[key].astype("float")
            _df[ExcelMetricsKeys.DATA_SAMPLE_RATE] = _df[ExcelMetricsKeys.DATA_SAMPLE_RATE].astype("float")
            _df[ExcelMetricsKeys.DATA_SAMPLE_RATE] = _df[ExcelMetricsKeys.DATA_SAMPLE_RATE] \
                .replace(-1, ExcelMetricAnalysis.BASE_SAMPLE_RATE)
            self._out_home = os.path.join(self._out_home, self._get_exp_name(_df))
            return _df
        except:
            traceback.print_exc()

    def get_best_founds_metrics(self):
        return self.output_analysis()

    def output_analysis(self):
        """
        找到每一个数据集对应的最佳的抽样比例

        ori, best_found,founds, =output_analysis()
        ori: 全量数据对应的指标，每个data_id 对应一条数据
        best_found： 通过策略找到的指标，每个data_id 对应一条数据
        found： 通过策略找到的指标，每个data_id 对应多条数据。用来计算寻找总时间的，每找一次就会多一条数据

        输出结构：
        Precision		Recall
        mean	std	mean	std

        0.0027	0.0047	0.0005	0.0009
        0.0000	0.0000	0.0000	0.0000
        0.5856	0.0154	0.0839	0.0071


        Returns
        -------

        """
        _data = self.merge_seeds()
        # The best sample rate, one record for every model and data id
        out_best_founds = []

        # The all records for founding the best sample rate. More than one record for model and data id
        out_founds = []

        # The baseline sample rate, one record for every model and data id
        out_ori = []
        for (_model_name, _data_set), _group_data in _data.groupby(
                by=[ExcelMetricsKeys.MODEL_NAME, ExcelMetricsKeys.DATASET_NAME],
                as_index=False):
            ori, found, best_found = \
                self.get_calculate_metrics_for_method_and_dataset(_group_data)
            assert len(ori) == len(best_found)
            self.output_results(_model_name, _data_set, ori, found, best_found)
            # aaa
            out_founds.append(found)
            out_best_founds.append(best_found)
            out_ori.append(ori)

        return pd.concat(out_ori), pd.concat(out_best_founds), pd.concat(out_founds)

    def get_calculate_metrics_for_method_and_dataset(self, _group_data):
        targets_ori = []
        targets_found = []
        bargets_best_fount = []
        for _data_id, metric in _group_data.groupby(
                by=ExcelMetricsKeys.DATA_ID,
                as_index=False):
            # 类型转换，不然会出现获取不到的情况
            metric[(ExcelMetricsKeys.DATA_SAMPLE_RATE, "")] = metric[
                (ExcelMetricsKeys.DATA_SAMPLE_RATE, "")].astype(
                "float")
            # 升序排序，为了从小找
            metric = metric.sort_values(by=[(ExcelMetricsKeys.DATA_SAMPLE_RATE, "")], ascending=True)

            # 全量数据时对应的训练指标
            base_metric = metric[metric[(ExcelMetricsKeys.DATA_SAMPLE_RATE, "")] == self._baseline_sample_rate]

            if base_metric.shape[0] == 0:
                warnings.warn(
                    f"Could not find baseline metrics. Please check your base sample rate. Current selected baseline sample rate is {self._baseline_sample_rate}, but it has {list(metric[(ExcelMetricsKeys.DATA_SAMPLE_RATE, '')])} in data")
            # 去除全量数据对应的训练指标
            metric = metric[metric[(ExcelMetricsKeys.DATA_SAMPLE_RATE, "")] != self._baseline_sample_rate]

            pr = metric[(self._ana_metric)].to_numpy()

            current_metrics = None

            for i in range(pr.shape[0] - self._post_see):
                if is_stopped_training_v1(pr[i:i + self._post_see + 1], post_see=self._post_see,
                                          alpha=self._alpha) is True:
                    current_metrics = metric.iloc[:i + self._post_see + 1, :]
                    break

            # 如果一直都没有找到截止条件，那么就选选全部数据，即一直训练到1.
            if current_metrics is None:
                current_metrics = metric

            best_found = current_metrics.sort_values(by=(self._ana_metric, EK.MEAN), ascending=False).iloc[
                         0:1]

            # _metrics = {
            #     'original_metric': base_metric[self._ana_metric].iloc[0].mean(),
            #     'found_metric': float(current_metrics[self._ana_metric].mean().max()),
            #     'found_srs': current_metrics[ExcelMetricsKeys.DATA_SAMPLE_RATE].mean().to_list(),
            #     'original_train_time(s)': float(base_metric[ExcelMetricsKeys.ELAPSED_TRAIN].iloc[0]),
            #     'real_train_time(s)': float(current_metrics[ExcelMetricsKeys.ELAPSED_TRAIN].sum()),
            #     'data_id': base_metric[ExcelMetricsKeys.DATA_ID].iloc[0],
            #     'found_max_sr': current_metrics[ExcelMetricsKeys.DATA_SAMPLE_RATE].max(),
            #     'found_best_sample_rate': best_found[ExcelMetricsKeys.DATA_SAMPLE_RATE]
            # }

            targets_ori.append(base_metric)
            targets_found.append(current_metrics)
            bargets_best_fount.append(best_found)
        return pd.concat(targets_ori), pd.concat(targets_found), pd.concat(bargets_best_fount)

    def output_results(self, model_name, data_set, ori, found, best_found):
        home = os.path.join(self._out_home, model_name, data_set)
        make_dirs(home)
        PDUtil.save_to_excel(ori, "04_model_baseline_metrics", home=home)
        PDUtil.save_to_excel(best_found, "04_model_found_best_metrics", home=home)
        ori_acc_mean = ori[(self._ana_metric, EK.MEAN)].mean()
        ori_acc_std = ori[(self._ana_metric, EK.STD)].mean()
        found_acc_mean = best_found[(self._ana_metric, EK.MEAN)].mean()
        found_acc_std = best_found[(self._ana_metric, EK.STD)].mean()
        PDUtil.save_to_dat(pd.DataFrame([
            ["ori.", ori_acc_mean, ori_acc_std],
            ["FastUTS", found_acc_mean, found_acc_std],

        ]), "perf", home=home)
        PDUtil.save_to_dat(pd.DataFrame([
            ["ori.",
             ori[ExcelMetricsKeys.ELAPSED_TRAIN][EK.MEAN].sum(),
             ori[ExcelMetricsKeys.ELAPSED_TRAIN][EK.MEAN].std()
             ],
            ["FastUTS.",
             found[ExcelMetricsKeys.ELAPSED_TRAIN].sum(),
             found[ExcelMetricsKeys.ELAPSED_TRAIN].std()
             ],

        ]), "train_time", home=home)
        # [ori_acc_mean, ori_acc_std]
        PDUtil.save_to_dat(pd.DataFrame({
            "data_id": best_found[ExcelMetricsKeys.DATA_ID].to_list(),
            "ori.": ori[(self._ana_metric, EK.MEAN)].to_list(),
            "FastUTS": best_found[(self._ana_metric, EK.MEAN)].to_list()
        }), "ori_perf", home=home)

    def _found_best_performance_metric(self, metrics):
        """
        在一组很多性能指标的dataframe中，找出最优的那个
        Parameters
        ----------
        metrics :

        Returns
        -------

        """
        data = metrics.sort_values(by=(self._ana_metric, EK.MEAN), ascending=False).iloc[:1]
        return data

    def _get_one_record_mean_and_std(self, metrics):
        ori_acc_mean = metrics[self._ana_metric].iloc[0]
        ori_acc_std = metrics[self._ana_metric][EK.STD].iloc[0]
        return ori_acc_mean, ori_acc_std

    def _found_auc_mean(self, found):
        accs = []
        for _data_id in found[ExcelMetricsKeys.DATA_ID].unique():
            _found_data_id_metric = found[found[ExcelMetricsKeys.DATA_ID] == _data_id]

            found_acc_mean, found_acc_std = self._get_one_record_mean_and_std(
                self._found_best_performance_metric(_found_data_id_metric))
            accs.append(found_acc_mean)

        return pd.Series(accs)

    def _found_all_best(self, found):
        accs = []
        for _data_id in found[ExcelMetricsKeys.DATA_ID].unique():
            _found_data_id_metric = found[found[ExcelMetricsKeys.DATA_ID] == _data_id]
            accs.append(self._found_best_performance_metric(_found_data_id_metric))
        return pd.Series(accs)

    def _get_ori_training_time(self, ori, found):
        ori_train_time_mean = ori[ExcelMetricsKeys.ELAPSED_TRAIN].mean()
        ori_train_time_std = ori[ExcelMetricsKeys.ELAPSED_TRAIN].std()
        found_train_time_mean = found[ExcelMetricsKeys.ELAPSED_TRAIN].mean()
        found_train_time_std = found[ExcelMetricsKeys.ELAPSED_TRAIN].std()
        return ori_train_time_mean, ori_train_time_std, found_train_time_mean, found_train_time_std

    def output_overall_results(self):
        ori, best, founds = self.get_best_founds_metrics()
        outputs = []
        for _model_name, _best in best.groupby(by=ExcelMetricsKeys.MODEL_NAME):
            _sr = f"{self._convert_to_percent(_best[(ExcelMetricsKeys.DATA_SAMPLE_RATE, '')].mean())}(±{self._convert_to_percent(_best[(ExcelMetricsKeys.DATA_SAMPLE_RATE, '')].std())})"

            _ori = ori[ori[ExcelMetricsKeys.MODEL_NAME] == _model_name]
            _founds = founds[founds[ExcelMetricsKeys.MODEL_NAME] == _model_name]

            _speedup = _ori[(EK.MEAN, ExcelMetricsKeys.ELAPSED_TRAIN)].mean() / \
                       _founds[
                           (EK.MEAN, ExcelMetricsKeys.ELAPSED_TRAIN)].mean()
            _ori_metrics = _ori[(self._ana_metric, EK.MEAN)].to_list()
            _best_metrics = _best[(self._ana_metric, EK.MEAN)].to_list()
            outputs.append({
                "Models": _model_name,
                "$s-r^*(\%)$": _sr,
                "p-val.": np.round(UtilHypo.welchs_test(_ori_metrics, _best_metrics), 6),
                "Ori. AUC": self._get_metrics_error(_ori, self._ana_metric),
                "$AUC^*$": self._get_metrics_error(_best, self._ana_metric),
                "Ori. Time": self._get_metrics_error(_ori, ExcelMetricsKeys.ELAPSED_TRAIN),
                "$Time^*$": self._get_metrics_error(_founds, ExcelMetricsKeys.ELAPSED_TRAIN),
                "Speedup": self._round(_speedup),
            })
        label = "tab:results_overview"
        caption = "The overall results of FastUTS."
        df = pd.DataFrame(outputs)
        latex_table = df.to_latex(index=False, label=label, caption=caption)
        path = os.path.join(UtilComm.get_runtime_directory(), 'results_01.tex')
        UtilSys.is_debug_mode() and log.info(f"The overall results of FastUTS is saved to {os.path.abspath(path)}")

        latex_table = latex_table.replace("{table", "{table*")
        latex_table = latex_table.replace("decision_tree", "decision\_tree")
        with open(path, 'w') as f:
            f.write(latex_table)

    def _round(self, params, keep=2):
        return np.round(params, keep)

    def _convert_to_percent(self, param):
        return np.round(param * 100, 2)

    def output_relation_sr_and_perf(self, data_sample_method="random"):
        """
        Figure 4: The relationship between the percentage of the training data and model precision.

        Returns
        -------

        """
        self._out_home = os.path.abspath(self._out_home)
        metric = self.merge_perf_folds(save_to_file=False)

        metric[ExcelMetricsKeys.DATA_SAMPLE_RATE] = metric[ExcelMetricsKeys.DATA_SAMPLE_RATE].astype(
            "float")
        for (_sample_method, _model), _metric in metric.groupby(
                by=[ExcelMetricsKeys.DATA_SAMPLE_METHOD, ExcelMetricsKeys.MODEL_NAME]):
            out_metrics = []
            for _sr, _sr_metric in _metric.groupby(by=ExcelMetricsKeys.DATA_SAMPLE_RATE):
                out_metrics.append({
                    ExcelMetricsKeys.DATA_SAMPLE_METHOD: _sample_method,
                    ExcelMetricsKeys.MODEL_NAME: _model,
                    ExcelMetricAnalysis.KEY_SR_START: self._round(1 if _sr == -1 else _sr, keep=4),
                    ExcelMetricsKeys.VUS_ROC: _sr_metric[ExcelMetricsKeys.VUS_ROC].mean(),
                    ExcelMetricsKeys.VUS_PR: _sr_metric[ExcelMetricsKeys.VUS_PR].mean(),
                    ExcelMetricsKeys.RPRECISION: _sr_metric[ExcelMetricsKeys.RPRECISION].mean(),
                    ExcelMetricsKeys.RECALL: _sr_metric[ExcelMetricsKeys.RECALL].mean(),
                    ExcelMetricsKeys.RF: _sr_metric[ExcelMetricsKeys.RF].mean(),
                    ExcelMetricsKeys.ELAPSED_TRAIN: _sr_metric[ExcelMetricsKeys.ELAPSED_TRAIN].mean(),
                    ExcelMetricsKeys.R_AUC_ROC: _sr_metric[ExcelMetricsKeys.R_AUC_ROC].mean(),
                })
            df = pd.DataFrame(out_metrics)
            df[ExcelMetricAnalysis.KEY_SR_START] = df[ExcelMetricAnalysis.KEY_SR_START].astype("float")
            df = df.sort_values(by=ExcelMetricAnalysis.KEY_SR_START, ascending=True)
            PDUtil.save_to_dat(df, f"{_sample_method}_{_model}", self._out_home)

        plot = get_gnuplot_command_for_sr_on_perf()
        FileUtil.save_txt_to_gnu(plot, "pic", self._out_home)
        CMD.exe_cmd("gnuplot pic.gnu", self._out_home)

    def output_relation_sr_and_perf_per_model_and_dataset(self):
        """
        单个模型在每个数据集上的单独效果。
        例如：
            decision_tree 在 IOPS 数据集上的效果
            decision_tree 在 MITB 数据集上的效果

        Returns
        -------

        """
        metric = self.merge_perf_folds(save_to_file=False)
        exp_name = self._get_exp_name(metric)
        home = os.path.join(os.path.abspath(self._out_home), exp_name, 'per_model_and_dataset')
        make_dirs(home)

        for (_sample_method, _model, _data_set), _metric in metric.groupby(
                by=[ExcelMetricsKeys.DATA_SAMPLE_METHOD, ExcelMetricsKeys.MODEL_NAME, ExcelMetricsKeys.DATASET_NAME]):
            out_metrics = []
            for _sr, _sr_metric in _metric.groupby(by=ExcelMetricsKeys.DATA_SAMPLE_RATE):
                out_metrics.append({
                    ExcelMetricsKeys.DATA_SAMPLE_METHOD: _sample_method,
                    ExcelMetricsKeys.MODEL_NAME: _model,
                    ExcelMetricAnalysis.KEY_SR_START: self._round(1 if _sr == -1 else _sr, keep=4),
                    ExcelMetricsKeys.VUS_ROC: _sr_metric[ExcelMetricsKeys.VUS_ROC].mean(),
                    ExcelMetricsKeys.VUS_PR: _sr_metric[ExcelMetricsKeys.VUS_PR].mean(),
                    ExcelMetricsKeys.RPRECISION: _sr_metric[ExcelMetricsKeys.RPRECISION].mean(),
                    ExcelMetricsKeys.RECALL: _sr_metric[ExcelMetricsKeys.RECALL].mean(),
                    ExcelMetricsKeys.RF: _sr_metric[ExcelMetricsKeys.RF].mean(),
                    ExcelMetricsKeys.ELAPSED_TRAIN: _sr_metric[ExcelMetricsKeys.ELAPSED_TRAIN].mean(),
                    ExcelMetricsKeys.R_AUC_ROC: _sr_metric[ExcelMetricsKeys.R_AUC_ROC].mean(),
                })
            df = pd.DataFrame(out_metrics)
            df[ExcelMetricAnalysis.KEY_SR_START] = df[ExcelMetricAnalysis.KEY_SR_START].astype("float")
            df = df.sort_values(by=ExcelMetricAnalysis.KEY_SR_START, ascending=True)
            PDUtil.save_to_dat(df, f"{_sample_method}_{_model}_{_data_set}", home)

        plot = get_gnuplot_command_for_sr_on_perf_for_every_model_and_dataset()
        FileUtil.save_txt_to_gnu(plot, "pic", home)
        CMD.exe_cmd("gnuplot pic.gnu", home)

    def _get_metrics_error(self, param, key, round=4):
        return f"{np.round(param[(key, EK.MEAN)].mean(), round)}(±{np.round(param[(key, EK.MEAN)].std(), round)})"

    def merge_seeds_v2(self, save_to_file=True):
        df = self.load_all_metrics_from_dirs()

        by_keys = [
            ExcelMetricsKeys.JOB_ID,
            ExcelMetricsKeys.EXP_NAME,
            ExcelMetricsKeys.ANOMALY_WINDOW_TYPE,
            ExcelMetricsKeys.WINDOW_SIZE,
            ExcelMetricsKeys.DATA_SAMPLE_METHOD,
            ExcelMetricsKeys.MODEL_NAME,
            ExcelMetricsKeys.DATASET_NAME,
            ExcelMetricsKeys.TEST_RATE,
            ExcelMetricsKeys.DATA_ID]
        output = []
        for _group_keys_item, _data in df.groupby(by=by_keys):
            # 计算不同seed 的平均值
            for _sr, _sr_data in _data.groupby(by=ExcelMetricsKeys.DATA_SAMPLE_RATE):
                _out_mean = _sr_data[ExcelMetricsKeys.get_performance_keys()].mean()
                _info = _sr_data[ExcelMetricsKeys.get_none_performance_keys()].iloc[0]
                _out_data = pd.concat([_out_mean, _info])
                output.append(_out_data.to_dict())
        out = pd.DataFrame(output)

        job_id = pd.unique(out[(ExcelMetricsKeys.JOB_ID)])[0]
        exp_name = pd.unique(out[(ExcelMetricsKeys.EXP_NAME)])[0]
        if self._out_home.find(job_id) == -1:
            self._out_home = os.path.join(self._out_home, exp_name, job_id)
        if save_to_file is True:
            PDUtil.save_to_excel(out, "02_merged_mean_metric_v2", home=self._out_home)
        return out

    @DeprecationWarning
    def merge_perf_baseline(self, metric_dir=None, df=None):
        if df is None:
            df = self.load_all_metrics_from_dirs(metric_dir)

        output = []
        for _group_keys_item, _data in df.groupby(by=self.GROUP_BY_KEYS):
            # 计算不同seed 的平均值
            _mean = _data[ExcelMetricsKeys.get_performance_keys()].mean()
            _std = _data[ExcelMetricsKeys.get_performance_keys()].std()
            _median = _data[ExcelMetricsKeys.get_performance_keys()].median()
            _other_info = _data[ExcelMetricsKeys.get_none_performance_keys(_data.columns.tolist())]

            _columns_1 = ExcelMetricsKeys.get_performance_keys() * 3
            len_perf_keys = len(ExcelMetricsKeys.get_performance_keys())
            _columns_2 = [EK.MEAN] * len_perf_keys + [EK.STD] * len_perf_keys + ['median'] * len_perf_keys
            _columns_values = _mean.to_list() + _std.to_list() + _median.to_list()

            temp_res = pd.DataFrame([_columns_values], columns=list(zip(_columns_1, _columns_2)))

            for _key in _other_info.columns.tolist():
                _item_value = _other_info[_key].unique()
                if _item_value.shape[0] == 1:
                    _insert_val = _item_value[0]
                else:
                    _insert_val = ";".join(_item_value.astype("str"))
                temp_res.insert(0, _key, _insert_val)
            output.append(temp_res)
        out = pd.concat(output)
        if self._save_to_file:
            PDUtil.save_to_excel(df, f"{self._get_exp_name(df)}_01_baseline_metrics", home=self._out_home)
            PDUtil.save_to_excel(out, f"{self._get_exp_name(df)}_02_baseline_merged_mean_std_metric",
                                 home=self._out_home)

        # df: original data
        # out: merged data. Merges the different k fold for each model
        return df, out

    def merge_kfold_data(self, metric_dir=None, df=None):
        """
        ema = ExcelMetricAnalysis(endswith="_metrics.csv", save_to_file=False)
        original,merged_kfold =  ema.merge_kfold_data(metric_dir)

        original: 没有合并kfold 的原始数据
        merged_kfold: 合并了kfold 的数据, 统计了每个fold 的均值和方差


        Parameters
        ----------
        metric_dir :
        df :

        Returns
        -------

        """
        if df is None:
            df = self.load_all_metrics_from_dirs(metric_dir)

        for _model in PaperFormat.EXCLUDE_METHODS:
            df = df[df[EK.MODEL_NAME] != 'decision_tree']
        output = []
        for _group_keys_item, _data in df.groupby(by=self.GROUP_BY_KEYS):
            # 计算多折交叉验证的均值
            _mean = _data[ExcelMetricsKeys.get_performance_keys()].mean()
            _std = _data[ExcelMetricsKeys.get_performance_keys()].std()
            _other_info = _data[ExcelMetricsKeys.get_none_performance_keys(_data.columns.tolist())]

            _columns_1 = ExcelMetricsKeys.get_performance_keys() * 3
            len_perf_keys = len(ExcelMetricsKeys.get_performance_keys())
            _columns_2 = [EK.MEAN] * len_perf_keys + [EK.STD] * len_perf_keys
            _columns_values = _mean.to_list() + _std.to_list()

            temp_res = pd.DataFrame([_columns_values], columns=list(zip(_columns_1, _columns_2)))

            for _key in _other_info.columns.tolist():
                _item_value = _other_info[_key].unique()
                if _item_value.shape[0] == 1:
                    _insert_val = _item_value[0]
                else:
                    _insert_val = ";".join(_item_value.astype("str"))
                temp_res.insert(0, _key, _insert_val)
            output.append(temp_res)

        # df: original data, not merag kfold
        # out: merged data. Merges the different k fold for each model

        avg_perf = pd.concat(output)

        # 计算时间*5, 因为上面计算时间只是一折交叉验证的结果
        avg_perf[(EK.MEAN, EK.ELAPSED_TRAIN)] = avg_perf[(EK.MEAN, EK.ELAPSED_TRAIN)] * 5
        avg_perf[(EK.MEAN, EK.DATA_PROCESSING_TIME)] = avg_perf[(EK.MEAN, EK.DATA_PROCESSING_TIME)] * 5

        return df, avg_perf

    @staticmethod
    def merge_kfold_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        merge_metric = ExcelMetricAnalysis.merge_kfold_df(df)

        merge_metric: 合并了kfold 的数据, 统计了每个fold 的均值和方差


        Parameters
        ----------
        df :

        Returns
        -------

        """

        # pd.crosstab(exp_index=ExcelMetricAnalysis.GROUP_BY_KEYS, columns=[EK.VUS_ROC, EK.VUS_PR, EK.RF], values=[''])
        # remove debug data
        df = df[df[EK.DEBUG] == 0]
        # remove NaN data
        df = df[pd.notnull(df[EK.VUS_ROC])]

        def _merge_kfold_data(_data: pd.DataFrame):
            # computes mean, std, median
            temp_res = pd.pivot_table(_data,
                                      index=ExcelMetricAnalysis.GROUP_BY_KEYS,
                                      values=ExcelMetricsKeys.get_performance_keys(),
                                      aggfunc=[np.mean, np.std, np.median])
            #####
            # append more info
            _other_info = _data[ExcelMetricsKeys.get_none_performance_keys(_data.columns.tolist())].iloc[0:1]
            # remove duplicated data
            # _other_info = _other_info.drop(ExcelMetricAnalysis.GROUP_BY_KEYS, axis=1)
            # append info
            temp_res[_other_info.columns] = _other_info.values.tolist()

            # add fold indexs to list
            temp_res[EK.FOLD_INDEX] = str(_data[EK.FOLD_INDEX].to_list())

            return temp_res

        from pandarallel import pandarallel

        # 初始化 Pandas Parallel
        pandarallel.initialize()
        # avg_perf = df.groupby(by=ExcelMetricAnalysis.GROUP_BY_KEYS).apply(_merge_kfold_data)
        avg_perf = df.groupby(by=ExcelMetricAnalysis.GROUP_BY_KEYS).parallel_apply(_merge_kfold_data)
        # output = []
        # for _group_keys_item, _data in df.groupby(by=ExcelMetricAnalysis.GROUP_BY_KEYS):
        #     # _data 包含同一个模型和数据的多折交叉验证结果
        #     temp_res = pd.pivot_table(_data,
        #                               exp_index=ExcelMetricAnalysis.GROUP_BY_KEYS,
        #                               values=ExcelMetricsKeys.get_performance_keys(),
        #                               aggfunc=[np.mean, np.std, np.median])
        #     _other_info = _data[ExcelMetricsKeys.get_none_performance_keys(_data.columns.tolist())].iloc[0:1]
        #     _other_info = _other_info.drop(ExcelMetricAnalysis.GROUP_BY_KEYS, axis=1)
        #     temp_res[_other_info.columns] = _other_info.values.tolist()
        #     temp_res[EK.FOLD_INDEX] = str(_data[EK.FOLD_INDEX].to_list())
        #     output.append(temp_res)
        #
        #
        # avg_perf = pd.concat(output)

        # 计算时间*5, 上面的时间是每一折的平时时间, 总时间要乘以5
        # 时间均值
        avg_perf[(EK.MEAN, EK.ELAPSED_TRAIN)] = avg_perf[(EK.MEAN, EK.ELAPSED_TRAIN)] * 5
        avg_perf[(EK.MEDIAN, EK.ELAPSED_TRAIN)] = avg_perf[(EK.MEDIAN, EK.ELAPSED_TRAIN)] * 5

        # 时间中位数
        avg_perf[(EK.MEAN, EK.DATA_PROCESSING_TIME)] = avg_perf[(EK.MEAN, EK.DATA_PROCESSING_TIME)] * 5
        avg_perf[(EK.MEDIAN, EK.DATA_PROCESSING_TIME)] = avg_perf[(EK.MEDIAN, EK.DATA_PROCESSING_TIME)] * 5

        return avg_perf

    def merge_perf_found(self, df=None, metric_dir=None):
        if df is None:
            df = self.load_all_metrics_from_dirs(metric_dir)

        output = []
        for _group_keys_item, _data in df.groupby(by=self.GROUP_BY_KEYS):
            # 计算不同seed 的平均值
            _mean = _data[ExcelMetricsKeys.get_performance_keys()].mean()
            _std = _data[ExcelMetricsKeys.get_performance_keys()].std()
            _median = _data[ExcelMetricsKeys.get_performance_keys()].median()
            _other_info = _data[ExcelMetricsKeys.get_none_performance_keys(_data.columns.tolist())]

            _columns_1 = ExcelMetricsKeys.get_performance_keys() * 3
            len_perf_keys = len(ExcelMetricsKeys.get_performance_keys())
            _columns_2 = [EK.MEAN] * len_perf_keys + [EK.STD] * len_perf_keys + ['median'] * len_perf_keys
            _columns_values = _mean.to_list() + _std.to_list() + _median.to_list()

            temp_res = pd.DataFrame([_columns_values], columns=list(zip(_columns_1, _columns_2)))

            for _key in _other_info.columns.tolist():
                _item_value = _other_info[_key].unique()
                if _item_value.shape[0] == 1:
                    _insert_val = _item_value[0]
                else:
                    _insert_val = ";".join(_item_value.astype("str"))
                temp_res.insert(0, _key, _insert_val)
            output.append(temp_res)
        out = pd.concat(output)
        if self._save_to_file:
            PDUtil.save_to_excel(df, f"{self._get_exp_name(df)}_01_found_original_metrics", home=self._out_home)
            PDUtil.save_to_excel(out, f"{self._get_exp_name(df)}_01_found_merged_mean_std_found_metrics",
                                 home=self._out_home)
        # df: original data
        # out: merged data. Merges the different k fold for each model
        return df, out

    def output_sr_in_model_and_dataset(self):
        """
        找出每个模型在每个数据集上的性能（sr*, Ori. AUC, AUC*, p-value
        Returns
        -------

        """
        data = self.merge_seeds_v2(False)
        ori, best_found, founds = self.found_metrics(data)

        out_metrics = []
        for (_data_sample_method, _model_name, _data_set), _g_metric in best_found.groupby(
                [ExcelMetricsKeys.DATA_SAMPLE_METHOD, ExcelMetricsKeys.MODEL_NAME, ExcelMetricsKeys.DATASET_NAME]):
            _ori = ori[ori[ExcelMetricsKeys.MODEL_NAME] == _model_name][
                ori[ExcelMetricsKeys.DATA_SAMPLE_METHOD] == _data_sample_method][
                ori[ExcelMetricsKeys.DATASET_NAME] == _data_set]
            out_metrics.append({
                ExcelMetricsKeys.DATA_SAMPLE_METHOD: _data_sample_method,
                ExcelMetricsKeys.MODEL_NAME: _model_name,
                ExcelMetricsKeys.DATASET_NAME: _data_set,
                "best_sr(\%)": self._round(_g_metric[ExcelMetricsKeys.DATA_SAMPLE_RATE].mean() * 100),
                "Ori. AUC": self._round(_ori[target_perf].mean()),
                "AUC*": self._round(_g_metric[target_perf].mean()),
                "p-val.": np.round(UtilHypo.welchs_test(
                    _ori[target_perf],
                    _g_metric[target_perf]), 6)
            })
        data = pd.DataFrame(out_metrics)
        PDUtil.save_to_excel(data, "output_sr_in_model_and_dataset", self._out_home)
        _convert_to_latex(data)

    def found_metrics(self, data):
        targets_ori = []
        targets_found = []
        bargets_best_fount = []
        for (_data_sample_method, _model_name, _data_id), _g_metric in data.groupby(
                by=[ExcelMetricsKeys.DATA_SAMPLE_METHOD, ExcelMetricsKeys.MODEL_NAME, ExcelMetricsKeys.DATA_ID]):
            # 类型转换，不然会出现获取不到的情况
            _g_metric[ExcelMetricsKeys.DATA_SAMPLE_RATE] = _g_metric[ExcelMetricsKeys.DATA_SAMPLE_RATE].astype(
                "float")
            # 升序排序，为了从小找
            _g_metric = _g_metric.sort_values(by=ExcelMetricsKeys.DATA_SAMPLE_RATE, ascending=True)

            # 全量数据时对应的训练指标
            _baseline_metric = _g_metric[_g_metric[ExcelMetricsKeys.DATA_SAMPLE_RATE] == self._baseline_sample_rate]

            if _baseline_metric.shape[0] == 0:
                warnings.warn(
                    f"Could not find baseline metrics. Please check your base sample rate. Current selected baseline sample rate is {self._baseline_sample_rate}, but it has {list(_g_metric[(ExcelMetricsKeys.DATA_SAMPLE_RATE, '')])} in data")
            # 去除全量数据对应的训练指标
            _ana_metric = _g_metric[_g_metric[ExcelMetricsKeys.DATA_SAMPLE_RATE] != self._baseline_sample_rate]

            _performances = _ana_metric[self._ana_metric].to_numpy()

            _current_metrics = None

            for i in range(_performances.shape[0] - self._post_see):
                if is_stopped_training_v1(_performances[i:i + self._post_see + 1], post_see=self._post_see) is True:
                    _current_metrics = _ana_metric.iloc[:i + self._post_see + 1, :]
                    break

            # 如果一直都没有找到截止条件，那么就选选全部数据，即一直训练到1.
            if _current_metrics is None:
                _current_metrics = _g_metric

            _best_found = _current_metrics.sort_values(by=self._ana_metric, ascending=False).iloc[
                          0:1]

            targets_ori.append(_baseline_metric)
            targets_found.append(_current_metrics)
            bargets_best_fount.append(_best_found)
        return pd.concat(targets_ori), pd.concat(bargets_best_fount), pd.concat(targets_found)

    def _get_exp_name(self, metric):
        return metric[ExcelMetricsKeys.EXP_NAME].iloc[0]

    @staticmethod
    def get_exp_name(metric):
        return metric[ExcelMetricsKeys.EXP_NAME].iloc[0]


def _convert_to_latex(df, label="tab:training_data_reduction_for_each_dataset_and_alg",
                      caption="The percentage of training data required by each algorithm to train a model, without losing model accuracy compared the full training set. ",
                      columns=[ExcelMetricsKeys.MODEL_NAME, ExcelMetricsKeys.DATASET_NAME, "best_sr(\%)", "Ori. AUC",
                               "AUC*", "p-val."]):
    latex_table = df.to_latex(label=label, caption=caption, columns=columns, index=False)
    path = os.path.join(UtilComm.get_runtime_directory(), 'results_02.tex')
    UtilSys.is_debug_mode() and log.info(f"The overall results of FastUTS is saved to {os.path.abspath(path)}")

    # latex_table = latex_table.replace("{table", "{table*")
    latex_table = latex_table.replace("decision_tree", "decision\_tree")
    latex_table = ExcelMetricsKeys.post_process_latex(latex_table)
    with open(path, 'w') as f:
        f.write(latex_table)


@cache_
def _load_metrics(load_dir):
    ema = ExcelMetricAnalysis(endswith="_metrics.csv", save_to_file=False)
    _, metrics = ema.merge_kfold_data(metric_dir=load_dir)
    return metrics


def _load_baseline_and_found(baseline_dir, found_dir):
    baseline = _load_metrics(baseline_dir)
    found = _load_metrics(found_dir)
    _comm_headers = found.columns.intersection(baseline.columns)
    return baseline[_comm_headers], found[_comm_headers]


def _load_baseline_and_found_find(baseline_dir, found_dir):
    baseline = _load_metrics(baseline_dir)
    found = _load_metrics(found_dir)
    _comm_headers = found.columns.intersection(baseline.columns)
    return baseline[_comm_headers], found[_comm_headers]


@DeprecationWarning
def calculate_speed_up(_data):
    return (_data[EK.ORI_TRAIN_TIME_MEAN].sum()) \
        / (_data[EK.FAST_TRAIN_TIME_MEAN].sum() + _data[EK.FAST_DATA_PROCESSING_TIME_MEAN].sum())


def calculate_save_time(_data):
    ori_time = _data[EK.ORI_TRAIN_TIME_MEAN].sum()
    fast_time = _data[EK.FAST_TRAIN_TIME_MEAN].sum() + _data[EK.FAST_DATA_PROCESSING_TIME_MEAN].sum()

    return ori_time - fast_time


@DeprecationWarning
# @cache_
def load_overall_results_from_different_opt_target():
    """
    返回不同优化目标下的FastUTS 的性能. 最佳配置: 随机抽样+间隔256+三种目标指标

    v2001_03_fastuts_sup1_rf_0.001_random.bz2
    v2001_03_fastuts_sup1_vus_roc_0.001_random.bz2
    v2001_03_fastuts_sup1_vus_pr_0.001_random.bz2

    Returns
    -------

    """

    _out_arr = []
    for __target_perf in ResultsConf.TARGET_PERFS:
        rt = ResultsConf(version=JobConfV1.EXP_VERSION,
                         target_perf=__target_perf)
        perf = ResTools.parse_model_perf_over_uts_scale(baseline=rt.get_baseline(),
                                                        target=rt.get_best_conf(),
                                                        target_perf=__target_perf)
        _out_arr.append(perf)
    return pd.concat(_out_arr)


@cache_
def load_merge_perf_default():
    """
    合并了 VUS PR 和 VUS ROC, ML 和 DL
    Returns
    -------

    """
    JobConfV1.EXP_VERSION = "v1923"
    _out_arr = []
    for __target_perf in ResultsConf.TARGET_PERFS:
        rt = ResultsConf(version=JobConfV1.EXP_VERSION,
                         target_perf=__target_perf)

        perf = ResTools.parse_model_perf_over_uts_scale(baseline=rt.get_baseline(),
                                                        target=rt.get_best_conf(),
                                                        target_perf=__target_perf)
        _out_arr.append(perf)
    return pd.concat(_out_arr)


def load_merge_perf_diff_sample_method():
    """
    合并了 VUS PR 和 VUS ROC, ML 和 DL
    Returns
    -------

    """
    _out_arr = []
    for __target_perf in ResultsConf.TARGET_PERFS:
        rt = ResultsConf(version=JobConfV1.EXP_VERSION,
                         target_perf=EK.VUS_ROC)

        target_dirs = rt.get_diff_sample_method()
        for _target_dir in target_dirs:
            perf = ResTools.parse_model_perf_over_uts_scale(baseline=rt.get_baseline(),
                                                            target=_target_dir,
                                                            target_perf=__target_perf)
            # rc = ResConfV1(
            #     rt.get_baseline(),
            #     _target_dir,
            #     opt_target=__target_perf,
            # )
            # perf = parse_model_perf_over_uts_scale(rc)
            perf[EK.DATA_SAMPLE_METHOD] = ResTools.get_target_sample_method(_target_dir)

            _out_arr.append(perf)
    return pd.concat(_out_arr)


from pylibs.utils.util_joblib import cache_


@cache_
def load_merge_perf_diff_stop_alpha() -> pd.DataFrame:
    """
    合并了 VUS PR 和 VUS ROC, ML 和 DL
    Returns
    -------

    'v2001_03_fastuts_sup1_VUS_ROC_0.001_random.bz2'
    'v2001_03_fastuts_sup1_VUS_ROC_0.01_random.bz2'
    'v2001_03_fastuts_sup1_VUS_ROC_0.1_random.bz2'
    'v2001_03_fastuts_sup1_VUS_ROC_0.5_random.bz2'

    """
    _out_arr = []
    for target_perf in JobConfV1.OPT_TARGETS:
        rts = ResultsConf(version=JobConfV1.EXP_VERSION,
                          target_perf=target_perf)
        target_dirs = rts.get_diff_stop_alpha()

        for __target_dir in target_dirs:
            perf = ResTools.parse_model_perf_over_uts_scale(baseline=rts.get_baseline(),
                                                            target=__target_dir,
                                                            target_perf=target_perf)

            perf[EK.STOP_ALPHA] = ResTools.get_stop_alpha(__target_dir)
            # ss
            _out_arr.append(perf)
    return pd.concat(_out_arr)


def get_model_dataset_view(c, target_perf):
    perf_model_over_uts = parse_model_perf_over_uts(c, target_perf)

    # Level2: 模型 over 数据集
    out_metrics2 = []
    for (_model_name, _dataset_name), _data in perf_model_over_uts.groupby(by=[EK.MODEL_NAME, EK.DATASET_NAME]):
        _ori_perf, _found_perf, _p_value = get_ori_found_p(_data)

        _metric = {
            EK.MODEL_NAME: _model_name,
            "model_typ": model_type,
            EK.DATASET_NAME: _dataset_name,
            "p value": ER.format_float(_p_value),
            "best sr": ER.format_perf_mean_and_std(_data[EK.FAST_BEST_SR].mean(),
                                                   _data[EK.FAST_BEST_SR].std()),
            f"ori. acc": ER.format_perf_mean_and_std(_data[EK.ORI_PERF_MEAN].mean(),
                                                     _data[EK.ORI_PERF_STD].mean()),
            f"fast. acc": ER.format_perf_mean_and_std(_data[EK.FAST_PERF_MEAN].mean(),
                                                      _data[EK.FAST_PERF_STD].mean()),
            "ori. train. time": ER.format_perf_mean_and_std(_data[EK.ORI_TRAIN_TIME_MEAN].mean(),
                                                            _data[EK.ORI_TRAIN_TIME_STD].mean()),
            "fast. train. time": ER.format_perf_mean_and_std(_data[EK.FAST_TRAIN_TIME_MEAN].mean(),
                                                             _data[EK.FAST_TRAIN_TIME_STD].mean()),
            "data proc. time": ER.format_perf_mean_and_std(_data[EK.FAST_DATA_PROCESSING_TIME_MEAN].mean(),
                                                           _data[EK.FAST_DATA_PROCESSING_TIME_STD].mean()),
            "speedup": ER.format_float(_data[EK.ORI_TRAIN_TIME_MEAN].mean() / (
                    _data[EK.FAST_DATA_PROCESSING_TIME_MEAN].mean() + _data[EK.FAST_TRAIN_TIME_MEAN].mean())),
            "sort_key": ER.format_float(_data[EK.ORI_PERF_MEAN].mean())
        }

        out_metrics2.append(_metric)

    _model_dataset2 = pd.DataFrame(out_metrics2)
    # PDUtil.save_to_excel(_model_dataset2, f"{c.get_file_name()}_l2_model_dataset_metrics", append_entry=True)
    # Level2: 模型 over 数据集
    return _model_dataset2


def get_ori_found_p(_data):
    """
    获取原始性能, 找到的性能, 原始性能和找到性能的均值
    Parameters
    ----------
    _data :

    Returns
    -------

    """
    _ori_perf = _data[EK.ORI_PERF_MEAN].astype("float").values
    _found_perf = _data[EK.FAST_PERF_MEAN].astype("float").values

    _p_value = get_hypo_test_p(_ori_perf, _found_perf)
    return _ori_perf, _found_perf, _p_value


def get_model_view(perf_model_over_uts):
    # level_3: 模型视角: 模型在所有实验室上的平均值,  模型 over all
    model_view_arr = []
    for _model_name, _data in perf_model_over_uts.groupby(by=EK.MODEL_NAME):
        _metric = {
            EK.MODEL_NAME: _model_name,
            "best sr": ER.format_perf_mean_and_std(_data[EK.FAST_BEST_SR].mean(),
                                                   _data[EK.FAST_BEST_SR].std(), scale=100),
            "p value": np.round(_p_value, 4),
            f"ori. perf.": ER.format_perf_mean_and_std(_data[EK.ORI_PERF_MEAN].mean(),
                                                       _data[EK.ORI_PERF_STD].mean(), 100),
            f"fast. perf.": ER.format_perf_mean_and_std(_data[EK.FAST_PERF_MEAN].mean(),
                                                        _data[EK.FAST_PERF_STD].mean(), 100),
            "ori. time (min)": ER.format_perf_mean_and_std(_data[EK.ORI_TRAIN_TIME_MEAN].sum(),
                                                           _data[EK.ORI_TRAIN_TIME_STD].sum(), 1 / 60, decimal=1),
            "fast. time (min)": ER.format_perf_mean_and_std(_data[EK.FAST_TRAIN_TIME_MEAN].sum(),
                                                            _data[EK.FAST_TRAIN_TIME_STD].mean(), 1 / 60, decimal=1),
            "proc. time (min)": ER.format_perf_mean_and_std(_data[EK.FAST_DATA_PROCESSING_TIME_MEAN].sum(),
                                                            _data[EK.FAST_DATA_PROCESSING_TIME_STD].mean(),
                                                            1 / 60, decimal=1),
            "speedup": ER.format_float(_data[EK.ORI_TRAIN_TIME_MEAN].mean() / (
                    _data[EK.FAST_DATA_PROCESSING_TIME_MEAN].mean() + _data[EK.FAST_TRAIN_TIME_MEAN].mean())),
            "sort_key": ER.format_float(_data[EK.ORI_PERF_MEAN].mean())
        }
        model_view_arr.append(_metric)

    _df = pd.DataFrame(model_view_arr)

    return _df


@typechecked
def get_hypo_test_p(baseline: list, found: list):
    """
    原假设: Fast.(found) 找到的性能显著低于原始性能(baseline)

    备选假设: 原始性能的均值高于Fast. 的性能

    方差不知道,所以不假设等方差

    Parameters
    ----------
    baseline :
    found :

    Returns
    -------

    """
    # 第一组样本
    # baseline = np.array(
    #     [0.953913107, 0.782083942, 0.774895328, 0.756125958, 0.773328474, 0.72601737, 0.780610077, 0.826074917,
    #      0.793391605,
    #      0.957709402, 0.975320827, 0.925120113, 0.921922346, 0.960023155, 0.885068632])

    # 第二组样本
    # found = np.array(
    #     [0.949470358, 0.785949623, 0.787545364, 0.751927228, 0.768896488, 0.719820047, 0.788408813, 0.819025978,
    #      0.784500375, 0.96086185, 0.973773848, 0.93475086, 0.916693948, 0.957867802, 0.88381055])

    # found = np.array(
    #     [0.949470358, 0.785949623, 0.787545364, 0.751927228, 0.768896488, 0.719820047, 0.788408813, 0.819025978,
    #      0.784500375, 0.96086185, 0.973773848, 0.93475086, 0.916693948, 0.957867802, 0.88381055]) * 0.01

    # 执行Welch's t-test
    import scipy
    t_stat, p_value = scipy.stats.stats.ttest_ind(baseline, found, equal_var=False, alternative="greater")
    return p_value


# to call ResTools.split_baseline_and_founds
@DeprecationWarning
def split_baseline_and_founds(item, base_sr, perf_key):
    _item = item.sort_values(ExcelMetricsKeys.DATA_SAMPLE_RATE, ascending=False)
    if _item.shape[0] > 1:
        assert _item[(ExcelMetricsKeys.MODEL_NAME)].unique().shape[0] == 1
        _baseline: pd.Series = _item[_item[EK.DATA_SAMPLE_RATE] == base_sr].iloc[0]
        _founds = _item[_item[EK.DATA_SAMPLE_RATE] != base_sr]
        _best_found = _founds.iloc[_founds[(EK.MEAN, perf_key)].argmax()]
        return _baseline, _best_found, _founds

    return None, None, None


def plot_fastuts_every_model_and_uts(
        baseline_dir="/Users/sunwu/SW-Research/sw-research-code/A01_paper_exp/data/baseline",
        found_dir="/Users/sunwu/SW-Research/sw-research-code/A01_paper_exp/data/found"
):
    """
    把每个模型和每个单变量时间序列对应的性能曲线画出来

    结果:https://uniplore.feishu.cn/wiki/BsKKwfTHHiM7kikGQUNc7dFynvc

    Returns
    -------

    """
    # 统计模型基线性能

    baseline, found = _load_baseline_and_found(baseline_dir, found_dir)

    exp_name = ExcelMetricAnalysis.get_exp_name(found)
    all = pd.concat([baseline, found])
    _group_keys = ExcelMetricAnalysis.GROUP_BY_KEYS.copy()
    _group_keys.remove(ExcelMetricsKeys.DATA_SAMPLE_RATE)
    g = gnuplot.Gnuplot(log=True)
    for _keys, _items in all.groupby(by=_group_keys):
        if _items.shape[0] < 4:
            continue
        _data = _items[[(EK.MEAN, ExcelMetricsKeys.TRAIN_LEN),
                        (EK.MEAN, target_perf),
                        (ExcelMetricsKeys.AUC_PR, EK.MEAN),
                        (ExcelMetricsKeys.RF, EK.MEAN),
                        (ExcelMetricsKeys.PRECISION, EK.MEAN),
                        (EK.MEAN, ExcelMetricsKeys.ELAPSED_TRAIN)
                        ]]
        # 1: sample rate
        _plot_data = _data.sort_values(by=(EK.MEAN, ExcelMetricsKeys.TRAIN_LEN), ascending=True)

        file_name = os.path.join(UtilComm.get_runtime_directory(), "plot_images",
                                 exp_name, "-".join(_keys) + ".pdf")

        make_dirs(os.path.dirname(file_name))
        title = f"Model = {_keys[-3]}   Dataset={_keys[-2]}    DATA ID= {_keys[-1]}"
        xlabel = _plot_data[(EK.MEAN, ExcelMetricsKeys.TRAIN_LEN)].tolist()
        xtics = _generate_xlabel(xlabel)
        g.cmd(f"set xtics {xtics}")
        g.cmd("set xtics rotate by -20 ")
        g.cmd('set rmargin 10')
        g.cmd('set bmargin 5')
        # set key spacing 30
        g.cmd("""
    set key horizontal
    set key bottom center  
    set ytics nomirror
    set y2tics  
                """)
        g.plot_data(_plot_data,
                    'using 0:3  axis x1y1 with lp lt 30 title "AUC ROC"',
                    'using 0:4  axis x1y2 with lp lt 31 title "AUC PR"',
                    'using 0:6 axis x1y2  with lp lt 33 title "Precision"',
                    grid="",
                    terminal='pdfcairo font "arial,10"  ',
                    output=f'"{file_name}"',
                    title=f'"{title}"',
                    xlabel='"Train Windows (Data Ratio %)"',
                    ylabel='"Model Performance" noenhanced',
                    )

        log.info(f"Saving {os.path.abspath(file_name)}")


mem = JLUtil.get_memory()


@mem.cache
def load_excel(filename):
    return pd.read_excel(filename)


def _is_perf_decrease(_baseline, _best_found, key=ExcelMetricsKeys.VUS_ROC, alpha=0.05):
    assert _baseline.shape[0] == _best_found.shape[0]
    return _best_found[(key, EK.MEAN)] - alpha - _baseline[(key, EK.MEAN)] >= 0


def static_fast_uts_info(baseline_dir, found_dir):
    df = _get_post_process_metrics(baseline_dir, found_dir)
    uv = UTSViewGnuplot()
    # uv.plot_ori_and_opt_perf(df["ori_auc"].to_numpy(), df['fastuts_auc'].to_numpy())
    uv.plot_ori_and_opt_perf_each_model(df)

    # PDUtil.save_to_excel(df, "my_results")


def _get_model_dataset(baseline_dir, found_dir):
    # JLUtil.clear_all_calche()
    baseline, found = _load_baseline_and_found(baseline_dir, found_dir)
    PDUtil.save_to_excel(baseline, "baseline_deep")
    _baseline_sr = baseline[ExcelMetricsKeys.DATA_SAMPLE_RATE].unique()[0]
    _all = pd.concat([baseline, found])

    for (_model, _dataset), _item in _all.groupby(by=[ExcelMetricsKeys.MODEL_NAME, ExcelMetricsKeys.DATASET_NAME]):
        pass


def _get_baseline_and_found_iter(baseline_dir, found_dir, g_keys=[ExcelMetricsKeys.MODEL_NAME],
                                 perf_key=ExcelMetricsKeys.VUS_ROC):
    baseline, found = _load_baseline_and_found(baseline_dir, found_dir)
    _baseline_sr = baseline[ExcelMetricsKeys.DATA_SAMPLE_RATE].unique()[0]
    _all = pd.concat([baseline, found])
    for _keys, _item in _all.groupby(by=g_keys):
        _item[(ExcelMetricAnalysis.VUS_ROC, EK.MEAN)] = _item[(ExcelMetricAnalysis.VUS_ROC, EK.MEAN)].astype("float")
        _item = _item.sort_values(ExcelMetricsKeys.DATA_SAMPLE_RATE, ascending=False)
        if _item.shape[0] > 3:
            assert _item[(ExcelMetricsKeys.MODEL_NAME)].unique().shape[0] == 1
            _baseline = _item.iloc[0]
            if _baseline[ExcelMetricsKeys.DATA_SAMPLE_RATE] == _baseline_sr:
                _founds = _item.iloc[1:]
                _best_found = _founds.loc[_founds[(EK.MEAN, perf_key)].idmax()]
                yield _keys, _baseline, _best_found, _founds
            else:
                log.error(f"❌❌❌ Skip for not found baseline {_keys} ")
                continue


@DeprecationWarning
def _get_post_process_metrics(baseline_dir, found_dir, _baseline_sr=None, target_perf="VUS_PR"):
    """
    每个模型在每个UTS上的表现. 包括

    Parameters
    ----------
    baseline_dir :
    found_dir :
    _baseline_sr :
    target_perf :

    Returns
    -------

    """
    baseline, found = _load_baseline_and_found(baseline_dir, found_dir)

    if _baseline_sr is None:
        _baseline_sr = baseline[ExcelMetricsKeys.DATA_SAMPLE_RATE].unique()[0]
    _all = pd.concat([baseline, found])
    _group_keys = ExcelMetricAnalysis.GROUP_BY_KEYS.copy()
    _group_keys.remove(ExcelMetricsKeys.DATA_SAMPLE_RATE)
    _best_founds = []
    out_metrics = []
    for _keys, _item in _all.groupby(by=_group_keys):
        _item[(EK.MEAN, target_perf)] = _item[(EK.MEAN, target_perf)].astype("float")
        _item = _item.sort_values(ExcelMetricsKeys.DATA_SAMPLE_RATE, ascending=False)
        if _item.shape[0] > 0:
            assert _item[(ExcelMetricsKeys.MODEL_NAME)].unique().shape[0] == 1
            _baseline = _item.iloc[0]
            if _baseline[ExcelMetricsKeys.DATA_SAMPLE_RATE] == _baseline_sr:
                _founds = _item.iloc[1:]
                if _founds.shape[0] == 0:
                    _founds = pd.DataFrame([[0] * _founds.shape[1]], columns=_founds.columns)
                _best_found_perf = _founds[(EK.MEAN, target_perf)].max()
                _ori_perf = _baseline[(EK.MEAN, target_perf)]
                _metrics = {
                    EK.MODEL_NAME: _baseline[ExcelMetricsKeys.MODEL_NAME],
                    "ori_time": _baseline[(EK.MEAN, ExcelMetricsKeys.ELAPSED_TRAIN)],
                    "fastuts_time": _founds[(EK.MEAN, ExcelMetricsKeys.ELAPSED_TRAIN)].sum(),
                    "ori. len": _baseline[(EK.MEAN, ExcelMetricsKeys.TRAIN_LEN)],
                    "try_len": _founds[(EK.MEAN, ExcelMetricsKeys.TRAIN_LEN)].sum(),
                    "try_len_list": str(_founds[(EK.MEAN, ExcelMetricsKeys.TRAIN_LEN)].tolist()),
                    "not_decrease": _ori_perf - _best_found_perf >= 0,
                    "is_decrease_0.5%": _ori_perf - _best_found_perf - 0.005 >= 0,
                    "is_decrease_1%": _ori_perf - _best_found_perf - 0.01 >= 0,
                    "is_decrease_3%": _ori_perf - _best_found_perf - 0.03 >= 0,
                    "is_decrease_5%": _ori_perf - _best_found_perf - 0.05 >= 0,
                    "is_decrease_7%": _ori_perf - _best_found_perf - 0.07 >= 0,
                    "is_decrease_10%": _ori_perf - _best_found_perf - 0.10 >= 0,
                    "is_decrease_15%": _ori_perf - _best_found_perf - 0.15 >= 0,
                    "dist_cal. time (sec)": _founds[(EK.MEAN, ExcelMetricsKeys.DATA_PROCESSING_TIME)].mean(),
                    f"ori_acc": _ori_perf,
                    f"fastuts_acc": _best_found_perf,
                    "decrease": _ori_perf - _best_found_perf,
                    "train_ratio": _founds[(EK.MEAN, ExcelMetricsKeys.TRAIN_LEN)].max() / _baseline[
                        (EK.MEAN, ExcelMetricsKeys.TRAIN_LEN)]
                }
                _metrics.update(list(zip(_group_keys, _keys)))
                out_metrics.append(_metrics)
            else:
                log.error(f"❌❌❌ Skip for not found baseline {_keys} ")
    df = pd.DataFrame(out_metrics)
    return df


from pylibs.utils.util_joblib import cache_


# @cache_


@cache_
def _load_metrics_from_npz(baseline_file, target_file):
    """
    Load the  metrics merged k-folds.

    Parameters
    ----------
    baseline_file :
    target_file :

    Returns
    -------

    """
    target_pd = ER.load_merge_metrics_bz2(target_file.lower())
    baseline_pd = ER.load_merge_metrics_bz2(baseline_file.lower())

    # process
    _comm_headers = target_pd.columns.intersection(baseline_pd.columns)
    return baseline_pd[_comm_headers], target_pd[_comm_headers]


@DeprecationWarning
def _load_metrics_from_excel(baseline_file, target_file):
    """
    这个方法有问题. 表头是元祖是,无法正常读取.


    Parameters
    ----------
    baseline_file :
    target_file :

    Returns
    -------

    """
    baseline_pd = pd.read_excel(baseline_file.lower())
    target_pd = pd.read_excel(target_file.lower())

    # process
    # baseline = ExcelMetricAnalysis.merge_kfold_df(baseline_pd)
    # found = ExcelMetricAnalysis.merge_kfold_df(target_pd)
    _comm_headers = target_pd.columns.intersection(baseline_pd.columns)
    return baseline_pd[_comm_headers], target_pd[_comm_headers]


@DeprecationWarning
class ResultDirConf:
    # 显著性水平
    SIGNIFICANT_LEVEL = 0.05
    TIME_SCALE = 3600
    MODEL_TYPES = ["dl", "ml"]
    TARGET_PERFS = ["VUS_PR", "VUS_ROC", "RF"]

    # TARGET_PERFS = ["VUS_ROC"]

    def __init__(self, version=JobConfV1.get_exp_version(), model_type="dl", target_perf="VUS_ROC"):
        self.version = version
        self.model_type = model_type
        self.target_perf = target_perf

    @classmethod
    def get_observation(cls):
        return "V404_observation_VUS_ROC_0.001_random"

    @classmethod
    def get_observation_final(cls):
        # return "V407_observation_VUS_ROC_0.001_random"
        # return "V408_observation_v2_VUS_ROC_0.001_random"
        return "V410_observation_v4_VUS_ROC_0.001_random"

    def get_dl_baseline_100(self):
        return f"{self.version}_baseline_{self.model_type}_VUS_ROC_0.001_random"

    def get_dl_baseline_0(self):
        return f"{self.version}_baseline_{self.model_type}_0_VUS_ROC_0.001_random"

    def get_best_conf(self):
        return f"{self.version}_fastuts_{self.model_type}_VUS_ROC_0.001_lhs"

    def get_diff_sample_method(self):
        if self.target_perf == ExcelMetricAnalysis.VUS_ROC:
            return self.get_diff_sample_method_vus_roc()
        else:
            return self.get_diff_sample_method_vus_pr()

    def get_diff_stop_alpha(self):
        if self.target_perf == ExcelMetricAnalysis.VUS_ROC:
            return self.get_different_stop_alpha_vus_roc()
        else:
            return self.get_different_stop_alpha_vus_pr()

    def get_diff_sample_method_vus_roc(self):
        return [
            f"{self.version}_fastuts_{self.model_type}_VUS_ROC_0.001_lhs",
            f"{self.version}_fastuts_{self.model_type}_VUS_ROC_0.001_random",
            f"{self.version}_fastuts_{self.model_type}_VUS_ROC_0.001_dist1",

        ]

    def get_diff_sample_method_vus_pr(self):
        return [
            f"{self.version}_fastuts_{self.model_type}_sup3_VUS_PR_0.001_lhs",
            f"{self.version}_fastuts_{self.model_type}_sup3_VUS_PR_0.001_random",
            f"{self.version}_fastuts_{self.model_type}_sup3_VUS_PR_0.001_dist1",
        ]

    def get_different_stop_alpha_vus_roc(self):
        return [
            f"{self.version}_fastuts_{self.model_type}_VUS_ROC_0.001_lhs",
            f"{self.version}_fastuts_{self.model_type}_sup2_VUS_ROC_0.01_lhs",
            f"{self.version}_fastuts_{self.model_type}_sup2_VUS_ROC_0.1_lhs",
            f"{self.version}_fastuts_{self.model_type}_sup2_VUS_ROC_0.5_lhs"
        ]

    def get_different_stop_alpha_vus_pr(self):
        return [
            f"{self.version}_fastuts_{self.model_type}_sup3_VUS_PR_0.001_lhs",
            f"{self.version}_fastuts_{self.model_type}_sup2_VUS_PR_0.01_lhs",
            f"{self.version}_fastuts_{self.model_type}_sup2_VUS_PR_0.1_lhs",
            f"{self.version}_fastuts_{self.model_type}_sup2_VUS_PR_0.5_lhs",
        ]


class ResultsConf:
    # 显著性水平

    SIGNIFICANT_LEVEL = 0.05
    TIME_SCALE = 3600
    TARGET_PERFS = ["VUS_PR", "VUS_ROC", "RF"]
    RESULT_DIR = "/Users/sunwu/Documents/download_metrics/p1"

    def __init__(self,
                 version,
                 target_perf="VUS_ROC",
                 ext=".bz2"):
        self.version = version
        self.target_perf = target_perf
        self.ext = ext

    @classmethod
    def get_observation(cls):
        return "V404_observation_VUS_ROC_0.001_random"

    @classmethod
    def get_observation_final(cls):
        # return "V407_observation_VUS_ROC_0.001_random"
        # return "V408_observation_v2_VUS_ROC_0.001_random"
        return "V410_observation_v4_VUS_ROC_0.001_random"

    def get_baseline(self):
        return "v4000_01_baseline_vus_roc_0.001_random.bz2"

    @DeprecationWarning
    def get_dl_baseline_0(self):
        return f"{self.version}_baseline_{self.model_type}_0_VUS_ROC_0.001_random"

    def get_best_conf(self):
        # return f"v2001_03_fastuts_sup1_{self.target_perf}_0.001_random.bz2"
        return f"v4020_03_fastuts_sup1_{self.target_perf}_0.001_random.bz2"

    def get_diff_sample_method(self):
        return [
            f"{self.version}_02_fastuts_{self.target_perf}_0.001_random{self.ext}",
            f"{self.version}_02_fastuts_{self.target_perf}_0.001_lhs{self.ext}",
            f"{self.version}_02_fastuts_{self.target_perf}_0.001_dist1{self.ext}",
        ]

    def get_diff_stop_alpha(self):
        return [
            f"v4020_03_fastuts_sup1_{self.target_perf}_0.001_random{self.ext}",
            f"v4020_03_fastuts_sup1_{self.target_perf}_0.01_random{self.ext}",
            f"v4020_03_fastuts_sup1_{self.target_perf}_0.1_random{self.ext}",
            f"v4020_03_fastuts_sup1_{self.target_perf}_0.5_random{self.ext}",

        ]


class ResConfV1:
    def __init__(self,
                 base_dir,
                 fastuts_dir,
                 target_perf,
                 home=ResultsConf.RESULT_DIR,
                 ):
        self.base_dir = os.path.join(home, base_dir).lower()
        self.target_dir = os.path.join(home, fastuts_dir).lower()
        self.opt_target = target_perf

    def get_target_sample_method(self):
        if self.target_dir.find("lhs") > 0:
            return "lhs"
        if self.target_dir.find("random") > 0:
            return "random"

        if self.target_dir.find("dist") > 0:
            return "dist"

        raise ValueError("Unsupported sample method")

    def get_stop_alpha(self):
        return self.target_dir.split("_")[-2]


class IMetricsLoader:
    @abc.abstractmethod
    def load_metrics(self):
        pass

    @classmethod
    def save_to_file(cls,
                     df: pd.DataFrame,
                     exp_name: str, home=UtilComm.get_runtime_directory()
                     ):
        """
        保存为 .bz2, 保存完 xlsx 读取不到多层表头.

        Parameters
        ----------
        df :
        exp_name :
        home :

        Returns
        -------

        """
        PDUtil.save_to_excel(df,
                             exp_name,
                             home=home)
        return PDUtil.save_to_bz2(df,
                                  exp_name,
                                  home=home)


class RedisMetricsLoader(IMetricsLoader):
    def __init__(self, exp_name: str, redis_conf: ExpServerConf = None):
        self.exp_name = exp_name
        self.conf = redis_conf
        self.rd = RedisUtil(self.conf)

    def load_metrics(self) -> pd.DataFrame:
        _metrics = self.rd.keys(self.exp_name + "*_metrics")
        outs = []
        for _m in _metrics:
            outs.append(json.loads(_m))

        ori_metrics = pd.DataFrame(outs)
        out_metrics = ExcelMetricAnalysis.merge_kfold_df(ori_metrics)
        self.save_to_file(out_metrics,
                          self.exp_name)
        return out_metrics


def _load_file(filename):
    try:
        df = pd.read_pickle(filename)
        _target = pd.DataFrame([df.iloc[:, 1].tolist()], columns=df.iloc[:, 0].tolist())
        _target["ORI_FILE"] = filename
        return _target
    except Exception as e:
        print(filename, f"is error: [{e}]", file=sys.stderr)
        return None


from pylibs.utils.util_joblib import cache_


@cache_
def _load_files_to_pd(metric_home, end_with):
    if not os.path.isdir(metric_home):
        print(f"⚠️ {metric_home} is not a directory!")
    files = FileUtils().get_all_files(metric_home, ext=end_with)
    out_metrics = Parallel(n_jobs=int(get_num_cpus() * 0.8), verbose=0, batch_size=1)(
        delayed(_load_file)(f) for f in tqdm(files))
    df = pd.concat(out_metrics)
    return df


class FileMetricsLoader(IMetricsLoader):
    def __init__(self,
                 exp_home: str,
                 end_with: str = "_metrics.bz2",
                 save_home: str = "./"
                 ):
        if exp_home.endswith("/"):
            exp_home = exp_home[:-1]
        self.exp_home = exp_home
        self.end_with = end_with
        self.save_home = save_home

    def load_metrics(self, save_origin=False) -> pd.DataFrame:
        """
        Load the metrics merged k-folds datasets.

        This method has been verified and it works correctly.

        Returns
        -------

        """
        metric_home = self.exp_home
        df = _load_files_to_pd(metric_home, self.end_with)
        if save_origin:
            PDUtil.save_to_excel(df, f"{os.path.basename(self.exp_home)}_original_metrics", home=self.save_home)
        outputs = ExcelMetricAnalysis.merge_kfold_df(df)
        file_name = self.save_to_file(
            outputs,
            exp_name=os.path.basename(self.exp_home),
            home=self.save_home
        )
        return file_name


def _post_process_metrics_dir(target_file):
    return os.path.join(ResultsConf.RESULT_DIR, target_file).lower()


from pylibs.utils.util_joblib import cache_


@cache_
def get_post_process_metrics(target_file,
                             baseline_file,
                             target_perf="VUS_PR"):
    """
    每个模型在每个UTS上的表现. 包括

    Parameters
    ----------
    metric_home :
    target_perf :

    Returns
    -------

    """

    target_file = _post_process_metrics_dir(target_file)
    baseline_file = _post_process_metrics_dir(baseline_file)

    print(f"Loading files with [acc]:"
          f"\ntarget: {os.path.basename(target_file)}"
          f"\nbasefile: {os.path.basename(baseline_file)}\n")

    baseline, found = _load_metrics_from_npz(baseline_file, target_file)
    # PDUtil.save_to_excel(baseline, "baseline")

    # parse the baseline sr
    base_srs = baseline[ExcelMetricsKeys.DATA_SAMPLE_RATE].unique()
    assert base_srs.shape[0] == 1, "baseline metrics error"
    _baseline_sr = base_srs[0]

    # merge
    _all = pd.concat([baseline, found])
    _group_keys = ExcelMetricAnalysis.GROUP_BY_KEYS.copy()
    _group_keys.remove(ExcelMetricsKeys.DATA_SAMPLE_RATE)
    _best_founds = []
    out_metrics = []
    for _keys, _item in _all.groupby(by=_group_keys):
        if _item.shape[0] == 1:
            warnings.warn("Only find one items")
            continue
        _baseline, _best_found, _founds = ResTools.split_baseline_and_founds(_item,
                                                                             baseline[
                                                                                 ExcelMetricsKeys.DATA_SAMPLE_RATE].iloc[
                                                                                 0],
                                                                             target_perf)
        _best_found_perf = _best_found[(EK.MEAN, target_perf)]
        _ori_perf = _baseline[(EK.MEAN, target_perf)]
        _metrics = {
            "model": _baseline[ExcelMetricsKeys.MODEL_NAME],
            "target_acc": target_perf,
            EK.DATA_SAMPLE_METHOD: _founds[EK.DATA_SAMPLE_METHOD].unique()[0],
            "ori_time": _baseline[(EK.MEAN, ExcelMetricsKeys.ELAPSED_TRAIN)],
            "fastuts_time": _founds[(EK.MEAN, ExcelMetricsKeys.ELAPSED_TRAIN)].sum(),
            "ori. len": _baseline[(EK.MEAN, ExcelMetricsKeys.TRAIN_LEN)],
            "try_len": _founds[(EK.MEAN, ExcelMetricsKeys.TRAIN_LEN)].sum(),
            "try_len_list": str(_founds[(EK.MEAN, ExcelMetricsKeys.TRAIN_LEN)].tolist()),
            "not_decrease": _ori_perf - _best_found_perf >= 0,
            "is_decrease_0.5%": _ori_perf - _best_found_perf - 0.005 >= 0,
            "is_decrease_1%": _ori_perf - _best_found_perf - 0.01 >= 0,
            "is_decrease_3%": _ori_perf - _best_found_perf - 0.03 >= 0,
            "is_decrease_5%": _ori_perf - _best_found_perf - 0.05 >= 0,
            "is_decrease_7%": _ori_perf - _best_found_perf - 0.07 >= 0,
            "is_decrease_10%": _ori_perf - _best_found_perf - 0.10 >= 0,
            "is_decrease_15%": _ori_perf - _best_found_perf - 0.15 >= 0,
            "dist_cal. time (sec)": _founds[(EK.MEAN, ExcelMetricsKeys.DATA_PROCESSING_TIME)].mean(),
            f"ori_acc": _ori_perf,
            f"fastuts_acc": _best_found_perf,
            "decrease": _ori_perf - _best_found_perf,
            "train_ratio": _founds[(EK.MEAN, ExcelMetricsKeys.TRAIN_LEN)].max() / _baseline[
                (EK.MEAN, ExcelMetricsKeys.TRAIN_LEN)]
        }
        _metrics.update(list(zip(_group_keys, _keys)))
        out_metrics.append(_metrics)
    df = pd.DataFrame(out_metrics)
    return df


class ResTools:
    @staticmethod
    def load_overall_results_from_different_opt_target():
        """
        返回不同优化目标下的FastUTS 的性能. 最佳配置: 随机抽样+间隔256+三种目标指标

        v2001_03_fastuts_sup1_rf_0.001_random.bz2
        v2001_03_fastuts_sup1_vus_roc_0.001_random.bz2
        v2001_03_fastuts_sup1_vus_pr_0.001_random.bz2

        Returns
        -------

        """

        _out_arr = []
        for __target_perf in ResultsConf.TARGET_PERFS:
            rt = ResultsConf(version=JobConfV1.EXP_VERSION,
                             target_perf=__target_perf)
            perf = ResTools.parse_model_perf_over_uts_scale(baseline=rt.get_baseline(),
                                                            target=rt.get_best_conf(),
                                                            target_perf=__target_perf)
            _out_arr.append(perf)
        return pd.concat(_out_arr)

    @staticmethod
    def _parse_model_perf_over_uts_scale(c: ResConfV1):
        """
        获取每个模型在每个单变量时间序列数据上面的 原始性能数据 和 找到的最佳数据. 性能数据包括3方面: 1)平均性能的均值和方差(由target perf指定) 2)

        key: model_name 	dataset_name	data_id
        性能: ori. perf. mean	ori. perf. std	fast. perf. mean	fast. perf. std
        模型训练时间(单位s): ori. train time mean	ori. train time std	fast. train time mean	fast. train time std
        数据处理时间: fast. data proc. time mean	fast. data proc. time std
        抽样: fast. best sr	fast. best train len


        主要工作:
        合并不同的kfold

        Parameters
        ----------
        c :

        Returns
        -------

        """
        # level 0: 原始数据
        baseline, found = _load_metrics_from_npz(baseline_file=c.base_file, target_file=c.target_file)
        # PDUtil.save_to_excel(df0, f"{c.get_file_name()}_l0_model_uts_metrics", append_entry=True)
        # level 0: 原始数据
        # level 1: 模型 over 单变量时间序列(UTS)
        _baseline_sr_list = baseline[ExcelMetricsKeys.DATA_SAMPLE_RATE].unique()
        assert _baseline_sr_list.shape[0] == 1, "baseline sampling rate is error. Expected one baseline (-1)"
        out_metrics1 = []
        for (_model_name, _dataset_name, _data_id), _data in pd.concat([baseline, found]).groupby(
                by=[ExcelMetricsKeys.MODEL_NAME, ExcelMetricsKeys.DATASET_NAME, ExcelMetricsKeys.DATA_ID]):
            _baseline, _best_found, _founds = ResTools.split_baseline_and_founds(_data,
                                                                                 _baseline_sr_list[0],
                                                                                 c.opt_target)
            if _baseline is None:
                print("Skip: ", _model_name, _dataset_name, _data_id)
                continue

            try:
                assert _founds.shape[0] == _data.shape[0] - 1
            except Exception as e:
                raise e

            # _ori_train_len = _baseline[(EK.TRAIN_LEN, EK.MEAN)]
            # _best_train_len = _best_found[(EK.TRAIN_LEN, EK.MEAN)]

            # 通过测试集来计算最佳抽样率.因为训练集会被抽样.
            _train_len = _best_found[(EK.MEAN, EK.TEST_LEN)] * (int(JobConfV1.K_FOLD) - 1)

            # 超过100%的,算100%
            _best_sample_rate = np.min([_best_found[(EK.DATA_SAMPLE_RATE, EK.EMPTY)] / _train_len, 1])

            _metric = {
                # 基本信息
                EK.TARGET_PERF: c.opt_target,
                EK.MODEL_NAME: _model_name,
                EK.DATASET_NAME: _dataset_name,
                EK.DATA_ID: _data_id,

                # Ori 性能
                EK.ORI_PERF_MEAN: _baseline[(EK.MEAN, c.opt_target)] * 100,
                EK.ORI_PERF_STD: _baseline[(EK.STD, c.opt_target)] * 100,

                # FastUTS 性能
                EK.FAST_PERF_MEAN: _best_found[(EK.MEAN, c.opt_target)] * 100,
                EK.FAST_PERF_STD: _best_found[(EK.STD, c.opt_target)] * 100,

                # 原始训练时间
                EK.ORI_TRAIN_TIME_MEAN: _baseline[(EK.MEAN, EK.ELAPSED_TRAIN)],
                EK.ORI_TRAIN_TIME_STD: _baseline[(EK.STD, EK.ELAPSED_TRAIN)],

                # FastUTS 的训练时间
                EK.FAST_TRAIN_TIME_MEAN: _founds[(EK.MEAN, EK.ELAPSED_TRAIN)].sum(),
                EK.FAST_TRAIN_TIME_STD: _founds[(EK.STD, EK.ELAPSED_TRAIN)].mean(),

                # 数据处理时间
                EK.FAST_DATA_PROCESSING_TIME_MEAN: _founds[(EK.MEAN, EK.DATA_PROCESSING_TIME)].sum(),
                EK.FAST_DATA_PROCESSING_TIME_STD: _founds[(EK.STD, EK.DATA_PROCESSING_TIME)].mean(),

                # 抽样率
                EK.FAST_BEST_SR: _best_sample_rate * 100,
                EK.FAST_BEST_TRAIN_LEN: _best_sample_rate * _train_len

            }

            _metric = round_dict(_metric, decimals=2)
            # 保留4位小数
            out_metrics1.append(_metric)
        model_uts_metrics_1 = pd.DataFrame(out_metrics1)
        # PDUtil.save_to_excel(model_uts_metrics_1)
        # sys.exit()
        # level 1: 模型 over 单变量时间序列(UTS)
        # =================================================================
        return model_uts_metrics_1

    @staticmethod
    def get_post_process_metrics(baseline, target, target_perf='VUS_ROC'):
        return get_post_process_metrics(baseline_file=baseline, target_file=target, target_perf=target_perf)

    @staticmethod
    def calculate_speed_up(_data):
        ori_train_time = _data[EK.ORI_TRAIN_TIME_MEAN].sum()
        fast_train_time = (_data[EK.FAST_TRAIN_TIME_MEAN].sum() + _data[EK.FAST_DATA_PROCESSING_TIME_MEAN].sum())
        return ((ori_train_time - fast_train_time) / ori_train_time) * 100

    @staticmethod
    def get_stop_alpha(target_dir):
        return target_dir.split("_")[-2]

    @staticmethod
    def parse_model_perf_over_uts_scale(baseline, target, target_perf):
        conf = ResConfV1(
            base_dir=baseline,
            fastuts_dir=target,
            target_perf=target_perf
        )
        return ResTools._parse_model_perf_over_uts_scale(conf)

    @classmethod
    def get_target_sample_method(cls, target_dir):
        return target_dir.split("_")[-1].split(".")[0]

    @classmethod
    def split_baseline_and_founds(cls, item, base_sr, perf_key):
        _item = item.sort_values(ExcelMetricsKeys.DATA_SAMPLE_RATE, ascending=False)
        if _item.shape[0] > 1:
            assert _item[ExcelMetricsKeys.MODEL_NAME].unique().shape[0] == 1, "Except one model."
            _baseline: pd.Series = _item[_item[EK.DATA_SAMPLE_RATE] == base_sr].iloc[0]
            _founds = _item[_item[EK.DATA_SAMPLE_RATE] != base_sr]
            _best_found = _founds.iloc[_founds[(EK.MEAN, perf_key)].argmax()]
            return _baseline, _best_found, _founds
        return None, None, None

    @classmethod
    def get_mean(cls, data, key):
        return data.groupby(by=EK.DATASET_NAME)[key].mean().round(2).mean()

    @classmethod
    def get_sum(cls, data, key):
        return data.groupby(by=EK.DATASET_NAME)[key].sum().round(2).sum()

    @classmethod
    def get_std(cls, data, key):
        return data.groupby(by=EK.DATASET_NAME)[key].std().round(2).mean()


class RT(ResTools):
    def __init__(self):
        pass


class ResConfV1:
    def __init__(self,
                 base_dir,
                 fastuts_dir,
                 target_perf,
                 home=ResultsConf.RESULT_DIR,
                 ):
        self.base_file = os.path.join(home, base_dir).lower()
        self.target_file = os.path.join(home, fastuts_dir).lower()
        pprint.pprint({
            "baseline metrics file": self.base_file,
            "target metrics file": self.target_file
        })
        assert self.base_file.index("baseline") > -1
        self.opt_target = target_perf

    def get_target_sample_method(self):
        if self.target_file.find("lhs") > 0:
            return "lhs"
        if self.target_file.find("random") > 0:
            return "random"

        if self.target_file.find("dist") > 0:
            return "dist"

        raise ValueError("Unsupported sample method")

    def get_stop_alpha(self):
        return self.target_file.split("_")[-2]


class ResConf:
    def __init__(self,
                 base_dir,
                 fastuts_dir,
                 model_type,
                 target_perf,
                 home="/Users/sunwu/Downloads/download_metrics"
                 ):
        self.base_dir = os.path.join(home, base_dir)
        self.target_dir = os.path.join(home, fastuts_dir)
        assert os.path.exists(self.base_dir), f"base_dir [{self.base_dir}] does not exist"
        assert os.path.exists(self.target_dir), f"target_dir [{self.target_dir}] does not exist"
        self.model_type = model_type
        self.target_perf = target_perf

    def get_file_name(self):
        return f"{self.target_perf}_{self.model_type}"

    def get_target_sample_method(self):
        if self.target_dir.find("lhs") > 0:
            return "lhs"
        if self.target_dir.find("random") > 0:
            return "random"

        if self.target_dir.find("dist") > 0:
            return "dist"

        raise ValueError("Unsupported sample method")

    def get_stop_alpha(self):
        return self.target_dir.split("_")[-2]


@DeprecationWarning
def parse_model_perf_over_uts(c: ResConf):
    """
    获取每个模型在每个单变量时间序列数据上面的 原始性能数据 和 找到的最佳数据. 性能数据包括3方面: 1)平均性能的均值和方差(由target perf指定) 2)

    key: model_name 	dataset_name	data_id
    性能: ori. perf. mean	ori. perf. std	fast. perf. mean	fast. perf. std
    模型训练时间: ori. train time mean	ori. train time std	fast. train time mean	fast. train time std
    数据处理时间: fast. data proc. time mean	fast. data proc. time std
    抽样: fast. best sr	fast. best train len


    主要工作:
    合并不同的kfold

    Parameters
    ----------
    c :

    Returns
    -------

    """
    # level 0: 原始数据
    baseline, found = _load_baseline_and_found(c.base_dir, c.target_dir)
    # PDUtil.save_to_excel(df0, f"{c.get_file_name()}_l0_model_uts_metrics", append_entry=True)
    # level 0: 原始数据
    # level 1: 模型 over 单变量时间序列(UTS)
    out_metrics1 = []
    for (_model_name, _dataset_name, _data_id), _data in pd.concat([baseline, found]).groupby(
            by=[ExcelMetricsKeys.MODEL_NAME, ExcelMetricsKeys.DATASET_NAME, ExcelMetricsKeys.DATA_ID]):
        _baseline, _best_found, _founds = split_baseline_and_founds(_data,
                                                                    baseline[
                                                                        ExcelMetricsKeys.DATA_SAMPLE_RATE].iloc[0],
                                                                    c.target_perf)
        if _baseline is None:
            print("Skip: ", _model_name, _dataset_name, _data_id)
            continue

        try:
            assert _founds.shape[0] == _data.shape[0] - 1
        except Exception as e:
            raise e

        _metric = {
            # 基本信息
            EK.TARGET_PERF: c.target_perf,
            EK.MODEL_NAME: _model_name,
            EK.DATASET_NAME: _dataset_name,
            EK.DATA_ID: _data_id,

            # Ori 性能
            EK.ORI_PERF_MEAN: _baseline[(EK.MEAN, c.target_perf)],
            EK.ORI_PERF_STD: _baseline[(EK.MEAN, c.target_perf)],

            # FastUTS 性能
            EK.FAST_PERF_MEAN: _best_found[(EK.MEAN, c.target_perf)],
            EK.FAST_PERF_STD: _best_found[(EK.MEAN, c.target_perf)],

            # 原始训练时间
            EK.ORI_TRAIN_TIME_MEAN: _baseline[(EK.MEAN, EK.ELAPSED_TRAIN)],
            EK.ORI_TRAIN_TIME_STD: _baseline[(EK.MEAN, EK.ELAPSED_TRAIN)],

            # FastUTS 的训练时间
            EK.FAST_TRAIN_TIME_MEAN: _founds[(EK.MEAN, EK.ELAPSED_TRAIN)].sum(),
            EK.FAST_TRAIN_TIME_STD: _founds[(EK.MEAN, EK.ELAPSED_TRAIN)].mean(),

            # 数据处理时间
            EK.FAST_DATA_PROCESSING_TIME_MEAN: _founds[(EK.MEAN, EK.DATA_PROCESSING_TIME)].sum(),
            EK.FAST_DATA_PROCESSING_TIME_STD: _founds[(EK.MEAN, EK.DATA_PROCESSING_TIME)].mean(),

            # 抽样率
            EK.FAST_BEST_SR: _best_found[EK.DATA_SAMPLE_RATE],
            EK.FAST_BEST_TRAIN_LEN: _best_found[(EK.TRAIN_LEN, EK.MEAN)],

        }

        _metric = round_dict(_metric)
        # 保留4位小数
        out_metrics1.append(_metric)
    model_uts_metrics_1 = pd.DataFrame(out_metrics1)

    # level 1: 模型 over 单变量时间序列(UTS)
    # =================================================================
    return model_uts_metrics_1


@DeprecationWarning
def parse_model_perf_over_uts_scale_v2(c: ResConf, df: pd.DataFrame = None):
    """
    获取每个模型在每个单变量时间序列数据上面的 原始性能数据 和 找到的最佳数据. 性能数据包括3方面: 1)平均性能的均值和方差(由target perf指定) 2)

    key: model_name 	dataset_name	data_id
    性能: ori. perf. mean	ori. perf. std	fast. perf. mean	fast. perf. std
    模型训练时间: ori. train time mean	ori. train time std	fast. train time mean	fast. train time std
    数据处理时间: fast. data proc. time mean	fast. data proc. time std
    抽样: fast. best sr	fast. best train len


    主要工作:
    合并不同的kfold

    Parameters
    ----------
    c :

    Returns
    -------

    """
    # level 0: 原始数据
    baseline, found = _load_baseline_and_found(c.base_dir, c.target_dir)
    # PDUtil.save_to_excel(df0, f"{c.get_file_name()}_l0_model_uts_metrics", append_entry=True)
    # level 0: 原始数据
    # level 1: 模型 over 单变量时间序列(UTS)

    # 过滤掉不需要的模型

    out_metrics1 = []
    for (_model_name, _dataset_name, _data_id), _data in pd.concat([baseline, found]).groupby(
            by=[ExcelMetricsKeys.MODEL_NAME, ExcelMetricsKeys.DATASET_NAME, ExcelMetricsKeys.DATA_ID]):
        _baseline, _best_found, _founds = split_baseline_and_founds(_data,
                                                                    baseline[
                                                                        ExcelMetricsKeys.DATA_SAMPLE_RATE].iloc[0],
                                                                    c.target_perf)
        if _baseline is None:
            print("Skip: ", _model_name, _dataset_name, _data_id)
            continue

        try:
            assert _founds.shape[0] == _data.shape[0] - 1
        except Exception as e:
            raise e

        _metric = {
            # 基本信息
            EK.TARGET_PERF: c.target_perf,
            EK.MODEL_NAME: _model_name,
            EK.DATASET_NAME: _dataset_name,
            EK.DATA_ID: _data_id,

            # Ori 性能
            EK.ORI_PERF_MEAN: _baseline[(EK.MEAN, c.target_perf)] * 100,
            EK.ORI_PERF_STD: _baseline[(EK.MEAN, c.target_perf)] * 100,

            # FastUTS 性能
            EK.FAST_PERF_MEAN: _best_found[(EK.MEAN, c.target_perf)] * 100,
            EK.FAST_PERF_STD: _best_found[(EK.MEAN, c.target_perf)] * 100,

            # 原始训练时间
            EK.ORI_TRAIN_TIME_MEAN: _baseline[(EK.MEAN, EK.ELAPSED_TRAIN)],
            EK.ORI_TRAIN_TIME_STD: _baseline[(EK.MEAN, EK.ELAPSED_TRAIN)],

            # FastUTS 的训练时间
            EK.FAST_TRAIN_TIME_MEAN: _founds[(EK.MEAN, EK.ELAPSED_TRAIN)].sum(),
            EK.FAST_TRAIN_TIME_STD: _founds[(EK.MEAN, EK.ELAPSED_TRAIN)].mean(),

            # 数据处理时间
            EK.FAST_DATA_PROCESSING_TIME_MEAN: _founds[(EK.MEAN, EK.DATA_PROCESSING_TIME)].sum(),
            EK.FAST_DATA_PROCESSING_TIME_STD: _founds[(EK.MEAN, EK.DATA_PROCESSING_TIME)].mean(),

            # 抽样率
            EK.FAST_BEST_SR: _best_found[EK.DATA_SAMPLE_RATE] * 100,
            EK.FAST_BEST_TRAIN_LEN: _best_found[(EK.TRAIN_LEN, EK.MEAN)],

        }

        _metric = round_dict(_metric, decimals=2)
        # 保留4位小数
        out_metrics1.append(_metric)
    model_uts_metrics_1 = pd.DataFrame(out_metrics1)

    # level 1: 模型 over 单变量时间序列(UTS)
    # =================================================================
    return model_uts_metrics_1


def parse_model_perf_observation(c: ResConf):
    """
    获取每个模型在每个单变量时间序列数据上面的 原始性能数据 和 找到的最佳数据. 性能数据包括3方面: 1)平均性能的均值和方差(由target perf指定) 2)

    key: model_name 	dataset_name	data_id
    性能: ori. perf. mean	ori. perf. std	fast. perf. mean	fast. perf. std
    模型训练时间: ori. train time mean	ori. train time std	fast. train time mean	fast. train time std
    数据处理时间: fast. data proc. time mean	fast. data proc. time std
    抽样: fast. best sr	fast. best train len


    主要工作:
    合并不同的kfold

    Parameters
    ----------
    c :

    Returns
    -------

    """
    # level 0: 原始数据
    metric = _load_metrics(c.target_dir)
    # PDUtil.save_to_excel(df0, f"{c.get_file_name()}_l0_model_uts_metrics", append_entry=True)
    # level 0: 原始数据
    # level 1: 模型 over 单变量时间序列(UTS)
    out_metrics1 = []
    for (_model_name, _dataset_name, _data_id), _data in metric.groupby(
            by=[ExcelMetricsKeys.MODEL_NAME, ExcelMetricsKeys.DATASET_NAME, ExcelMetricsKeys.DATA_ID]):
        _baseline, _best_found, _founds = split_baseline_and_founds(_data,
                                                                    metric[EK.DATA_SAMPLE_RATE].max(),
                                                                    c.target_perf)
        if _baseline is None:
            print("Skip: ", _model_name, _dataset_name, _data_id)
            continue

        try:
            assert _founds.shape[0] == _data.shape[0] - 1
        except Exception as e:
            raise e

        _metric = {
            # 基本信息
            EK.TARGET_PERF: c.target_perf,
            EK.MODEL_NAME: _model_name,
            EK.DATASET_NAME: _dataset_name,
            EK.DATA_ID: _data_id,

            # Ori 性能
            EK.ORI_PERF_MEAN: _baseline[(EK.MEAN, c.target_perf)] * 100,
            EK.ORI_PERF_STD: _baseline[(EK.MEAN, c.target_perf)] * 100,

            # FastUTS 性能
            EK.FAST_PERF_MEAN: _best_found[(EK.MEAN, c.target_perf)] * 100,
            EK.FAST_PERF_STD: _best_found[(EK.MEAN, c.target_perf)] * 100,

            # 原始训练时间
            EK.ORI_TRAIN_TIME_MEAN: _baseline[(EK.MEAN, EK.ELAPSED_TRAIN)],
            EK.ORI_TRAIN_TIME_STD: _baseline[(EK.MEAN, EK.ELAPSED_TRAIN)],

            # FastUTS 的训练时间
            EK.FAST_TRAIN_TIME_MEAN: _founds[(EK.MEAN, EK.ELAPSED_TRAIN)].sum(),
            EK.FAST_TRAIN_TIME_STD: _founds[(EK.MEAN, EK.ELAPSED_TRAIN)].mean(),

            # 数据处理时间
            EK.FAST_DATA_PROCESSING_TIME_MEAN: _founds[(EK.MEAN, EK.DATA_PROCESSING_TIME)].sum(),
            EK.FAST_DATA_PROCESSING_TIME_STD: _founds[(EK.MEAN, EK.DATA_PROCESSING_TIME)].mean(),

            # 抽样率
            EK.FAST_BEST_SR: _best_found[EK.DATA_SAMPLE_RATE] * 100,
            EK.FAST_BEST_TRAIN_LEN: _best_found[(EK.TRAIN_LEN, EK.MEAN)],

        }

        _metric = round_dict(_metric, decimals=2)
        # 保留4位小数
        out_metrics1.append(_metric)
    model_uts_metrics_1 = pd.DataFrame(out_metrics1)

    # level 1: 模型 over 单变量时间序列(UTS)
    # =================================================================
    return model_uts_metrics_1


if __name__ == '__main__':
    pass
    data = load_overall_results_from_different_opt_target()
    PDUtil.save_to_excel(data)
    # outputs = ExcelMetricAnalysis.merge_kfold_df(df)
    # ml = RedisMetricsLoader(exp_name="v1921_02_fastuts_vus_roc_0.001_random",
    #                         redis_conf=GCFV3.get_server_conf(ServerNameV2.REDIS_219, net_type="lan"))
    # metrics = ml.load_metrics()
    # PDUtil.save_to_excel(metrics)

    # _m = _load_metrics_from_redis()
    # print(_m)

    # rc = ResultDirConf(model_type="ml")
    # print(rc.get_baseline())
    # print(rc.get_dl_baseline_0())
    # print(rc.get_diff_sample_method_vus_pr())
    # print(rc.get_diff_sample_method_vus_roc())
    # print(rc.get_different_stop_alpha_vus_pr())
    # print(rc.get_different_stop_alpha_vus_roc())
    #
    # res = load_merge_perf_diff_sample_method()
    # print(res)
    # pass
    # print("Deel Models")
    # baseline_dir_ = "/Users/sunwu/SW-Research/sw-research-code/A01_paper_exp/data/baseline/tf"
    # found_dir_ = "/Users/sunwu/SW-Research/sw-research-code/A01_paper_exp/data/found/main_fastuts_tf-2023-07-18-v7-debug-False"
    # static_fast_uts_info(baseline_dir_, found_dir_)
    #
    # print("Classical Models")
    # baseline_dir_ = "/Users/sunwu/SW-Research/sw-research-code/A01_paper_exp/data/baseline/sklearn"
    # found_dir_ = "/Users/sunwu/SW-Research/sw-research-code/A01_paper_exp/data/found/main_fastuts_sklearn-2023-07-18-v11-debug-False"
    # static_fast_uts_info(baseline_dir_, found_dir_)
