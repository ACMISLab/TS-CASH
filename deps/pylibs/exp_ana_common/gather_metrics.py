import argparse
import os
import sys
import traceback
import warnings

from tqdm import tqdm

from pylibs.exp_ana_common.ExcelOutKeys import ExcelMetricsKeys
from pylibs.utils.util_common import UtilComm
from pylibs.utils.util_system import UtilSys

sys.path.append(os.path.abspath("../../libs/datasets"))
sys.path.append(os.path.abspath("../../libs/py-search-lib"))
sys.path.append(os.path.abspath("../../libs/timeseries-models"))
import numpy as np
import pandas as pd
from joblib import Memory

from pylibs.utils.util_directory import make_dirs, DUtil
from pylibs.utils.util_log import get_logger
from pylibs.utils.util_stack import StackHelper
from pylibs.utils.util_statistical import is_significant_decrease_fix_test

memory = Memory(".cache", verbose=0)
log = get_logger()
from pylibs.utils.util_file import FileUtils

HOME = "./output_analysis"
DUtil.mkdir(HOME)

KEY_MODEL_NAME = "model_name"
KEY_DATASET_NAME = "dataset_name"
KEY_DATA_SAMPLE_METHOD = "data_sample_method"
KEY_DATA_SAMPLE_RATE = "data_sample_rate"
KEY_TRAIN_ELAPSED = "elapsed_train"
KEY_DATA_ID = "data_id"

_column_names = [
    "model_name",
    "data_set",
    "data_set",
]


class FileGathers:
    _metrics_all = None
    _FLOAT_COLUMNS = ["Precision",
                      "Recall",
                      "F",
                      "AUC_ROC",
                      "AUC_PR",
                      "Precision_at_k",
                      "Rprecision",
                      "Rrecall",
                      "RF",
                      "R_AUC_ROC",
                      "R_AUC_PR",
                      "VUS_ROC",
                      "VUS_PR",
                      "elapsed_train"
                      ]

    def __init__(self, metric_dir,
                 is_save_all=True,
                 baseline_sr=-1,
                 target_metric="VUS_ROC",
                 out_home=UtilComm.get_runtime_directory(),
                 endswith="_metrics.csv",
                 job_id=None,
                 headers=None
                 ):
        """
        Args:
            args :
            is_save_all:
        """
        self._baseline_sample_rate = baseline_sr
        self._target_metric = target_metric
        self._is_save_all = is_save_all
        self._metric_dir = metric_dir
        self._result_home = out_home
        self._exp_name = ""
        self._job_id = job_id
        self._endswith = endswith
        self._headers = headers
        make_dirs(self._result_home)

        # the file name for collecting all metrics
        self._output_metric_file_name = os.path.join(self._result_home, f"{self._exp_name}_original_metrics.xlsx")

    def __round_metrics(self, metrics: pd.DataFrame, decimals=4):
        """
        Round the metrics.
        """
        metrics[self._FLOAT_COLUMNS] = metrics[self._FLOAT_COLUMNS].astype("float").round(decimals)
        return metrics

    def load_all_metrics(self):
        if self._metrics_all is not None:
            return self._metrics_all

        if os.path.exists(self._output_metric_file_name) and self._is_save_all is True:
            UtilSys.is_debug_mode() and log.info(
                "Loading all metrics from existing file: \n%s " % os.path.abspath(self._output_metric_file_name))
            self._metrics_all = pd.read_excel(self._output_metric_file_name)
        else:
            fu = FileUtils()
            files = fu.get_all_files(self._metric_dir, ext=self._endswith)
            if len(files) == 0:
                raise RuntimeError(f"No metrics file found in {self._metric_dir} ")
                sys.exit(-1)
            _headers_names = None
            _data_array = []
            n_empty = 0
            for file in tqdm(files, desc="Loading metrics files"):
                try:
                    if os.path.getsize(file) == 0:
                        # print(f"{file} is empty!")
                        n_empty += 1
                        continue
                    data = pd.read_csv(file, header=None)
                    if data.shape[1] > 2:
                        # 数据结构是 DataFrame,每一行是一条有效数据

                        if _headers_names is None:
                            _headers_names = data.iloc[0].to_list()
                        data.columns = data.iloc[0].to_list()
                        _data = data.iloc[1:]
                        _target = _data[_headers_names]
                        _target['filename'] = file
                        # 确保header是相同的
                        _data_array.append(_target)
                    else:
                        # 数据结构是 Series
                        if _headers_names is None:
                            _headers_names = data.iloc[:, 0].values.tolist()

                        # 将列数据变为行数据
                        data = data.T

                        # 设置列名
                        data.columns = data.iloc[0, :]

                        # 移除第一行,因为第一行是列名
                        data = data.iloc[1:, :]

                        # 保证列相同
                        _target = data[_headers_names]
                        _target['filename'] = file
                        _data_array.append(_target.copy())
                except:
                    traceback.print_exc()
                    sys.exit(-1)
            print(f"{self._metric_dir}: all:{len(files)}, empty files: {n_empty}. ")
            if len(_data_array) == 0:
                return None
            _metrics_all = pd.concat(_data_array)

            UtilSys.is_debug_mode() and log.info(f"✅✅✅ All metrics have loaded, shapes: {_metrics_all.shape}")
            self._metrics_all = _metrics_all
        return self._metrics_all

    @classmethod
    @DeprecationWarning
    def get_model_accuracy(cls, metrics, target_metric: str = "AUC_ROC", sample_rate: float = 1):
        _data = metrics[metrics["data_sample_rate"].astype('float') == sample_rate][target_metric].astype('float')
        return np.asarray(_data)

    @classmethod
    def get_model_metrics_within_sample_rate(cls, metrics, sample_rate: float = -1):
        _data = metrics[metrics["data_sample_rate"].astype('float') == sample_rate]
        return _data

    @DeprecationWarning
    def statistical_analysis(self):
        """
        这个版本的假设检验是用错了的
        """
        metrics_all = self.load_all_metrics()
        statistics_results = []
        for (_model_name, _dataset_name, _data_sample_method), _metrics in metrics_all.groupby(
                by=[KEY_MODEL_NAME, KEY_DATASET_NAME, KEY_DATA_SAMPLE_METHOD]):
            _statistics_tmp = {
                "model_name": _model_name,
                "_dataset": _dataset_name,
                KEY_DATA_SAMPLE_METHOD: _data_sample_method
            }

            baseline_metrics = self.get_model_metrics_within_sample_rate(_metrics, self._baseline_sample_rate)

            # Verify normal distribution

            _data_sample_rates = self._get_sample_rates_from_metrics_descending(_metrics)

            for _cur_data_sample_rate in _data_sample_rates:
                _cur_metrics = self.get_model_metrics_within_sample_rate(_metrics, _cur_data_sample_rate)
                _baseline_metrics, _sampled_metrics = self._post_process_paired_metrics(baseline_metrics, _cur_metrics)

                is_sig_decrease_, t_, p_ = is_significant_decrease_fix_test(_baseline_metrics,
                                                                            _sampled_metrics,
                                                                            return_t_and_p=True)
                if np.isnan(p_):
                    p_ = 999
                _statistics_tmp[f'sr={_cur_data_sample_rate}'] = np.round(p_, 4)

            statistics_results.append(_statistics_tmp)

        df = pd.DataFrame(statistics_results)

        self._save_df_to_excel(df, "hypothesis_test_on_{}".format(self._target_metric))

    def statistical_analysis_fix_test(self):
        """
        这个版本的假设检验是用错了的
        """
        metrics_all = self.load_all_metrics()
        statistics_results = []
        for (_model_name, _dataset_name, _data_sample_method), _metrics in metrics_all.groupby(
                by=[KEY_MODEL_NAME, KEY_DATASET_NAME, KEY_DATA_SAMPLE_METHOD]):
            _statistics_tmp = {
                "model_name": _model_name,
                "_dataset": _dataset_name,
                KEY_DATA_SAMPLE_METHOD: _data_sample_method
            }

            baseline_metrics = self.get_model_metrics_within_sample_rate(_metrics, self._baseline_sample_rate)

            # Verify normal distribution

            _data_sample_rates = self._get_sample_rates_from_metrics_descending(_metrics)

            for _cur_data_sample_rate in _data_sample_rates:
                _cur_metrics = self.get_model_metrics_within_sample_rate(_metrics, _cur_data_sample_rate)
                _baseline_metrics, _sampled_metrics = self._post_process_paired_metrics(baseline_metrics, _cur_metrics)

                is_sig_decrease_, t_, p_ = is_significant_decrease_fix_test(_baseline_metrics,
                                                                            _sampled_metrics,
                                                                            return_t_and_p=True)
                if np.isnan(p_):
                    p_ = 999
                _statistics_tmp[f'sr={_cur_data_sample_rate}'] = np.round(p_, 4)

            statistics_results.append(_statistics_tmp)

        df = pd.DataFrame(statistics_results)

        self._save_df_to_excel(df, "hypothesis_test_on_{}".format(self._target_metric))

    def statistical_analysis_single(self):
        metrics_all = self.load_all_metrics()
        statistics_results = []
        for (_model_name, _dataset_name, _data_sample_method), _metrics in metrics_all.groupby(
                by=[KEY_MODEL_NAME, KEY_DATASET_NAME, KEY_DATA_SAMPLE_METHOD]):
            _statistics_tmp = {
                "model_name": _model_name,
                "_dataset": _dataset_name,
                KEY_DATA_SAMPLE_METHOD: _data_sample_method
            }

            baseline_metrics = self.get_model_metrics_within_sample_rate(_metrics, self._baseline_sample_rate)

            # Verify normal distribution

            _data_sample_rates = self._get_sample_rates_from_metrics_descending(_metrics)

            for _cur_data_sample_rate in _data_sample_rates:
                _cur_metrics = self.get_model_metrics_within_sample_rate(_metrics, _cur_data_sample_rate)
                _baseline_metrics, _sampled_metrics = self._post_process_paired_metrics(baseline_metrics, _cur_metrics)

                is_sig_decrease_, t_, p_ = is_significant_decrease_fix_test(_baseline_metrics,
                                                                            _sampled_metrics,
                                                                            return_t_and_p=True)
                if np.isnan(p_):
                    p_ = 999
                _statistics_tmp[f'sr={_cur_data_sample_rate}'] = np.round(p_, 4)

            statistics_results.append(_statistics_tmp)

        df = pd.DataFrame(statistics_results)

        self._save_df_to_excel(df, "hypothesis_test_on_{}".format(self._target_metric))

    def _post_process_paired_metrics(self, baseline_metrics, sampled_metrics):
        """
         _baseline_metrics, _sampled_metrics = self._post_process_paired_metrics(baseline_metrics, _cur_metrics)

        Parameters
        ----------
        baseline_metrics :
        sampled_metrics :

        Returns
        -------

        """
        _merged_metrics = baseline_metrics.merge(sampled_metrics, on=KEY_DATA_ID)
        baseline_perf = _merged_metrics[self._target_metric + "_x"].values
        sampled_perf = _merged_metrics[self._target_metric + "_y"].values
        return baseline_perf, sampled_perf

    def get_saved_file_name(self, file_name):
        file_name = os.path.join(self._result_home, self._exp_name, file_name)
        UtilSys.is_debug_mode() and log.info(f"OptMetricsType file: {os.path.abspath(file_name)}")
        return file_name

    def accuracy_decrease(self):
        statistics_results = []
        metrics_all = self.load_all_metrics()
        for _model_name, metrics in metrics_all.groupby(by=KEY_MODEL_NAME):
            for _dataset_name, metrics_model_dataset in metrics.groupby(by=KEY_DATASET_NAME):
                for _data_sample_method, _metrics_model_dataset in metrics_model_dataset.groupby(
                        by=KEY_DATA_SAMPLE_METHOD):

                    _statistics_tmp = {
                        "model_name": _model_name,
                        "_dataset": _dataset_name,
                        KEY_DATA_SAMPLE_METHOD: _data_sample_method
                    }

                    _base_line_metrics = self.get_model_accuracy(_metrics_model_dataset,
                                                                 target_metric=self._target_metric,
                                                                 sample_rate=self._baseline_sample_rate)
                    _base_line_mean = _base_line_metrics.mean()
                    _base_line_std = _base_line_metrics.std()

                    _data_sample_rates = self._get_sample_rates_from_metrics_descending(metrics_model_dataset)
                    _data_sample_rates.sort()
                    for _cur_data_sample_rate in _data_sample_rates:
                        _cur_metrics = self.get_model_accuracy(_metrics_model_dataset,
                                                               target_metric=self._target_metric,
                                                               sample_rate=_cur_data_sample_rate)
                        _cur_metrics_mean = _cur_metrics.mean()
                        _cur_metrics_std = _cur_metrics.std()

                        _statistics_tmp[f'sr={_cur_data_sample_rate}_mean'] = np.round(
                            (_base_line_mean - _cur_metrics_mean), 4)
                        _statistics_tmp[f'sr={_cur_data_sample_rate}_std'] = np.round(
                            (_base_line_mean - _cur_metrics_mean), 4)
                    statistics_results.append(_statistics_tmp)
                # end for

        # end for

        file_name = os.path.join(args.out_home, self._exp_name, "accuracy_decrease.xlsx")
        results = pd.DataFrame(statistics_results)
        results.to_excel(file_name)

        groups_results = results.groupby(by=["model_name", "data_sample_method"])

        sample_rate_means = ["sr=0.5_mean", "sr=0.7_mean", "sr=0.9_mean"]

        for srm in sample_rate_means:
            UtilSys.is_debug_mode() and log.info(f"sample_rate_mean: {srm}" + "=" * 20)
            UtilSys.is_debug_mode() and log.info(groups_results[srm].mean().round(4))

    def training_time_within_accuracy_loss(self, agg=np.mean, acc_loss=0):
        """

        Parameters
        ----------
        agg :
        acc_loss : float
             0: means not significant decrease, using statistical significant test
             a float number between > 1, such 0.01: the mean decrease 0.01

        Returns
        -------

        """
        data = self.load_all_metrics()

        # analyze only the data with the given sample method
        time_metrics = []

        for (_model_name, _dataset_name, _data_sample_method), _metric in data.groupby(
                by=[KEY_MODEL_NAME, KEY_DATASET_NAME, KEY_DATA_SAMPLE_METHOD]):
            target_sample_rate = self._find_max_sample_rate_within_acc_decrease(_model_name, _dataset_name, _metric)
            baseline_training_time = self._get_training_time(_metric, self._baseline_sample_rate, agg)
            sampled_training_time = self._get_training_time(_metric, target_sample_rate, agg)
            time_metrics.append({
                KEY_MODEL_NAME: _model_name,
                KEY_DATASET_NAME: _dataset_name,
                KEY_DATA_SAMPLE_METHOD: _data_sample_method,
                "baseline_sample_rate": self._baseline_sample_rate,
                "baseline_training_time": baseline_training_time,
                "sampled_training_time": sampled_training_time,
                "found_sample_sr": target_sample_rate
            }
            )

        model_train_time = pd.DataFrame(time_metrics)
        model_train_time["faster_than"] = model_train_time["baseline_training_time"] / \
                                          model_train_time["sampled_training_time"]
        model_train_time["faster_than"] = model_train_time["faster_than"]
        model_train_time["time_reduction"] = model_train_time["baseline_training_time"] - model_train_time[
            "sampled_training_time"]
        file_name = f"faster_than_within_acc_loss_{acc_loss}"
        self._save_df_to_excel(model_train_time, file_name)
        self._save_df_to_latex(model_train_time, file_name)

        _count_all = model_train_time.groupby(by=[KEY_MODEL_NAME, KEY_DATA_SAMPLE_METHOD], as_index=False)
        output = _count_all['time_reduction'].sum()
        output['avg_faster_than'] = _count_all["faster_than"].mean()
        self._save_df_to_excel(output, f"count_time_{acc_loss}")

    def _get_training_time(self, data, sample_rate, agg=np.sum):
        """

        Parameters
        ----------
        data :
        agg : np.function
            such np.sum, not end with ().
            you can't specify np.sum().
            Must be np.sum

        sample_rate :

        Returns
        -------

        """

        try:
            return agg(data[data[KEY_DATA_SAMPLE_RATE].astype("float") == float(sample_rate)][KEY_TRAIN_ELAPSED].astype(
                "float").values)
        except Exception:
            traceback.print_exc()
            raise Exception

    def _find_max_sample_rate_within_acc_decrease(self, model_name, dataset_name, metric, acc_decrease=0,
                                                  min_sample_rate=0.015625):
        """
        None means for all.


        Parameters
        ----------
        model_name :
        dataset_name :
        metric :
        acc_decrease: float
            0: means not significant decrease, using statistical significant test
            a float between > 1, such 0.01: the mean decrease 0.01


        Returns
        -------

        """

        baseline_metrics = self.get_model_metrics_within_sample_rate(metric, self._baseline_sample_rate)

        _data_sample_rates = self._get_sample_rates_from_metrics_descending(metric)
        stack = StackHelper()
        for _cur_data_sample_rate in _data_sample_rates:
            stack.append(_cur_data_sample_rate)
            # is_normal_distribution(baseline_metrics[self._target_metric].values)
            _cur_metrics = self.get_model_metrics_within_sample_rate(metric, _cur_data_sample_rate)

            # Fix the issue for data is not equal.
            baseline_perf, sampled_perf = self._post_process_paired_metrics(baseline_metrics, _cur_metrics)

            if acc_decrease == 0:
                if is_significant_decrease_fix_test(baseline_perf, sampled_perf):
                    stack.pop()
                    break
            elif acc_decrease > 0:
                if sampled_perf + acc_decrease < baseline_perf:
                    stack.pop()
                    break

        # not found the specified sample rate
        _found_sample_rate = stack.pop()
        if _found_sample_rate is None:
            return self._baseline_sample_rate
        elif _found_sample_rate == self._baseline_sample_rate:
            # in all cases, the model accuracy is not decreased
            return min_sample_rate
        else:
            return _found_sample_rate

    def _find_sample_rate_within_given_accuracy(self, model_name, dataset_name, metric, accuracy_loss=0.01):
        """
        None means for all.


        Parameters
        ----------
        model_name :
        dataset_name :
        metric :

        Returns
        -------

        """
        # baseline_metrics = self.get_model_accuracy(
        #     metric,
        #     target_metric=self.args.target_metric,
        #     sample_rate=self._baseline_sample_rate
        # )
        baseline_metrics = self.get_model_metrics_within_sample_rate(metric, self._baseline_sample_rate)

        _data_sample_rates = self._get_sample_rates_from_metrics_descending(metric)

        _first_significance_decrease_index = -1
        for index_, _cur_data_sample_rate in enumerate(_data_sample_rates):
            # is_normal_distribution(baseline_metrics[self._target_metric].values)
            _cur_metrics = self.get_model_metrics_within_sample_rate(metric, _cur_data_sample_rate)

            # Fix the issue for data is not equal.
            merged_metrics = baseline_metrics.merge(_cur_metrics, on=KEY_DATA_ID)
            baseline_perf = merged_metrics[self._target_metric + "_x"].values
            sampled_perf = merged_metrics[self._target_metric + "_y"].values
            _base_mean = baseline_perf.mean()
            _sample_mean = sampled_perf.mean()
            if _sample_mean + accuracy_loss < _base_mean:
                _first_significance_decrease_index = index_
                break

        _first_significance_decrease_index = _first_significance_decrease_index - 1
        if _first_significance_decrease_index < 0:
            # all are significant decrease.
            return None

        return _data_sample_rates[_first_significance_decrease_index]

    @classmethod
    def _get_sample_rates_from_metrics_descending(cls, metric):
        ret_metric = metric[KEY_DATA_SAMPLE_RATE].sort_values(ascending=False).unique()
        return ret_metric

    def _save_df_to_latex(self, df, file_name):
        ext = ".tex"
        if not file_name.endswith(ext):
            file_name = file_name + ext
        df.to_latex(self.get_saved_file_name(file_name), float_format="%.2f", index=False)

    def _save_df_to_excel(self, df, file_name):
        ext = ".xlsx"
        if not file_name.endswith(ext):
            file_name = file_name + ext
        df.to_excel(self.get_saved_file_name(file_name), index=False)


def get_analysis_args(command=None):
    """
    args=get_analysis_args(["--metric_dir", "results/exp_classic_all_rep/lof/MGAB"])

    Parameters
    ----------
    command :

    Returns
    -------

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_metric", default="VUS_ROC", help="The target metric to analyze")
    parser.add_argument("--out_home", default="./output_analysis", help="The target metric to analyze")
    parser.add_argument("--analysis_data_sample_method", default="normal_random",
                        help="The data sample method to analyze. "
                             "One of random, normal_random, stratified, and None."
                             "None means  all")
    parser.add_argument("--baseline_sr", default=-1, help="A number for disabling sample (rate) feature.", type=float)
    parser.add_argument("--metric_dir",
                        default="/Users/sunwu/SW-Research/sw-research-code/A01_paper_exp/results/test_task_2023_07_07_v1/iforest/ECG",
                        help="The directory to load metrics")
    if command is not None:
        _args = parser.parse_args(command)
    else:
        _args = parser.parse_args()
    return _args


if __name__ == '__main__':
    # metrics_types = ["VUS_ROC","VUS_PR","R_AUC_ROC",'Precision_at_k',"F","RF"]
    args = get_analysis_args(["--metric_dir",
                              "/Users/sunwu/SW-Research/sw-research-code/A01_paper_exp/results/test_iforest_Daphnet_2023_07_07_v1"])
    ay = FileGathers(args)
    ay.load_all_metrics()
