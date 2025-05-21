import pandas as pd

from pylibs.utils.util_joblib import JLUtil
from pylibs.utils.util_log import get_logger
from pylibs.utils.util_number import float_str_to_fraction_str
from pylibs.utils.util_stack import StackHelper

log=get_logger()

class AnaCommon:
    MODEL_ORDER = ["decision_tree", "iforest", "random_forest", "pca", "hbos", "lof",
                   "svm", "ocsvm"]


class P1CK:
    training_time = "elapsed_train"
    data_sample_rate = "data_sample_rate"
    model_name = "model_name"
    data_sample_method = "data_sample_method"


memory = JLUtil.get_memory()


@memory.cache
def get_training_time_from_original_metrics(data_sample_rate, data_sample_method, model_name,
                                            metric_file="/Users/sunwu/SW-Research/sw-research-code/A01_paper_exp/results_analys_script/output_analysis/exp_classic_all_v2.1/original_metrics.xlsx",
                                            time_unit="sec"):
    """
    Given the data_sample_method,  data_sample_rate, and model_name, return the average training time (sec).

    time_unit: sec, min
    """
    df = P1Util.get_metrics_from_excel(metric_file)
    df_filter = df[(df[P1CK.data_sample_method] == data_sample_method) & (df[P1CK.model_name] == model_name) & (
                df[P1CK.data_sample_rate].astype("float") == data_sample_rate)]
    avg_train_time = df_filter[P1CK.training_time].mean()
    if time_unit == "sec":
        pass
    elif time_unit == "min":
        avg_train_time = avg_train_time / 60
    return avg_train_time


class P1Util:

    original_metrics=None
    @classmethod
    def get_training_time(cls, sample_rate, data_sample_method, model_name):
        return get_training_time_from_original_metrics(sample_rate, data_sample_method, model_name)

    @classmethod
    def get_metrics_from_excel(cls, metric_file):
        if cls.original_metrics is None:
            cls.original_metrics=pd.read_excel(metric_file)
        return cls.original_metrics
    @staticmethod
    def excat_sr(df, start_with="sr=", removed_sr=[-1, 0], asscending=False):
        """
        Extract  sorted columns names with prefix start_with
        removed_sr: the sr in removed will be removed

        for example:
            input pd columns:
                Index(['model_name', '_dataset', 'data_sample_method', 'sr=0.95', 'sr=0.9',
                   'sr=0.8', 'sr=0.7', 'sr=0.6', 'sr=0.5', 'sr=0.25', 'sr=0.125',
                   'sr=0.0625', 'sr=0.03125', 'sr=0.015625', 'sr=0.0078125',
                   'sr=0.00390625', 'sr=0.00195312', 'sr=0.0', 'sr=-1.0'],
                  dtype='object')
            output:
                ['sr=0.95', 'sr=0.9', 'sr=0.8', 'sr=0.7', 'sr=0.6', 'sr=0.5', 'sr=0.25', 'sr=0.125', 'sr=0.0625', 'sr=0.03125', 'sr=0.015625', 'sr=0.0078125', 'sr=0.00390625', 'sr=0.00195312']
        """
        t = []
        for c in df.columns:
            if str(c).startswith(start_with):
                float_sr = float(str(c).split("=")[-1])
                if not float_sr in removed_sr:
                    t.append([c, float_sr])
        df_ = pd.DataFrame(t, columns=["sr", "srf"])
        df_ = df_.sort_values(by='srf', ascending=asscending)
        return df_['sr'].values.tolist()
    @staticmethod
    def find_max_sr_without_sig_decrease(df):
        srs = P1Util.excat_sr(df, asscending=False)

        tmp = []
        for (model_name, dataset_name, data_sample_method), model_data in df.groupby(
                by=["model_name", "_dataset", "data_sample_method"]):
            # 从大的 sr 开始找, 只要发现模型性能显著降低就停止,并返回没有显著降低的 sr
            stack = StackHelper()
            # 保证栈底部有元素
            stack.append("sr=-1")

            found = False
            for sr in srs:
                if model_data[sr].min() < 0.01:
                    # 找到一个就出栈, prove that accuracy is significantly decrease
                    sr_start = stack.pop()
                    tmp.append([model_name, dataset_name, data_sample_method, sr_start, sr_start.split("=")[-1],
                                float_str_to_fraction_str(sr_start.split("=")[-1])])
                    found = True
                    break
                else:
                    stack.append(sr)
            # 到最后还没有找到, 那么就用最小的一个
            if not found:
                sr_start = stack.pop()
                tmp.append([model_name, dataset_name, data_sample_method, sr_start, sr_start.split("=")[-1],
                            float_str_to_fraction_str(sr_start.split("=")[-1])])

        return pd.DataFrame(tmp, columns=['model_name', 'dataset_name', 'data_sample_method', 'sr*', "float_sr*",
                                          "frac_sr*"])

    @staticmethod
    def get_data_in_dataset(data_with_method, selected_datasets):
        """
        Input:
        name, dataset_name, data_sample_method
        decision_tree,MGAB,normal_random
        decision_tree,SMD,normal_random
        decision_tree,IOPS,normal_random
        decision_tree,OPPORTUNITY,normal_random
        decision_tree,YAHOO,normal_random
        hbos,OPPORTUNITY,normal_random
        hbos,YAHOO,normal_random
        hbos,SMD,normal_random
        hbos,MGAB,normal_random
        hbos,IOPS,normal_random
        iforest,MGAB,normal_random
        iforest,YAHOO,normal_random
        iforest,SMD,normal_random
        iforest,IOPS,normal_random
        iforest,OPPORTUNITY,normal_random

        output:
        Return the pd where the dataset_name is in selected_datasets
        if selected_datasets=['SMD'], it equals  df[df['dataset_name'] == "SMD"]

        """
        _arr = []
        for _data_name in selected_datasets:
            _arr.append(data_with_method[data_with_method['dataset_name'] == _data_name])
        current_data = pd.concat(_arr, axis=0)
        return current_data


if __name__ == '__main__':
    t = get_training_time_from_original_metrics(0.9,"normal_random","decision_tree")
    # t = get_training_time_from_original_metrics(-1, "random", 'svm')
    # print(t)
    # t = get_training_time_from_original_metrics(-1, "random", 'ocsvm')
    # print(t)
