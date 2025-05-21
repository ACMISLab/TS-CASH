"""
@summary
AutoCASH 自动剪枝算法的超参数 (Auto-CASH: 3.5.1. Potential priority)



作用:
1. 从 `c00_baseline_n500_madelon_original_20241029_1434.csv.gz` 中提取每个算法及其超参数对应的精度
[
(Configuration(values={
  '__choice__': 'adaboost',
  'adaboost:algorithm': 'SAMME.R',
  'adaboost:learning_rate': 0.1835637959226,
  'adaboost:max_depth': 3,
  'adaboost:n_estimators': 414,
}),
RunValue(default=0.4351, roc_auc=0.5649, f1=0.3889, accuracy=0.56, recall=0.3333, log_loss=7.0146, precision=0.4667, elapsed_seconds=1.0019, error_msg='', run_job=None, exp_conf=None)),


(Configuration(values={
  '__choice__': 'adaboost',
  'adaboost:algorithm': 'SAMME',
  'adaboost:learning_rate': 0.4408660466885,
  'adaboost:max_depth': 5,
  'adaboost:n_estimators': 380,
}),
RunValue(default=0.4729, roc_auc=0.5271, f1=0.303, accuracy=0.54, recall=0.2381, log_loss=7.3335, precision=0.4167, elapsed_seconds=0.8701, error_msg='', run_job=None, exp_conf=None))
]
"""
import pickle
import sys

sys.path.append("/Users/sunwu/SW-Research/AutoML-Benchmark")
sys.path.append("/Users/sunwu/SW-Research/AutoML-Benchmark/deps")
# PERF_HOME = "perf"
# os.makedirs(PERF_HOME, exist_ok=True)
from tshpo.automl_libs import *


class Configuration:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model_name = kwargs['values']['__choice__']

        # 算法的名称,例如: 'adaboost'

        hpys = kwargs['values']
        del hpys['__choice__']
        self.hpys = hpys


class RunValue:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.accuracy = kwargs['accuracy']
        self.roc_auc = kwargs['roc_auc']

        # auto-cash 的fscore定义
        self.fscore_autocash = self.accuracy * self.roc_auc

        # 模型的训练时间
        self.runtime = kwargs['elapsed_seconds']


metircs = AnaHelper.get_all_metrics()
df_baseline = AnaHelper.load_csv_file("c00_baseline_n500_madelon_original_20241029_1434.csv.gz")
all_data = {}
for k, v in df_baseline.iterrows():

    config_and_value = v['configs_and_metrics']
    dataset = v['dataset']
    conf_and_perf = eval(config_and_value)
    # 'dresses-sales:::gaussian_nb'
    for _c, _v in conf_and_perf:
        hpys = _c.hpys
        model_name = _c.model_name

        data_key = f"{dataset}:::{model_name}"
        hpys['label'] = _v.fscore_autocash
        # {'adaboost:algorithm': 'SAMME.R', 'adaboost:learning_rate': 0.1835637959226, 'adaboost:max_depth': 3, 'adaboost:n_estimators': 414, 'label': 0.316344}
        if data_key not in all_data.keys():
            all_data[data_key] = []
        all_data[data_key].append(hpys)

pickle.dump(all_data, open("../meta_datas/all_data.pkl", "wb"))
