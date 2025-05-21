"""
将历史训练数据(数据,算法及其超参数配置,性能)搜集起来, 组成一个数据库, 避免同样的算法训练两次,以节省时间

"""
import os
import sys

from pyutils.kvdb.kvdb_mysql import KVDBMySQL

sys.path.append("/Users/sunwu/SW-Research/AutoML-Benchmark")
sys.path.append("/Users/sunwu/SW-Research/AutoML-Benchmark/deps")
from sota.auto_cash.auto_cash_helper import get_model_args_from_dict_by_model_name, KVDB
from tshpo.automl_libs import *

# from pyutils.kvdb.kvdb_sqlite import KVDBSqlite
# dbsqlite = KVDBSqlite(dbfile="/Users/sunwu/SW-Research/AutoML-Benchmark/tshpo/tshpo_alg_perf.sqlite")
dbsqlite = KVDBMySQL()


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


def main(file="c00_baseline_n500_madelon_original_20241029_1434.csv.gz"):
    df_baseline = AnaHelper.load_csv_file(file)
    for k, v in tqdm(df_baseline.iterrows(), total=df_baseline.shape[0]):

        config_and_value = v['configs_and_metrics']
        dataset = v['dataset']
        conf_and_perf = eval(config_and_value)
        for _c, _v in tqdm(conf_and_perf):
            hpys = _c.hpys
            hpys["__choice__"] = _c.model_name
            item = get_model_args_from_dict_by_model_name(hpys, _c.model_name)
            model_name = _c.model_name
            _key = copy.deepcopy(item)
            _key.update({
                "model": model_name,
                "dataset": dataset,
                "fold_index": v['fold_index']})

            _val = {"elapsed_seconds": _v.kwargs['elapsed_seconds'],
                    "f1": _v.kwargs['f1'],
                    "precision": _v.kwargs['precision'],
                    "recall": _v.kwargs['recall'],
                    "roc_auc": _v.kwargs['roc_auc'],
                    "log_loss": -1,
                    "accuracy": _v.kwargs['accuracy'],
                    "error_msg": _v.kwargs.get('error_msg')
                    }

            # key按键排序
            _key_sort = KVDB.sort_dict_by_key(_key)

            dbsqlite.add(_key_sort, _val)


if __name__ == '__main__':
    datahome = "/Users/sunwu/SW-Research/AutoML-Benchmark/exp_results/tshpo"
    for f in os.listdir(datahome):
        if f.endswith(".gz") and f.find("baseline_n") > -1:
            print(f"load {f}")
            main(f)
