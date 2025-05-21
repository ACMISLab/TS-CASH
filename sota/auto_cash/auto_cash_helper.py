import copy
import dataclasses
import os
import pickle

import numpy as np
import pandas as pd

DB_HOME = "/Users/sunwu/SW-Research/AutoML-Benchmark/sota/auto_cash/algdb"


class AuthCashHelper:

    @staticmethod
    def get_model_name_by_model_id(id):
        """
        输入: 0,
        输出: extra_trees
        """
        maps = pickle.load(open("auto_cash_algorithm_mapping.pkl", "rb"))
        return maps[id]


@dataclasses.dataclass
class ModelTrainConf:
    model: str  # 算法名称
    model_configs: dict
    dataset_name: str
    fold_index: int
    n_fold: int = 5
    seed: int = 42

    def get_model_confs(self):
        # fix: sklearn.utils._param_validation.InvalidParameterError: The 'epsilon' parameter of MLPClassifier must be a float in the range (0.0, inf). Got 0.0 instead.
        if self.model == "mlp" and self.model_configs['epsilon'] == 0:
            self.model_configs['epsilon'] = 1e-8

        # fix:sklearn.utils._param_validation.InvalidParameterError: The 'epsilon' parameter of SGDClassifier must be a float in the range [0.0, inf). Got nan instead.
        if self.model == "sgd" and np.isnan(self.model_configs['epsilon']):
            self.model_configs['epsilon'] = 1e-8

        # fix: sklearn.utils._param_validation.InvalidParameterError: The 'l1_ratio' parameter of SGDClassifier must be a float in the range [0.0, 1.0]. Got nan instead.
        if self.model == "sgd" and np.isnan(self.model_configs['l1_ratio']):
            self.model_configs['l1_ratio'] = 1e-8
        # sklearn.utils._param_validation.InvalidParameterError: The 'power_t' parameter of SGDClassifier must be a float in the range (-inf, inf). Got nan instead.
        if self.model == "sgd" and np.isnan(self.model_configs['power_t']):
            self.model_configs['power_t'] = 1e-8
        return self.model_configs

    def get_model_cash_file(self):
        return os.path.join(DB_HOME, f"{self.model}.csv")

    def get_model_cash_file_tmp(self):
        """运行的时候更新的文件"""
        return os.path.join(DB_HOME, f"{self.model}_tmp.csv")

    def get_cache_metric(self):
        df_base = pd.read_csv(self.get_model_cash_file(), index_col=0)
        df_tmp = pd.read_csv(self.get_model_cash_file(), index_col=0)
        df = pd.concat([df_base, df_tmp])

        query = ""
        for key, value in self.get_model_xargs().items():
            if value == "True" or value == "False":
                query += f"{key}=={bool(value)} and "
            elif isinstance(value, (float, int)):
                query += f"{key}=={value} and "
            elif isinstance(value, str):
                query += f"{key}=='{value}' and "

        #  activation=='relu' and alpha==0.015429 and batch_size=='auto' and beta_1==0.9 and beta_2==0.999 and early_stopping=='train' and epsilon==0.0 and hidden_layer_depth==1 and learning_rate_init==0.000163 and n_iter_no_change==32 and num_nodes_per_layer==152 and shuffle=='True' and solver=='adam' and tol==0.0001 and validation_fraction==0.1 and random_state==42
        _query = query[:-4]

        # early_stopping=='train'
        r = df.query(_query)
        if r.shape[0] == 0:
            return None
        else:
            return r.iloc[0].to_dict()

    def get_model_xargs(self):
        """获取算法的超参数配置"""
        return get_model_args_from_dict_by_model_name(self.get_model_confs(), self.model, self.seed)

    def get_kvdb_dict(self):
        hpys = get_model_args_from_dict_by_model_name(self.get_model_confs(), self.model, self.seed)
        hpys.update({
            "model": self.model,
            "dataset": self.dataset_name,
            "fold_index": self.fold_index})

        return hpys


@dataclasses.dataclass
class ModelTrainValue:
    elapsed_seconds: float
    f1: float
    precision: float
    recall: float
    roc_auc: float
    log_loss: float
    accuracy: float
    error_msg: str = ""


class KVDB:
    @staticmethod
    def sort_dict_by_key(sort_d):
        """字典安装key排序,最后的目标是直接str(dict)作为map的key"""
        return {key: sort_d[key] for key in sorted(sort_d.keys())}

    def __init__(self, db_name="algdb.dump", db_home=DB_HOME):

        self.file = os.path.join(db_home, db_name)
        if os.path.exists(self.file):
            with open(self.file, "rb") as f:
                self.db = pickle.load(f)
        else:
            self.db = {}

    def add(self, k, v):
        """向数据库中增加内容"""
        # print(f"add: {k}\n{v}")
        with open(self.file, "wb") as f:
            self.db[k] = v
            pickle.dump(self.db, f)

    def adds(self, kvs: list[str, str]):
        """批量向数据库中增加内容"""
        for k, v in kvs:
            self.db[k] = v

        with open(self.file, "wb") as f:
            pickle.dump(self.db, f)

    def add_by_dict(self, model_hpys: dict, model_name: str, dataset: str, fold_index: int, value: dict):
        """添加算法超参数的时候,需要提供其他参数如模型名称/数据集名称/交叉验证的折数"""
        _key = copy.deepcopy(model_hpys)
        _key.update({
            "model": model_name,
            "dataset": dataset,
            "fold_index": fold_index}
        )
        _sort_key = KVDB.sort_dict_by_key(_key)
        self.add(str(_sort_key), value)

    def query(self, k):
        """查询指定的key

        """
        if self.db.get(k) is None:
            return None

        try:
            return ModelTrainValue(**self.db.get(k))
        except:
            return self.db.get(k)

    def keys(self):
        return list(self.db.keys())

    def values(self):
        return list(self.db.values())

    def query_by_dict(self, q_dict: dict):
        sort_d = KVDB.sort_dict_by_key(q_dict)
        return self.query(str(sort_d))


def is_number(value):
    if value is None:
        return False
    try:
        float(value)  # 尝试转换为浮点数
        return True
    except ValueError:
        return False


def get_model_args_from_dict(dict_configs: dict, seed=42) -> dict:
    """将 ConfigSpace.Configuration 转为dict,作为算法的参数.
    另外, 如果是float,那么就保留6位小数,如果是科学计数法,不变

    输入:
    {'mlp:activation': 'relu', 'mlp:alpha': 0.0001, 'mlp:batch_size': 'auto', 'mlp:beta_1': 0.9, 'mlp:beta_2': 0.999, 'mlp:early_stopping': 'train', 'mlp:epsilon': 1e-08, 'mlp:hidden_layer_depth': 1, 'mlp:learning_rate_init': 0.001, 'mlp:n_iter_no_change': 32, 'mlp:num_nodes_per_layer': 32, 'mlp:shuffle': 'True', 'mlp:solver': 'adam', 'mlp:tol': 0.0001, 'random_state': 42}

    输出:
    {'activation': 'relu', 'alpha': 0.0001, 'batch_size': 'auto', 'beta_1': 0.9, 'beta_2': 0.999, 'early_stopping': 'train', 'epsilon': 1e-08, 'hidden_layer_depth': 1, 'learning_rate_init': 0.001, 'n_iter_no_change': 32, 'num_nodes_per_layer': 32, 'random_state': 42, 'shuffle': 'True', 'solver': 'adam', 'tol': 0.0001}
    """
    model_name = dict_configs['__choice__']
    del dict_configs['__choice__']
    # dict_configs.update({"random_state": seed})
    #
    # values = {}
    # for k, v in dict_configs.items():
    #     if is_number(v):
    #         v = round(v, 6)
    #     values.update({k.replace(f"{_model_name}:", ""): v})
    # return values
    return get_model_args_from_dict_by_model_name(dict_configs, model_name, seed)


def get_model_args_from_dict_by_model_name(dict_configs: dict, model_name: str, seed=42) -> dict:
    """将 ConfigSpace.Configuration 转为dict,作为算法的参数.
    另外, 如果是float,那么就保留6位小数,如果是科学计数法,不变

    输入:
    {'mlp:activation': 'relu', 'mlp:alpha': 0.0001, 'mlp:batch_size': 'auto', 'mlp:beta_1': 0.9, 'mlp:beta_2': 0.999, 'mlp:early_stopping': 'train', 'mlp:epsilon': 1e-08, 'mlp:hidden_layer_depth': 1, 'mlp:learning_rate_init': 0.001, 'mlp:n_iter_no_change': 32, 'mlp:num_nodes_per_layer': 32, 'mlp:shuffle': 'True', 'mlp:solver': 'adam', 'mlp:tol': 0.0001, 'random_state': 42}

    输出:
    {'activation': 'relu', 'alpha': 0.0001, 'batch_size': 'auto', 'beta_1': 0.9, 'beta_2': 0.999, 'early_stopping': 'train', 'epsilon': 1e-08, 'hidden_layer_depth': 1, 'learning_rate_init': 0.001, 'n_iter_no_change': 32, 'num_nodes_per_layer': 32, 'random_state': 42, 'shuffle': 'True', 'solver': 'adam', 'tol': 0.0001}
    """
    dict_configs.update({"random_state": seed})

    values = {}
    for k, v in dict_configs.items():
        if is_number(v):
            v = round(v, 8)
        values.update({k.replace(f"{model_name}:", ""): v})
    return values


if __name__ == '__main__':
    print(AuthCashHelper.get_model_name_by_model_id(0))
