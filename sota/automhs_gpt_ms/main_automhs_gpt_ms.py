from hp_builder import FCHyperparameterSpaceBuilder
from pyutils.kvdb.kvdb_json import KVDBJson

kvdb_processed = KVDBJson("authmhs_gpt_ms_process.json")

import os
import time

import pandas as pd
from ConfigSpace import ConfigurationSpace

from pylibs.utils.util_numpy import enable_numpy_reproduce
from sota.auto_cash.auto_cash_helper import ModelTrainConf, get_model_args_from_dict, \
    get_model_args_from_dict_by_model_name
from sota.auto_cash.model_trainer import ModelTrainer
from tshpo.automl_libs import get_auto_sklearn_classification_search_space, load_dataset_at_fold, train_model_smac_ms

from tshpo.lib_class import AnaHelper, RunJobMS, ExpConf

SEED = 42
enable_numpy_reproduce(SEED)


def narrow_alg_search_space(alg_important_hpy, cs):
    """减少单个算法的超参数搜索空间, 只保留cs中重要的超参数alg_important_hpy
    Auto-CASH: 3.5.1. Potential priority
    """
    # 创建目标配置空间
    cstmp = ConfigurationSpace()
    # 根据条件筛选超参数（例如：只复制默认值为 0 的超参数）
    for hpy in cs.get_hyperparameters():
        if hpy.name in alg_important_hpy:
            cstmp.add_hyperparameter(hpy)
    return cstmp


def get_model_args_from_cs(conf):
    """将ConfigSpace的配置转为dict, 作为算法的参数

    输入:
    {'mlp:activation': 'relu', 'mlp:alpha': 0.0001, 'mlp:batch_size': 'auto', 'mlp:beta_1': 0.9, 'mlp:beta_2': 0.999, 'mlp:early_stopping': 'train', 'mlp:epsilon': 1e-08, 'mlp:hidden_layer_depth': 1, 'mlp:learning_rate_init': 0.001, 'mlp:n_iter_no_change': 32, 'mlp:num_nodes_per_layer': 32, 'mlp:shuffle': 'True', 'mlp:solver': 'adam', 'mlp:tol': 0.0001, 'random_state': 42}

    输出:
    {'activation': 'relu', 'alpha': 0.0001, 'batch_size': 'auto', 'beta_1': 0.9, 'beta_2': 0.999, 'early_stopping': 'train', 'epsilon': 1e-08, 'hidden_layer_depth': 1, 'learning_rate_init': 0.001, 'n_iter_no_change': 32, 'num_nodes_per_layer': 32, 'random_state': 42, 'shuffle': 'True', 'solver': 'adam', 'tol': 0.0001}
    """
    model_configs = conf.get_dictionary()
    _model_name = model_configs['__choice__']
    del model_configs['__choice__']
    model_configs.update({"random_state": SEED})

    values = {}
    for k, v in model_configs.items():
        values.update({k.replace(f"{_model_name}:", ""): v})
    return values


class AutoMHSGPT:
    def __init__(self, dataset, fold_index, metric, gpt_version, random_state=42):
        self.dataset = dataset
        self.fold_index = fold_index
        self.random_state = random_state
        self.metric = metric
        suggest_res = kvdb_processed.query({
            "dataset": dataset,
            "gpt_version": gpt_version,
        })
        self.suggest_alg_name = suggest_res["suggest_model_name"]
        self.suggest_hpys = suggest_res['suggest_model_hpy']
        # self.suggest_important_hpy_name=suggest_res['suggest_model_hpy']
        cs = get_auto_sklearn_classification_search_space(y_train=[0, 1],
                                                          include=[self.suggest_alg_name])
        print(cs)
        # fix: sklearn.utils._param_validation.InvalidParameterError: The 'max_leaf_nodes' parameter of RandomForestClassifier must be an int in the range [2, inf) or None. Got 0 instead.
        if self.suggest_alg_name == "random_forest":
            if self.suggest_hpys['max_leaf_nodes'] in [None, 0]:
                self.suggest_hpys['max_leaf_nodes'] = 2

        # fix: sklearn.utils._param_validation.InvalidParameterError: The 'max_features' parameter of RandomForestClassifier must be an int in the range [1, inf), a float in the range (0.0, 1.0], a str among {'log2', 'sqrt'} or None. Got 'auto' instead.
        if self.suggest_alg_name == "random_forest":
            if self.suggest_hpys['max_features'] in [None, "auto"]:
                self.suggest_hpys['max_features'] = "sqrt"
        # fix: ValueError: could not convert string to float: 'scale'
        if self.suggest_alg_name == "libsvm_svc":
            if self.suggest_hpys['gamma'] in ["scale"]:
                self.suggest_hpys['gamma'] = 0.1
        if self.suggest_alg_name == "decision_tree":
            if self.suggest_hpys['max_features'] in ["auto"]:
                self.suggest_hpys['max_features'] = 1.0

        # fix:s klearn.utils._param_validation.InvalidParameterError: The 'max_depth' parameter of RandomForestClassifier must be an int in the range [1, inf) or None. Got 0 instead.
        if self.suggest_alg_name == "random_forest":
            if self.suggest_hpys['max_depth'] in [0]:
                self.suggest_hpys['max_depth'] = None

        # fix: ValueError: could not convert string to float: 'sqrt' for extra_trees
        if self.suggest_alg_name == "extra_trees":
            if self.suggest_hpys['max_features'] in ["sqrt", "auto"]:
                self.suggest_hpys['max_features'] = 0.5

        # fix: sklearn.utils._param_validation.InvalidParameterError: The 'max_depth' parameter of ExtraTreesClassifier must be an int in the range [1, inf) or None. Got 0 instead.
        if self.suggest_alg_name == "extra_trees":
            if self.suggest_hpys['max_depth'] in [0]:
                self.suggest_hpys['max_depth'] = None

        # fix: sklearn.utils._param_validation.InvalidParameterError: The 'max_leaf_nodes' parameter of ExtraTreesClassifier must be an int in the range [2, inf) or None. Got 0 instead.
        if self.suggest_alg_name == "extra_trees":
            if self.suggest_hpys['max_leaf_nodes'] in [None, 0]:
                self.suggest_hpys['max_leaf_nodes'] = None

        self.X_train, self.y_train, self.X_test, self.y_test = load_dataset_at_fold(dataset_name=dataset, n_fold=5,
                                                                                    fold_index=self.fold_index,
                                                                                    seed=random_state)

    def eval(self):

        # 评估算法
        _start_time = time.time()

        model_configs = self.suggest_hpys
        mt = ModelTrainer(ModelTrainConf(
            model=self.suggest_alg_name,
            model_configs=model_configs,
            dataset_name=self.dataset,
            fold_index=self.fold_index,
            seed=self.random_state
        ))
        train_res = mt.train()
        if self.metric == AnaHelper.METRIC_ACCURACY:
            ret_metric = train_res.accuracy
        elif self.metric == AnaHelper.METRIC_ROC_AUC:
            ret_metric = train_res.roc_auc
        else:
            raise NotImplementedError("指标未实现")
        print(model_configs, train_res)

        return ret_metric


class AutoMHSGPTMS:
    def __init__(self, dataset, fold_index, metric, gpt_version, random_state=42):
        self.dataset = dataset
        self.fold_index = fold_index
        self.random_state = random_state
        self.metric = metric
        suggest_res = kvdb_processed.query({
            "dataset": dataset,
            "gpt_version": gpt_version,
        })
        self.suggest_alg_name = suggest_res["suggest_model_name"]
        self.suggest_hpys = suggest_res['suggest_model_hpy']
        if self.suggest_alg_name == "fc_rf":
            if self.suggest_hpys['max_depth'] in [0]:
                self.suggest_hpys['max_depth'] = None

        # self.suggest_important_hpy_name=suggest_res['suggest_model_hpy']
        builder = FCHyperparameterSpaceBuilder(seed=42)
        self.cs = builder.build_configuration_space(include=[self.suggest_alg_name])
        print(self.cs)

    def eval(self):
        # 评估算法
        _start_time = time.time()

        model_configs = self.suggest_hpys
        default_hpy_dict = dict(model_configs)
        default_hpy_dict_processed = get_model_args_from_dict_by_model_name(default_hpy_dict, self.suggest_alg_name, 42)
        self.suggest_hpys['__choice__'] = self.suggest_alg_name
        res = train_model_smac_ms(RunJobMS(
            X_train=None,
            y_train=None,

            X_test=None,
            y_test=None,
            # 精度指标，如f1，acc，recall
            metric=self.metric,

            # 模型的超参数配置
            config=self.suggest_hpys,
            # 超参数搜索空间
            cs=self.cs,

            debug=False,

            seed=42,
            exp_conf=ExpConf(dataset=self.dataset),
            mode="max"
        ))

        return res.default


db = KVDBJson("metric.json")
if __name__ == '__main__':
    small_dataset = ["D1",
                     "D2",
                     ]
    metrics = ["prec", "recall", "f1"]
    gpt_versions = ["gpt-4o"]
    SEED = 42
    N_INDIVIDUALS = 10
    COST_LIMIT = 500
    for _dataset in small_dataset:
        for _metric in metrics:
            for _gpt_version in gpt_versions:
                _key = {
                    "dataset": _dataset,
                    "gpt_version": _gpt_version,
                    "metric": _metric,
                }

                if db.is_exist(_key):
                    print("{} is exist".format(_key))
                    continue
                automhs = AutoMHSGPTMS(dataset=_dataset, fold_index=0, gpt_version=_gpt_version, metric=_metric)
                _best_metric = automhs.eval()

                _val = {
                    "dataset": _dataset,
                    "gpt_version": _gpt_version,
                    "metric": _metric,
                    "value": _best_metric,
                }
                db.add(_key, _val)
    db.to_csv("results_automhs_gpt_ms.csv")
