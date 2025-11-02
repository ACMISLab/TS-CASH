from sota.auto_cash.auto_cash_helper import KVDB

automhs_hpy = KVDB("authmhs_gpt_processed.dump", "./")

import os
import time
import time
import traceback

import numpy as np
import pandas as pd
import pygmo as pg
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, CategoricalHyperparameter, Configuration
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, log_loss, accuracy_score

from pylibs.utils.util_numpy import enable_numpy_reproduce
from sota.auto_cash.auto_cash import AutoCASHUtil
from sota.auto_cash.auto_cash_helper import get_model_args_from_dict, ModelTrainConf
from sota.auto_cash.model_trainer import ModelTrainer
from tshpo.automl_libs import get_auto_sklearn_classification_search_space, load_dataset_at_fold
from autosklearn.pipeline.components.classification import ClassifierChoice

from tshpo.lib_class import AnaHelper

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
        suggest_res = automhs_hpy.query(f"{dataset}:::{gpt_version}")
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


DATA_HOME = "automhs_gpt_perf"
os.makedirs(DATA_HOME, exist_ok=True)
if __name__ == '__main__':
    small_dataset = ["dresses-sales",
                     "climate-model-simulation-crashes",
                     "cylinder-bands",
                     "ilpd",
                     "credit-approval",
                     "breast-w",
                     "diabetes",
                     "tic-tac-toe",
                     "credit-g",
                     "qsar-biodeg",
                     "pc1",
                     "pc4",
                     "pc3",
                     "kc1",
                     "ozone-level-8hr",
                     "madelon",
                     "kr-vs-kp",
                     "Bioresponse",
                     "sick",
                     "spambase",
                     "wilt",
                     "churn",
                     "phoneme",
                     "jm1",
                     "PhishingWebsites",
                     "nomao",
                     "bank-marketing",
                     "electricity",
                     ]
    fold_indexes = [0, 1, 2, 3, 4]
    metrics = [AnaHelper.METRIC_ACCURACY, AnaHelper.METRIC_ROC_AUC]
    gpt_versions = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
    SEED = 42
    N_INDIVIDUALS = 10
    COST_LIMIT = 500

    for _dataset in small_dataset:
        results = []
        metric_file_name = os.path.join(DATA_HOME, f"{_dataset}.csv")
        if os.path.exists(metric_file_name):
            continue
        for _fold_index in fold_indexes:
            for _metric in metrics:
                for _gpt_version in gpt_versions:
                    automhs = AutoMHSGPT(dataset=_dataset, fold_index=_fold_index, gpt_version=_gpt_version,
                                         metric=_metric)
                    _best_metric = automhs.eval()

                    results.append({
                        "dataset": _dataset,
                        "gpt_version": _gpt_version,
                        "fold_index": _fold_index,
                        "metric": _metric,
                        "value": _best_metric,
                    })

        pd.DataFrame(results).to_csv(metric_file_name)
