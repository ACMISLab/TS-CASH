"""
1算法选择: 基于Meta-feature来选择算法
2超参数优化: 基于GA来优化

"""
import pickle
from pathlib import Path

import cachetools.func
import numpy as np
import pandas as pd

from sota.auto_cash.alg_select import get_best_alg_by_datasetname

DATA_HOME = Path(__file__).parent / "meta_datas"


class AutoCASHUtil:

    def get_important_fea(self, dataset, model_name):
        """实现Auto-CASH 3.5.1. Potential priority

        返回每个具体算法的重要超参数, 使用随机深林实现.

        预处理脚本 :
        1) hpo_pruning/01_prepare_hp_and_per.py
        2) hpo_pruning/02_feature_importance.py
        """
        key = dataset + ":::" + model_name
        pickle_file = DATA_HOME / "select_features.pkl"
        conf = pickle.load(open(pickle_file, "rb"))
        return conf[key]

    def get_alg_by_dataset_name(self, dataset, fold_index):
        """实现Auto-CASH: 3.4. Automatic algorithm selection

        输入数据集名称, 根据meta-features获取推荐的算法名称.

        """
        return get_best_alg_by_datasetname(dataset, fold_index)

    def get_init_hpy(self, dataset, model_name):
        """
        3.5.2. Fast-forward initialization

        简化: 相同的数据集视为相同的任务, 使用相同任务上的模型最佳算法的超参数配置初始化算法


        返回 {'adaboost:algorithm': 'SAMME.R', 'adaboost:learning_rate': 1.9970105234092, 'adaboost:max_depth': 10, 'adaboost:n_estimators': 185}
        """
        pickle_file = DATA_HOME / "all_data.pkl"
        conf = pickle.load(open(pickle_file, "rb"))
        # print(conf)
        pd.DataFrame(conf[f'{dataset}:::{model_name}'])
        best_conf = pd.DataFrame(conf[f'{dataset}:::{model_name}']).sort_values(by=['label'], ascending=False).iloc[0]
        init_hpy = best_conf.to_dict()
        del init_hpy['label']
        return init_hpy

    @cachetools.func.lru_cache
    def init_auto_cash(self, dataset, fold_index):
        """初始化Auto-CASH选择的算法/重要超参数/超参数默认值"""
        # 3.4. Automatic algorithm selection
        # alg_name: 根据meta-features 获取到的算法的推荐名称
        alg_name = self.get_alg_by_dataset_name(dataset, fold_index)

        # 3.5.1. Potential priority
        # alg_important_hpy: 使用RF 获取到的重要超参数
        alg_important_hpy = self.get_important_fea(dataset, alg_name)

        # 3.5.2. Fast-forward initialization
        # Our observation is that the hyperparameter settings are similar for the same algorithm for different tasks.
        # 简化: 在数据集上最佳算法对应的超参数
        # 最佳的初始化超参数
        best_hpy_from_history = self.get_init_hpy(dataset, alg_name)

        # 处理MLP中'mlp:validation_fraction' 为空的情况
        if alg_name == 'mlp' and np.isnan(best_hpy_from_history['mlp:validation_fraction']):
            best_hpy_from_history['mlp:validation_fraction'] = 0.1
        """
        alg_name: 3.4. Automatic algorithm selection 选出来的算法名称
        alg_important_hpy: 3.5.1. Potential priority 选出来的重要超参数
        best_hpy_from_history: 3.5.2. Fast-forward initialization 初始化的超参数
        """
        return alg_name, alg_important_hpy, best_hpy_from_history


if __name__ == '__main__':
    ac = AutoCASHUtil()
    acf = ac.init_auto_cash(dataset="bank-marketing", fold_index=0)
    print(acf)

    # print(ac.get_important_fea("bank-marketing", "extra_trees"))
