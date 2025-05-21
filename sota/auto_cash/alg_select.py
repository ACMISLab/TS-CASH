"""
使用从baseline中学到的算法来训练一个meta-learning. 训练的meta-learner是一个RF算法
"""
import json
import logging
import os.path
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sota.auto_cash.auto_cash_helper import AuthCashHelper

DATA_HOME = "/Users/sunwu/SW-Research/AutoML-Benchmark/sota/auto_cash"


def get_best_alg_by_datasetname(dataset_name="churn", fold_index=0):
    """通过Meta-learn 获取给定数据集的预测到的最好的算法
    输入: 数据集
    输出: 表现最好的算法名称,  通过 meta-learner 学到的. Meta-learner 是一个随机深林RF

    """

    current_dataset_meta_normalize, normalize_train_data, train_label = \
        meta_features_preprocessing(dataset_name, fold_index=fold_index)

    # build model
    rf = RandomForestClassifier()
    rf.fit(normalize_train_data, train_label)
    # 训练每个数据集的时候, 数据集中不能包括这个数据集的信息
    res = rf.predict(current_dataset_meta_normalize)

    return AuthCashHelper.get_model_name_by_model_id(res[0])


def meta_features_preprocessing(dataset_name="churn", fold_index=0):
    """处理元数据特征
    返回:
    current_dataset_meta_normalize: 当前数据集的特征, 标准化之后的
    normalize_train_data: 标准化之后的训练数据集
    train_label: 训练数据集的meta
    """
    remove_dataset_name = dataset_name
    df = pd.read_csv(os.path.join(DATA_HOME, "auto_cash_meta_data.csv"))
    # 将 `best_model` 转换为分类编码
    # ,dataset,model,fscore
    df['label'] = df['model'].astype('category').cat.codes
    # 获取当前数据集的meta
    # current_dataset_meta = df[df['dataset'] == remove_dataset_name]
    current_dataset_meta = df.query(f"dataset=='{dataset_name}' and fold_index=={fold_index}")
    # 移除当前数据的元数据信息, 不能放到训练集里面
    # df = df[df['dataset'] != remove_dataset_name]
    df = df.query(f"dataset!='{dataset_name}'")
    assert remove_dataset_name not in df['dataset'].tolist()
    # 获取算法名称及其对应编号
    # {0: 'adaboost', 1: 'bernoulli_nb', 2: 'decision_tree', 3: 'extra_trees', 4: 'gaussian_nb', 5: 'k_nearest_neighbors', 6: 'lda', 7: 'liblinear_svc', 8: 'libsvm_svc', 9: 'mlp', 10: 'multinomial_nb', 11: 'qda', 12: 'random_forest', 13: 'sgd'}
    algorithm_mapping = dict(enumerate(df['model'].astype('category').cat.categories))
    pickle.dump(algorithm_mapping, open("auto_cash_algorithm_mapping.pkl", "wb"))
    json.dump(algorithm_mapping, open("auto_cash_algorithm_mapping.json", "w"))
    # print(algorithm_mapping)
    # all column: ['dataset', 'model', 'fscore', 'dataid', 'data_name', 'mf0', 'mf2', 'mf4', 'mf6', 'mf7', 'mf9', 'mf13', 'label']
    train_data_col = ['mf0', 'mf2', 'mf4', 'mf6', 'mf7', 'mf9', 'mf13']
    # ydata_col=[['label']]
    # {0: 'adaboost', 1: 'bernoulli_nb', 2: 'decision_tree', 3: 'extra_trees', 4: 'gaussian_nb', 5: 'k_nearest_neighbors', 6: 'lda', 7: 'liblinear_svc', 8: 'libsvm_svc', 9: 'mlp', 10: 'multinomial_nb', 11: 'qda', 12: 'random_forest', 13: 'sgd'}
    train_data = df[train_data_col]
    current_dataset_meta = current_dataset_meta[train_data_col]
    # 用来归一化的
    _mean = train_data.mean()
    _std = train_data.std() + 1e-9
    train_label = df[['label']]
    normalize_train_data = (train_data - _mean) / _std
    current_dataset_meta_normalize = (current_dataset_meta - _mean) / _std
    logging.debug(normalize_train_data)
    return current_dataset_meta_normalize, normalize_train_data, train_label


if __name__ == '__main__':
    dataset_name = 'dresses-sales'
    fold_index = 5
    print(get_best_alg_by_datasetname())
