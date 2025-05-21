"""
目标:
提取OpemML中的特征, 为训练Auto-CASH做准备
提取到的特征存在文件 auto_cash_fea.csv 中

requests 必须为下面版本:
pip install requests==2.28.2
否者会报错:
ImportError: cannot import name 'DEFAULT_CIPHERS' from 'urllib3.util.ssl_' (/Users/sunwu/miniforge3/envs/autosklearn/lib/python3.9/site-packages/urllib3/util/ssl_.py)

AttributeError: `np.sctypes` was removed in the NumPy 2.0 release. Access dtypes explicitly instead.
"""
import json
import sys

import pandas as pd
from tqdm import tqdm

from datasets.openml.meta_helper import MetaAutoCASH
from tshpo.automl_libs import load_dataset_at_fold

sys.path.append("/Users/sunwu/SW-Research/AutoML-Benchmark/deps")
import os.path
import warnings
import numpy as np
import openml
from pathlib import Path
from amlb.datasets import OpenmlLoader

warnings.filterwarnings('ignore')

api_key = "c1994bdb7ecb3c6f3c8f3b35f4b47f1f"


def task_id_2_data_names(task_id):
    """
    task_id 是一个数字

    """
    task = openml.tasks.get_task(task_id, download_qualities=False)
    _, original_task_folds, _ = task.get_split_dimensions()
    dl = OpenmlLoader(api_key=api_key)
    dataset = dl.load(task_id=task_id)
    dataset_name = dataset._oml_dataset.name
    return dataset_name


def taskid2datafeatures(task_id):
    """
    task_id 是一个数字
    返回的特征中包括了目标类

    """
    task = openml.tasks.get_task(task_id, download_qualities=False)
    _, original_task_folds, _ = task.get_split_dimensions()
    dl = OpenmlLoader(api_key=api_key)
    dataset = dl.load(task_id=task_id)
    return dataset.features


def load_dataset_by_task_id(task_id, fold_index=0, cache_dir=".cache"):
    try:
        print(f"Parse task task_id: {task_id}, fold_index: {fold_idex}")
        task = openml.tasks.get_task(task_id, download_qualities=False)
        _, original_task_folds, _ = task.get_split_dimensions()
        dl = OpenmlLoader(api_key=api_key, cache_dir=cache_dir)
        dataset = dl.load(task_id=task_id, fold=fold_index)
        dataset_name = dataset._oml_dataset.name
        assert fold_index < original_task_folds, f"fold_index is out of range total folds: {original_task_folds} for dataset {dataset_name}"
        home = Path(dataset_name)
        if not home.exists():
            home.mkdir(parents=True)
        npz_file_name = os.path.abspath(f"{home}/{dataset_name}_fold_{fold_index}.npz")
        if os.path.exists(npz_file_name):
            print(f"File {npz_file_name} is already exists, skipping")
            return
        # 找出分类变量和连续变量
        cat_features = []
        num_features = []
        for _f in dataset.features:
            if _f.is_target == False:
                if _f.data_type == 'category':
                    cat_features.append(_f.name)
                elif _f.data_type == "number":
                    num_features.append(_f.name)
                    pass
                else:
                    raise RuntimeError(f"Unsupported data type: {_f.data_type}")

        # 异常值填充：用每一列的均值进行填充
        X = np.concatenate([dataset.train.X_enc, dataset.test.X_enc])
        mean_values = np.nan_to_num(np.nanmean(X, axis=0), nan=0.0)
        std_values = np.nan_to_num(np.nanstd(X, axis=0), nan=0.0)
        std_values = std_values + 1e-8

        # 异常值填充 -> 标准化每一列 -> 存储为.npz (文件名： 数据集名称+fold_id ）
        X_train = (np.where(np.isnan(dataset.train.X_enc), mean_values, dataset.train.X_enc) - mean_values) / std_values
        X_test = (np.where(np.isnan(dataset.test.X_enc), mean_values, dataset.test.X_enc) - mean_values) / std_values
        y_train = dataset.train.y_enc
        y_test = dataset.test.y_enc

        assert np.isnan(X_train).any() == False
        assert np.isnan(X_test).any() == False
        assert np.isinf(X_train).any() == False
        assert np.isinf(X_test).any() == False
        print(f"File is saved to {npz_file_name}")
        np.savez(npz_file_name, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

        # print(dataset.train.X_enc)
        # print(dataset.train.y_enc)
        # print(dataset.test.X_enc)
        # print(dataset.test.y_enc)
        # print(dataset)
    except Exception as e:
        print(f"❌❌❌Error at task_id={task_id}, since {e}")


from collections import Counter


def number_of_numeric_attributes(features):
    """Number of numeric attributes."""
    counter = 0
    for f in features:
        if f.data_type == 'number':
            counter += 1
    return counter


def number_of_features(features):
    """Number of numeric attributes."""
    counter = 0
    for f in features:
        if not f.is_target:
            counter += 1
    return counter


def max_class_proportion(target_list):
    """Maximum proportion of single class"""
    # 统计每个类别出现的次数
    count = Counter(target_list)

    # 计算每个类别的比例
    total = len(target_list)
    proportions = {num: freq / total for num, freq in count.items()}

    # 找到最大比例
    max_proportion = max(proportions.values())

    return max_proportion


def feature_list(data_feataures):
    """获取数据集中y标签的所有类别列表[nowin, won]
    """
    for f in data_feataures:
        if f.is_target:
            return list(f.label_encoder.classes)
    raise RuntimeError("Cant find")


def class_number_in_cat_attr_with_least_class(data_feataures):
    """
    Class number in category attribute with the least classes.
    """

    class_counts = {}

    for feature in data_feataures:
        if feature.data_type == 'category':
            class_counts[feature.name] = len(feature.label_encoder.classes)
    return np.min(list(class_counts.values()))


def class_number_in_cat_attr_with_most_class(data_feataures):
    """
    Class number in category attribute with the most classes.
    """

    class_counts = {}

    for feature in data_feataures:
        if feature.data_type == 'category':
            class_counts[feature.name] = len(feature.label_encoder.classes)
    return np.max(list(class_counts.values()))


if __name__ == '__main__':
    print("Starting ...")
    task_ids = [
        3,
        15,
        29,
        31,
        37,
        3021,
        43,
        49,
        219,
        3902,
        3903,
        3904,
        3913,
        3917,
        3918,
        14965,
        10093,
        10101,
        # 14970, # multiply labels
        9971,
        9976,
        9977,
        9978,
        9952,
        9957,
        9946,
        # 7592, # only contain one fold
        9910,
        14952,
        14954,
        125920,
        167120,
        167141,
        167125,
        146820,
        146819
    ]

    auto_cash_feas = []
    for task_id in tqdm(task_ids):
        data_name = task_id_2_data_names(task_id)
        data_feataures = taskid2datafeatures(task_id)
        # Number of numeric attributes.
        num_numeric_attributes = number_of_numeric_attributes(data_feataures)

        # Number of numeric attributes
        num_features = number_of_features(data_feataures)  # 返回的特征中不包括目标特征

        # 所有的类别数
        label_list = feature_list(data_feataures)
        for fold_index in range(5):
            X_train, y_train, X_test, y_test = load_dataset_at_fold(dataset_name=data_name, n_fold=fold_index)
            fea = MetaAutoCASH(
                id=str(data_name) + str(fold_index),
                dataid=task_id,
                data_name=data_name,
                mf0=len(label_list),  # 多少个类别
                mf2=max_class_proportion(list(y_train) + list(y_test)),
                mf4=num_numeric_attributes,
                mf6=num_numeric_attributes / num_features,
                mf7=num_features,
                mf9=class_number_in_cat_attr_with_least_class(data_feataures),
                mf13=class_number_in_cat_attr_with_most_class(data_feataures)
            )
            auto_cash_feas.append(fea.model_dump())
    pd.DataFrame(auto_cash_feas).to_csv("auto_cash_data_meta_fea.csv")
