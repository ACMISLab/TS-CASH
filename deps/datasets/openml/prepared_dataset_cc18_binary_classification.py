"""
requests 必须为下面版本:
pip install requests==2.28.2
否者会报错:
ImportError: cannot import name 'DEFAULT_CIPHERS' from 'urllib3.util.ssl_' (/Users/sunwu/miniforge3/envs/autosklearn/lib/python3.9/site-packages/urllib3/util/ssl_.py)
"""
import json
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

    """
    task = openml.tasks.get_task(task_id, download_qualities=False)
    _, original_task_folds, _ = task.get_split_dimensions()
    dl = OpenmlLoader(api_key=api_key)
    dataset = dl.load(task_id=task_id)
    dataset_name = dataset._oml_dataset.name
    return dataset_name


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

    id_name_maps = {}
    for task_id in task_ids:
        id_name_maps[task_id] = task_id_2_data_names(task_id)
        for fold_idex in range(10):
            load_dataset_by_task_id(task_id=task_id, fold_index=fold_idex)
    with open("binary_classification_maps.json", "w") as f:
        json.dump(id_name_maps, f, indent=3)
