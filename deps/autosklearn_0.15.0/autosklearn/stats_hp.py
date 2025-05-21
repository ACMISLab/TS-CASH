#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/9/6 07:11
# @Author  : gsunwu@163.com
# @File    : stats_hp.py
# @Description:
from automl_libs import load_dataset_at_fold, get_auto_sklearn_classification_search_space

X_train, y_train, X_test, y_test = load_dataset_at_fold(dataset_name="credit-g", n_fold=5,
                                                        fold_index=1, seed=1)
cs = get_auto_sklearn_classification_search_space(y_train=y_train, random_state=42)

print(f"算法数量：{len(cs['__choice__'].choices)},算法：{cs['__choice__'].choices}")
print(f"超参数数量：{len(list(cs.get_hyperparameters()))}")