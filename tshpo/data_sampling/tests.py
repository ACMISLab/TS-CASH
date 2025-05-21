#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/10/25 08:31
# @Author  : gsunwu@163.com
# @File    : tests.py
# @Description:
from __init__ import get_data_sampling_method

if __name__ == '__main__':
    from lib_class import ExpConf
    from automl_libs import load_dataset_at_fold

    econf = ExpConf(feature_selec_rate=1)
    X_train, y_train, X_test, y_test = load_dataset_at_fold(dataset_name=econf.dataset, n_fold=10,
                                                            fold_index=0, seed=42)

    for _fpm in ["LHS", "RS", "Sobol", 'Halton', 'Stratified']:
        print(f"============\n{_fpm}")
        model = get_data_sampling_method(_fpm)

        econf = ExpConf(data_sample_rate=1)
        _X_train, _y_train, _xtest, _ = model.transform(X_train, y_train, X_test, y_test, econf)
        assert _X_train.shape[1] == X_train.shape[1]
        print(_X_train.shape, _xtest.shape)

        econf = ExpConf(data_sample_rate=0.1)
        _X_train, _y_train, _xtest, _ = model.transform(X_train, y_train, X_test, y_test, econf)
        assert _X_train.shape[1] <= X_train.shape[1]
        print(_X_train.shape, _xtest.shape)
