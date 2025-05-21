#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/10/25 08:31
# @Author  : gsunwu@163.com
# @File    : tests.py
# @Description:
from __init__ import get_feature_select_method
from tshpo.automl_libs import load_dataset_at_fold
from tshpo.lib_class import ExpConf

if __name__ == '__main__':

    econf = ExpConf(feature_selec_rate=1)
    X_train, y_train, X_test, y_test = load_dataset_at_fold(dataset_name=econf.dataset, n_fold=10,
                                                            fold_index=0, seed=42)

    # for _fpm in ["PCA", "MIC", "RF", "SP", "FastICA"]:
    # for _fpm in ["TSNEkhorn"]:
    for _fpm in ["UMAP"]:
        print(f"============\n{_fpm}")
        model = get_feature_select_method(_fpm)

        econf = ExpConf(feature_selec_rate=1)
        _X_train, _y_train, _xtest, _ = model.transform(X_train, y_train, X_test, y_test, econf)
        assert _X_train.shape[1] == X_train.shape[1]
        print(_X_train.shape, _xtest.shape)

        econf = ExpConf(feature_selec_rate=0)
        _X_train, _y_train, _xtest, _ = model.transform(X_train, y_train, X_test, y_test, econf)
        assert _X_train.shape[1] <= X_train.shape[1]
        print(_X_train.shape, _xtest.shape)
