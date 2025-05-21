#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/10/25 07:31
# @Author  : gsunwu@163.com
# @File    : SP.py
# @Description:
"""
它的工作原理是根据特征的重要性评分选择前 50% 的特征

"""
from sklearn.feature_selection import SelectPercentile
from tshpo.feature_selection.api import FS
from tshpo.lib_class import ExpConf


class SP(FS):
    def transform(self, X_train, y_train, X_test, y_test, econf: ExpConf):
        per = econf.feature_selec_rate * 100
        # 最少保留1%
        per = max(per, 0.01)
        clf = SelectPercentile(percentile=per)
        clf.fit(X_train, y_train)
        X_train_new = clf.transform(X_train)
        X_test_new = clf.transform(X_test)
        return X_train_new, y_train, X_test_new, y_test


if __name__ == '__main__':
    from automl_libs import load_dataset_at_fold

    econf = ExpConf(feature_selec_rate=1)

    X_train, y_train, X_test, y_test = load_dataset_at_fold(dataset_name=econf.dataset, n_fold=10,
                                                            fold_index=0, seed=42)

    _X_train, _y_train, X_test, y_test = SP().transform(X_train, y_train, X_test, y_test, econf)
    assert _X_train.shape[1] == _X_train.shape[1]
    print(X_train.shape, _X_train.shape)

    econf = ExpConf(feature_selec_rate=0.1)

    _X_train1, _y_train1, X_test1, y_test1 = SP().transform(X_train, y_train, X_test, y_test, econf)
    assert _X_train1.shape[1] < _X_train.shape[1]
    print(X_train.shape, _X_train1.shape)
