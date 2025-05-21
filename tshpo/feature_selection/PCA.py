#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/10/25 07:31
# @Author  : gsunwu@163.com
# @File    : PCA.py
# @Description:
"""
它的工作原理是根据特征的重要性评分选择前 50% 的特征

"""
from sklearn.feature_selection import SelectPercentile

from tshpo.feature_selection.api import FS
from tshpo.lib_class import ExpConf

from sklearn.decomposition import PCA as PCA_


class PCA(FS):
    def transform(self, X_train, y_train, X_test, y_test, econf: ExpConf):
        k = self.check_n_feature(X_train.shape[1] * econf.feature_selec_rate)
        pca = PCA_(n_components=k)
        X_train_trans = pca.fit_transform(X_train)
        X_test_trans = pca.transform(X_test)
        return X_train_trans, y_train, X_test_trans, y_test


if __name__ == '__main__':
    from automl_libs import load_dataset_at_fold

    econf = ExpConf(feature_selec_rate=1)

    X_train, y_train, X_test, y_test = load_dataset_at_fold(dataset_name=econf.dataset, n_fold=10,
                                                            fold_index=0, seed=42)

    _X_train, _y_train, X_test, y_test = PCA().transform(X_train, y_train, X_test, y_test, econf)
    assert _X_train.shape[1] == _X_train.shape[1]
    print(X_train.shape, _X_train.shape)

    econf = ExpConf(feature_selec_rate=0.1)

    _X_train1, _y_train1, X_test1, y_test1 = PCA().transform(X_train, y_train, X_test, y_test, econf)
    assert _X_train1.shape[1] < _X_train.shape[1]
    print(X_train.shape, _X_train1.shape)
