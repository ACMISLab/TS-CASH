#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/10/25 07:35
# @Author  : gsunwu@163.com
# @File    : FastICA.py
# @Description:
"""
它基于统计学中的独立性假设，旨在找到一组彼此统计独立的非高斯信号
"""
from sklearn.decomposition import FastICA as FastICA_
from tshpo.feature_selection.api import FS
from tshpo.lib_class import ExpConf


class FastICA(FS):
    def transform(self, X_train, y_train, X_test, y_test, econf: ExpConf):
        k = self.check_n_feature(X_train.shape[1] * econf.feature_selec_rate)
        lda = FastICA_(n_components=k)  # 选择降维到2维
        # 3. 拟合模型并转换数据
        lda.fit(X_train)
        X_train_trans = lda.fit_transform(X_train)
        X_test_trans = lda.fit_transform(X_test)
        return X_train_trans, y_train, X_test_trans, y_test


if __name__ == '__main__':
    from tshpo.automl_libs import load_dataset_at_fold, TSHPOFramework

    econf = ExpConf(feature_selec_rate=1)

    X_train, y_train, X_test, y_test = load_dataset_at_fold(dataset_name=econf.dataset, n_fold=10,
                                                            fold_index=0, seed=42)

    _X_train, _y_train, X_test, y_test = FastICA().transform(X_train, y_train, X_test, y_test, econf)
    assert _X_train.shape[1] == _X_train.shape[1]
    print(X_train.shape, _X_train.shape)

    econf = ExpConf(feature_selec_rate=0.1)

    _X_train1, _y_train1, X_test1, y_test1 = FastICA().transform(X_train, y_train, X_test, y_test, econf)
    assert _X_train1.shape[1] < _X_train.shape[1]
    print(X_train.shape, _X_train1.shape)
