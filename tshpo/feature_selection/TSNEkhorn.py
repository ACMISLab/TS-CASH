#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2025/5/9 10:51
# @Author  : gsunwu@163.com
# @File    : kernelPCA.py
# @Description:
"""
pip install torchdr
TSNEkhorn
"""
import numpy as np
from torchdr import TSNEkhorn as TN

from tshpo.feature_selection.api import FS
from tshpo.lib_class import ExpConf


class TSNEkhorn(FS):
    def transform(self, X_train, y_train, X_test, y_test, econf: ExpConf):
        # err: ValueError: [TorchDR] ERROR Affinity: NaN at iter 92, consider decreasing the learning rate.

        train_len = X_train.shape[0]
        k = self.check_n_feature(X_train.shape[1] * econf.feature_selec_rate)
        tran_data: np.ndarray = np.concatenate([X_train, X_test])
        tran_data = np.nan_to_num(tran_data, nan=0)

        # model
        clf = TN(n_components=k, lr=0.0001)
        X_train_trans = clf.fit_transform(tran_data)
        X_train_new = X_train_trans[:train_len]
        X_test_new = X_train_trans[train_len:]
        return X_train_new, y_train, X_test_new, y_test
