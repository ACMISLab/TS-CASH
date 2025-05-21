#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/10/25 07:35
# @Author  : gsunwu@163.com
# @File    : FastICA.py
# @Description:
import random

import numpy as np

from tshpo.data_sampling.api import DS
from tshpo.automl_libs import TSHPOFramework
from tshpo.lib_class import ExpConf


class RS(DS):
    def transform(self, X_train, y_train, X_test, y_test, econf: ExpConf):
        random.seed(econf.random_state)
        np.random.seed(econf.random_state)
        k = self.check_n_samples(X_train.shape[0] * econf.data_sample_rate)
        train_sample_index = np.random.choice(X_train.shape[0], size=k, replace=True)
        # 抽样
        X_train_sampled, y_train_sampled = X_train[train_sample_index], y_train[train_sample_index]
        # 是否要降维
        return X_train_sampled[:TSHPOFramework.max_n_rows], y_train_sampled[:TSHPOFramework.max_n_rows], X_test, y_test


