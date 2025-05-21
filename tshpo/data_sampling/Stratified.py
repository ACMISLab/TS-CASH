#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/10/25 07:35
# @Author  : gsunwu@163.com
# @File    : FastICA.py
# @Description:
"""
"""
import random

import numpy as np
from tshpo.data_sampling.api import DS
from tshpo.automl_libs import TSHPOFramework
from tshpo.lib_class import ExpConf


class Stratified(DS):
    def transform(self, X_train, y_train, X_test, y_test, econf: ExpConf):
        random.seed(econf.random_state)
        np.random.seed(econf.random_state)
        k = self.check_n_samples(X_train.shape[0] * econf.data_sample_rate)
        _n_ss_sample = int(k / 2)
        X_train_neg_index = np.where(y_train == 0)[0]
        X_train_pos_index = np.where(y_train == 1)[0]

        _n_neg_sample = np.min([len(X_train_neg_index), _n_ss_sample])
        _n_pos_sample = np.min([len(X_train_pos_index), _n_ss_sample])

        sampled_x_neg = \
            X_train_neg_index[np.random.choice(len(X_train_neg_index), size=_n_neg_sample, replace=False)]

        sampled_x_pos = \
            X_train_pos_index[np.random.choice(len(X_train_pos_index), size=_n_pos_sample, replace=True)]

        # 合并两个向量
        ret_index = np.concatenate((sampled_x_neg, sampled_x_pos))

        # 打乱合并后的向量s
        np.random.shuffle(ret_index)

        # 抽样
        X_train_sampled, y_train_sampled = X_train[ret_index], y_train[ret_index]
        # assert y_train_sampled.mean() == 0.5, "分层抽样结果不正确"

        # 是否要降维, 最多允许3000个样本
        return X_train_sampled[:TSHPOFramework.max_n_rows], y_train_sampled[:TSHPOFramework.max_n_rows], X_test, y_test
