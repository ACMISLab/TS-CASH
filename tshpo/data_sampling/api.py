#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/10/25 07:36
# @Author  : gsunwu@163.com
# @File    : api.py
# @Description:
import abc
import math
from tshpo.lib_class import ExpConf


class DS(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def transform(self, X_train, y_train, X_test, y_test, econf: ExpConf):
        """
        Get the training data without labels

        Returns
        -------

        """

        pass

    def check_n_samples(self, rsr):
        """
        控制样本数量。
        最多 2000 个样本, 最小300

        Parameters
        ----------
        rsr :

        Returns
        -------

        """
        if rsr >= 2000:
            return 2000
        elif rsr <= 300:
            return 300
        else:
            return int(rsr)
