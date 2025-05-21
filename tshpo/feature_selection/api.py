#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/10/25 07:36
# @Author  : gsunwu@163.com
# @File    : api.py
# @Description:
import abc
import math

from tshpo.lib_class import ExpConf


class FS(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def transform(self, X_train, y_train, X_test, y_test, econf: ExpConf):
        """
        Get the training data without labels

        Returns
        -------

        """

        pass

    def check_n_feature(self, fsr):
        """
        保证至少有一个特征，最多30个特征

        Parameters
        ----------
        fsr :

        Returns
        -------

        """
        fsr = math.ceil(fsr)
        if fsr <= 3:
            return 3
        elif fsr >= 30:
            return 30
        else:
            return fsr
