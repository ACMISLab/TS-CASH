#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/10/25 07:36
# @Author  : gsunwu@163.com
# @File    : api.py
# @Description:
import abc
import dataclasses
from ConfigSpace import ConfigurationSpace
from tshpo.lib_class import ExpConf
from tshpo.automl_libs import TrainingHistory
from smac.runhistory import TrialInfo, TrialValue


@dataclasses.dataclass
class HPO(metaclass=abc.ABCMeta):
    history: TrainingHistory
    cs: ConfigurationSpace
    econf: ExpConf

    @abc.abstractmethod
    def ask(self):
        """
        Get the training data without labels

        Returns
        -------

        """

        pass

    @abc.abstractmethod
    def tell(self, info: TrialInfo, value: TrialValue, save: bool = True):
        """
        Get the training data without labels

        Returns
        -------

        """

        pass
