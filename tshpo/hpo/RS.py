#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/10/27 10:23
# @Author  : gsunwu@163.com
# @File    : RS.py
# @Description:
import dataclasses
from smac.runhistory import TrialInfo, TrialValue
from tshpo.hpo.api import HPO


@dataclasses.dataclass
class RS(HPO):
    counter: int = -1
    def __post_init__(self):
        self.counter = -1
        self.configs = self.cs.sample_configuration(size=10000)

    def tell(self, info: TrialInfo, value: TrialValue, save: bool = True) -> None:
        pass

    def ask(self):
        self.counter += 1
        return TrialInfo(config=self.configs[self.counter])
