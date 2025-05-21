#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/4/6 22:28
# @Author  : gsunwu@163.com
# @File    : util_configure_space.py
# @Description:
import numpy as np


# https://pypi.org/project/ConfigSpace/
def config_space_to_array(x_tries_confs):
    rets = []
    for _conf in x_tries_confs:
        rets.append(_conf.get_array())
    return np.nan_to_num(rets)