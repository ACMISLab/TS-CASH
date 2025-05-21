#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/4/22 09:02
# @Author  : gsunwu@163.com
# @File    : black_box_function.py
# @Description:
import numpy as np
def black_function_1(x):
    return np.exp(-(x - 2) ** 2) + np.exp(-(x - 6) ** 2 / 10) + 1/ (x ** 2 + 1)

