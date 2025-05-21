#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/1/25 09:17
# @Author  : gsunwu@163.com
# @File    : upload_12.py
# @Description:
from pylibs.utils.util_env import Servers

s = Servers.S164
s.prepare_env()
s.prepare_dask_worker()
