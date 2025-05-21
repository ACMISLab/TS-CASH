#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/1/25 09:17
# @Author  : gsunwu@163.com
# @File    : upload_12.py
# @Description:
from pylibs.utils.util_servers import Servers

servers = [
    # Servers.S100_9,
    Servers.S164,
    # Servers.S219,
    # Servers.S220,
    # Servers.S215,
    # Servers.DASK_SCHEDULER

]
for server in servers:
    server.prepare_dataset()
