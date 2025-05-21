#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/4/17 16:58
# @Author  : gsunwu@163.com
# @File    : test_dask_client.py
# @Description:
from pylibs.utils.util_servers import Servers

client=Servers.S219.get_dask_client()
print(client.has_what())
print(client.status)
client.close()
print(client.status)