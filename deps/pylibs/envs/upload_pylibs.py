#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/1/25 09:17
# @Author  : gsunwu@163.com
# @File    : upload_12.py
# @Description:
from pathlib import Path
from pylibs.utils.util_servers import Servers
servers = [
    Servers.S100_9,
]
for server in servers:
    server.upload_dir(Path("/Users/sunwu/SW-Research/pylibs"), Path("/remote-home/cs_acmis_sunwu/pylibs"))
