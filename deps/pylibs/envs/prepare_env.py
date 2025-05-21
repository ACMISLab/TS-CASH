#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/2/29 16:32
# @Author  : gsunwu@163.com
# @File    : prepare_env.py
# @Description:
import time

from pylibs.utils.util_servers import Servers

# your_server_ip:24697
servers = [
    Servers.S100_9,
    Servers.S164,
    Servers.S219,
    Servers.S220,

]

for server in servers:
    # 上传tailscale 文件代理 启动tailscale
    ssh = server.get_ssh()
    ssh.exec("pip install -i https://pypi.tuna.tsinghua.edu.cn/simple qiniu aligo")
