#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/11/1 18:42
# @Author  : gsunwu@163.com
# @File    : debug.py
# @Description:
import os.path

from pylibs.utils.util_rsync import Rsync
from pylibs.utils.util_servers import Servers
from tshpo.lib_class import AnaHelper

Rsync.upload_file(Servers.S219, local_file=os.path.abspath("debug.py"))
AnaHelper.prepare_csv("debug.py")
