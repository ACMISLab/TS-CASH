#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/9/28 08:49
# @Author  : gsunwu@163.com
# @File    : test_rsync.py
# @Description:

import unittest
import logging

from pylibs.utils.util_rsync import Rsync
from pylibs.utils.util_servers import Servers

log = logging.getLogger(__name__)


class TestDatasetLoader(unittest.TestCase):

    def test_demo(self):
        Rsync.upload_file(Servers.S100_9_HOST,local_file="/Users/sunwu/SW-OpenSourceCode/AutoML-Benchmark/deps/pylibs/pylibs/utils/test.pdf",remote_file="/remote-home4/sunwu/exp_results/test_pdf")
