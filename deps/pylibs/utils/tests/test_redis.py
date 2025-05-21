#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/8/28 16:10
# @Author  : gsunwu@163.com
# @File    : test_redis.py
# @Description:

import unittest

from pylibs.config import GCFV3, ServerNameV2
from pylibs.utils.util_log import get_logger
from pylibs.utils.util_redis import RedisUtil

log = get_logger()


class TestDatasetLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.rd = RedisUtil(GCFV3.get_server_conf(ServerNameV2.REDIS_LOCAL, net_type="lan"))

    def test_demo(self):
        self.rd.set("v0120_1", "v0120_1")
        self.rd.set("v0120_2", "v0120_2")
        self.rd.set("v1122_2", "v1122_2")

    def test_get_keys(self):
        print(self.rd.keys("v01*"))
