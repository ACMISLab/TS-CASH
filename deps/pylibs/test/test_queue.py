#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/2/4 14:52
# @Author  : gsunwu@163.com
# @File    : test_queue.py
# @Description:
import os
import unittest

from pylibs.utils.util_bash import BashUtil
from pylibs.utils.util_log import get_logger
from pylibs.utils.util_queue import RedisQueue

log = get_logger()


class TestDatasetLoader(unittest.TestCase):

    def test_demo(self):
        rq = RedisQueue(queue_name="dq")
        rq.produce("111")
        rq.produce("222")
        r1 = rq.consume_one_item()
        print(r1)
        assert r1 == "111"
        assert rq.consume_one_item() == "222"
        assert rq.consume_one_item() is None

    def test_get_res(self):
        rq = RedisQueue(queue_name="demo_11")
        rq.produce("111")
        rq.produce("222")
        for i in range(10):
            print(f"==>{i}",rq.get_all())
        assert rq.is_exist("111") is True
        assert rq.is_exist("234234234") is False

    def test_get_res3(self):
        rq = RedisQueue(queue_name="demo_222")
        rq.produce({"a":"werwer","b":"323"})
        rq.produce({"a":"werwer","b":"323"})
        for i in range(10):
            print(f"==>{i}",rq.get_all())
        assert rq.is_exist({"a":"werwer","b":"323"}) is True
        assert rq.is_exist({"a":"werwer","b":"323444444"}) is False
    def test_run_cmds(self):
        BashUtil.run_command_print_progress(f"nnictl create --config /Users/sunwu/SW-Research/p2/src/exps/debug/debug.yaml --port 9980 -f")
        print("finished")
    def test_res(self):
        file=os.path.join(os.sep.join(os.path.dirname(__file__).split(os.sep)[:4]),"nni_experiments")

        print(file)