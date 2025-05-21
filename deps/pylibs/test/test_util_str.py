#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/1/20 09:47
# @Author  : gsunwu@163.com
# @File    : test_util_str.py
# @Description:

import unittest

from pylibs.utils.util_log import get_logger
from pylibs.utils.util_str import *

log = get_logger()


class TestDatasetLoader(unittest.TestCase):

    def test_demo(self):
        s = "这是1个示例字符串! 包含i数字123和lov符号#*。\ne\n\nyou"
        r = get_str_letters(s)

        self.assertEqual(r, "iloveyou")
        print("====+")
        print(s, "\n==>", r)

    def test_demo2(self):
        # 测试函数
        list1 = ["apple", "banana", "cherry"]
        list2 = ["banana", "kiwi", "apple"]
        print(list1)
        print()


if __name__ == '__main__':
    unittest.main()
