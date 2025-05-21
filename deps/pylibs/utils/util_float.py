#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/4/9 16:28
# @Author  : gsunwu@163.com
# @File    : util_float.py
# @Description:

class UtilFloat:
    @classmethod
    def format_float(cls, _v: float):
        """
        将float 填充为6位小数
        Parameters
        ----------
        _v :

        Returns
        -------

        """
        return f"{_v:.6f}"
