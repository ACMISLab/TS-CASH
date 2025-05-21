#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/2/27 11:56
# @Author  : gsunwu@163.com
# @File    : util_jupyter.py
# @Description:
class UtilJupyter:
    @staticmethod
    def get_file_path():
        import os
        from IPython.core.getipython import get_ipython
        # 获取当前notebook的绝对路径
        notebook_path = os.path.abspath(get_ipython().starting_dir)
        return notebook_path