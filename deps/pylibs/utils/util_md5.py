#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/8/8 09:56
# @Author  : gsunwu@163.com
# @File    : util_md5.py
# @Description:
import hashlib
import warnings
warnings.filterwarnings("ignore")

def get_str_md5(input_str:str):
    """
    范围给定字符串的md5

    Parameters
    ----------
    input_str :

    Returns
    -------

    """
    md5 = hashlib.md5(input_str.encode("utf8"))
    return md5.hexdigest()

if __name__ == '__main__':
    print(get_str_md5("111"))
    print(get_str_md5("222"))
    assert get_str_md5("111")==get_str_md5("111")
    assert get_str_md5("111")!=get_str_md5("222")