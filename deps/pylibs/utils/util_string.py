#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/9/14 10:35
# @Author  : gsunwu@163.com
# @File    : util_string.py
# @Description:


class UtilString:
    @staticmethod
    def center_str(string):
        """
         return '***********<string>***********'

        Parameters
        ----------
        string :

        Returns
        -------

        """
        return '{:*^30}'.format(string)
