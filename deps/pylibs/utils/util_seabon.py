#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/4/28 16:03
# @Author  : gsunwu@163.com
# @File    : util_seabon.py
# @Description:
import seaborn


class UtilSeabon:

    @staticmethod
    def fix_chinese(sns: seaborn):
        """
        解决画图时中文不显示的问题

        Parameters
        ----------
        sns :

        Returns
        -------

        """
        from matplotlib.font_manager import FontProperties
        myfont = FontProperties(fname=r"/System/Library/Fonts/PingFang.ttc")
        sns.set(font=myfont.get_name())
