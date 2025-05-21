#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/3/3 17:31
# @Author  : gsunwu@163.com
# @File    : 02_merge_sample_rate.py
# @Description:
import pandas as pd

from pylibs.utils.util_pandas import PDUtil

df = pd.read_csv("best_sample_rate_ori.csv")
res = pd.pivot_table(df,
                     index=["dataset_name", "data_id"],
                     values=['best_sample_rate'],
                     aggfunc=['mean'])
res=res.reset_index()
PDUtil.save_to_csv(res,"best_sample_rate.csv",home="./",index=False)
