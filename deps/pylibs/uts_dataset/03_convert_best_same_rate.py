#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/4/14 00:36
# @Author  : gsunwu@163.com
# @File    : 03_convert_best_same_rate.py
# @Description:
import pandas as pd
import rich

df=pd.read_csv("best_sample_rate.csv")
# dataset_name,data_id,repeat_run,best_sample_rate
df=df.loc[:,["dataset_name","data_id","best_sample_rate"]]
rich.print(df.values)