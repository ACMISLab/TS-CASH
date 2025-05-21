#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/4/14 00:18
# @Author  : gsunwu@163.com
# @File    : select_dataset.py
# @Description:
import rich

from pylibs.uts_dataset.dataset_loader import UTSDataset
datasets = UTSDataset.select_datasets_split(
            dataset_names=["YAHOO","NAB","SMD"],
            top_n=10,
            test_ratio=0.4,
            seed=42
        )
rich.print(datasets)