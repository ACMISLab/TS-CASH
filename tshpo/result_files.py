#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/10/28 08:18
# @Author  : gsunwu@163.com
# @File    : results_files.py
# @Description:
import dataclasses
import os.path

import yaml


@dataclasses.dataclass
class ResultFiles:
    baseline_file: str = None
    model_select: str = None

    @staticmethod
    def load_files():
        _f = os.path.join(os.path.dirname(__file__), "result_files.yaml")
        with open(_f, 'r') as f:
            cf = yaml.safe_load(f)
        return ResultFiles(baseline_file=cf['baseline']['file'])

    @staticmethod
    def get_baseline_file():
        """
        获取基线文件

        Returns
        -------

        """
        return ResultFiles.load_files().baseline_file


if __name__ == '__main__':
    print(ResultFiles.get_baseline_file())
