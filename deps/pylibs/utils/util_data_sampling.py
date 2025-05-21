#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/7/17 11:32
# @Author  : gsunwu@163.com
# @File    : util_data_sampling.py
# @Description:
import numpy as np


class UtilDataSampling:

    @staticmethod
    def random_sampling_data(x: np.ndarray, y: np.ndarray, size: float = None):
        """

        Parameters
        ----------
        data :
        size : None or int
            None means return all input samples
            Int means return the number of size samples randomly sampled with output replacement.

        Returns
        -------

        """
        if size is not None or size != 1:
            # 获取数据的行数
            num_rows = x.shape[0]

            # 从行索引中随机抽取10个样本（假设数据行数大于等于10）
            random_indices = np.random.choice(num_rows, size=int(num_rows * size), replace=False)

            # 使用随机索引从原数组中抽取相应的行
            return x[random_indices, :], y[random_indices]
        else:
            return x, y
