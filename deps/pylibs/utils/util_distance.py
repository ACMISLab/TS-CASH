#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/9/12 14:46
# @Author  : gsunwu@163.com
# @File    : euclidean_distance.py
# @Description:
import numpy as np


class EuclideanDistance:
    """
    d(p,q)=|p-q|=sqrt( (p-q)**2) )
    """

    def __init__(self):
        pass

    def score(self, real: np.ndarray, predict: np.ndarray):
        """

        Parameters
        ----------
        real : 1d np.ndarray
        predict : 1d np.ndarray

        Returns
        -------

        """
        score = np.sqrt((real - predict) ** 2)
        return score


if __name__ == '__main__':
    ed = EuclideanDistance()
    real = np.asarray([1, 2, 3])
    predict = np.asarray([0, 0, 3])
    print(ed.score(real, predict))
