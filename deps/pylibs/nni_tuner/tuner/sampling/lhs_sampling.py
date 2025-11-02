import random
import numpy as np
import pandas as pd

from tuner.sampling.base_sampling import BaseSampling


class LHSSampling(BaseSampling):
    def __init__(self, sample_size: int, search_space: dict, middle=False, seed=0):
        """
        拉丁超立方体LHS抽样算法
        Parameters
        ----------
        n_sample：int
            样本的数量
        cur_search_space_：dict
            参数空间，格式参考 https://nni.readthedocs.io/en/stable/hpo/search_space.html
            如：
            constraint = {
                    'latent_dim': {'_type': 'quniform', '_value': [2, 20, 1]},
                    'res': {'_type': 'quniform', '_value': [3, 8, 1]},
                    "ho":{"_type":"choice","_value":["s","r","p"]}
                }

        middle: bool
            是否取中间的数
        seed:int
            随机种子
        """
        super().__init__(search_space, sample_size, seed=seed)
        self.sample_size = int(sample_size)
        self.middle = middle

    def get_samples(self):
        """
        Generate samples.
        Examples
        -------

        Returns
        -------
        pandas.DataFrame
                    Like:
                       Fx        Fy         Fz
                    0  6.381344  42.681766  1805.250611
                    1  7.212056  81.165833  1954.240508
                    2  4.919377  83.294901  1075.293490
                    3  3.776898  95.213013  1467.894749
                    4  0.453047  14.260119  2418.262023
                    5  5.845662  53.932508  2804.997941
                    6  9.702729  24.696960  2485.764319
                    7  2.661000  63.913773  1267.641317
                    8  8.287838  28.919920   397.066531
                    9  1.341232  66.298533   608.920244
        """
        range_search_space = super().get_range_search_space()

        names = []
        constraints = []
        for key in range_search_space.keys():
            names.append(key)
            _val = range_search_space[key]["_value"]
            constraints.append([_val[0], _val[1]])
        res = self._parameter_array(limit_array=constraints, num_samples=self.sample_size)
        return pd.DataFrame(res, columns=names)

    def _representative(self, partition_range):
        """
        计算单个随机代表数

        Parameters
        ----------
        partition_range
            一个shape为 (m,N,2) 的三维矩阵，m为变量个数、n为样本个数、2代表区间上下限的两列
            partition_range[0]: 第一个参数的上下限数组，如：
            10.00000,9.00000 （第一个区间上限(10)下限(9)）
            9.00000,8.00000  （第二个区间）
            8.00000,7.00000
            7.00000,6.00000
            6.00000,5.00000
            5.00000,4.00000
            4.00000,3.00000
            3.00000,2.00000
            2.00000,1.00000
            1.00000,0.00000 （第 10个区间）

        Returns
        -------
        返回由各变量分区后区间随机代表数组成的矩阵，每列代表一个变量
        """
        # 获得变量个数
        number_of_value = partition_range.shape[0]
        # 获得区间/分层个数
        numbers_of_row = partition_range.shape[1]
        # 创建随机系数矩阵
        coefficient_random = np.zeros((number_of_value, numbers_of_row, 2))
        representative_random = np.zeros((numbers_of_row, number_of_value))

        if self.middle is True:
            # 计算每个变量各区间内的随机代表数，行数为样本个数n，列数为变量个数m
            for j in range(number_of_value):
                temp_random = (partition_range[j, :, 0] + partition_range[j, :, 1]) / 2
                representative_random[:, j] = temp_random
        else:
            for m in range(number_of_value):
                for i in range(numbers_of_row):
                    y = random.random()
                    # 第M个参数的第i个参数的下限
                    coefficient_random[m, i, 0] = 1 - y
                    # 第M个参数的第i个参数的上限
                    coefficient_random[m, i, 1] = y
            # 利用*乘实现公式计算（对应位置进行乘积计算），计算结果保存于临时矩阵 temp_arr 中
            temp_arr = partition_range * coefficient_random
            # 计算每个变量各区间内的随机代表数，行数为样本个数n，列数为变量个数m
            for j in range(number_of_value):
                temp_random = temp_arr[j, :, 0] + temp_arr[j, :, 1]
                representative_random[:, j] = temp_random
        return representative_random

    def _partition_lower_and_upper(self, number_of_sample, limit_array):
        """
        为各变量的变量区间按样本数量进行划分，返回划分后的各变量区间矩阵
        :param number_of_sample: 需要输出的 样本数量
        :param limit_array: 所有变量范围组成的矩阵,为(m, 2)矩阵，m为变量个数，2代表上限和下限
        :return: 返回划分后的个变量区间矩阵（三维矩阵），三维矩阵每层对应于1个变量
        Parameters
        ----------
        number_of_sample
        limit_array

        Returns
        -------
        ndarray
            partition_range[0]: 第一个变量的上下限
            partition_range[1]: 第二个变量的上下限

        """
        coefficient_lower = np.zeros((number_of_sample, 2))
        coefficient_upper = np.zeros((number_of_sample, 2))
        for i in range(number_of_sample):
            coefficient_lower[i, 0] = 1 - i / number_of_sample
            coefficient_lower[i, 1] = i / number_of_sample
        for i in range(number_of_sample):
            coefficient_upper[i, 0] = 1 - (i + 1) / number_of_sample
            coefficient_upper[i, 1] = (i + 1) / number_of_sample

        partition_lower = coefficient_lower @ limit_array.T  # 变量区间下限
        partition_upper = coefficient_upper @ limit_array.T  # 变量区间上限

        partition_range = np.dstack((partition_lower.T, partition_upper.T))  # 得到各变量的区间划分，三维矩阵每层对应于1个变量

        return partition_range  # 返回区间划分上下限

    def _rearrange(self, arr_random):
        """
        打乱矩阵各列内的数据
        Parameters
        ----------
        arr_random 一个N行, m列的矩阵

        Returns
        -------
        matrix
            每列打乱后的矩阵

        """
        for i in range(arr_random.shape[1]):
            np.random.shuffle(arr_random[:, i])
        return arr_random

    def _parameter_array(self, limit_array, num_samples):
        """
        根据输入的各变量的范围矩阵以及希望得到的样本数量，输出样本参数矩阵

        Parameters
        ----------
        limit_array: list
            下限矩阵，shape为(m,2),m为变量个数
        num_samples:int
            样本的数量

        Returns
        -------
        list

        """
        if type(limit_array) == list:
            limit_array = np.asarray(limit_array)
        arr = self._partition_lower_and_upper(num_samples, limit_array)
        parameters_matrix = self._rearrange(self._representative(arr))
        return parameters_matrix
