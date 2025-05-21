#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/2/27 13:41
# @Author  : gsunwu@163.com
# @File    : util_ust_test.py
# @Description:
import numpy as np

from pylibs.uts_dataset.dataset_loader import KFDL

import unittest
import logging

log = logging.getLogger(__name__)


class TestDatasetLoader(unittest.TestCase):

    def test_kfdl(self):
        sample_rate = 2
        dl = KFDL(dataset_name="YAHOO",
                  data_id="Yahoo_A1real_24_data.out",
                  sample_rate=sample_rate
                  )
        train_x, train_y, test_x, test_y = dl.load_train_and_test()
        assert train_x.shape[0] == sample_rate
        assert train_y.shape[0] == sample_rate

        sample_rate = 10
        dl = KFDL(dataset_name="YAHOO",
                  data_id="Yahoo_A1real_24_data.out",
                  sample_rate=sample_rate
                  )
        train_x, train_y, test_x, test_y = dl.load_train_and_test()
        assert train_x.shape[0] == sample_rate
        assert train_y.shape[0] == sample_rate

    def test_kfdl_2(self):
        sample_rate = 1
        dl = KFDL(dataset_name="YAHOO",
                  data_id="Yahoo_A1real_24_data.out",
                  sample_rate=sample_rate
                  )
        train_x_1, train_y, test_x, test_y = dl.load_train_and_test()

        sample_rate = -1
        dl = KFDL(dataset_name="YAHOO",
                  data_id="Yahoo_A1real_24_data.out",
                  sample_rate=sample_rate
                  )
        train_x_2, train_y, test_x, test_y = dl.load_train_and_test()
        assert train_x_1.shape[0] == train_x_2.shape[0]

    def test_kfdl_3(self):
        sample_rate = 0
        dl = KFDL(dataset_name="YAHOO",
                  data_id="Yahoo_A1real_24_data.out",
                  sample_rate=sample_rate
                  )
        train_x_1, train_y, test_x, test_y = dl.load_train_and_test()
        assert train_x_1.shape[0] == 2
        assert np.sum(train_y) == 1


if __name__ == '__main__':
    unittest.main()
