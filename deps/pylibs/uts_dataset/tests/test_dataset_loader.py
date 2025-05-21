#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/3/10 09:38
# @Author  : gsunwu@163.com
# @File    : test_dataset_loader.py
# @Description:
import unittest
import logging
from pathlib import PosixPath

import rich
from numpy.testing import assert_equal

from pylibs.uts_dataset.dataset_loader import KFoldDatasetLoader, UTSDataset, KFDL

log = logging.getLogger(__name__)


class TestDatasetLoader(unittest.TestCase):

    def test_demo(self):
        kf = KFoldDatasetLoader()
        self.assertTrue(kf.is_test_y_valid([0, 1]))
        self.assertTrue(kf.is_test_y_valid([1, 0]))
        self.assertFalse(kf.is_test_y_valid([0, 0]))
        self.assertFalse(kf.is_test_y_valid([1, 1]))

    def test_02(self):
        dts = UTSDataset.select_datasets_split(dataset_names=[UTSDataset.DATASET_NASA_SMAP, UTSDataset.DATASET_SMD],
                                               top_n=1)
        rich.print(dts)

    def test_03(self):
        dts = UTSDataset.select_datasets_split(dataset_names=[UTSDataset.DATASET_NASA_SMAP, "SMD"], top_n=9999)
        rich.print(dts)

    def test_04(self):
        # [['NASA-SMAP', 'E-5.test.out']]
        # NASA-SMAP : E-5.test.out

        res = UTSDataset.select_datasets_split(dataset_names=['NASA-SMAP'])
        print(res)
        opt = OptConf(
            exp_name='all_debug1',
            job_name='dataset_name_NASA-SMAP-data_id_E-5.test.out-autoscaling_data_True'
            ,
            dataset_name='NASA-SMAP',
            data_id='E-5.test.out',
            seed=0,
            is_auto_data_scaling=False,
            client_type='164',
            is_debug=True,
            facade='hyperparameter_optimization',
            out_dir=PosixPath('automl_results/debug/all_debug1/hyperparameter_optimization/NASA-SMAP'),
            index=1,
            total=-1,
            test_rate=0.3
        )
        print(opt.check_test_y())

    def test_dataset_select(self):
        datasets1= UTSDataset.select_datasets_split(
            dataset_names=["YAHOO", "NAB", "SMD"],
            top_n=10,
            test_ratio=0.4,
            seed=42
        )
        datasets2= UTSDataset.select_datasets_split(
            dataset_names=["YAHOO", "NAB",  "SMD"],
            top_n=10,
            test_ratio=0.4,
            seed=42
        )

        # [['NAB', 'NAB_data_Traffic_3.out'], ['NAB', 'NAB_data_Traffic_4.out'], ['NAB', 'NAB_data_KnownCause_5.out'], ['NAB', 'NAB_data_CloudWatch_6.out'], ['NAB', 'NAB_data_CloudWatch_9.out'], ['NAB', 'NAB_data_KnownCause_6.out'], ['NAB', 'NAB_data_tweets_5.out'], ['NAB', 'NAB_data_art1_0.out'], ['NAB', 'NAB_data_Exchange_3.out'], ['NAB', 'NAB_data_Traffic_1.out'], ['SMD', 'machine-2-3.test.csv@25.out'], ['SMD', 'machine-2-8.test.csv@30.out'], ['SMD', 'machine-1-5.test.csv@1.out'], ['SMD', 'machine-3-11.test.csv@28.out'], ['SMD', 'machine-3-3.test.csv@6.out'], ['SMD', 'machine-3-11.test.csv@20.out'], ['SMD', 'machine-3-4.test.csv@34.out'], ['SMD', 'machine-3-5.test.csv@31.out'], ['SMD', 'machine-2-8.test.csv@17.out'], ['SMD', 'machine-3-8.test.csv@20.out'], ['YAHOO', 'Yahoo_A1real_46_data.out'], ['YAHOO', 'YahooA4Benchmark-TS55_data.out'], ['YAHOO', 'Yahoo_A1real_31_data.out'], ['YAHOO', 'YahooA3Benchmark-TS64_data.out'], ['YAHOO', 'YahooA3Benchmark-TS73_data.out'], ['YAHOO', 'YahooA3Benchmark-TS84_data.out'], ['YAHOO', 'YahooA3Benchmark-TS30_data.out'], ['YAHOO', 'Yahoo_A2synthetic_31_data.out'], ['YAHOO', 'YahooA4Benchmark-TS26_data.out'], ['YAHOO', 'Yahoo_A1real_25_data.out']]
        assert_equal(datasets1,datasets2)
        print(datasets2)
        print(datasets1)
