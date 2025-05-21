#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/3/27 19:24
# @Author  : gsunwu@163.com
# @File    : test_replicated.py
# @Description:
import copy
import shutil

from numpy.testing import assert_equal

from pylibs.smac3.search_pace import ExpRun, FacadeType
import unittest
import logging

log = logging.getLogger(__name__)


class TestDatasetLoader(unittest.TestCase):

    def test_demo(self):
        er = ExpRun(
            data_autoscaling=False,
            exp_name="test_replicated_load",
            is_debug=False,
            datasets=["DEBUG1"],
            facades=[FacadeType.TYPE_HYPERPARAMETER_OPTIMIZATION],
            client_type="local",
            n_uts=2,
            n_seed=2,
            is_joblib_parallel=True,
            n_jobs=2,
            n_trials=[5],
            opt_metrics=["VUS_ROC"],
            skip_confirm=True,
            skip_upload_qiniu=True,
            test_rate=0.8
        )
        res1 = er.run().iloc[:, 0].values

        print("Removing dir", er.get_outhome())
        # shutil.rmtree(er.get_outhome())

        er2 = copy.deepcopy(er)
        res2 = er2.run().iloc[:, 0].values
        print("="*30)
        print(res1)
        print(res2)
        print("=" * 30)
        assert_equal(res1, res2)

    def test_rerun(self):
        er = ExpRun(
            data_autoscaling=False,
            exp_name="test_replicated_recreated",
            is_debug=False,
            datasets=["DEBUG1"],
            facades=[FacadeType.TYPE_HYPERPARAMETER_OPTIMIZATION],
            client_type="local",
            n_uts=2,
            n_seed=2,
            is_joblib_parallel=False,
            n_jobs=2,
            n_trials=[5],
            opt_metrics=["VUS_ROC"],
            skip_confirm=True,
            skip_upload_qiniu=True,
            test_rate=0.8
        )
        res1 = er.run().iloc[:, 0].values

        print("Removing dir", er.get_outhome())
        shutil.rmtree(er.get_outhome())

        er2 = copy.deepcopy(er)
        res2 = er2.run().iloc[:, 0].values
        print("=" * 30)
        print(res1)
        print(res2)
        print("=" * 30)
        assert_equal(res1, res2)

if __name__ == '__main__':
    unittest.main()