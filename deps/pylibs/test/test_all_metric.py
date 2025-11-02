from unittest import TestCase

import numpy as np

from pylibs.metrics.uts.uts_metric_helper import UTSMetricHelper


class TestConfig(TestCase):
    def test_1(self):
        score = np.random.random(3)
        labels = np.asarray([1, 0, 0])
        UTSMetricHelper.get_metrics_all(labels, score, 2)

    def test_2(self):
        score = np.random.random(3)
        score[0] = np.inf
        labels = np.asarray([1, np.nan, np.inf])
        UTSMetricHelper.get_metrics_all(labels, score, 2)

    def test_3(self):
        labels=np.random.random((20000,)).round().astype("int")
        score=np.random.random((20000,))
        t2=UTSMetricHelper.get_metrics_all_cache(labels, score, 2)
        print("mark_as_finished cached")
        t1=UTSMetricHelper.get_metrics_all(labels, score, 2)
        print("done1")
        assert t1==t2

    def test_4(self):
        labels = np.random.random((20000,)).round().astype("int")
        score = np.random.random((20000,))
        t2 = UTSMetricHelper.get_metrics_all_cache(labels, score, 2)
        print("mark_as_finished cached")