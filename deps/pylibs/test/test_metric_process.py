from unittest import TestCase

from pylibs.evaluation.contextual import contextual_precision, contextual_recall


class TestKFoldMetricProcess(TestCase):

    def test_contest_score(self):
        print()
        expected = [[2, 3], [10, 20]]
        predict = [[2, 2], [3, 3], [2, 2.5]]
        print(contextual_precision(expected, predict, weighted=False))
        print(contextual_recall(expected, predict, weighted=False))
