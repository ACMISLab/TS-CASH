from unittest import TestCase

from numpy.testing import assert_almost_equal

from pylibs.evaluation.utils import from_list_points_labels
from pylibs.utils.utils import modified_score


class TestDonutPytorch(TestCase):
    def test_modified_score(self):
        y_true = [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1]
        score = [-0.1, -0.3, -0.1, 0, 0, 0, 0, 0, 0, -0.5, 0, 0]

        modified = [-0.3, -0.3, -0.3, 0, 0, 0, -0.5, -0.5, -0.5, -0.5, 0, 0]
        assert_almost_equal(modified_score(y_true, score), modified)
