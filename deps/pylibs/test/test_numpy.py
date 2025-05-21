import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from pylibs.utils.util_numpy import fill_nan_inf


class TestNumpy(unittest.TestCase):
    def test_fill_array(self):
        assert_almost_equal(fill_nan_inf(np.asarray([np.NAN, np.inf]), -1), [-1, -1])
