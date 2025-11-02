import timeit
from unittest import TestCase

import numpy as np
import torch
from numpy.testing import assert_almost_equal
from torch import Tensor

from pylibs.affiliation.generics import convert_vector_to_events
from pylibs.utils.util_timeseries_sliding_windows import unroll_ts_torch, unroll_ts
from timeseries_models.base_model import ModelInterface


class TestTimeSeries(TestCase):
    def test_unroll_ts_torch(self):
        x = Tensor([
            [1, 2, 3, 4],
            [5, 6, 7, 10],
        ])
        print(unroll_ts_torch(x))
        assert_almost_equal([1., 3.5, 4.5, 5.5, 10.], unroll_ts_torch(x))

    def test_unroll_ts(self):
        # unroll_ts(np.eye(size))
        # print(unroll_ts_torch(torch.eye(size)))
        run_time = timeit.timeit("unroll_ts_torch(torch.eye(3000))",
                                 setup="from pylibs.utils.util_timeseries_sliding_windows import unroll_ts_torch, unroll_ts;import torch; ",
                                 number=10)
        print("unroll_ts_torch：", run_time)

        run_time = timeit.timeit("unroll_ts(np.eye(3000))",
                                 setup="from pylibs.utils.util_timeseries_sliding_windows import unroll_ts_torch, unroll_ts;import torch; import numpy as np;",
                                 number=10)
        print("unroll_ts：", run_time)

    def test_unroll_ts_01(self):
        res = Tensor([
            [1, 2, 3, 4],
            [5, 6, 7, 10],
        ])
        assert_almost_equal(unroll_ts(res.numpy(), full=True), unroll_ts_torch(res).numpy())

    def test_unroll_ts_02(self):
        res = torch.randn((30, 30))
        assert_almost_equal(unroll_ts(res.numpy(), full=True), unroll_ts_torch(res).numpy())

    def test_aa(self):
        print(convert_vector_to_events())
