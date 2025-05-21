from numpy.testing import assert_almost_equal
from unittest import TestCase

import tensorflow as tf
from sklearn.metrics import precision_recall_curve

from pylibs.utils.util_bash import exec_cmd_and_return_str
from pylibs.utils.util_tf import log_computed_device_tf2, using_gpu_without_memory_growth_tf2, \
    using_gpu_with_memory_growth_tf2, specify_gpus, get_specify_gpus_from_str, set_all_devices_memory_growth, \
    get_model_save_home
from pylibs.utils.util_log import get_logger

log = get_logger()


class TestBash(TestCase):
    def test_retry(self):
        res = exec_cmd_and_return_str("pwd", retry=2)
        UtilSys.is_debug_mode() and log.info(f"exe result: {res}")

        try:
            exec_cmd_and_return_str("sdfs", retry=2)
        except Exception as e:
            assert str(e).find("Execute [sdfs] error after retry 2 times") > -1
