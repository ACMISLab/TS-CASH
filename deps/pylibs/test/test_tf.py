from numpy.testing import assert_almost_equal
from unittest import TestCase

import tensorflow as tf
from sklearn.metrics import precision_recall_curve

from pylibs.utils.util_tf import log_computed_device_tf2, using_gpu_without_memory_growth_tf2, \
    using_gpu_with_memory_growth_tf2, specify_gpus, get_specify_gpus_from_str, set_all_devices_memory_growth, \
    get_model_save_home
from pylibs.utils.util_log import get_logger

log = get_logger()


class TestTF(TestCase):
    def testRead(self):
        m = tf.keras.metrics.AUC(num_thresholds=3)
        m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])
        print(f"right:{m.result().numpy()}")
        m.reset_state()
        m.update_state([0, 0, 0, 0], [0, 0, 0, 0])
        print(f"All zero:{m.result().numpy()}")

        print(precision_recall_curve([0, 0, 0], [0, 0.5, 1]))

    def testUseDefaultGPU(self):
        log_computed_device_tf2()
        using_gpu_with_memory_growth_tf2()
        a = tf.constant(1)
        b = tf.constant(2)

        print(a + b)

    def test_gpus(self):
        try:
            devices = tf.config.list_logical_devices('GPU')
            if len(devices) > 0:
                assert specify_gpus()[0].name == devices[0].name
                assert specify_gpus('0')[0].name == devices[0].name
                assert specify_gpus('0,')[0].name == devices[0].name
        except:
            pass

    def test_gpus2(self):
        using_gpu_with_memory_growth_tf2(-1)
        using_gpu_with_memory_growth_tf2(None)
        using_gpu_with_memory_growth_tf2(0)
        try:
            using_gpu_with_memory_growth_tf2(9999)
        except:
            assert True

    def test_get_gpus(self):
        assert_almost_equal(get_specify_gpus_from_str("1,2,3"), [1, 2, 3])
        assert_almost_equal(get_specify_gpus_from_str("1,2,"), [1, 2])
        assert_almost_equal(get_specify_gpus_from_str(), [0])

    def test_using_gpus(self):
        for sequence_id in range(32):  # the sequence id
            gpus = [1, 3]  # specify the gpus
            target_gpus_index = sequence_id % len(gpus)
            print(gpus[target_gpus_index])

    def test_get_model_save_dir(self):
        name = get_model_save_home()
        print(name)
