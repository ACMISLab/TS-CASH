import unittest

from pylibs.experiments.exp_config import ExpConf
from pylibs.experiments.fast_uts_helper import main_fast_uts
from pylibs.utils.util_log import get_logger

log = get_logger()


class TestDatasetLoader(unittest.TestCase):

    def test_demo(self):
        conf_json = {'anomaly_window_type': 'coca', 'batch_size': 256, 'data_id': '3.test.out',
                     'data_sample_method': 'random', 'data_sample_rate': 64, 'dataset_name': 'MGAB', 'debug': False,
                     'epoch': 1, 'exp_index': 20, 'exp_name': 'verify_fast_uts_verify_main_fast_uts-2',
                     'exp_total': 56, 'fold_index': 0, 'gpu_memory_limit': 1024, 'is_send_message_to_feishu': True,
                     'job_id': '654a33fde5d7a204077e13abd5eed05b15b369a4', 'kfold': 5,
                     'metrics_save_home': '/remote-home/sunwu/cs_acmis_sunwu/sw_research_code/A01_paper_exp/runtime',
                     'model_name': 'pca', 'seed': 0, 'test_rate': 0.3, 'verbose': 0, 'window_size': 64}
        conf = ExpConf(**conf_json)
        main_fast_uts(conf)
