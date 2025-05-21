from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal

from pylibs.evaluation.utils import normalize_score
from pylibs.util_pandas import save_pandas_to_mysql
from pylibs.utils.util_feishu import send_msg_to_feishu
from pylibs.utils.util_nni import get_all_experiment, get_experiments_from_profile_file, get_experiments_by_name
from pylibs.utils.utils import yaml_to_json, valid_search_space, get_random_idle_port


class TestConfig(TestCase):
    def test_send_msg_to_feishu(self):
        send_msg_to_feishu("aaa")

    def test_valid_search_space(self):
        mode = "AE"
        search_space = {
            'hidden_layers': 2,
            'hidden_neurons': 120,
            'latent_dim': 5,
            'window_size': 120,
            'hidden_activation': "relu",
            'output_activation': "tanh",
            'epochs': 255,
            'batch_size': 120,
            'dropout_rate': 0.1,
            'l2_regularizer': 0.1,
        }
        ret_search_space = valid_search_space(search_space, mode)
        res = {'latent_dim': 5, 'window_size': 120, 'hidden_activation': 'relu',
               'output_activation': 'tanh', 'epochs': 255, 'batch_size': 120, 'dropout_rate': 0.1,
               'l2_regularizer': 0.1,
               'encoder_neurons': [120, 120], 'decoder_neurons': [120, 120]}
        assert ret_search_space == res

    def test_normalize_score(self):
        assert_almost_equal(normalize_score([0, 0]), [0, 0])
        assert_almost_equal(normalize_score([100, 100]), [0, 0])
        assert_almost_equal(normalize_score([0, 1]), [0, 1])
        assert_almost_equal(normalize_score([np.NAN, 1]), [0, 1])
        assert_almost_equal(normalize_score([np.inf, 1]), [0, 1])

    def test_run_cmd(self):
        # cmd="nnictl experiment list --all|grep deepsvdd_smtp_DDS "
        print(get_all_experiment())

    def test_get_experiment_id(self):
        get_experiments_from_profile_file()

    def test_get_all_experiment(self):
        get_all_experiment()

    def test_random_port_sss(self):
        print(get_random_idle_port())

    def test_export_to_pandas(self):
        df = pd.DataFrame({
            "a": np.linspace(1, 100),
        })

        save_pandas_to_mysql(df, table_name="test")

    def test_get_exp_by_name(self):
        get_experiments_by_name("VAE_SHUTTLE_1668681263")
