import numpy as np
from numpy.testing import assert_almost_equal

from pylibs.evaluation.utils import normalize_score
from pylibs.utils.util_feishu import send_msg_to_feishu
from pylibs.utils.util_nni import get_all_experiment
from pylibs.utils.utils import yaml_to_json, valid_search_space, get_random_idle_port


def test_send_msg_to_feishu():
    send_msg_to_feishu("测试消息")


def test_yaml_to_json():
    # yaml_to_json("/Users/sunwu/SyncResearch/pytorch-donut-master/experiments/exp03/d302_donut_10.yaml")
    yaml_to_json("/Users/sunwu/SyncResearch/pytorch-donut-master/experiments/exp03/d301_donut_baseline.yaml")


def test_valid_search_space():
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
           'output_activation': 'tanh', 'epochs': 255, 'batch_size': 120, 'dropout_rate': 0.1, 'l2_regularizer': 0.1,
           'encoder_neurons': [120, 120], 'decoder_neurons': [120, 120]}
    assert ret_search_space == res


def test_file_sep():
    with open("./test_str.txt", 'r') as f:
        exp_ids = []
        for line in f.readlines():
            if line.find("Experiment ID") > -1:
                exp_id = line.split("[")[-2][:-1]
                exp_ids.append(exp_id)
        print(exp_ids)


def test_normalize_score():
    assert_almost_equal(normalize_score([0, 0]), [0, 0])
    assert_almost_equal(normalize_score([100, 100]), [0, 0])
    assert_almost_equal(normalize_score([0, 1]), [0, 1])
    assert_almost_equal(normalize_score([np.NAN, 1]), [0, 1])
    assert_almost_equal(normalize_score([np.inf, 1]), [0, 1])


def test_run_cmd():
    # cmd="nnictl experiment list --all|grep deepsvdd_smtp_DDS "
    print()
    print(get_all_experiment())


def test_random_port_sss():
    print(get_random_idle_port())
