import argparse
import time
from unittest import TestCase

import numpy as np
import torch

from nni.tools.nnictl.config_utils import Experiments
from nni.tools.nnictl.launcher import view_experiment
from pylibs.utils.util_nni import parse_exp_name_to_dict
from pylibs.utils.util_nnictl import NNICTL
import nni


class TestNNIReport(TestCase):

    def test_nni_tools(self):
        string = "model_entry_file=main_sk.py&search_space_file=none&exp_name=exp_aiops_if_final_v24&max_trial_number=32&trial_concurrency=12&template_file=temp_baseline.yaml&round_rate=none&seed=1&data_sample_rate=0.125&data_sample_method=random&dry_run=false&dataset=iopscompetition&data_id=6d1114ae-be04-3c46-b5aa-be1a003a57cd&nni_tuner=rs&model=isolationforest&gpus=none&wait_interval_time=3&iter_counter=81&n_experiments=162&n_experiment_restart=3"

        print(parse_exp_name_to_dict(string))
        assert parse_exp_name_to_dict(string) == {'model_entry_file': 'main_sk.py', 'search_space_file': 'none',
                                                  'exp_name': 'exp_aiops_if_final_v24', 'max_trial_number': '32',
                                                  'trial_concurrency': '12', 'template_file': 'temp_baseline.yaml',
                                                  'round_rate': 'none', 'seed': '1', 'data_sample_rate': '0.125',
                                                  'data_sample_method': 'random', 'dry_run': 'false',
                                                  'dataset': 'iopscompetition',
                                                  'data_id': '6d1114ae-be04-3c46-b5aa-be1a003a57cd', 'nni_tuner': 'rs',
                                                  'model': 'isolationforest', 'gpus': 'none', 'wait_interval_time': '3',
                                                  'iter_counter': '81', 'n_experiments': '162',
                                                  'n_experiment_restart': '3'}

    def test_nni_tools2(self):
        string = "model_entry_filexmain_sk.py&search_space_fi"

        print(parse_exp_name_to_dict(string))

    def test_nni_ctl(self):
        # NNICTL.view_experiment("jlpq984h", 31000)
        # time.sleep(5)
        NNICTL.stop_all()

    def test_get_all_experiments(self):
        exps = NNICTL.get_all_experiments()
        print(exps)

    def test_get_experiment(self):
        experiments_config = Experiments()
        experiments_dict = experiments_config.get_all_experiments()
        print(experiments_dict)

    def test_get_experiment_by_name(self):
        NNICTL.get_experiment_by_name_from_shell_v2("sdfs")

    def test_torch(self):
        arr = torch.Tensor(np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        print(arr)
        print(arr * 2)
        rand_arr = torch.randn_like(arr)
        print(rand_arr)
        print(rand_arr * arr)

    def test_torch2(self):
        arr = torch.Tensor(np.asarray([[1, 2],
                                       [5, 6],
                                       [8, 9]]))
        print(torch.sum(arr, dim=0))
        # tensor([14., 17.])

        print(torch.sum(arr, dim=1))
        # tensor([3., 11., 17.])

    def test_resume_experiment(self):
        id = "n93bpwrz"
        NNICTL.view_experiment(id, 50000)
        assert NNICTL.is_view(id) is True
        NNICTL.stop_experiment(id)
        assert NNICTL.is_view(id) is False
