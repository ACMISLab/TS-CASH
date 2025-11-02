import os
from unittest import TestCase
from pylibs.utils.util_nni_sqlite import NNISqlite


class TestNNISqlite(TestCase):

    def test_error_nni_experiments(self):

        experiment_working_directory = os.path.join(os.getcwd(),"nni_exp/error_with_to_exp/db/nni.sqlite")
        ndb=NNISqlite(experiment_working_directory)
        assert ndb.is_all_trials_success() is False

