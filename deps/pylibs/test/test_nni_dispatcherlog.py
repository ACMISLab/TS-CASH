import os
from unittest import TestCase

from pylibs.utils.util_nni_dispatcherlog import NNIDispatcherLog


class TestNNIDispatcher(TestCase):

    def test_nni_tools(self):
        experiment_working_directory = os.path.join(os.getcwd(),
                                                    "data/nnilog")
        id = "err_connect_to_nni_server"
        ndp = NNIDispatcherLog(experiment_working_directory=experiment_working_directory,
                               id=id)
        assert ndp.has_error() is True
