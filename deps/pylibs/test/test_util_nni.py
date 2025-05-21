from unittest import TestCase

import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.metrics import roc_auc_score

from pylibs.common import ConstMetric
from pylibs.nni_report import report_nni_final_result_with_loss
from pylibs.utils.util_log import get_logger
from pylibs.utils.util_nni import get_experiments_by_name, is_running_experiment, get_running_experiments, \
    print_experiment_error_info, NNIManagerLog, NNIDotExperiment, NNICTL

log = get_logger()


class TestKFoldMetricProcess(TestCase):
    def test_nni_experiment(self):
        # cmd="nnictl experiment list --all|grep deepsvdd_smtp_DDS "
        print()
        # ex=get_experiment_by_name("None")
        # assert ex is not None

        ex = get_experiments_by_name("jsldjflsjdlkfjlksjljxljljldsjfljsd")
        assert ex is None

    def test_nni_experiment1(self):
        print(is_running_experiment())

    def test_nni_experiment2(self):
        print(get_running_experiments())

    def test_log_print(self):
        print_experiment_error_info()

    def testAUC(self):
        # disable_eager_execution()
        # assure to fix: ValueError: Only one class present in y_true. ROC AUC score is not defined '
        report_nni_final_result_with_loss(np.zeros(shape=(100,)), np.zeros(shape=(100,)))
        report_nni_final_result_with_loss(np.ones(shape=(100,)), np.ones(shape=(100,)))

        for i in range(100):
            y_true = np.random.randint(2, size=100)
            score = np.random.uniform(0, 1, 100)
            auc = roc_auc_score(y_true, score)
            res = report_nni_final_result_with_loss(y_true, score)

            assert_almost_equal(auc, res[ConstMetric.KEY_AUC], decimal=2)

    def testClassCat(self):
        res = report_nni_final_result_with_loss([0, 0, 0], [0, 0, 0])
        assert res[ConstMetric.BEST_RECALL] == 0

    # is_experiment_done("/Users/sunwu/nni-experiments/debug_1675855568130661000", "qc85ane1")
    def test_is_experiment_done(self):
        nm = NNIManagerLog("./data", "p8a9fo0k", "nnimanager_done.txt")
        assert nm.is_experiment_done()

        nm = NNIManagerLog("./data", "p8a9fo0k", "nnimanager_error.txt")
        assert nm.is_experiment_error()

    def test_nni_experiment_1(self):
        nde = NNIDotExperiment()
        test_exp_id = "EXP_DEBUG_DAT-IOPSCOMPETITION_NTR-RANDOM_MTN-2_MTC-1_RET-0_SED-0_DSM-RANDOM_DSR-1.0_MON-DEMO_MEF-MAIN_TORCH.PY"
        print(nde.get_experiment_info(test_exp_id))
        expinfo = nde.get_experiment_info("xx")
        assert expinfo.is_existed() == False

    def test_nni_experiment_dir(self):
        nde = NNIDotExperiment()
        test_exp_id = "EXP_DEBUG_DAT-IOPSCOMPETITION_NTR-RANDOM_MTN-2_MTC-1_RET-0_SED-0_DSM-RANDOM_DSR-1.0_MON-DEMO_MEF-MAIN_TORCH.PY"
        print(nde.get_all_log_dirs())

    def test_algo(self):
        nc = NNICTL()
        nc.get_all_tuners()
