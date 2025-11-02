from unittest import TestCase

from pylibs.utils.nni_report import report_nni_final_metric
from pylibs.utils.util_nni import get_all_experiments_from_shell
from pylibs.utils.util_nni_exp_info import get_exp_status_by_id


class TestNNIReport(TestCase):

    def test_nni_tools(self):
        res = get_all_experiments_from_shell()
        print(res)

    def test_nni_exp_status(self):
        res = get_exp_status_by_id("5o2qcs30", retry_times=3)
        print(res)

        res = get_exp_status_by_id("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", retry_times=2)

    def test_rusult(self):
        res = report_nni_final_metric(**{'auc_0': 0.0,
                                         'auc_1': 0.0,
                                         'auc_2': 0.0,
                                         'auc_3': 0.0,
                                         'auc_4': 0.0,
                                         'auc_5': 0.0,
                                         'auc_6': 0.0,
                                         'auc_7': 0.0,
                                         'auc_8': 0.0,
                                         'auc_9': 0.0,
                                         'evaluate_end': 1669252940145021000,
                                         'evaluate_start': 1669252939731556000,
                                         'program_end': 1669252941233617000,
                                         'program_start': 1669252939451356000,
                                         'test_loss_0': {'kl_loss': 15.397006034851074,
                                                         'loss': 89.717041015625,
                                                         'recon_loss': 74.3200454711914},
                                         'test_loss_1': {'kl_loss': 15.397006034851074,
                                                         'loss': 90.60455322265625,
                                                         'recon_loss': 75.2075424194336},
                                         'test_loss_2': {'kl_loss': 15.397006034851074,
                                                         'loss': 88.45326232910156,
                                                         'recon_loss': 73.05625915527344},
                                         'test_loss_3': {'kl_loss': 15.397006034851074,
                                                         'loss': 90.01097106933594,
                                                         'recon_loss': 74.61396026611328},
                                         'test_loss_4': {'kl_loss': 15.397006034851074,
                                                         'loss': 88.1444091796875,
                                                         'recon_loss': 72.74740600585938},
                                         'test_loss_5': {'kl_loss': 15.397006034851074,
                                                         'loss': 87.21097564697266,
                                                         'recon_loss': 71.81397247314453},
                                         'test_loss_6': {'kl_loss': 15.397006034851074,
                                                         'loss': 84.75272369384766,
                                                         'recon_loss': 69.35570526123047},
                                         'test_loss_7': {'kl_loss': 15.397006034851074,
                                                         'loss': 84.53620147705078,
                                                         'recon_loss': 69.13919067382812},
                                         'test_loss_8': {'kl_loss': 15.397006034851074,
                                                         'loss': 85.92872619628906,
                                                         'recon_loss': 70.53173065185547},
                                         'test_loss_9': {'kl_loss': 15.397006034851074,
                                                         'loss': 88.47504425048828,
                                                         'recon_loss': 73.07804107666016},
                                         'train_end': 1669252939731553000,
                                         'train_start': 1669252939519440000})
        res = report_nni_final_metric(auc1=3, auc2=4)
        assert res == {'auc1': 3, 'auc2': 4}

        res = report_nni_final_metric(auc1=3, auc2=4, **{'key1': 3})
        assert res == {'auc1': 3, 'auc2': 4, 'key1': 3}
