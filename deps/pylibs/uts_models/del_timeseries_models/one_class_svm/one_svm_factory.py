from pylibs.utils.util_log import get_logger
from timeseries_models.model_factory import ModelFactory
from timeseries_models.one_class_svm.one_svm_conf import OneClassSVMConf
from timeseries_models.one_class_svm.one_svm_model import OneClassSVMModel

log = get_logger()


class OneClassSVMFactory(ModelFactory):
    def __init__(self, hpconfig, device="cpu", seed=None):
        super().__init__(hpconfig, device, seed=seed)

    def get_model(self):
        cfg = OneClassSVMConf()
        cfg.update_parameters(self.get_hyperparameters())
        return OneClassSVMModel(cfg, self.device)
