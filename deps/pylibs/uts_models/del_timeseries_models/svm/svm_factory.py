from pylibs.utils.util_log import get_logger
from timeseries_models.model_factory import ModelFactory
from timeseries_models.svm.svm_conf import SVMConf
from timeseries_models.svm.svm_model import SVMModel

log = get_logger()


class SVMFactory(ModelFactory):
    def __init__(self, hpconfig, device="cpu", seed=None):
        super().__init__(hpconfig, device, seed=seed)

    def get_model(self):
        cfg = SVMConf()
        cfg.update_parameters(self.get_hyperparameters())
        return SVMModel(cfg, self.device)
