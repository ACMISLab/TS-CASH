from pylibs.utils.util_log import get_logger
from timeseries_models.isolation_forest.isolation_forest_conf import IsolationForestConf
from timeseries_models.isolation_forest.isolation_forest_model import IsolationForestModel
from timeseries_models.model_factory import ModelFactory

log = get_logger()


class IsolationForestFactory(ModelFactory):
    def __init__(self, hpconfig, device="cpu", seed=None):
        super().__init__(hpconfig, device, seed=seed)

    def get_model(self):
        cfg = IsolationForestConf()
        cfg.update_parameters(self.get_hyperparameters())
        return IsolationForestModel(cfg, self.device)
