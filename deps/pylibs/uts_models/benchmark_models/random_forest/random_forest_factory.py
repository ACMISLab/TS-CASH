from pylibs.utils.util_log import get_logger
from timeseries_models.model_factory import ModelFactory
from timeseries_models.random_forest.random_forest_conf import RandomForestConf
from timeseries_models.random_forest.random_forest_model import RandomForestModel

log = get_logger()


class RandomForestFactory(ModelFactory):
    def __init__(self, hpconfig, device="cpu", seed=None):
        super().__init__(hpconfig, device, seed=seed)

    def get_model(self):
        cfg = RandomForestConf()
        cfg.update_parameters(self.get_hyperparameters())
        return RandomForestModel(cfg, self.device)
