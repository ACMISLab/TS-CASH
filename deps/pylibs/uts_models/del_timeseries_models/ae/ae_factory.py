from pylibs.utils.util_log import get_logger
from timeseries_models.ae.ae_confi import AEConf
from timeseries_models.ae.ae_model import AEModel
from timeseries_models.model_factory import ModelFactory

log = get_logger()


class AEFactory(ModelFactory):
    def __init__(self, hpconfig, device=None, seed=None):
        """

        Parameters
        ----------
        hpconfig :
        device : int
            A gpu exp_index for training the model
        seed :
        """
        super().__init__(hpconfig, device=device, seed=seed)

    def get_model(self) -> AEModel:
        ccf = AEConf()
        ccf.update_parameters(self.get_hyperparameters())
        return AEModel(ccf, device=self.device, seed=self.seed)
