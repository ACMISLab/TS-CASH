from pylibs.utils.util_log import get_logger
from timeseries_models.model_factory import ModelFactory
from timeseries_models.vae.vae_confi import VAEConf
from timeseries_models.vae.vae_model import VAEModel

log = get_logger()


class VAEFactory(ModelFactory):
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

    def get_model(self) -> VAEModel:
        ccf = VAEConf()
        ccf.update_parameters(self.get_hyperparameters())
        return VAEModel(ccf, device=self.device, seed=self.seed)
