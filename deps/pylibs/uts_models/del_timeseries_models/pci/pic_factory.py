from pylibs.utils.util_log import get_logger
from timeseries_models.model_factory import ModelFactory
from timeseries_models.pci.pci_config import PCIConf
from timeseries_models.pci.pci_model import PCIModel
from timeseries_models.vae.vae_model import VAEModel

log = get_logger()


class PCIFactory(ModelFactory):
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

    def get_model(self) -> PCIModel:
        ccf = PCIConf()
        ccf.update_parameters(self.get_hyperparameters())
        return PCIModel(ccf, device=self.device, seed=self.seed)
