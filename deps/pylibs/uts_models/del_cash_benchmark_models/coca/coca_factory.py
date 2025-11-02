from typeguard import typechecked

from pylibs.utils.util_log import get_logger
from timeseries_models.coca.coca_config import COCAConf
from timeseries_models.coca.coca_model import COCAModel
from timeseries_models.coca.coca_model_remove_batch_norm import COCAModelRemoveBatchNorm1D
from timeseries_models.model_factory import ModelFactory

log = get_logger()


class COCAFactory(ModelFactory):
    @typechecked
    def __init__(self, hpconfig, device: str = None, seed: int = None):
        """

        Parameters
        ----------
        hpconfig :
        device :
            "cuda:0", "cuda:1", "cuda:2",...
            or "cpu"
        seed :
        """
        super().__init__(hpconfig, device=device, seed=seed)

    def get_model(self) -> COCAModelRemoveBatchNorm1D:
        ccf = COCAConf()
        ccf.update_parameters(self.get_hyperparameters())
        return COCAModelRemoveBatchNorm1D(ccf, device=self.device)
