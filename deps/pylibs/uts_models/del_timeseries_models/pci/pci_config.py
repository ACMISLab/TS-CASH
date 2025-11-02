import numpy as np

from pylibs.utils.util_log import get_logger
from timeseries_models.base_conf import BaseModelConfig

log = get_logger()


class PCIConf(BaseModelConfig):

    def __init__(self, window_size=20, thresholding_p=0.05, random_status=42):
        super().__init__()
        self.window_size: int = window_size
        self.thresholding_p: float = thresholding_p
        self.random_state: int = random_status

    def update_parameters(self, parameters: dict):
        # int parameters
        key_of_int_parameter = ['window_size', 'random_state', 'batch_size']
        count = 0
        for key, val in parameters.items():
            if hasattr(self, key):
                if key in key_of_int_parameter:
                    val = int(np.round(val))
                    assert isinstance(val, int)
                UtilSys.is_debug_mode() and log.info(f"Updating parameters {key:20} = {val}")
                setattr(self, key, val)
            else:
                raise KeyError(f"{__class__} has not the hyper-parameter key named {key} ")
            count += 1
        UtilSys.is_debug_mode() and log.info(f"The number of parameters updated: {count:10}")
