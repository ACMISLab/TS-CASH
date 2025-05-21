import numpy as np

from pylibs.utils.util_log import get_logger
from timeseries_models.base_conf import BaseModelConfig

log = get_logger()


class OneClassSVMConf(BaseModelConfig):
    def __init__(self, verbose=1):
        super().__init__()
        self.kernel = "rbf"
        self.verbose = verbose
        self.max_iter = 100
        self.tol = 1e-3
        self.nu = 0.5
        self.gamma = "scale"

    def update_parameters(self, parameters: dict):
        # window_size
        # time_step
        # batch_size
        key_of_int_parameter = ['n_estimators', 'window_size', 'time_step', 'batch_size']
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
