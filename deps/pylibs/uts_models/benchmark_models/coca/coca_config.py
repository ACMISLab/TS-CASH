import numpy as np

from pylibs.utils.util_log import get_logger
from pylibs.utils.util_system import UtilSys

from timeseries_models.base_conf import BaseModelConfig

log = get_logger()


class COCAConf(BaseModelConfig):
    def __init__(self):
        # datasets
        super().__init__()
        # 1 if univariate time series.
        # M if m dimensional time series
        self.input_channels = 1
        # model configs

        self.kernel_size = 4
        self.stride = 1
        self.device = "cpu"

        self.final_out_channels = 32
        self.hidden_size = 64
        self.num_layers = 3
        self.project_channels = 20
        self.dropout = 0.45
        # 👈
        self.features_len = 6
        # 👈
        self.window_size = 16
        # 👈
        self.time_step = 2
        # 👈
        self.num_epoch = 5
        # 👈
        self.center_eps = 0.1
        # 👈
        self.lr = 1e-4
        # 👈
        self.batch_size = 64
        # training configs
        self.weight_decay = 5e-4

        self.freeze_length_epoch = 2
        self.change_center_epoch = 1

        self.omega1 = 1
        self.omega2 = 0.1

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99

        # data parameters
        self.drop_last = False

        # Anomaly Detection parameters
        self.nu = 0.001
        # Anomaly quantile of fixed threshold
        self.detect_nu = 0.0015
        # Methods for determining thresholds ("fix","floating","one-anomaly")
        self.threshold_determine = 'floating'
        # Specify COCA objective ("one-class" or "soft-boundary")
        self.objective = 'soft-boundary'
        # Specify loss objective ("arc1","arc2","mix","no_reconstruction", or "distance")
        self.loss_type = 'distance'

    def update_parameters(self, parameters: dict):
        key_of_int_parameter = ['num_layers', 'project_channels', 'features_len', 'window_size', 'time_step',
                                'num_epoch', 'batch_size', 'hidden_size', 'final_out_channels', 'num_layers']

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
