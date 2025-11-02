import numpy as np
from pylibs.utils.util_log import get_logger
from timeseries_models.base_conf import BaseModelConfig

log = get_logger()


class LSTMConf(BaseModelConfig):
    def __init__(self):

        self.input_size = 1
        self.latent_size = 5
        self.rnn_hidden_size = 5
        self.window_size: int = 6
        self.batch_size: int = 128
        self.lstm_layers: int = 10

    def update_parameters(self, parameters: dict):
        key_of_int_parameter = ['batch_size', 'window_size', 'epoch', 'latent_dim', 'n_neurons_layer1',
                                'n_neurons_layer2', 'n_neurons_layer3']

        count = 0
        for key, val in parameters.items():
            if hasattr(self, key):
                if key in key_of_int_parameter:
                    val = int(np.round(val))
                    assert isinstance(val, int)
                UtilSys.is_debug_mode() and log.info(f"Updating parameters {key:20} = {val}")
                setattr(self, key, val)
            else:
                raise KeyError(f"{__class__} has not the hyper-parameter key named [{key}] ")
            count += 1

        UtilSys.is_debug_mode() and log.info(f"The number of parameters updated: {count:10}")

        if self.n_neurons_layer3 is not None:
            if self.n_neurons_layer1 is None or self.n_neurons_layer2 is None:
                raise ValueError(f"self.n_neurons_layer2 and self.n_neurons_layer1 can't be None.")
            self.encoder_neurons = [
                self.n_neurons_layer1,
                self.n_neurons_layer2,
                self.n_neurons_layer3
            ]

        elif self.n_neurons_layer2 is not None:
            if self.n_neurons_layer1 is None:
                raise ValueError(f"self.n_neurons_layer1 can't be None.")
            self.encoder_neurons = [
                self.n_neurons_layer1,
                self.n_neurons_layer2
            ]
        elif self.n_neurons_layer1 is not None:
            self.encoder_neurons = [
                self.n_neurons_layer1,
            ]
            assert self.n_neurons_layer2 is None
            assert self.n_neurons_layer3 is None

        self.decoder_neurons = self.encoder_neurons[::-1]
