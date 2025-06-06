import numpy as np

from pylibs.utils.util_log import get_logger
from pylibs.utils.util_system import UtilSys

log = get_logger()


class VAEConfig:
    def __init__(self):
        super().__init__()
        self.batch_size = 64
        self.window_size = 64
        self.hidden_activation = 'relu'
        self.output_activation = 'relu'
        self.l2_regularizer = 0.001
        self.drop_rate = 0.001
        self.epochs = 100
        self.latent_dim = 8
        self.learning_rate = 0.001
        self.n_neurons_layer1 = 64
        self.n_neurons_layer2 = 32
        self.n_neurons_layer3 = None

        # Autogenerated parameters
        self.encoder_neurons = [32]
        self.decoder_neurons = [32]

    def update_parameters(self, parameters: dict):
        key_of_int_parameter = ['batch_size', 'window_size', 'epochs', 'latent_dim', 'n_neurons_layer1',
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
                raise KeyError(f"{__class__} has not the hyper-parameter key named {key} ")
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
