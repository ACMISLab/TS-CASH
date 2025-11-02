import abc
from pylibs.utils.util_log import get_logger

log = get_logger()


class BaseModelConfig(metaclass=abc.ABCMeta):
    def __init__(self):
        self.lr = 1e-4
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.weight_decay = 5e-4
        self.window_size = 16
        self.batch_size = 256
        self.drop_last = False
        self.seed = 0

    @abc.abstractmethod
    def update_parameters(self, parameters: dict):
        """
        Update the parameters grabbed from nni.get_next_parameters()

        Examples
        --------

        .. code-block:: python

                    count = 0
                    for key, val in parameters.items():
                        if hasattr(self, key):
                            UtilSys.is_debug_mode()  and log.info(f"Updating parameters {key}={val}")
                            setattr(self, key, val)
                        else:
                            raise KeyError(f"{__class__} has not the hyper-parameter key named {key} ")
                        count += 1
                    UtilSys.is_debug_mode()  and log.info(f"The number of parameters updated: {count}")

        Parameters
        ----------
        parameters :

        Returns
        -------

        """
        pass
