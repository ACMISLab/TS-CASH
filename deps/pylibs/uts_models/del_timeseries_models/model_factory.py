import json
import os
import pprint
from abc import ABCMeta, abstractmethod
import nni
from pylibs.utils.util_log import get_logger
from pylibs.utils.util_message import logs

log = get_logger()


class ModelFactory(metaclass=ABCMeta):

    def __init__(self, hpconfig, device, seed=None):
        if seed is None:
            self.seed = 0
        else:
            self.seed = seed
        self.hpconfig = hpconfig
        self.device = device

    def get_hyperparameters(self):
        return self.parse_hyperparameters(self.hpconfig)

    @abstractmethod
    def get_model(self):
        """
        Update the model configuration(hyperparams)
        Generate a new model.


        Parameters
        ----------

        Returns
        -------

        """
        pass

    def parse_hyperparameters(self, hpconfig_file):
        """
        Load the hyperparameters from NNI or file file

        Parameters
        ----------
        hpconfig_file : Union[None,str]
            None means to grab the hyperparameters from NNI.
            str means to grab the hyperparameters from a json file.

        Returns
        -------

        """
        parameters = None
        logs(f"Hyperparameters config: {hpconfig_file}")
        if hpconfig_file is None or hpconfig_file == "None":
            logs("Loading hyperparameters from Microsoft NNI.")
        elif type(hpconfig_file) is str and os.path.exists(hpconfig_file):
            logs("Loading hyperparameters from file %s." % os.path.abspath(hpconfig_file))
            parameters = self.__load_json_file(hpconfig_file)
        elif type(hpconfig_file) is dict:
            logs("Loading hyperparameters from dict.")
            parameters = hpconfig_file
        elif str(type(hpconfig_file)).index("<class ") == 0:
            logs("Loading hyperparameters from class file.")
            parameters = hpconfig_file.__dict__
        else:
            raise RuntimeError("Unsupported hyperparameter config type!")
        if parameters is None:
            parameters = nni.get_next_parameter()

        logs(f"Loaded hyperparameters: \n"
             f"{parameters}\n\n")

        # fix errors:
        # AssertionError('nni.get_next_parameter() needs to be called before report_final_result')
        # nni.get_next_parameter()

        return parameters

    @staticmethod
    def __load_json_file(file):
        """
        Load hyperparameters from a json file.

        Parameters
        ----------
        file :

        Returns
        -------

        """
        logs(f"Load data from file {os.path.abspath(file)}")
        if os.path.exists(file):
            with open(file) as f:
                value: dict = json.load(f)
                if value is not None:
                    return value
                else:
                    raise RuntimeError(f"hpconfig parameter is not available, got {pprint.pformat(value)}")
        else:
            raise FileNotFoundError(f"Hyperparameter config file is not find {file}")
