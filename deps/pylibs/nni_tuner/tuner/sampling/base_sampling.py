import abc
import copy
import random
import numpy as np
import pandas as pd
from pylibs.utils.util_log import get_logger

log = get_logger()


class BaseSampling(metaclass=abc.ABCMeta):
    """
    Doc inner class
    """

    def __init__(self, search_space: dict, sample_size, seed=0):
        """Init the base sample_base

        Parameters
        ----------
        sample_size : int
            The number of samples

        search_space : dict
           a dict from nni https://nni.readthedocs.io/en/stable/hpo/search_space.html
           {
             'latent_dim': {'_type': 'quniform', '_value': [2, 20, 1]},
             'res': {'_type': 'quniform', '_value': [3, 8, 1]}
            }
        middle : bool
            Whether to select the middle value between the lower and the upper.
        seed : int
            The seed of numpy or python, in order to reproduce.

        """
        self.TYPE_CHOICE = "choice"

        self.n_sample = sample_size
        if sample_size <= 0:
            raise ValueError(f"Sample size excepted > 0, but received {sample_size}")
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.search_space: dict = search_space
        self._clone_search_space: dict = {}

    def get_range_search_space(self):
        """
        Convert category to integer.


        Returns
        -------
        dict
            The range of search space
        """
        if self._clone_search_space:
            return self._clone_search_space
        clone_search_space = copy.deepcopy(self.search_space)
        for key in self.search_space:
            val = self.search_space[key]
            if val['_type'] == self.TYPE_CHOICE:
                clone_search_space[key]['_value'] = [0, len(val["_value"])]
                assert clone_search_space[key]['_value'] != self.search_space[key]['_value']

        self._clone_search_space = clone_search_space

        return self._clone_search_space

    def get_samples_by_sample_name(self, name, is_convert_number_to_choice=True):
        """
        Generate `n_samples` samples with `n_dim` dimension by method `user`, where
        `n_dim` and `n_samples` are automatically found.

        Parameters
        ----------
        name : str
            The sample_base method,one of random,lhs,halton, sobol

        Returns
        -------
        pd.DataFrame
            The sample result.
        """
        range_search_space = self.get_range_search_space()

        names = []

        # a array contain multi-sample with [lower,upper,upper-lower]
        constraints = []
        # exp_index,_type,_value
        # hidden_activation,choice,[0, 9]
        # latent_dim,quniform,[2, 20, 1]

        for key in range_search_space.keys():
            names.append(key)
            _val = range_search_space[key]["_value"]
            constraints.append([_val[0], _val[1], _val[1] - _val[0]])

        constraints = np.asarray(constraints)
        const_t = constraints.T
        parameter_span = const_t[-1, :][None, :]
        parameter_lower = const_t[0, :]
        X = sample(name, self.n_sample, len(names))
        hp = X * parameter_span + parameter_lower

        # map number to columns
        res = pd.DataFrame(hp, columns=names)
        if is_convert_number_to_choice:
            self.convert_number_to_choice(res)
        else:
            UtilSys.is_debug_mode() and log.info("Pass to convert number to choice")
        UtilSys.is_debug_mode() and log.info(f"Samples result: \n{res}")
        return res

    def convert_number_to_choice(self, res):
        """
        Convert the  number to category. E.g.,



        Examples
        --------
        input:

        f1,f2,f3

        16.92865,1.06248,8.37653

        2.01740,0.45669,7.29109


        Output:

        f1,f2,f3

        16.92865,li,8.37653

        2.01740,zhang,7.29109

        Parameters
        ----------
        res : int
            The result for pure number

        Returns
        -------
        pd.DataFrame
            The hyper-parameters for choice

        """
        UtilSys.is_debug_mode() and log.info("Convert number to choice")
        columns = pd.DataFrame(self.get_range_search_space()).T.reset_index()
        _type_choice = columns.loc[columns['_type'] == self.TYPE_CHOICE]
        # convert number to choice
        _cat_hp: pd.DataFrame = res.loc[:, _type_choice["exp_index"].to_list()]
        for key, val in _cat_hp.iteritems():
            # [3, 8, 4, 7, 0]
            values_codes = val.astype("int").to_list()
            codes, unique = pd.factorize(self.search_space[key]["_value"])
            maps = dict(zip(codes, unique))
            res[key] = values_codes
            res[key] = res[key].map(maps)
        assert res is not None
        return res

    @abc.abstractmethod
    def get_samples(self):
        """
        Get samples.

        Returns
        -------
        pd.DataFrame
            A pd.DataFrame Like column `[hp_1,hp_2,hp_3]`, where each column is a key of hyper-parameters.
        """
