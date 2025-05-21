import sys

import numpy as np
import pandas as pd
from tuner.sampling.base_sampling import BaseSampling


class RandomSampling(BaseSampling):
    def __init__(self, sample_size: int, search_space: dict, seed=0):
        """
        Parameters
        ----------
        sample_size
        search_space:dict
            a dict from nni https://nni.readthedocs.io/en/stable/hpo/search_space.html
            {
              'latent_dim': {'_type': 'quniform', '_value': [2, 20, 1]},
              'res': {'_type': 'quniform', '_value': [3, 8, 1]}
             }
        """

        super().__init__(search_space, sample_size, seed=seed)
        if type(sample_size) != int:
            sample_size = int(sample_size)

        self.result = None
        self.type = "Random"
        self.search_space = search_space
        self.num_samples = sample_size

    def get_samples(self):
        """
        Generate samples.

        Examples
        -------
        Returns
        -------
        pandas.DataFrame
                    Like:
                       Fx        Fy         Fz
                    0  6.381344  42.681766  1805.250611
                    1  7.212056  81.165833  1954.240508
                    2  4.919377  83.294901  1075.293490
                    3  3.776898  95.213013  1467.894749
                    4  0.453047  14.260119  2418.262023
                    5  5.845662  53.932508  2804.997941
                    6  9.702729  24.696960  2485.764319
                    7  2.661000  63.913773  1267.641317
                    8  8.287838  28.919920   397.066531
                    9  1.341232  66.298533   608.920244
        """

        result = {}
        for key in self.search_space.keys():
            val = self.search_space.get(key)
            if val is None:
                print("Val is None", file=sys.stderr)
                continue
            if val['_type'] == "uniform":
                result[key] = np.random.random(size=self.num_samples) * (val['_value'][1] - val['_value'][0])
            if val['_type'] == "quniform":
                result[key] = np.clip(np.round(
                    np.random.uniform(val['_value'][0], val['_value'][1], self.num_samples) / val['_value'][2]) *
                                      val['_value'][2], val['_value'][0], val['_value'][1])
                # result[key] = np.repeat(np.random.random(size=self.num_samples) * (val['_value'][1] - val['_value'][0]))
            if val['_type'] == "choice":
                result[key] = np.random.randint(0, len(val['_value']), size=self.num_samples)

        sample_data = pd.DataFrame(result)
        return sample_data
