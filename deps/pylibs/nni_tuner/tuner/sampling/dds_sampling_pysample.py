import numpy as np

from tuner.sampling.base_sampling import BaseSampling
from tuner.sampling.lhs_sampling_pysample import LHSSamplingPySample


class DDSSamplingPySample(BaseSampling):
    """
    DDS sample_base methods
    """

    def __init__(self, sample_size: int, search_space: dict, middle=False, seed=0, n_lhs_samples=100):
        super().__init__(search_space, sample_size, seed=seed)
        self.sample_size = int(sample_size)
        self.middle = middle
        self.n_lhs_samples = n_lhs_samples

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
        lhs = LHSSamplingPySample(sample_size=self.sample_size, search_space=self.search_space, seed=self.seed,
                                  middle=self.middle)
        samples = []
        distance = []
        for i in range(self.n_lhs_samples):
            _sample = lhs.get_samples(is_convert_number_to_choice=False)
            samples.append(_sample)
            distance.append(self.calculate_sample_distance(_sample))
        res = samples[np.argmax(distance)]
        return self.convert_number_to_choice(res)

    def calculate_sample_distance(self, sample):
        dis = 0
        for i in range(sample.shape[0] - 1):
            dis = dis + np.sqrt(np.square((sample.iloc[i + 1, :] - sample.iloc[i, :]).to_numpy()).sum())
        return dis
