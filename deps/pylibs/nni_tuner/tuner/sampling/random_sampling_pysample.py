from tuner.sampling.base_sampling import BaseSampling


class RandomSamplingPySample(BaseSampling):
    """
    A wrapper of random sample_base for pysampling
    """

    def __init__(self, sample_size: int, search_space: dict, seed=0):
        super().__init__(search_space, sample_size, seed=seed)
        if type(sample_size) != int:
            sample_size = int(sample_size)

        self.result = None
        self.type = "Random"
        self.search_space = search_space
        self.num_samples = sample_size

    def get_samples(self):
        return self.get_samples_by_sample_name("random")
