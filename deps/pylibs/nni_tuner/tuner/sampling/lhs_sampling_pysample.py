from tuner.sampling.base_sampling import BaseSampling


class LHSSamplingPySample(BaseSampling):
    """
    A wrapper of LHS sample_base for pysampling
    """

    def __init__(self, sample_size: int, search_space: dict, middle=False, seed=0):
        super().__init__(search_space, sample_size, seed=seed)
        self.n_sample = int(sample_size)
        self.middle = middle

    def get_samples(self, is_convert_number_to_choice=True):
        """Get samples

        Returns
        -------

        """
        return self.get_samples_by_sample_name("lhs", is_convert_number_to_choice=is_convert_number_to_choice)
