import numpy as np

from pylibs.utils.util_numpy import enable_numpy_reproduce
from pylibs.utils.util_joblib import cache_


# @cache_
def latin_hypercube_sampling(data_size, num_samples):
    """
    一维数据的拉丁超立方体抽样.


    Parameters
    ----------
    data_size : 数量大小
    num_samples :  要抽取的数量

    Returns
    -------
    返回抽样的索引

    """
    if num_samples >= data_size:
        return np.arange(data_size)
    # Generate random uniform intervals for each sample
    intervals = np.linspace(0, 1, num_samples + 1)

    # Calculate the start_or_restart and end indices for each sample
    start_indices = (data_size * intervals[:-1]).astype(int)
    end_indices = (data_size * intervals[1:]).astype(int)

    # Choose a random exp_index within each interval for each sample
    indices = [np.random.randint(start, end) for start, end in zip(start_indices, end_indices)]

    return indices


if __name__ == '__main__':
    # Usage example
    # enable_numpy_reproduce()
    # data_size = 10000  # Data size
    # num_samples = 40  # Number of samples
    # sample_indices = latin_hypercube_sampling(data_size, num_samples)
    # print(sample_indices)
    # [102, 429, 592, 764, 1106, 1321, 1688, 1770, 2102, 2371, 2710, 2964, 3074, 3452, 3587, 3866, 4099, 4353, 4651, 4880, 5149, 5302, 5501, 5837, 6235, 6407, 6537, 6879, 7191, 7437, 7520, 7910, 8203, 8307, 8521, 8985, 9088, 9298, 9718, 9808]
    print(latin_hypercube_sampling(100, 1000))
    print(latin_hypercube_sampling(10, 10))
    print(latin_hypercube_sampling(10, 5))
    print(latin_hypercube_sampling(10, 0))
