import random

import numpy
import numpy as np

import logging

log = logging.getLogger(__name__)


def set_display_style(prec=4):
    # 设置显示4位小数(打印的时候)
    np.set_printoptions(precision=prec)


def enable_numpy_reproduce(seed=42):
    log.debug(f"Enable reproducible generation: {seed}")
    random.seed(seed)
    numpy.random.seed(seed)


def fill_nan_inf(arr: np.ndarray, fill_value):
    """
    Fill the np.nan, np.inf as fill_value

    Parameters
    ----------
    fill_value : float

    arr : np.ndarray
        The array to fill
    Returns
    -------

    """
    assert isinstance(arr, np.ndarray), "arr must be a np.ndarray"

    _array = arr.copy()
    _array[np.isnan(arr)] = fill_value
    _array[np.isinf(arr)] = fill_value

    return _array


def dist_eu(x1, x2):
    return np.sum(np.power(x1 - x2, 2), axis=1)


def feature_(x1):
    _max = np.percentile(x1, 95, axis=1)
    _min = np.percentile(x1, 5, axis=1)
    # 考虑不同区间,平方防止正反相减
    _f1 = np.power(_max, 2) + np.power(_min, 2)

    _dist = np.sum(np.power(x1 - 0, 2), axis=1)
    # 考虑窗口与0的距离
    _f2 = _f1 + _dist

    # 归一化
    return (_f2 - _f2.min()) / (_f2.max() - _f2.min())


def fill_nan_by_mean(nd_array: np.ndarray):
    nd_array[np.isnan(nd_array)] = np.nanmean(nd_array)
    return nd_array


if __name__ == '__main__':
    # x1 = np.asarray([
    #     [-101, -2, 3],
    #     [1, 2, 3],
    #     [4, 5, 3],
    #     [100, 200, 150],
    #     [100, 200, 2000],
    # ])
    # print(feature_(x1))
    # print(dist_eu(x1, x1[0]))
    # print(dist_eu(x1, x1))
    # print(np.max(x1, axis=1))

    data = np.array([5.0, 2.0, np.nan, 7.0, np.nan, 3.0, 8.0, np.nan])
    print(fill_nan_by_mean(data))
