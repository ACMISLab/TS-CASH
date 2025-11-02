import numpy as np
from typeguard import typechecked


def unroll_ts(y_hat, full: bool = False):
    """
    To reassemble or “unroll” the predicted signal X_hat we can choose different aggregation methods (e.g., mean,
    max, etc). In our implementation, we chose it to as the median value.

    如: 预测结果是 [[1,     2,     3],  [4,      5,    5]], 那么我们将窗口排列(一个窗口表示一个数据点):

    [1,     2,      3]
           [4,      5,     5]

    取每个位置对应的均值
    [1     4+2/2   5+3/2   5]

    将会得到最终的预测数据:

    [1,3,4,5], 但是由于两个窗口只有两个有效值(两个标签, 因此, 最终我们会得到 [4,5]


    details see: https://medium.com/mit-data-to-ai-lab/time-series-anomaly-detection-in-the-era-of-deep-learning-f0237902224a

    Parameters
    ----------
    full : bool
        Whether return the all dataset.
    y_hat :

    Returns
    -------

    """
    predictions = list()
    pred_length = y_hat.shape[1]
    num_errors = y_hat.shape[1] + (y_hat.shape[0] - 1)

    for i in range(num_errors):
        intermediate = []

        for j in range(max(0, i - num_errors + pred_length), min(i + 1, pred_length)):
            intermediate.append(y_hat[i - j, j])

        if intermediate:
            predictions.append(np.median(np.asarray(intermediate)))

    if full:
        return np.asarray(predictions)
    else:
        return np.asarray(predictions[pred_length - 1:])


@typechecked
def unroll_ts_torch(y_hat, remove_front=False):
    """
    A pytorch implementation of unroll_ts.

    This function is faster than unroll_ts 22x
    unroll_ts_torch： 0.6283605829999999
    unroll_ts： 14.383021291

    Parameters
    ----------
    y_hat :

    Returns
    -------

    """
    import torch
    batch_size = y_hat.shape[0]
    window_size = y_hat.shape[1]

    # 水平翻转
    v_flip_x = torch.flip(y_hat, dims=[1])
    list_arr = []
    for _offset in torch.arange(start=-(batch_size - 1), end=window_size, step=1):
        cur_x = torch.diagonal(v_flip_x, offset=_offset)
        list_arr.insert(0, torch.quantile(cur_x, 0.5))
    if remove_front:
        return torch.stack(list_arr[window_size - 1:])
    else:
        return torch.stack(list_arr)


if __name__ == '__main__':
    from numpy.testing import assert_almost_equal

    a = unroll_ts(np.asarray([[1, 2, 3], [4, 5, 5]]))
    assert_almost_equal(a, [4, 5])
    print(a)
