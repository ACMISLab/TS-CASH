import numpy as np

from pylibs.utils.util_message import log_metric_msg
from pylibs.utils.util_metrics import affiliation_metrics
from pylibs.utils.util_numpy import fill_nan_inf


class AffiliationMetrics:

    def __init__(self, threshold: float = None, percentile: float = None, precision: float = None,
                 recall: float = None,
                 f1: float = None):
        # threshold(the threshold(float) corresponding to the best _percentile) ,
        self.threshold = threshold
        # percentile(the best percentile with best f1)
        self.percentile = percentile
        # precision(metric),
        self.precision = precision
        # recall(metric),
        self.recall = recall
        # f1(affiliation f1, metric)
        self.f1 = f1


class ModelPerformanceHelper:
    def __init__(self):
        pass

    @staticmethod
    def find_best_affiliation_score_and_threshold(label, scores, log_best=True):
        """

        Parameters
        ----------
        label : np.ndarray
            The ground truth
        scores : np.ndarray
            Predicted scores
        log_best : bool
            Whether to print the best metrics.

        Returns
        -------

        """
        _aff_all = []
        _percentiles = np.arange(0.0, 10.5, 0.5)
        for _percentile in _percentiles:
            _threshold = np.percentile(scores, 100 - _percentile)
            predict = np.where(scores > _threshold, 1, 0)
            # Affiliation metrics
            aff_precision, aff_recall, aff_f1 = affiliation_metrics(label, predict)
            _aff_all.append([_percentile, _threshold, aff_precision, aff_recall, aff_f1])

        # Find the maximum affiliation F1 score
        _aff_all = np.asarray(_aff_all)
        _index_aff_all_max = np.argmax(fill_nan_inf(_aff_all[:, -1], -9999))

        # best_affiliation is 5 dimension array:
        # 0: (0-based): _percentile(the best percentile with best f1)
        # 1: _threshold(the threshold(float) corresponding to the best _percentile) ,
        # 2: precision(metric),
        # 3: recall(metric),
        # 4: f1(metric)
        best_affiliation = _aff_all[_index_aff_all_max]
        from prettytable import PrettyTable
        if log_best:
            x = PrettyTable()
            x.field_names = ["type", "percentile", "threshold", "precision", "recall", "f1"]
            x.add_rows(
                [
                    np.concatenate([['affiliation'], best_affiliation]),
                ]
            )
            log_metric_msg(f"Found best metric:\n{x}")

        return AffiliationMetrics(percentile=best_affiliation[0],
                                  threshold=best_affiliation[1],
                                  precision=best_affiliation[2],
                                  recall=best_affiliation[3],
                                  f1=best_affiliation[4])


def subsequences(sequence, window_size, time_step):
    # An array of non-contiguous memory is converted to an array of contiguous memory
    sq = np.ascontiguousarray(sequence)
    a = (sq.shape[0] - window_size + time_step) % time_step
    # label array
    if sq.ndim == 1:
        shape = (int((sq.shape[0] - window_size + time_step) / time_step), window_size)
        stride = sq.itemsize * np.array([time_step * 1, 1])
        if a != 0:
            sq = sq[:sq.shape[0] - a]
    # data array
    elif sq.ndim == 2:
        shape = (int((sq.shape[0] - window_size + time_step) / time_step), window_size, sq.shape[1])
        stride = sq.itemsize * np.array([time_step * sq.shape[1], sq.shape[1], 1])
        if a != 0:
            sq = sq[:sq.shape[0] - a, :]
    else:
        raise RuntimeError('Array dimension error')
    sq = np.lib.stride_tricks.as_strided(sq, shape=shape, strides=stride)
    return sq
