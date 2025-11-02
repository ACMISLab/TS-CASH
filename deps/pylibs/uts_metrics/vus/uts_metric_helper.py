import traceback

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from pylibs.utils.util_log import get_logger
from pylibs.uts_metrics.vus.metrics import get_metrics_cacheable, VUSMetrics

log = get_logger()


class UTSMetrics:
    @staticmethod
    def metrics(*, score: np.ndarray, label: np.ndarray, window_size: int):
        """
        VUS_ROC, VUS_PR, BEST_F1_SCORE =
        Parameters
        ----------
        score :
        label :
        window_size :

        Returns
        -------

        """
        VUS_ROC, VUS_PR = UTSMetrics.vus_metrics(score=score, label=label, window_size=window_size)
        BEST_F1_SCORE = UTSMetrics.best_f1_score(score=score, label=label)
        return VUS_ROC, VUS_PR, BEST_F1_SCORE

    @staticmethod
    def vus_metrics(*, score: np.ndarray, label: np.ndarray, window_size: int):
        """
        Return the VUS ROC and VUS PR scores

        VUS_ROC, VUS_PR=UTSMetrics.vus_metrics(score, label, window_size)

        Parameters
        ----------
        score :
        label :
        window_size :

        Returns
        -------

        """
        assert score.shape[0] == label.shape[0]
        score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
        mr = VUSMetrics()
        tpr_3d, fpr_3d, prec_3d, window_3d, VUS_ROC, VUS_PR = mr.RangeAUC_volume(
            labels_original=label,
            score=score,
            windowSize=2 * window_size)

        VUS_ROC = np.nan_to_num(VUS_ROC)
        VUS_PR = np.nan_to_num(VUS_PR)
        return float(VUS_ROC), float(VUS_PR)

    @staticmethod
    def best_f1_score(*, score: np.ndarray, label: np.ndarray):
        """
        Return best f1 score

        best_f1_score=UTSMetrics.BEST_F1_SCORE(score, label, window_size)

        Parameters
        ----------
        score :
        label :

        Returns
        -------

        """
        score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
        # 计算F1分数
        f1s = []
        for _thr in np.linspace(0.000001, 1, 101):
            _predict = score >= _thr
            _predict = np.asarray(_predict, dtype=int)
            f1 = f1_score(label, _predict)
            f1s.append(f1)

        return float(np.max(f1s))


class UTSMetricHelper:

    def __init__(self, window_size):
        self.window_size = window_size
        pass

    @staticmethod
    def get_metrics_all(labels, score, window_size, metric_type="all"):
        """
        return:
        {'Precision': 0.9392670157068063, 'Recall': 0.49778024417314093, 'F': 0.6507072905331882, 'AUC_ROC': 0.8743892252460557, 'AUC_PR': 0.6949331870273381, 'Precision_at_k': 0.49778024417314093, 'Rprecision': 0.942357420007146, 'Rrecall': 0.03808188252647562, 'RF': 0.07320543857009269, 'R_AUC_ROC': 0.8576044003109524, 'R_AUC_PR': 0.4500519990413418, 'VUS_ROC': 0.8404911845193188, 'VUS_PR': 0.4398182985228424}

        Parameters
        ----------
        labels :
        score :
        window_size :
        metric_type :

        Returns
        -------

        """
        # Post-processing
        # all
        # if np.sum(labels) == 0:
        #     return None
        # try:
        #     score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
        #     return get_metrics(score, labels, metric=metric_type, window_size=window_size)
        # except:
        #     log.error(traceback.format_exc())
        #     return None
        return UTSMetricHelper.get_metrics_all_cache(labels, score, window_size, metric_type)

    @staticmethod
    def get_metrics_all_cache(labels, score, window_size, metric_type="all"):
        # Post-processing
        try:
            score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
            return get_metrics_cacheable(score, labels, metric=metric_type, window_size=window_size)
        except KeyboardInterrupt:
            return None
        except Exception as e:
            log.error(e)
            return None

    @staticmethod
    def get_none_metrics():
        return {'Precision': -1, 'Recall': -1, 'F': -1, 'AUC_ROC': -1, 'AUC_PR': -1,
                'Precision_at_k': -1, 'Rprecision': -1, 'Rrecall': -1, 'RF': -1,
                'R_AUC_ROC': -1, 'R_AUC_PR': -1, 'VUS_ROC': -1, 'VUS_PR': -1}

    @staticmethod
    def vus_accuracy(*, score, label, window_size=64):
        """
        Return the VUS ROC and VUS PR scores

        VUS_ROC, VUS_PR=UTSMetricHelper.vus_accuracy(score, label, window_size)

        Parameters
        ----------
        score :
        label :
        window_size :

        Returns
        -------

        """
        score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
        tpr_3d, fpr_3d, prec_3d, window_3d, VUS_ROC, VUS_PR = VUSMetrics().RangeAUC_volume(
            labels_original=label,
            score=score,
            windowSize=2 * window_size)
        return VUS_ROC, VUS_PR

    @staticmethod
    def BEST_F1_SCORE(*, score, label):
        """
        Return the VUS ROC and VUS PR scores

        VUS_ROC, VUS_PR=UTSMetricHelper.vus_accuracy(score, label, window_size)

        Parameters
        ----------
        score :
        label :

        Returns
        -------

        """
        score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
        # 计算F1分数
        f1s = []
        for _thr in np.linspace(0.000001, 1, 101):
            _predict = score >= _thr
            _predict = np.asarray(_predict, dtype=int)
            f1 = f1_score(label, _predict)
            f1s.append(f1)

        return np.max(f1s)


if __name__ == '__main__':
    print(UTSMetrics.best_f1_score(score=np.asarray([0.1, 0.5, 1]), label=np.asarray([0, 1, 1])))
    print(UTSMetrics.vus_metrics(score=np.asarray([0.1, 0.5, 1]), label=np.asarray([0, 1, 1])))
