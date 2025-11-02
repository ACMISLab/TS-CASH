import pprint
import nni
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve

from pylibs.common import ConstMetric
from pylibs.utils.util_log import get_logger

log = get_logger()


def report_nni_final_result(y_true, score):
    auc = roc_auc_score(y_true, score)

    precisions, recalls, thresholds = precision_recall_curve(y_true, score)

    # 拿到最优结果以及索引
    _l = np.asarray(1e-10) if np.sum(precisions + recalls) == 0 else precisions + recalls

    f1_scores = (2 * precisions * recalls) / _l
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])

    # 阈值
    result = {
        "default": best_f1_score,
        "best_prec": precisions[best_f1_score_index],
        "best_recall": recalls[best_f1_score_index],
        "best_f1": best_f1_score,
        "auc": auc

    }
    UtilSys.is_debug_mode() and log.info(f"Intermediate result:\n{pprint.pformat(result)}")
    nni.report_final_result(result)


def report_nni_final_result_with_loss(y_true, score, valid_loss=None, model_id=None, time_elapse=None):
    """
    Report valid_loss(default), best_prec,best_recall,best_f1,auc to final result of nni

    Parameters
    ----------
    time_elapse : dict
        The time elapse dict
    model_id :
    y_true : list
        The label of the data
    score : list
        The predict score of the record
    valid_loss :float
        The valid_loss of the test data

    Returns
    -------

    """
    # todo: to fix ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.
    # by replace `roc_auc_score(y_true, score)` to `keras.metrics.AUC()`
    # auc = roc_auc_score(y_true, score)
    from tensorflow import keras
    m = keras.metrics.AUC()
    # m.reset_state()
    m.update_state(y_true, score)
    auc = m.result().numpy()

    precisions, recalls, thresholds = precision_recall_curve(y_true, score)

    # Fix bugs that divided by 0
    divisor = (precisions + recalls) + 1e-10
    f1_scores = (2 * precisions * recalls) / divisor
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])

    # The result
    result = {
        ConstMetric.KEY_DEFAULT: valid_loss,
        ConstMetric.KEY_BEST_PRECISION: precisions[best_f1_score_index],
        ConstMetric.KEY_BEST_RECALL: recalls[best_f1_score_index],
        ConstMetric.KEY_BEST_F1: best_f1_score,
        ConstMetric.KEY_AUC: auc,
        ConstMetric.KEY_VALID_LOSS: valid_loss,
        ConstMetric.KEY_MODEL_ID: model_id

    }

    if time_elapse is not None:
        for key, val in time_elapse.items():
            result[key] = val

    UtilSys.is_debug_mode() and log.info(f"Final result:\n{pprint.pformat(result)}")
    click.echo(result)

    nni.report_final_result(result)
    return result


def report_nni_final_valid_loss(valid_loss=None, auc=None, model_id=None, time_elapse=None):
    """
    Report  metric to nni.

    Parameters
    ----------
    auc :
    time_elapse : dict
        The time elapse dict
    model_id :
    valid_loss :float
        The valid_loss of the test data

    Returns
    -------

    """
    # by replace `roc_auc_score(y_true, score)` to `keras.metrics.AUC()`
    # auc = roc_auc_score(y_true, score)

    # The result
    result = {
        ConstMetric.KEY_METRIC_AUC: auc,
        ConstMetric.KEY_DEFAULT: valid_loss,
        ConstMetric.KEY_VALID_LOSS: valid_loss,
        ConstMetric.KEY_MODEL_ID: model_id

    }

    if time_elapse is not None:
        for key, val in time_elapse.items():
            result[key] = val

    UtilSys.is_debug_mode() and log.info(f"Final result:\n{pprint.pformat(result)}")

    nni.report_final_result(result)
    return result


def report_nni_final_metric(**kwargs):
    """
    Report  metric to nni.

    Examples
    --------
    .. code-block::

        res = report_nni_final_metric(auc1=3, auc2=4)
        assert res == {'auc1': 3, 'auc2': 4}

        res = report_nni_final_metric(auc1=3, auc2=4, **{'key1': 3})
        assert res == {'auc1': 3, 'auc2': 4, 'key1': 3}

    """

    # The result
    if kwargs.get("default") is None:
        raise ValueError("Report metric must contain a key named default according to the NNI. More details see "
                         "https://nni.readthedocs.io/en/stable/reference/hpo.html#nni.report_final_result")
    # log.info(f"Metric report: \n{pprint.pformat(kwargs)}")
    nni.report_final_result(kwargs)

    return kwargs
