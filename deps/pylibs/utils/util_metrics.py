import numpy as np
from pylibs.affiliation.generics import convert_vector_to_events
from pylibs.affiliation.metrics import pr_from_events


def affiliation_metrics(ground_truth: np.ndarray, predict: np.ndarray):
    """
    Calculate the affiliation precision, recall,  f1

    Parameters
    ----------
    ground_truth : np.ndarray
    predict : np.ndarray

    Returns
    -------

    """
    events_gt = convert_vector_to_events(ground_truth)  # [(3, 4), (7, 10)]
    events_pred = convert_vector_to_events(predict)  # [(4, 5), (8, 9)]
    t_range = (0, len(predict))  # (0, 10)
    _metric = pr_from_events(events_pred, events_gt, t_range)

    # the _metric is:
    # {
    #     'individual_precision_distances': [0.5, 0.0],
    #     'individual_precision_probabilities': [0.6363636363636364, 1.0],
    #     'individual_recall_distances': [0.5, 0.3333333333333333],
    #     'individual_recall_probabilities': [0.8181818181818181, 0.8703703703703703],
    #     'precision': 0.8181818181818181,
    #     'recall': 0.8442760942760943
    # }

    precision = _metric['precision']
    recall = _metric['recall']
    if np.isnan(precision):
        precision = 0
    if np.isnan(recall):
        recall = 0

    _dividend = precision + recall
    if _dividend == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1
