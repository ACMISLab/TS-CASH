from pylibs.evaluation.utils import from_list_points_labels

from pylibs.evaluation.contextual import contextual_confusion_matrix, contextual_precision, contextual_recall, \
    contextual_f1_score


def test_get_sliding_windows():
    anomalies = [0, 0, 1, 1, 1, 1, 0, 0, 1, 1]
    returned = from_list_points_labels(anomalies)
    print("")
    print(returned)
    print(contextual_precision(returned, [[2, 3]], weighted=False))
    print(contextual_recall(returned, [[2, 3]], weighted=False))
    print(contextual_f1_score(returned, [[2, 3]], weighted=False))
    print(contextual_confusion_matrix(returned, [[2, 3]], weighted=False))
