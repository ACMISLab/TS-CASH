import datetime
import os

import numpy as np
import sklearn

from pylibs.utils.util_log import get_logger

log = get_logger()
from pylibs.utils.util_network import get_random_idle_port as idle_port


def proprocess(x, y, slide_win=120):
    """
    Standardize(zero mean ??) and fill missing with zero

    :param sliding_window: sliding widget size
    :param x: 1-D array
        origin kpi
    :param y: 1-D array
        label
    :return: zero mean standardized kpi,
    """
    ret_x, ret_y = slide_sampling(x, y, slide_win=slide_win)
    return ret_x, ret_y


def slide_sampling(x, y, slide_win):
    ret_x = []
    ret_y = []
    for i in range(len(x) - slide_win + 1):
        ret_x.append(x[i: i + slide_win])
        ret_y.append(y[i: i + slide_win])
    ret_x = np.array(ret_x)
    ret_y = np.array(ret_y)
    return ret_x, ret_y


def modified_score(y_true, score):
    """
    Modified metrics;that is, identify the anomaly by segment, not a point.
    If any point in an anomaly segment in the ground truth can be detected by a chosen threshold,
    we say this segment is detected correctly, and all points in this segment are treated as if they can be detected
    by this threshold.
    Examples
    ----------
    y_true =    [1,      1,      1,       0, 0, 0,   1,    1,      1,      1,     0,    1]
    score  =    [-0.1,  -0.3,   -0.1,     0, 0, 0,   0,    0,      0,     -0.5,   0,    0]

    modified =  [-0.3,  -0.3,   -0.3,     0, 0, 0,  -0.5,  -0.5,  -0.5,   -0.5,   0,    0]
    Parameters
    ----------
    score: 1-D np.array
    y_true: 1-D np.array

    Returns
    -------
    1-D np.array

    """

    y_true = np.asarray(y_true, dtype=float)
    score = np.asarray(score, dtype=float)

    assert y_true.shape[0] == score.shape[0]
    assert len(y_true.shape) == 1
    assert len(score.shape) == 1
    modified_score = score.copy()

    anomaly_span_index_of_truth = []
    size_of_y = len(y_true)
    _s = 0
    _n = 0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            _n = i + 1
            if i == size_of_y - 1:
                anomaly_span_index_of_truth.append(np.arange(start=_s, stop=_n, step=1))
        else:
            if _n > _s:
                anomaly_span_index_of_truth.append(np.arange(start=_s, stop=_n, step=1))
            _s = i + 1

    for ts in anomaly_span_index_of_truth:
        modified_score[ts] = np.repeat(np.min(modified_score[ts]), len(ts))
    return modified_score


def yaml_to_json(yamlPath):
    import yaml
    import json
    # yaml文件内容转换成json格式
    with open(yamlPath, encoding="utf-8") as f:
        datas = yaml.load(f, Loader=yaml.FullLoader)  # 将文件的内容转换为字典形式
    jsonDatas = json.dumps(datas, indent=5)  # 将字典的内容转换为json格式的字符串
    print(jsonDatas)


def valid_search_space(search_space, model="AE"):
    """
    Convert search space to valid search space
    Parameters
    ----------
    search_space: dict

    model:str
        AE
        cur_search_space_ = {
             'hidden_layers': 2,
            'hidden_neurons': 120,
            'latent_dim': 5,
            'window_size':120,
            'hidden_activation': "relu",
            'output_activation': "tanh",
            'epochs': 255,
            'batch_size': 120,
            'dropout_rate': 0.1,
            'l2_regularizer': 0.1,
        }
        # encoder_neurons = None, *******
        # decoder_neurons = None, ******

        # 1     latent_dim = 2,
        # 2     hidden_activation = 'relu',
        # output_activation = 'sigmoid',
        # epochs = 100,
        # batch_size = 32,
        # dropout_rate = 0.2,
        # l2_regularizer = 0.1

    Returns
    -------

    """
    # Convert user of search space
    search_space["latent_dim"] = int(search_space["latent_dim"])
    search_space["window_size"] = int(search_space["window_size"])
    search_space["epochs"] = int(search_space["epochs"])
    search_space["batch_size"] = int(search_space["batch_size"])
    search_space["batch_size"] = int(search_space["batch_size"])

    # pass dropout_rate for float
    # pass l2_regularizer for float
    if search_space.get("lr_step"):
        search_space.update({
            "lr_step": int(search_space.get("lr_step"))
        })

    search_space["hidden_neurons"] = int(search_space["hidden_neurons"])
    search_space["hidden_layers"] = int(search_space["hidden_layers"])
    search_space["encoder_neurons"] = []
    search_space["decoder_neurons"] = []
    for _ in range(search_space['hidden_layers']):
        search_space["encoder_neurons"].append(search_space["hidden_neurons"])
        search_space["decoder_neurons"].append(search_space["hidden_neurons"])

    del search_space["hidden_neurons"]
    del search_space["hidden_layers"]
    return search_space


def get_metrics(y_true, mscores):
    """
    Get metrics of auc, and the P、R、F1 for different probability threshold
    Parameters
    ----------
    y_true: label of shape (n_sample,)
    mscores: normal probability of shape (n_sample,)

    Returns
    -------
    list
         auc,thresholds,precisions,recalls,f1_scores
    """
    # Negative the scores, because it is the log probability

    mscores = -mscores
    _min = np.min(mscores)
    _max = np.max(mscores)

    # Normal the score
    scores_normal = (mscores - _min) / (_max - _min + 1e-8)

    # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)  from prediction scores.
    auc = sklearn.metrics.roc_auc_score(y_true, scores_normal)

    # Compute precision-recall pairs for different probability thresholds.
    precisions, recalls, thresholds = sklearn.metrics.precision_recall_curve(y_true, scores_normal)

    # Compute F1 for different probability threshold
    f1_scores = 2.0 * precisions * recalls / np.clip(precisions + recalls, a_min=1e-4, a_max=None)
    return auc, thresholds, precisions, recalls, f1_scores


def report_final_result_to_console(BEST_F1_SCOREs, aucs):
    print(f"=========== Final result ========="
          f"\nAuc: {np.round(np.mean(aucs) * 100, 4)},"
          f"\nbest F-Score: {np.round(np.mean(BEST_F1_SCOREs) * 100, 4)}")


def report_intermediate_result_to_console(k_fold_index, auc, BEST_F1_SCOREs, best_precsion=None, best_recall=None):
    print(f"------------ Intermediate result ------------"
          f"\nFold index: {k_fold_index}, "
          f"\nAuc: {auc}, "
          f"\nbest F-Score {BEST_F1_SCOREs}"
          f"\nbest Prec {best_precsion}"
          f"\nbest Recall {best_recall}")


def get_random_file_name(ext):
    home = "runtime"
    if not os.path.exists(home):
        os.makedirs(home)

    return os.path.join(home, f"{datetime.datetime.now().microsecond}.{ext}")


def get_random_idle_port():
    return idle_port()


def convert_str_to_float(float_str):
    if float_str is None or str(float_str).lower() == "none":
        return None
    elif isinstance(float_str, str) and float_str.find("/") > -1:
        dividend, divisor = float_str.split("/")
        float_str = int(dividend) / int(divisor)

    else:
        float_str = float(float_str)
    return float_str
