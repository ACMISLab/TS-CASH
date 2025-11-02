from pylibs.dataset.DatasetAIOPS2018 import DatasetAIOps2018, AIOpsKPIID
from timeseries_models.isolation_forest.isolation_forest_conf import IsolationForestConf

BEST_MODEL_NAME = "model_aiops_if_best.m"
WORST_MODEL_NAME = "model_aiops_if_worst.m"

_best_conf = {
    "window_size": 13.014162565200778,
    "n_estimators": 427.64040215616075
}

_worst_conf = {
    "window_size": 25.616942433075963,
    "n_estimators": 51.913167049576515
}


def get_best_conf() -> IsolationForestConf:
    ifc = IsolationForestConf()
    ifc.update_parameters(_best_conf)
    return ifc


def get_worst_conf() -> IsolationForestConf:
    ifc = IsolationForestConf()
    ifc.update_parameters(_worst_conf)
    return ifc


# best 431a8542 D7
# worst  6d1114ae D13
valid_rate = 0.5


def get_best_kpi_data():
    """
    x, train_y, valid_x, valid_y, test_x, test_y = get_best_kpi_data()

    Returns
    -------

    """

    da = DatasetAIOps2018(kpi_id=AIOpsKPIID.D7,
                          windows_size=get_best_conf().window_size,
                          is_include_anomaly_windows=True,
                          sampling_rate=1,
                          valid_rate=valid_rate)
    return da.get_sliding_windows_three_splits()


def get_worst_kpi_data():
    """
    x, train_y, valid_x, valid_y, test_x, test_y = get_worst_kpi_data()
    Returns
    -------

    """

    da = DatasetAIOps2018(kpi_id=AIOpsKPIID.D13,
                          windows_size=get_worst_conf().window_size,
                          is_include_anomaly_windows=True,
                          sampling_rate=1,
                          valid_rate=valid_rate)
    return da.get_sliding_windows_three_splits()
