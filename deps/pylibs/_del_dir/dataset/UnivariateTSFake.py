import numpy as np
import pandas as pd
from pylibs.dataset.DatasetAIOPS2018 import DatasetAIOps2018

from pylibs.utils.util_numpy import enable_numpy_reproduce

enable_numpy_reproduce(0)


def generate_fake_period_anomaly_unclear():
    """
    Returns

    x, train_y, valid_x, valid_y, test_x, test_y = da.get_sliding_windows_three_splits()

    """
    _n_samples = 1024
    x = np.sin(np.linspace(start=-30 * np.pi, stop=30 * np.pi, num=_n_samples) + np.random.random(_n_samples) * 0.2)
    y = np.zeros((_n_samples,))
    error_index = [30, 202, 334, 555, 888, 980, 981]
    # error_index = []
    for index in error_index:
        x[index] = x[index] + np.random.random() * 2
        y[index] = 1

    data = pd.DataFrame({
        'timestamp': x,
        'value': x,
        'label': y

    })
    DatasetAIOps2018.save_dataset_to_db_v2(kpi_id=DatasetAIOps2018.KPI_IDS.TEST_PERIOD_VAGUE,
                                           train_data=data,
                                           test_data=data,
                                           preprocessing="standardize")
    return DatasetAIOps2018.KPI_IDS.TEST_PERIOD_VAGUE


def generate_fake_liner():
    _n_samples = 1024

    x = np.random.standard_normal((_n_samples,))
    y = np.zeros((_n_samples,))
    error_index = [30, 202, 334, 555, 888, 980]
    for index in error_index:
        x[index] = 30 * np.random.random()
        y[index] = 1

    data = pd.DataFrame({
        'timestamp': x,
        'value': x,
        'label': y

    })
    n = DatasetAIOps2018.save_dataset_to_db_v2(kpi_id=DatasetAIOps2018.KPI_IDS.TEST_01, train_data=data, test_data=data,
                                               preprocessing="standardize")
    return DatasetAIOps2018.KPI_IDS.TEST_01


def generate_fake_liner_anomaly_unclear():
    _n_samples = 1024
    kpi_id = DatasetAIOps2018.KPI_IDS.TEST_LINE_VAGUE
    x = np.random.standard_normal((_n_samples,))
    y = np.zeros((_n_samples,))
    error_index = [30, 202, 334, 555, 888, 980]
    for index in error_index:
        x[index] = 5 * np.random.random()
        y[index] = 1

    data = pd.DataFrame({
        'timestamp': x,
        'value': x,
        'label': y

    })
    DatasetAIOps2018.save_dataset_to_db_v2(kpi_id=kpi_id, train_data=data,
                                           test_data=data,
                                           preprocessing="standardize")
    return kpi_id
