import numpy as np

from pylibs.utils.util_message import logi, logs
from pylibs.utils.util_numpy import enable_numpy_reproduce

from timeseries_models.isolation_forest.isolation_forest_conf import IsolationForestConf
from timeseries_models.isolation_forest.isolation_forest_model import IsolationForestModel

train_x = np.asarray([[-1.1], [0.3], [0.5], [100]])
test_x = np.asarray([[0.1], [0], [90]])
test_y = np.asarray([0, 0, 1])
enable_numpy_reproduce(1)
iforest = IsolationForestModel(IsolationForestConf())
iforest.fit(train_x)
logs(f"source: \n{test_x}")
logs(f"score: {iforest.score(test_x)}")
logs(f"predict: {iforest.predict(test_x)}.")
iforest.report_metrics(test_x, test_y, test_x, test_y)
from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView

av = UnivariateTimeSeriesView()
av.plot_kpi_with_anomaly_score_row2_with_best_threshold_numpy(test_x[:, -1], test_y, iforest.score(test_x))
