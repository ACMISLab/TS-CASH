import numpy as np
import pandas as pd

from pylibs.util_pandas import log_pretty_table
from pylibs.utils.util_message import logs
from pylibs.utils.util_numpy import enable_numpy_reproduce
from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView
from timeseries_models.isolation_forest.isolation_forest_conf import IsolationForestConf
from timeseries_models.isolation_forest.isolation_forest_model import IsolationForestModel

train_x = np.asarray([[-1.1, -2], [0.3, .5], [0.5, 0.2], [-3, 4], [100, 3]])
test_x = np.asarray([[0.1, 0.2], [0, -1], [90, 1], [1, 90], [50, 50], [-2, 2]])
test_y = np.asarray([0, 0, 1, 1, 1, 0])
enable_numpy_reproduce(1)
iforest = IsolationForestModel(IsolationForestConf())
iforest.fit(train_x)
predict = pd.DataFrame({
    "x0": test_x[:, 0],
    "x1": test_x[:, -1],
    "score": iforest.score(test_x),
    "predict": iforest.predict(test_x),
})
logs(f"train:\n{train_x}")
log_pretty_table(predict)
iforest.report_metrics(test_x, test_y, test_x, test_y)
uv = UnivariateTimeSeriesView()
uv.plot_kpi_with_anomaly_score_row2_with_best_threshold_numpy(test_x[:, -1], test_y, iforest.score(test_x))
