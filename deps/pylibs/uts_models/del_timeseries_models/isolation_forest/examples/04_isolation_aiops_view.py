"""
Before run this script, please make sure you have run the `isolation_aiops_train.py`
"""

import joblib
import argparse

from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView
from timeseries_models.isolation_forest.examples.isolation_aiops_lib import BEST_MODEL_NAME, get_best_kpi_data, \
    WORST_MODEL_NAME, get_worst_kpi_data
from timeseries_models.isolation_forest.isolation_forest_model import IsolationForestModel

parser = argparse.ArgumentParser()
parser.add_argument("--dtype", default="valid", help="The data set type, one of train, valid and test")
parser.add_argument("--mtype", default="worst", help="The model type, one of best or worst")
args = parser.parse_args()

if args.mtype == "best":
    train_x, train_y, valid_x, valid_y, test_x, test_y = get_best_kpi_data()
    _if: IsolationForestModel = joblib.load(BEST_MODEL_NAME)
elif args.mtype == "worst":
    train_x, train_y, valid_x, valid_y, test_x, test_y = get_worst_kpi_data()
    _if: IsolationForestModel = joblib.load(WORST_MODEL_NAME)
else:
    raise RuntimeError(f"Unknown model type {args.mtype}")

x = None
y = None
if args.dtype == "train":
    x = train_x
    y = train_y
elif args.dtype == "valid":
    x = valid_x
    y = valid_y
elif args.dtype == "test":
    x = test_x
    y = test_y
else:
    raise ValueError(f"Unknown dtype {args.dtype}")

score = _if.score(x)
av = UnivariateTimeSeriesView()
av.plot_kpi_with_anomaly_score_row1(x[:, -1], y, score)
