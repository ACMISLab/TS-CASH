import os
import sys

from pylibs.uts_models.benchmark_models.a01_observation.util_observation import ObservationsUtil
from pylibs.utils.util_common import UC
from pylibs.utils.util_gnuplot import Gnuplot, UTSViewGnuplot
from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView
import argparse
from pylibs.uts_dataset.dataset_loader import DatasetLoader, DatasetType, KFoldDatasetLoader, _get_selected_avaliable_data

par = argparse.ArgumentParser()
par.add_argument("-d", "--dataset", default="ECG")
par.add_argument("-n", "--number", default=10000, type=int, help="How many data points to show. ")
args = par.parse_args()

datas = _get_selected_avaliable_data(args.dataset, top_n=100000)
for k, v in datas:
    values, labels = ObservationsUtil.load_observation_data_origin(dataset_name=k, data_id=v)
    gp = UTSViewGnuplot(home=os.path.join(UC.get_entrance_directory(), "plots"))
    gp.plot_uts_data_without_score(values, labels, w=1600, h=400, fname=f"{k}_{v}")
