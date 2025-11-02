import pandas as pd

import numpy as np
import sklearn
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from timeseries_models.isolation_forest.isolation_forest_conf import IsolationForestConf
from timeseries_models.isolation_forest.isolation_forest_model import IsolationForestModel

_n_samples = 30

x = np.random.standard_normal((_n_samples))
x = np.concatenate([x, [5, 8, 10, 30]])
x = np.expand_dims(x, axis=1)
y = np.concatenate([np.zeros(_n_samples), [1, 1, 1, 1]])

cfg = IsolationForestConf()
model_if = IsolationForestModel(cfg, "cpu")
model_if.fit(x)

# the score has been negative.
scores = model_if.score(x)
print(f"Score samples: \n {scores[:10]}")

# 创建图
fig: Figure = plt.figure(figsize=(30, 8))
#
# Plot the source data
ax1: Axes = fig.add_subplot(1, 1, 1)

# https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
ax1.scatter(range(len(x)), x, c=np.where(y > 0, 'red', 'blue'), marker="x", label="data")
ax1.set_ylabel("Source Value")
ax1.legend(loc='lower left')
#
# PLot the scores data
ax = ax1.twinx()
ax.plot(scores, marker="^", label="score")
ax.set_ylabel("Score")
ax.legend(loc='center left')

#
# Calculate the f1 scores
max_f1 = 0
threshold_with_max_f1 = 0
adj_scores = scores
for _per in range(100):
    threshold = np.percentile(adj_scores, _per)
    f1 = sklearn.metrics.f1_score(y, adj_scores > threshold)
    print(f"At threshold={threshold}, f1={f1}")

    if f1 > max_f1:
        max_f1 = f1
        threshold_with_max_f1 = threshold
print(f"Max f1={max_f1}, where score (threshold)={threshold_with_max_f1}")
from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView

UnivariateTimeSeriesView._save_fig(fig)
