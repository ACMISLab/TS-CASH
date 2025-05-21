import os
import re
import sys
import unittest

import numpy as np

from pylibs.utils.util_bash import exec_cmd_and_return_str
from pylibs.utils.util_log import get_logger

log = get_logger()


class TestIsolationForest(unittest.TestCase):
    def test_factory(self):
        import matplotlib.pyplot as plt
        import numpy as np
        import sklearn
        from matplotlib.axes import Axes
        from matplotlib.figure import Figure

        from hyper_speed.models.isolation_forest.IsolationForestConfig import IsolationForestConf
        from hyper_speed.models.model_utils import parse_model, ModelType
        from pylibs.utils.util_file import grf
        from pylibs.utils.util_metrics import affiliation_metrics
        from pylibs.utils.util_numpy import enable_numpy_reproduce

        enable_numpy_reproduce(1)

        _n_samples = 30

        x = np.random.standard_normal((_n_samples))
        x = np.concatenate([x, [0, 8, 10, 30]])
        x = np.expand_dims(x, axis=1)
        y = np.concatenate([np.zeros(_n_samples), [1, 1, 1, 1]])
        cfg = IsolationForestConf()
        model = parse_model(ModelType.IsolationForest, cfg, "cpu")

        model.fit(x)

        # the score has been negative.
        scores = model.score(x)
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
        percentile_with_max_f1 = 0
        adj_scores = scores
        for _per in np.linspace(0, 20, 200):
            threshold = np.percentile(adj_scores, 100 - _per)
            predict = adj_scores > threshold
            f1 = sklearn.metrics.f1_score(y, predict)
            aff = affiliation_metrics(y, predict)
            print(f"At percentile={_per},threshold={threshold}, f1={f1}, aff={aff}")

            if f1 > max_f1:
                max_f1 = f1
                threshold_with_max_f1 = threshold
                percentile_with_max_f1 = _per
        print(
            f"Max f1={max_f1}, where score (threshold)={threshold_with_max_f1} and percentile={percentile_with_max_f1}")
        png_file = grf(ext=".png")
        fig.savefig(png_file)

        model.report_metrics(x, y, x, y)
        assert max_f1 == 0.8571428571428571

    def test_reproducibility(self):
        #
        # Sample seed
        # home = os.path.abspath("../../")
        # sys.path.append(home)
        # log.info("Start run experiment with same seed at OCSVM and IsolationForest...")
        # result = exec_cmd_and_return_str("make test_ios_forest", timeout=36000, retry=1)
        # log.info(f"Exec result: \n{result}")
        # exec_result = re.findall("Mean default metric is\s*([0-9.]+)", result, re.S)
        # # seed 相同时指标一样
        # res = np.diff(np.asarray(exec_result, dtype=float))
        #
        # assert res[0] == 0
        # assert res[1] != 0
        # assert res[2] == 0
        # log.info("All checking  is successful!")
        pass