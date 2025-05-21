import os.path

import numpy as np
import pandas as pd
from unittest import TestCase


class TestLogs(TestCase):

    def test_log_file(self):
        a = {
            "nni": 1,
            "val": 3
        }

        metric_01 = {"default": 3,
                     "best_f1": 2}

        names = np.concatenate([list(a.keys()), list(metric_01.keys())])
        values = np.concatenate([list(a.values()), list(metric_01.values())]).tolist()

        for i in range(2):
            csv_file = "debug.csv"
            header = False if os.path.exists(csv_file) is True else True
            pd.DataFrame([values], columns=names).to_csv(csv_file, mode="a", header=header, index_label=False)
