"""
This class represents a web interface of nni.

It only works when the server of a specific experiment is running.
"""
import json

import numpy as np
import pandas as pd
import requests

from pylibs.utils.util_http import http_json_get
from pylibs.utils.util_log import get_logger

log = get_logger()


class NNIWebInterface:
    KEY_TRIALJOBID = 'trialJobId'
    KEY_STATUS = 'status'
    KEY_HYPERPARAMETERS = 'hyperParameters'
    KEY_LOGPATH = 'logPath'
    KEY_STARTTIME = 'startTime'
    KEY_SEQUENCEID = 'sequenceId'
    KEY_ENDTIME = 'endTime'
    KEY_STDERRPATH = 'stderrPath'
    STATUS_FAILED = 'FAILED'

    def __init__(self, port=8080, ip="your_server_ip"):
        self._ip = ip
        self._port = port
        self._host = f"http://{self._ip}:{self._port}"

        if not self._is_backend_available():
            raise RuntimeError(f"Backend [{self._host}] is not available")

    @property
    def _api_trials_job(self):
        # method GET
        return f"{self._host}/api/v1/nni/trial-jobs"

    @property
    def _api_experiment_meta(self):
        """

        Returns
        -------

        """
        return f"{self._host}/api/v1/nni/experiment-metadata"

    @property
    def _api_experiment(self):
        return f"{self._host}/api/v1/nni/experiment"

    @property
    def _api_metric_data(self):
        """

        Returns
        -------

        """
        return f"{self._host}/api/v1/nni/metric-data"

    def is_all_trials_success(self):
        """
        True if all trials are successful.
        False if any trials is failed.

        Returns
        -------

        """
        return len(self.get_failed_trials()) == 0

    def has_failed_trials(self):
        return not self.is_all_trials_success()

    def get_failed_trials(self):
        trials = self.get_all_trials()
        failed_trials = trials.loc[trials[self.KEY_STATUS] == self.STATUS_FAILED]
        return failed_trials

    def get_all_trials(self):
        """
        Get all trials of the experiment with columns
        ['trialJobId', 'status', 'hyperParameters', 'logPath', 'startTime', 'sequenceId', 'endTime', 'stderrPath']

        Returns
        -------
        pd.DataFrame

        """
        data = http_json_get(self._api_trials_job)
        trials = pd.DataFrame(data)
        return trials

    def get_all_trials_metrics(self):
        """
        Get all metrics of every trial as a pd.DataFrame with keys

        Returns
        -------

        """
        data = http_json_get(self._api_metric_data)

        _trials_metrics = []
        if data is None:
            return None
        for trials in data:
            metric = json.loads(json.loads(trials['data']))
            _trials_metrics.append(metric)
        return pd.DataFrame(_trials_metrics)

    def get_all_default_metrics(self, ret_type="numpy"):
        if ret_type == "json":
            return self.get_all_trials_metrics().loc[:, ["test_affiliation_f1"]].to_json()
        elif ret_type == "numpy":
            return np.sort(np.reshape(self.get_all_trials_metrics().loc[:, ["test_affiliation_f1"]].to_numpy(), -1))
        else:
            return self.get_all_trials_metrics().loc[:, ["test_affiliation_f1"]]

    def get_mean_default_metrics(self):
        res = self.get_all_default_metrics(ret_type="numpy")
        res = res[~np.isnan(res)]
        return np.mean(res)

    def _is_backend_available(self):
        try:
            req = requests.get(self._api_experiment_meta)
            if req.status_code == 200:
                return True
            else:
                return False
        except Exception as e:
            return False


if __name__ == '__main__':
    trials = NNIWebInterface(port=50751).get_all_default_metrics()
    UtilSys.is_debug_mode() and log.info(f"get_all_default_metrics:\n {trials}\n\n\n")

    trials = NNIWebInterface(port=61885).get_all_trials_metrics()
    UtilSys.is_debug_mode() and log.info(f"get_all_trials_metrics:\n {trials}\n\n\n")

    status = NNIWebInterface(port=61885).is_all_trials_success()
    UtilSys.is_debug_mode() and log.info(f"is_all_trials_success: {status}")

    status = NNIWebInterface(port=333).is_all_trials_success()
    UtilSys.is_debug_mode() and log.info(f"is_all_trials_success: {status}")
