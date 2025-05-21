import datetime
import json
import os
import re
import time

import pandas as pd
import yaml

from nni.tools.nnictl.nnictl_utils import get_experiment_status
from pylibs.common import Emjoi
from pylibs.utils.util_bash import exec_cmd_and_return_str, exec_cmd
from pylibs.utils.util_feishu import send_msg_to_feishu
from pylibs.utils.util_log import get_logger
from pylibs.utils.util_message import log_warning_msg, log_setting_msg, logi
from pylibs.utils.util_network import get_host_ip
from pylibs.utils.util_nni import get_available_nni_experiment_port
from pylibs.utils.util_sqlite import exec_sql_by_sqlite_file

log = get_logger()


class NNIExperimentInfo:
    """
    Represents an experiment status in NNI.
    """

    def __init__(self, exp_id=None, exp_name=None, exp_port=None, exp_status=None, log_dir=None):
        self.exp_id = exp_id
        self.exp_name = exp_name
        self.exp_port = exp_port
        self.exp_status = exp_status
        self.log_dir = log_dir

    def is_existed(self):
        return self.exp_id is not None

    def get_exp_name(self):
        return self.exp_name

    def get_exp_port(self):
        return self.exp_port

    def get_exp_status(self):
        return self.exp_status

    def get_exp_id(self):
        return self.exp_id

    def __str__(self):
        return "exp_id: {}, exp_name: {}, exp_port: {}, exp_status: {}".format(
            self.exp_id, self.exp_name, self.exp_port, self.exp_status
        )

    def is_stopped_or_view(self):
        return self.is_stopped() or self.is_view()

    def is_stopped(self):
        return self.exp_status == 'STOPPED'

    def is_view(self):
        return self.exp_status == 'VIEWED'

    def is_done(self):
        return self.exp_status == 'DONE'

    def is_all_trails_finished(self):
        """
        Check if all trials of the current experiment is finished.

        Returns
        -------
        bool
            True if all trials of the experiment are finished, False otherwise.

        """
        exp_config_yaml = self._get_exp_config()
        home = self._get_experiment_working_directory(exp_config_yaml)
        db_file = os.path.join(home, self.exp_id, 'db', 'nni.sqlite')

        # max sequence number of the success experiment:
        # select max(sequenceId) from TrialJobEvent where event="SUCCEEDED"
        sql = "select max(sequenceId) from TrialJobEvent where event='SUCCEEDED'"
        records, columns = exec_sql_by_sqlite_file(db_file, sql)
        total_trials = exp_config_yaml['maxTrialNumber']
        if isinstance(records, list) and len(records) > 0:
            max_success_sequence = records[0][0]
            return total_trials - 1 == max_success_sequence
        else:
            return False

    def _get_experiment_working_directory(self, yaml_info):
        """
        Get experimentWorkingDirectory (nni-experiments) ,
        e.g., /Users/sunwu/SW-Research/py-search-lib/nni-experiments

        Parameters
        ----------
        yaml_info : dict
             a instance of self._get_exp_config()

        Returns
        -------



        """

        exp_home = yaml_info['experimentWorkingDirectory']
        return exp_home

    def _get_exp_config(self):
        _exp_config = exec_cmd_and_return_str(f"nnictl config show {self.exp_id}")
        yaml_info = yaml.safe_load(_exp_config)
        return yaml_info

    def stop_experiment(self):
        UtilSys.is_debug_mode() and log.info(f"Stop experiment {self.exp_id}")
        if self.is_stopped():
            log_warning_msg(f"Experiment {self.exp_id} was stopped.")
        else:
            stop_info = exec_cmd_and_return_str(f"nnictl stop {self.exp_id}", check=False)
            UtilSys.is_debug_mode() and log.info(f"{Emjoi.SETTING} Stop result: \n{stop_info}")

    def resume(self):
        UtilSys.is_debug_mode() and log.info(f"Resuming experiment {self.exp_id}")
        # Resume port
        resume_port = get_available_nni_experiment_port()
        resume_info = exec_cmd_and_return_str(f"nnictl resume {self.exp_id} --port {resume_port}")
        UtilSys.is_debug_mode() and log.info(f"{Emjoi.SETTING} Resuming result: \n{resume_info}")
        self.exp_port = resume_port

    def is_running(self):
        self._update_experiment_status_from_nnictl()
        return self.exp_status == 'RUNNING' or self.exp_status == "NO_MORE_TRIAL" or self.exp_status == "INITIALIZED"

    def print_running_info(self):
        UtilSys.is_debug_mode() and log.info(
            f"{Emjoi.WAITING}[{datetime.datetime.now()}] "
            f"Experiment is running at [http://{get_host_ip()}:{self.get_exp_port()}], waiting...")

    def get_view_url(self):
        """
        Return the url(http://xxxx) for the experiment.

        Returns
        -------

        """
        return f"http://{get_host_ip()}:{self.get_exp_port()}";

    def delete(self):
        """
        Delete current experiment.

        Returns
        -------

        """
        self.delete_exp(self.exp_id)

    @classmethod
    def delete_exp(cls, exp_id):
        del_msg = exec_cmd_and_return_str(f"echo y|nnictl experiment delete {exp_id}", check=False)
        logi(f"Deleted experiment status:\n {del_msg}")

    def is_error_alive(self):
        self._update_experiment_status_from_nnictl()
        return self.exp_status == "ERROR"

    def is_done_alive(self):
        # call
        self._update_experiment_status_from_nnictl()
        return self.exp_status == "DONE"

    def is_running_alive(self):
        return self.is_running()

    def is_done_or_error(self):
        self._update_experiment_status_from_nnictl()
        return self.exp_status == "DONE" or self.exp_status == "ERROR" or self.exp_status == "None" \
            or self.exp_status is None

    def _update_experiment_status_from_nnictl(self):
        self.exp_status = get_experiment_status(self.exp_port)
        logi(f"Updated status {self.exp_status}")


def get_exp_status_by_id(exp_id, retry_times=10) -> NNIExperimentInfo:
    cmd = f"nnictl experiment status {exp_id}"
    assert len(exp_id) > 0, 'experiment name must be specified'
    for retry_time in range(retry_times):
        try:
            stdout, stderr = exec_cmd(cmd, split_to_array=False)
            exp_info = json.loads(stdout)
            assert exp_info['status'] is not None, 'experiment status must be specified'
            return NNIExperimentInfo(exp_id=exp_id, exp_status=exp_info['status'])
        except Exception as e:
            UtilSys.is_debug_mode() and log.info(
                f"Getting experiment status failed, retry {retry_time}/{retry_times} after 3 seconds")
            time.sleep(3)

    return None


class NNIDotExperiment:
    """
    Represents an instance of .experiment: /Users/sunwu/nni-experiments/.experiment
    """

    def __init__(self, user_home=os.path.expanduser('~')):
        experiment_file = os.path.join(user_home, "nni-experiments", ".experiment")
        log_setting_msg(f"The path of .experiment: {experiment_file}")
        if not os.path.exists(experiment_file):
            log_setting_msg(f"{experiment_file} is not found")
            self.experiments = {}
            self.experiments_pd = pd.DataFrame()
        else:
            # Represent experiments in /Users/sunwu/nni-experiments/.experiment
            self.experiments = None
            with open(experiment_file, "r") as f:
                self.experiments = json.load(f)
            self.experiments_pd: pd.DataFrame = pd.DataFrame(self.experiments).T.reset_index(drop=True)

    def get_experiment_info(self, experiment_name) -> NNIExperimentInfo:
        """
        Get experiment information by name

        Parameters
        ----------
        experiment_name :

        Returns
        -------

        """
        if self.experiments_pd.shape[0] > 0:
            cur_expinfo = self.experiments_pd.loc[self.experiments_pd['experimentName'] == experiment_name, :]
            if cur_expinfo.shape[0] == 1:
                return NNIExperimentInfo(exp_id=cur_expinfo['id'].iloc[0],
                                         exp_name=cur_expinfo['experimentName'].iloc[0],
                                         exp_port=cur_expinfo['port'].iloc[0], exp_status=cur_expinfo['status'].iloc[0])
            elif cur_expinfo.shape[0] > 1:
                msg = f"There exists {cur_expinfo.shape[0]} duplicate experiments with name: \n {experiment_name}" \
                      f"\n Duplicate experiment ids: {cur_expinfo.loc[:, 'id'].values}"
                log_warning_msg(msg)
                send_msg_to_feishu(msg)
                raise RuntimeError(msg)
            else:
                return NNIExperimentInfo()
        else:
            return NNIExperimentInfo()

    def get_experiment_info_by_id(self, exp_id) -> NNIExperimentInfo:
        """
        If experiment
        Parameters
        ----------
        experiment_name :

        Returns
        -------

        """
        cur_expinfo = self.experiments_pd.loc[self.experiments_pd['experimentName'] == exp_id, :]
        if cur_expinfo.shape[0] == 1:
            return NNIExperimentInfo(exp_id=cur_expinfo['id'].iloc[0], exp_name=cur_expinfo['experimentName'].iloc[0],
                                     exp_port=cur_expinfo['port'].iloc[0], exp_status=cur_expinfo['status'].iloc[0])
        return NNIExperimentInfo()

    def get_all_experiments(self) -> pd.DataFrame:
        return self.experiments_pd

    def get_all_log_dirs(self, without_regrex="debug"):
        """
        Returns all subdirectory in the nni-experiments.

        Parameters
        ----------
        without_regrex : str
            The returned directories can't match `remove_regrex`

        Returns
        -------
        list

        """
        _all = self.get_all_experiments()['logDir'].unique().tolist()
        if without_regrex is None:
            return _all
        ret_all = []
        for dir in _all:
            if not re.search(without_regrex, dir):
                ret_all.append(dir)
        return ret_all

    def get_experiment_working_directory(self, without_regrex=None):
        return self.get_all_log_dirs(without_regrex=without_regrex)


def get_experiment_info(exp_name) -> NNIExperimentInfo:
    return NNIDotExperiment().get_experiment_info(exp_name)


if __name__ == '__main__':
    nei = NNIExperimentInfo(exp_port=50001)
    print(nei.is_done_alive())
    print(nei.is_running_alive())
