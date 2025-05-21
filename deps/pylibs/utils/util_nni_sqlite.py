# description: nni 3.0 sqlite 工具类
import json
import os
import traceback
from typing import Union
import shutil
import pandas as pd
from nni import Experiment
from pylibs.utils.util_log import get_logger
from pylibs.utils.util_message import logw
from pylibs.utils.util_sqlite import exec_sql_by_sqlite_file
from pylibs.utils.util_system import UtilSys
log = get_logger()

def is_experiment_down(exp: Experiment):
    """
    根据nni数据库判断nni的所有实验是否完成. 是:返回True,否:return false
    Parameters
    ----------
    exp :

    Returns
    -------

    """
    exp_dir = os.path.join(os.path.abspath(os.path.expanduser(exp.config.experiment_working_directory)), exp.id)
    db_file = os.path.join(exp_dir, "db", "nni.sqlite")
    ns = NNISqlite(db_file)
    is_success = ns.is_all_trials_success()
    if is_success:
        print(f"Experiment already exists: {exp_dir}")
        return True
    else:
        if os.path.exists(exp_dir):
            shutil.rmtree(exp_dir)
        return False

class DeleteNNISqliteBak:
    EXP_PROFILE_EXPERIMENTNAME = 'experimentName'
    EXP_PROFILE_EXPERIMENTTYPE = 'experimentType'
    EXP_PROFILE_SEARCHSPACEFILE = 'searchSpaceFile'
    EXP_PROFILE_SEARCHSPACE = 'searchSpace'
    EXP_PROFILE_TRIALCOMMAND = 'trialCommand'
    EXP_PROFILE_TRIALCODEDIRECTORY = 'trialCodeDirectory'
    EXP_PROFILE_TRIALCONCURRENCY = 'trialConcurrency'
    EXP_PROFILE_MAXTRIALNUMBER = 'maxTrialNumber'
    EXP_PROFILE_USEANNOTATION = 'useAnnotation'
    EXP_PROFILE_DEBUG = 'debug'
    EXP_PROFILE_LOGLEVEL = 'logLevel'
    EXP_PROFILE_EXPERIMENTWORKINGDIR = 'experimentWorkingDirectory'
    EXP_PROFILE_TUNER = 'tuner'
    EXP_PROFILE_TRAININGSERVICE = 'trainingService'

    def __init__(self, experiment_working_directory, id, file_base_name='nni.sqlite'):
        """

        Parameters
        ----------
        file_base_name : str
            The file name of nni.sqlite, default nni.sqlite. E.g.,
            /Users/sunwu/nni-experiments/test_exp/qc85ane1/log/nni.sqlite
        experiment_working_directory : str
            The location for nni-experiments, default is  `~/nni-experiments`.
            An example can be py-search-lib/pylibs/test/data/nnilog/normal/db/nni.sqlite
        id : str
            the experiment id
        """
        self.experiment_working_directory = experiment_working_directory
        self.experiment_id = id
        self.file_base_name = file_base_name

        # Represents the file /Users/sunwu/nni-experiments/test_exp/qc85ane1/log/nni.sqlite
        self.db_file = os.path.join(self.experiment_working_directory, self.experiment_id, "db",
                                    self.file_base_name)
        UtilSys.is_debug_mode() and log.info(f"nni.sqlite file path: {self.db_file}")

        # Represents the contents of file /Users/sunwu/nni-experiments/test_exp/qc85ane1/log/nni.sqlite

    def is_all_trials_success(self):
        """
        True if all trials in an experiment is successful(status is FINISHED). False otherwise.

        What is all the trials finished?
        the number of trials with status is SUCCEEDED = get_max_trial_number

        Returns
        -------

        """
        sql = "select distinct * from  TrialJobEvent where event=='SUCCEEDED'"

        try:
            records, columns_name = exec_sql_by_sqlite_file(self.db_file, sql)
            return len(records) == self.get_max_trial_number()
        except Exception as e:
            logw(e)
            return False

    def get_failed_trials(self):
        """
        Return a list which contains a list of trial ids. e.g. ['ZV0ym','xxx'].

        Return empty list [ ] if no trials.

        Returns
        -------

        """
        arr = []

        try:
            sql = "select distinct * from  TrialJobEvent where event=='FAILED'"
            records, columns_name = exec_sql_by_sqlite_file(self.db_file, sql)
            if len(records) == 0:
                return arr
            else:
                for r in records:
                    arr.append(r[1])
                return arr
        except Exception as e:
            logw(e)
            return arr

    def has_error(self) -> bool:
        """
        Get the status from /Users/sunwu/nni-experiments/test_exp/qc85ane1/log/nni.sqlite


        Returns
        -------
        bool
            True if experiment is mark_as_finished, False otherwise.
        """

        # Where the nni.sqlite:
        # "/Users/sunwu/nni-experiments/debug_1675855568130661000/qc85ane1/log"
        records, columns_name = exec_sql_by_sqlite_file(self.db_file,
                                                        "select * from  TrialJobEvent where event ='FAILED'")

        # Error when executing, report error
        if records is None:
            return True

        if len(records) > 0:
            return True
        else:
            return False

    def get_mean_default_metrics(self) -> Union[float, None]:
        _dm = self.get_default_metrics()
        if _dm is None:
            return None
        else:
            return _dm.mean()

    def get_all_metrics(self) -> Union[pd.DataFrame, None]:
        try:
            sql = "select data from MetricData where type='FINAL'"
            records, columns_name = exec_sql_by_sqlite_file(self.db_file, sql)
            if records is None:
                raise RuntimeError("Error getting all metrics from nni.sqlite")
            _all = []
            for _r in records:
                _metric = json.loads(json.loads(_r[0]))
                _all.append(_metric)
            if len(_all) == 0:
                # Return None if no metrics data found.
                return None
            else:
                data = pd.DataFrame(_all)
                return data
        except Exception as e:
            log.error(traceback.format_exc())
            return None

    def get_default_metrics(self) -> Union[pd.DataFrame, None]:
        _all = self.get_all_metrics()
        if _all is not None and _all.shape[0] > 0:
            return _all.loc[:, 'default']
        else:
            return None

    def get_max_trial_number(self):
        return self.get_experiment_profile()[self.EXP_PROFILE_MAXTRIALNUMBER]

    def get_experiment_profile(self):
        sql = "select params from ExperimentProfile limit 1"
        records, columns_name = exec_sql_by_sqlite_file(self.db_file, sql)
        try:
            return json.loads(records[0][0])
        except Exception as e:
            raise RuntimeError(f"Could not load experiment profile fro {self.db_file}")
class NNISqlite:
    EXP_PROFILE_EXPERIMENTNAME = 'experimentName'
    EXP_PROFILE_EXPERIMENTTYPE = 'experimentType'
    EXP_PROFILE_SEARCHSPACEFILE = 'searchSpaceFile'
    EXP_PROFILE_SEARCHSPACE = 'searchSpace'
    EXP_PROFILE_TRIALCOMMAND = 'trialCommand'
    EXP_PROFILE_TRIALCODEDIRECTORY = 'trialCodeDirectory'
    EXP_PROFILE_TRIALCONCURRENCY = 'trialConcurrency'
    EXP_PROFILE_MAXTRIALNUMBER = 'maxTrialNumber'
    EXP_PROFILE_USEANNOTATION = 'useAnnotation'
    EXP_PROFILE_DEBUG = 'debug'
    EXP_PROFILE_LOGLEVEL = 'logLevel'
    EXP_PROFILE_EXPERIMENTWORKINGDIR = 'experimentWorkingDirectory'
    EXP_PROFILE_TUNER = 'tuner'
    EXP_PROFILE_TRAININGSERVICE = 'trainingService'

    def __init__(self,db_file='nni.sqlite'):
        """

        Parameters
        ----------
        file_base_name : str
            The file name of nni.sqlite, default nni.sqlite. E.g.,
            /Users/sunwu/nni-experiments/test_exp/qc85ane1/log/nni.sqlite
        experiment_working_directory : str
            The location for nni-experiments, default is  `~/nni-experiments`.
            An example can be py-search-lib/pylibs/test/data/nnilog/normal/db/nni.sqlite
        id : str
            the experiment id
        """
        # Represents the file /Users/sunwu/nni-experiments/test_exp/qc85ane1/log/nni.sqlite
        self.db_file = db_file

        # Represents the contents of file /Users/sunwu/nni-experiments/test_exp/qc85ane1/log/nni.sqlite

    def is_all_trials_success(self):
        """
        True if all trials in an experiment is successful(status is FINISHED). False otherwise.

        What is all the trials finished?
        the number of trials with status is SUCCEEDED = get_max_trial_number

        Returns
        -------

        """

        # sql = "select distinct * from  TrialJobEvent where event=='SUCCEEDED'"
        sql="select distinct * from  MetricData where type=='FINAL'"
        try:
            records, columns_name = exec_sql_by_sqlite_file(self.db_file, sql)
            return len(records) == self.get_max_trial_number()
        except Exception as e:
            logw(e)
            return False

    def get_failed_trials(self):
        """
        Return a list which contains a list of trial ids. e.g. ['ZV0ym','xxx'].

        Return empty list [ ] if no trials.

        Returns
        -------

        """
        arr = []

        try:
            sql = "select distinct * from  TrialJobEvent where event=='FAILED'"
            records, columns_name = exec_sql_by_sqlite_file(self.db_file, sql)
            if len(records) == 0:
                return arr
            else:
                for r in records:
                    arr.append(r[1])
                return arr
        except Exception as e:
            logw(e)
            return arr

    def has_error(self) -> bool:
        """
        Get the status from /Users/sunwu/nni-experiments/test_exp/qc85ane1/log/nni.sqlite


        Returns
        -------
        bool
            True if experiment is mark_as_finished, False otherwise.
        """

        # Where the nni.sqlite:
        # "/Users/sunwu/nni-experiments/debug_1675855568130661000/qc85ane1/log"
        records, columns_name = exec_sql_by_sqlite_file(self.db_file,
                                                        "select * from  TrialJobEvent where event ='FAILED'")

        # Error when executing, report error
        if records is None:
            return True

        if len(records) > 0:
            return True
        else:
            return False

    def get_mean_default_metrics(self) -> Union[float, None]:
        _dm = self.get_default_metrics()
        if _dm is None:
            return None
        else:
            return _dm.mean()

    def get_all_metrics(self) -> Union[pd.DataFrame, None]:
        try:
            sql = "select data from MetricData where type='FINAL'"
            records, columns_name = exec_sql_by_sqlite_file(self.db_file, sql)
            if records is None:
                raise RuntimeError("Error getting all metrics from nni.sqlite")
            _all = []
            for _r in records:
                _metric = json.loads(json.loads(_r[0]))
                _all.append(_metric)
            if len(_all) == 0:
                # Return None if no metrics data found.
                return None
            else:
                data = pd.DataFrame(_all)
                return data
        except Exception as e:
            log.error(traceback.format_exc())
            return None

    def get_default_metrics(self) -> Union[pd.DataFrame, None]:
        _all = self.get_all_metrics()
        if _all is not None and _all.shape[0] > 0:
            return _all.loc[:, 'default']
        else:
            return None

    def get_max_trial_number(self):
        return self.get_experiment_profile()[self.EXP_PROFILE_MAXTRIALNUMBER]

    def get_experiment_profile(self):
        sql = "select params from ExperimentProfile limit 1"
        records, columns_name = exec_sql_by_sqlite_file(self.db_file, sql)
        if records is None:
            return None
        return json.loads(records[0][0])

    def get_trial_cmd_parameters(self):

        """
        python ../../algs/model.py  --params  eyJtb2RlbF9uYW1lIjogImhib3MiLCAidHJpYWxfY29uY3VycmVuY3kiOiAxLCAibWF4X3RyaWFsX251bWJlciI6IDUwLCAidHVuZXIiOiAiR3JpZFNlYXJjaCIsICJkYXRhc2V0X25hbWUiOiAiRGFwaG5ldCIsICJkYXRhX2lkIjogIlMwMlIwMUUwLnRlc3QuY3N2QDYub3V0IiwgInNhbXBsZV9yYXRlIjogMC4xLCAiaW5kZXBlbmRlbnRfcm91bmQiOiAyLCAibWF4X2V4cGVyaW1lbnRfZHVyYXRpb24iOiAiMzBtIn0=

        返回--params后面的参数
        Returns
        -------

        """
        if self.get_experiment_profile() is None:
            return None
        cmds = self.get_experiment_profile()['trialCommand'].strip().split(" ")
        return cmds[-1]

    def get_experiment_name(self):
        return self.get_experiment_profile()[NNISqlite.EXP_PROFILE_EXPERIMENTNAME]