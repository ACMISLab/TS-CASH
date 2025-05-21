import json
import json
import os.path
import sqlite3

import numpy as np
import pandas as pd

from pylibs.common import ConstNNI, ConstMetric, Emjoi
from pylibs.util_pandas import show_all_pandas
from pylibs.utils.util_file import generate_random_file
from pylibs.utils.util_log import get_logger

log = get_logger()


class NNIExperimentUtils:
    DB_ATTR_TRIAL_JOB_ID = "trialJobId"
    ATTR_HYPERPAREMETER = "hyper_parameters"

    def __init__(self, exp_ids, home_for_nni_experiments="/Users/sunwu/SyncResearch/nni-experiments", max_sequence=-1,
                 exp_name=""):

        """

        Parameters
        ----------
        home_for_nni_experiments:str
            存储实验的路径，即 nni-experiments 的绝对路径, 如 /Users/sunwu/nni-experiments
        max_sequence: int
            最大的实验数，用来限制查询的实验，（sequence 从 0 开始），：
            1. 我只想查询前20轮的结果，那么 max_trials = 20 （包含）

        exp_name:str
            The experiment user to identify experiments
        """

        if isinstance(exp_ids, str):
            raise TypeError("exp_ids must be a list.")
        show_all_pandas()
        self._exp_home = home_for_nni_experiments
        self._max_sequence = max_sequence
        self._conn = None
        self._exp_name = exp_name

        self._exp_results = []
        self._best_results = []

        for exp_id in exp_ids:
            # running time
            assert os.path.exists(self._get_db_path(exp_id))

            # auc and f1
            exp_name = self._get_exp_config_by_name(exp_id, key="experimentName")
            experiment_list = self._get_exp_list(exp_id)
            UtilSys.is_debug_mode() and log.info(f"Start calculating experiment result for exp_name [{exp_name}]")
            if experiment_list is not None:
                experiment_list[ConstMetric.KEY_TRIAL_CMD] = self._get_exp_config_by_name(exp_id, key="trialCommand")
                experiment_list[ConstMetric.KEY_RUNNING_TIME] = self._get_exp_running_seconds(exp_id)
                experiment_list[ConstMetric.KEY_EXP_NAME] = exp_name
                experiment_list[ConstMetric.KEY_TUNER] = self._get_exp_config_by_name(exp_id, key="nni_tuner")
                experiment_list[ConstMetric.KEY_EXP_ID] = exp_id
                self._exp_results.append(experiment_list)

                # get best result
                self._best_results.append(self._get_best_results(experiment_list))
                self._save_ch05_best_resut()
            else:
                log.warning(f"Skip None experiment_list for experiment {exp_id}")
                return

    def _save_exp_result(self):
        if self._exp_results is not None:
            res_file = generate_random_file(ext=".xlsx", prefix=f"all_{self._exp_name}_")
            self._exp_results.to_excel(res_file)
            UtilSys.is_debug_mode() and log.info(
                f"\n{Emjoi.SPARKLES_TRIBLE}Save experiment results to "
                f"\n{os.path.abspath(res_file)}")
        else:
            log.warning("Experiments result not find.")

    def _save_ch05_best_resut(self):

        if self._best_results is not None:
            for source_data in self._best_results:
                exp_name = source_data[ConstNNI.KEY_EXP_NAME].values[0]
                best_file = generate_random_file(ext=".xlsx", prefix=f"best_", name=exp_name)
                data = source_data.loc[:, [ConstMetric.BEST_F1, ConstMetric.BEST_PRECISION, ConstMetric.BEST_RECALL,
                                           ConstMetric.KEY_DEFAULT, ConstMetric.RUNNING_TIME]]
                data = data.astype(float).round(4)

                data[ConstMetric.KEY_EXP_ID] = source_data[ConstMetric.KEY_EXP_ID]
                data[ConstMetric.KEY_EXP_NAME] = source_data[ConstMetric.KEY_EXP_NAME]
                data[ConstMetric.KEY_TRIAL_CMD] = source_data[ConstMetric.KEY_TRIAL_CMD]

                data.to_excel(best_file, index=False,
                              header=[ConstMetric.BEST_F1, ConstMetric.BEST_PRECISION, ConstMetric.BEST_RECALL,
                                      ConstMetric.KEY_DEFAULT,
                                      ConstMetric.RUNNING_TIME, ConstMetric.KEY_EXP_ID,
                                      ConstMetric.KEY_EXP_NAME, ConstMetric.KEY_TRIAL_CMD])
                UtilSys.is_debug_mode() and log.info(f"Save best results to: "
                                                     f"\n{os.path.abspath(best_file)}")
        else:
            log.warning("Experiments result not find.")

    def _save_best_resut(self):

        if self._best_results is not None:
            best_file = generate_random_file(ext=".xlsx", prefix=f"best_{self._exp_name}_")
            pd.concat(self._best_results, axis=0).to_excel(best_file)
            UtilSys.is_debug_mode() and log.info(
                f"\nSave best results to\n"
                f"\n{os.path.abspath(best_file)}")
        else:
            log.warning("Experiments result not find.")

    def get_exp_results(self):
        """
        Get results of all experiment with experiment id specified by exp_ids

        Returns
        -------
        pd.DataFrame or None

        """
        if len(self._exp_results) > 0:
            experiments_result = pd.concat(self._exp_results)
            experiments_result = experiments_result.sort_values(by=ConstMetric.KEY_DEFAULT, ascending=False)
            experiments_result = experiments_result.reset_index()
            return experiments_result

        else:
            log.warning(
                f"\nExperiment {self._exp_name} has not result.")
            return None

    def get_best_results(self):
        """
        Get best result of ✅All experiments.

        Returns
        -------
        pd.DataFrame or None

        """
        if len(self._best_results) > 0:
            return pd.concat(self._best_results)
        else:
            log.warning(
                f"\nExperiment {self._exp_name} has not best result.")
            return None

    def _get_exp_running_seconds(self, exp_id):
        """
        获取整个实验(experiment) 的执行时间
        Parameters
        ----------
        exp_id

        Returns
        -------
        float
            实验的执行时间，单位：分钟

        """
        c = self._get_connection_by_exp_id(exp_id)
        c.execute(
            "SELECT trialJobId,max(timestamp) as end,min(timestamp) as start_or_restart,(max(timestamp)-min(timestamp))/1000 as elapse from TrialJobEvent WHERE event !='WAITING' GROUP BY trialJobId ")
        record = c.fetchall()
        time_elapse = pd.DataFrame(record).iloc[:, -1].sum()
        c.close()
        return time_elapse

    def _get_exp_list(self, exp_id):
        """
        获取整个实验(experiment) 的执行结果. 降序排序，即第一个是最优的实验结果(Default).

        获取第一个实验结果：res.iloc[0,:]

        Parameters
        ----------
        exp_id

        Returns
        -------
        pd.DataFrame
            实验的信息

        """
        c = self._get_connection_by_exp_id(exp_id)
        if self._max_sequence == -1:
            sql = f"SELECT {self.DB_ATTR_TRIAL_JOB_ID},data from MetricData WHERE type='{ConstNNI.STATUS_FINAL}'"
        else:
            sql = f"SELECT {self.DB_ATTR_TRIAL_JOB_ID},data from MetricData WHERE type='{ConstNNI.STATUS_FINAL}' and  sequence < {self._max_sequence}"

        c.execute(
            f"select {ConstNNI.KEY_TRAIL_JOB_ID},{ConstNNI.KEY_EVENT},{ConstNNI.KEY_LOG_PATH} from TrialJobEvent where {ConstNNI.KEY_EVENT} = '{ConstNNI.STATUS_FAILED}'")
        failed_record = c.fetchall()
        if len(failed_record) > 0:
            log.warning(f"Jobs [{','.join(np.asarray(failed_record)[:, 0])}] are failed")

        c.execute(sql)
        training_job = c.fetchall()

        c.execute("select trialJobId,data from TrialJobEvent where data is not null ")
        hyper_parameters = c.fetchall()
        hyper_parameters = pd.DataFrame(hyper_parameters, columns=[self.DB_ATTR_TRIAL_JOB_ID, self.ATTR_HYPERPAREMETER])

        keys = None
        metrics = []
        for r in training_job:
            _job_id = r[0]
            _metric = r[-1]
            _metric_dict = json.loads(json.loads(_metric))
            if keys is None:
                keys = list(_metric_dict.keys())
                keys = np.concatenate([keys, ['trialJobId', self.ATTR_HYPERPAREMETER]])

            # Find the hyperparameters with corresponding to the metrics
            _hyper_parameters = hyper_parameters[hyper_parameters[self.DB_ATTR_TRIAL_JOB_ID] == _job_id][
                self.ATTR_HYPERPAREMETER]
            _current_metrics = np.concatenate([list(_metric_dict.values()), [_job_id, hyper_parameters[
                hyper_parameters[self.DB_ATTR_TRIAL_JOB_ID] == _job_id][self.ATTR_HYPERPAREMETER].tolist()[0]]])
            metrics.append(_current_metrics)

        if len(metrics) < 1:
            return None
        else:
            data = pd.DataFrame(metrics, columns=keys)
            data[ConstMetric.KEY_EXP_ID] = exp_id
            return data

    def _get_connection_by_exp_id(self, exp_id):
        db_path = self._get_db_path(exp_id)
        try:
            self._conn = sqlite3.connect(db_path)
            c = self._conn.cursor()
            return c
        except:
            raise FileNotFoundError(f"{ConstNNI.NNI_DB_NAME} is not found: {db_path}")

    def _get_exp_config_by_name(self, exp_id, key="trialCommand"):
        """
        获取整个实验(experiment) 的执行之间，
        Parameters
        ----------
        key: 配置的键，如 trialCommand
        exp_id

        Returns
        -------
        float
            实验的执行时间，单位：分钟

        """
        c = self._get_connection_by_exp_id(exp_id)
        c.execute('SELECT params FROM ExperimentProfile')
        record = c.fetchone()
        preofile = json.loads(record[0])
        c.close()
        return preofile[key]

    def _get_best_results(self, experiment_list):
        """
        Save best result to a .xlsx file



        Parameters
        ----------
        experiment_list : pd.DataFrame
            Experiment lists.

        Returns
        -------

        """
        max_index = np.argmax(experiment_list.loc[:, [ConstMetric.BEST_F1]].values.astype(float))
        best = experiment_list.iloc[max_index:max_index + 1]
        assert best is not None
        return best

    def _get_db_path(self, exp_id):
        """
        Get absolute path of nni.sqlite by exp_id
        Parameters
        ----------
        exp_id :

        Returns
        -------

        """
        home = f"{self._exp_home}/{exp_id}/db/"
        db_path = os.path.join(home, ConstNNI.VALUE_DB_NAME)
        return db_path
