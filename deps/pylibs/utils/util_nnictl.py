import argparse
import pprint
import re
import subprocess
import sys
import time

import pandas as pd

from nni.tools.nnictl.config_utils import Experiments
from nni.tools.nnictl.launcher import view_experiment
from nni.tools.nnictl.nnictl_utils import stop_experiment
from pylibs.common import Emjoi
from pylibs.util_pandas import log_pretty_table
from pylibs.utils.util_bash import exec_cmd_return_stdout_and_stderr
from pylibs.utils.util_feishu import feishu_report_error_and_exit
from pylibs.utils.util_log import get_logger
from pylibs.utils.util_message import logi, loge, logw
from pylibs.utils.util_nni_exp_info import NNIExperimentInfo
from pylibs.utils.util_nni_mamager_status import NNIManagerStatus
from pylibs.utils.util_system import UtilSys

log = get_logger()


class NNICTL:
    Id = "Id"
    Name = "Name"
    Status = "Status"
    Port = "Port"

    # cache ✅All experiments
    EXPERIMENTS = None

    @staticmethod
    def reset():
        NNICTL.EXPERIMENTS = None

    @staticmethod
    def get_all_tuners():
        """
        Return the names of all tuners grabed by `nnictl algo list`

        Returns
        -------

        """
        std_out, std_err = exec_cmd_return_stdout_and_stderr("nnictl algo list", retry=3, check=False)
        if str(std_err).find("nnictl: command not found") > -1:
            raise RuntimeError("nni is not installed. Install nni by \npip install nni==2.10")
        patern_tuner_name = re.compile("^\|\s*(.*?)\s*\|\stuner")
        _tuner_names = []
        for v in std_out.split("\n"):
            _re = patern_tuner_name.search(v)
            if _re:
                _tuner_names.append(_re.group(1))
        UtilSys.is_debug_mode() and log.info(f"Tuners: \n {_tuner_names}")
        return _tuner_names

    @staticmethod
    def get_all_experiments(retry_times=3, retry_interval=3):
        """Returns ✅All experiments as a pd.DataFrame.

                 Id                                               Name   Status   Port
        0  jlpq984h  model_entry_file=main_sk.py&search_space_file=...  STOPPED  31000
        1  almk7nzt  model_entry_file=main_sk.py&search_space_file=...  STOPPED  50001
        2  2pjk6z7e  model_entry_file=main_sk.py&search_space_file=...  STOPPED  50002

        """
        logi(f"Fetch ✅All experiments from nni: \n{Emjoi.CAR * 30:*^88}")
        for retry_time in range(retry_times):
            try:
                all_experiment_info, err = exec_cmd_return_stdout_and_stderr("nnictl experiment list --all",
                                                                             check=False)
                UtilSys.is_debug_mode() and log.info(f"All experiment: \n{all_experiment_info}")

                pattern = "Id:\s+(.*?)\s+Name:\s+(.*?)\s+Status:\s+(.*?)\s+Port:\s+(\d+)\s+"
                res = re.findall(pattern, all_experiment_info)

                if not isinstance(res, list):
                    return None
                exps = pd.DataFrame(res, columns=[NNICTL.Id, NNICTL.Name, NNICTL.Status, NNICTL.Port])
                return exps

            except Exception as e:
                UtilSys.is_debug_mode() and log.info(
                    f"Getting experiment information failed, retry {retry_time} after {retry_interval} seconds")
                time.sleep(retry_interval)

    @staticmethod
    def get_experiment_by_name_from_shell(exp_name):
        return NNICTL.get_experiment_by_name_from_shell_v2(exp_name)

    @staticmethod
    @DeprecationWarning
    def get_experiment_by_name_from_shell_v1(exp_name, retry_times=3, retry_interval=3):

        if NNICTL.EXPERIMENTS is None:
            NNICTL.EXPERIMENTS = NNICTL.get_all_experiments()
        assert len(exp_name) > 0, 'Experiment name must be specified'
        for _retry_time in range(retry_times):
            try:
                # In the first time, refresh the experiment info from nnictl
                if _retry_time == 0 and NNICTL.EXPERIMENTS is None:
                    NNICTL.EXPERIMENTS = NNICTL.get_all_experiments()
                    continue

                found_exps = NNICTL.EXPERIMENTS.loc[NNICTL.EXPERIMENTS[NNICTL.Name] == exp_name, :]
                log_pretty_table(found_exps)

                if found_exps.shape[0] > 1:
                    log_pretty_table(found_exps)
                    msg = f"Duplicate experiments are found!!!"
                    loge(msg)
                    # NNIExperimentInfo.delete_exp(found_exps[NNICTL.Id].iloc[0])
                    sys.exit(-1)

                elif found_exps.shape[0] == 1:
                    logi(f"Found experiment: {found_exps.shape}")
                    return NNIExperimentInfo(exp_id=found_exps[NNICTL.Id].iloc[0],
                                             exp_name=found_exps[NNICTL.Name].iloc[0],
                                             exp_port=found_exps[NNICTL.Port].iloc[0],
                                             exp_status=found_exps[NNICTL.Status].iloc[0])
                else:
                    return NNIExperimentInfo()
            except:
                UtilSys.is_debug_mode() and log.info(
                    f"Getting experiment information failed, retry {_retry_time} after {retry_interval} seconds. ")
                time.sleep(retry_interval)

        return NNIExperimentInfo()

    @staticmethod
    def get_experiment_by_name_from_shell_v2(exp_name):
        experiments_config = Experiments()
        experiments_dict = experiments_config.get_all_experiments()
        for _exp_id, _exp_info in experiments_dict.items():
            if _exp_info['experimentName'] == exp_name:
                logi(f"Experiment info loaded from nni: {_exp_info}")
                return NNIExperimentInfo(exp_id=_exp_info["id"],
                                         exp_name=_exp_info["experimentName"],
                                         exp_port=_exp_info.get("port"),
                                         exp_status=_exp_info["status"],
                                         )
        logw("Experiment is not found!")
        return NNIExperimentInfo()

    @staticmethod
    def get_experiment_by_id(exp_id):
        experiments_config = Experiments()
        experiments_dict = experiments_config.get_all_experiments()
        for _exp_id, _exp_info in experiments_dict.items():
            if _exp_info['id'] == exp_id:
                logi(f"Experiment info loaded from nni: {_exp_info}")
                return NNIExperimentInfo(exp_id=_exp_info["id"],
                                         exp_name=_exp_info["experimentName"],
                                         exp_port=_exp_info.get("port"),
                                         exp_status=_exp_info["status"],
                                         )
        logw("Experiment is not found!")
        return NNIExperimentInfo()

    @staticmethod
    def stop_all():
        # args = argparse.ArgumentParser()
        # args.id = "--all"
        # args.all = True
        # stop_experiment(args)
        logi("Stop all nni experiments ... ")
        std_out, std_err = exec_cmd_return_stdout_and_stderr("nnictl stop --all", retry=10, check=False, timeout=99999)
        logi(f"NNI stop out: \n{std_err} \n{std_out} ")

    @staticmethod
    def view_experiment(exp_id, port=50099):
        # std_out, std_err = exec_cmd_return_stdout_and_stderr(f"nnictl view --port {port} {exp_id}", retry=10,
        #                                                      check=False,
        #                                                      timeout=999)
        # logi(f"NNI view out: \n{std_err} \n{std_out} ")
        args = NNICTL._create_args()
        args.id = exp_id
        args.port = port
        args.experiment_dir = None
        #  exp_id = args.id
        #  port = args.port
        #  exp_dir = args.experiment_dir
        try:
            view_experiment(args)
        except Exception as e:
            loge(e)

    @classmethod
    def stop_experiment(cls, exp_id):
        args = NNICTL._create_args()
        args.id = exp_id
        args.all = False
        args.port = None
        stop_experiment(args)

    @classmethod
    def _create_args(cls):
        return argparse.ArgumentParser()

    @classmethod
    def is_view(cls, id):
        exp_info = NNICTL.get_experiment_by_id(id)
        return exp_info.is_view()


def create_nni_experiment(command, retry=10, retry_interval=3, check=False):
    """
    Exec command.

    ret stdout,stderr
    Parameters
    ----------
    check : bool
    retry : int
        How many times to retry when command is failed
    command : str
        The command to execute
    retry_interval: int
        The interval between each retry

    Returns
    -------
    NNIExperimentInfo
        the id of the experiment. Raise an error if failed.
    """
    for i in range(retry):
        UtilSys.is_debug_mode() and log.info(f"Exec command: \n{pprint.pformat(command)}")
        var = subprocess.run(
            command,
            shell=True,
            encoding="utf-8",
            timeout=60,
            check=check,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE
        )
        stdout = var.stdout
        stderr = var.stderr

        exp_id_ser = re.search(NNIManagerStatus.NNICTL_CREATE_SUCCESS_EXP_ID_REGEX, stdout)
        if exp_id_ser:
            exp_id = exp_id_ser.group(1)
            exp_port = int(re.search(NNIManagerStatus.NNI_MANAGER_LOG_PORT_REGEX, stdout).group(1))
            nni_exp_info = NNIExperimentInfo(exp_id=exp_id, exp_port=exp_port)
            assert len(exp_id) == 8
            return nni_exp_info

        # The port is not idle
        res = re.search("RuntimeError: Port\s+\d+\s+is not idle", stderr)
        if res:
            log.error(f"Create experiment failed. \n{stderr}")
            feishu_report_error_and_exit()

        if re.search(NNIManagerStatus.NNICTL_CREATE_ERROR, stderr):
            log.error(f"Create experiment failed. \n{stderr}")
            feishu_report_error_and_exit()

        UtilSys.is_debug_mode() and log.info(f"Experiment created failed, waiting {retry_interval} to retry...")
        time.sleep(retry_interval)

    raise RuntimeError(f"Create experiment failed after retry {retry} times.")
