import json
import os
import pprint
import re
import sqlite3
import traceback

import numpy as np
import pandas as pd
from nni import Experiment
from pylibs.common import ConstNNI, Config, ConstMetric, Emjoi, EI
from pylibs.utils.util_args import parse_cmd_to_dict
from pylibs.utils.util_bash import exec_cmd
from pylibs.utils.util_file import gaf
from pylibs.utils.util_log import get_logger
from pylibs.utils.util_message import logw, log_warn_msg, loge
from pylibs.utils.util_network import get_host_ip, is_port_listing
from pylibs.utils.util_pandas import save_pandas_to_mysql, save_pandas_to_sqlite, save_pandas_to_csv, PDUtil
from pylibs.utils.util_sqlite import get_table_names_and_types, exec_sql
from pylibs.utils.util_system import UtilSys
from pylibs.utils.util_directory import make_dirs

log = get_logger()


class UtilNNI:
    @staticmethod
    def get_nni_experiments_dir(exp_name="./"):
        _path = os.path.abspath(
            os.path.join(os.sep.join(os.path.dirname(__file__).split(os.sep)[:4]), "nni-experiments", exp_name))

        make_dirs(_path)
        return _path


def get_environment_by_name(name):
    return os.environ.get(name) or "."


def get_nni_trial_dir():
    return get_environment_by_name("NNI_OUTPUT_DIR")


def get_all_experiment():
    """
    获取所有的NNI的实验

    Examples
    --------
              Id  Name     Port   Status
    0   bp2avqlh  None  STOPPED   8888
    ....


    Returns
    -------

    """
    #
    return get_all_experiments_from_shell()
    # return get_experiments_from_profile_file()


def get_experiments_from_profile_file(profile=os.path.join(os.environ.get("HOME"), "nni-experiments/.experiment")):
    """

    Returns
    -------

    """
    if not os.path.exists(profile):
        raise FileNotFoundError(f".experiment file is not exists at {profile}")
    with open(profile) as f:
        experiments = json.load(f)
    return pd.DataFrame(experiments.values())


def get_all_experiments_from_shell():
    """
    获取所有的NNI的实验

    Returns
    -------
    list
        ['Id', 'Name', 'Status', 'Port']
    list
        [
            ['bp2avqlh', 'None', 'STOPPED', '8888']
            ['bp2avqlh', 'None', 'STOPPED', '8888']
        ]



    """
    columns = ["id", 'experimentName', 'status', 'port']
    cmd = "nnictl experiment list --all"
    msg, err = exec_cmd(cmd)
    if len(err) > 1:
        raise RuntimeError(f"Error when getting experiments:\n {err}")
    # res
    # 0 = {str} 'w4yabkgn'
    # 1 = {str} 'EXPERIMENTS_VAE_04_DAT'
    # 2 = {str} 'STOPPED'
    # 3 = {str} '44145'
    res = []
    for item in msg:
        _items = str(item).split(" ")
        if _items[0] == "Id:":
            res.append([_items[1], _items[6], _items[11], _items[16]])
    # return columns, res
    return pd.DataFrame(res, columns=columns)


def get_experiments_by_name_from_shell(exp_name, retry_times=10):
    """
    获取所有的NNI的实验, headers ["id", 'experimentName', 'status', 'port']

    Parameters
    -------
    retry_times:int
        When the command `nnictl experiment list --all` failed, how many times to retry. Default 10.

    Returns
    -------
    pd.DataFrame
        With headers ["id", 'experimentName', 'status', 'port']



    """
    assert len(exp_name) > 0, 'experiment name must be specified'
    columns = ["id", 'experimentName', 'status', 'port']

    for i in range(retry_times):
        try:
            msg, err = exec_cmd(f"nnictl experiment list --all|grep {exp_name}")
            if len(err) > 1:
                err = err
                continue
            if "".join(msg) == "":
                return None
            res = []
            for item in msg:
                _items = str(item).split(" ")
                if _items[0] == "Id:":
                    res.append([_items[1], _items[6], _items[11], _items[16]])
            # return columns, res
            return pd.DataFrame(res, columns=columns)
        except Exception as e:
            raise e
    log.warning("Not find experiment.")
    return None


def get_experiments_by_name(name):
    """
    根据实验名称获取实验.

    None if not find the experiment with user, else pd.DataFrame

    Parameters
    ----------
    name :

    Returns
    -------
    None or pd.DataFrame

    """

    res = get_all_experiment()
    ret_array = []
    for key, value in res.iterrows():
        if value.loc['experimentName'].find(name) > -1:
            ret_array.append(res.loc[key:key, :])
    if len(ret_array) > 0:
        return pd.concat(ret_array, axis=0)
    return None


def is_running_experiment():
    """
    是否存在运行中的实验。
    在所有的实验中，只要一个是 running，那么都是运行

    Returns
    -------

    """
    all = get_all_experiment()
    for key, exp in all.iterrows():
        if exp.loc['status'] in [ConstNNI.STATUS_RUNNING, ConstNNI.STATUS_NO_MORE_TRIAL]:
            UtilSys.is_debug_mode() and log.info(
                f"Experiment is running at: [http://{get_host_ip()}:{int(exp['port'])}] ({exp.to_dict()})")
            return True
    else:
        return False


def is_experiment_running(exp_name):
    """
    Whether experiment with exp_name is running

    Returns
    -------

    """
    all = get_all_experiment()
    for key, exp in all.iterrows():
        if str(exp['experimentName']).find(exp_name) > -1 and exp.loc['status'] in [ConstNNI.STATUS_RUNNING,
                                                                                    ConstNNI.STATUS_NO_MORE_TRIAL]:
            UtilSys.is_debug_mode() and log.info(
                f"Experiment is running at: [http://{get_host_ip()}:{int(exp['port'])}] ({exp.to_dict()})")
            return True
    else:
        return False


def get_running_experiments():
    """
    获取所有运行中的实验（experiment)

    Returns
    -------
    pd.DataFrame
        The all running experiments.
    """
    ret_arr = []
    all = get_all_experiment()
    for key, exp in all.iterrows():
        if exp.loc['status'] in [ConstNNI.STATUS_RUNNING, ConstNNI.STATUS_NO_MORE_TRIAL]:
            ret_arr.append(exp)
    if len(ret_arr) > 0:
        return pd.concat(ret_arr, axis=0)
    else:
        return None


def stop_all_experiment():
    exec_cmd("nnictl stop --all")


def print_experiment_error_info(nni_experiment_dir="/Users/sunwu/SyncResearch/nni-experiments/", exp_id="048btrli"):
    if os.path.exists(nni_experiment_dir):
        with open(os.path.join(nni_experiment_dir, exp_id, f"log/dispatcher.log")) as f:
            disp_logs = f.readlines()
            log.error(f"Error in  NNI:\n{pprint.pformat(disp_logs)}")
    else:
        log.error(f"Not exist {nni_experiment_dir}")


def get_nni_db_file(home, exp_id):
    """
    Get nni.sqlite file.

    Parameters
    ----------
    home : str
        The nni-experiments home, e.g., /root/nni-experiments
    exp_id : str
        The experiment id.

    Returns
    -------
    str
        The nni.sqlite file, e.g., 'root/nni-experiments/mub2ygx8/db/nni.sqlite
    """
    _db = os.path.join(home, exp_id, "db", ConstNNI.NNI_DB_NAME)
    if not os.path.exists(_db):
        log.warning(f"Db file [{_db}] is not existed")
    return _db


def get_nni_connection_by_file_and_exp_id(home, exp_id):
    file = get_nni_db_file(home, exp_id)
    if file is not None and os.path.exists(file):
        return sqlite3.connect(file)
    else:
        return None


def get_nni_connection_by_file(file):
    if file is not None and os.path.exists(file):
        return sqlite3.connect(file)
    else:
        traceback.print_exc()
        return None


@DeprecationWarning
def get_dataset_from_trial_command(param):
    """
    Get m_dataset from trial command

    Parameters
    ----------
    param :str
        The trial command.
         e.g., 'python models/deepsvdd.py  --m_dataset SHUTTLE  --seed 0 --data_sample_method RS --data_sample_rate  0.0078125'

    Returns
    -------
    str
       Get the SHUTTLE in `--m_dataset SHUTTLE`

    """
    return param.split(Config.ARGS_KEY_DATASET)[1].lstrip().split(" ")[0]


@DeprecationWarning
def get_all_metric(conn: sqlite3.Connection):
    """
    Get all metric from table MetricData.

    Parameters
    ----------
    conn : sqlite3.Connection
        The connection of sqlite by get_nni_connection_by_file_and_exp_id()

    Returns
    -------
    pd.DataFrame
        The result. columns:
        ['timestamp', 'trialJobId', 'parameterId', 'type',
         'sequence', 'default', 'best_prec', 'best_recall',
          'best_f1', 'AUC', 'VALID_LOSS', 'model_id']

    """
    c = conn.cursor()
    c.execute("select * from MetricData where type='FINAL'")

    metrics = c.fetchall()
    c.execute("select * from ExperimentProfile order by revision desc limit  1")

    try:
        exp_profiles = c.fetchall()
        exp_profile = exp_profiles[0]
    except:
        traceback.print_exc()
    # params
    exp_info: dict = json.loads(exp_profile[0])

    all_output = []
    #     timestamp   INTEGER,
    #     trialJobId  TEXT,
    #     parameterId TEXT,
    #     type        TEXT,
    #     sequence    INTEGER,
    for record in metrics:
        performances_arr = {
            ConstMetric.KEY_TIMESTAMP: record[0],
            ConstMetric.KEY_TRIAL_JOB_ID: record[1],
            ConstMetric.KEY_PARAMETER_ID: record[2],
            ConstMetric.KEY_TYPE: record[3],
            ConstMetric.KEY_SEQUENCE: record[4],
            ConstMetric.KEY_RUNNING_TIME: get_experiment_runtime_time(conn)
        }

        training_service = pd.Series(exp_info['trainingService']).to_dict()
        performances_arr.update(training_service)

        # Parse running args
        for _k, _v in parse_cmd_to_dict(performances_arr[ConstMetric.KEY_TRIAL_COMMAND]).items():
            performances_arr[_k] = _v

        for key in exp_info.keys():
            performances_arr[key] = exp_info[key]

        performance: dict = json.loads(json.loads(record[-1]))
        for key in performance.keys():
            val_ = performance.get(key)
            performances_arr[key] = val_
        # Make unique column true.
        assert len(set(performances_arr.keys())) == len(performances_arr.keys())
        all_output.append(performances_arr)
    c.close()
    return pd.DataFrame(all_output)


def get_all_metric_of_nni_db(conn: sqlite3.Connection):
    """
    Get all accuracy metric from db.sqlite of NNI

    Parameters
    ----------
    conn : sqlite3.Connection
        The connection of sqlite by get_nni_connection_by_file_and_exp_id()

    Returns
    -------
    pd.DataFrame
        The result. columns:
        ['timestamp', 'trialJobId', 'parameterId', 'type',
         'sequence', 'default', 'best_prec', 'best_recall',
          'best_f1', 'AUC', 'VALID_LOSS', 'model_id']

    """
    c = conn.cursor()
    c.execute("select * from MetricData where type='FINAL'")

    metrics = c.fetchall()
    c.execute("select * from ExperimentProfile order by revision desc limit  1")

    try:
        exp_profiles = c.fetchall()
        exp_profile = exp_profiles[0]
        # params
        exp_info: dict = json.loads(exp_profile[0])

        all_output = []
        #     timestamp   INTEGER,
        #     trialJobId  TEXT,
        #     parameterId TEXT,
        #     type        TEXT,
        #     sequence    INTEGER,
        for record in metrics:
            performances_arr = {
                ConstMetric.KEY_TIMESTAMP: record[0],
                ConstMetric.KEY_TRIAL_JOB_ID: record[1],
                ConstMetric.KEY_PARAMETER_ID: record[2],
                ConstMetric.KEY_TYPE: record[3],
                ConstMetric.KEY_SEQUENCE: record[4],
                ConstMetric.KEY_RUNNING_TIME: get_experiment_runtime_time(conn)
            }

            training_service = pd.Series(exp_info['trainingService']).to_dict()
            performances_arr.update(training_service)

            # Parse running args
            for _k, _v in parse_cmd_to_dict(performances_arr[ConstMetric.KEY_TRIAL_COMMAND]).items():
                performances_arr[_k] = _v

            for key in exp_info.keys():
                performances_arr[key] = exp_info[key]

            performance: dict = json.loads(json.loads(record[-1]))
            for key in performance.keys():
                val_ = performance.get(key)
                performances_arr[key] = val_
            # Make unique column true.
            assert len(set(performances_arr.keys())) == len(performances_arr.keys())
            all_output.append(performances_arr)
        c.close()
        return all_output
    except Exception as e:
        traceback.print_exc()


def get_experiment_runtime_time(conn):
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
    c = conn.cursor()
    c.execute(
        "SELECT trialJobId,max(timestamp) as end,min(timestamp) as start_or_restart,(max(timestamp)-min(timestamp))/1000 as elapse from TrialJobEvent WHERE event !='WAITING' GROUP BY trialJobId ")
    record = c.fetchall()
    time_elapse = pd.DataFrame(record).iloc[:, -1].sum()
    c.close()
    return time_elapse


def get_nni_big_table(conn: sqlite3.Connection):
    """
    Get all metric merged all table of nni.

    Parameters
    ----------
    conn : sqlite3.Connection
        The connection of sqlite by get_nni_connection_by_file_and_exp_id()

    Returns
    -------
    pd.DataFrame
        The result. columns:
        ['timestamp', 'trialJobId', 'parameterId', 'type',
         'sequence', 'default', 'best_prec', 'best_recall',
          'best_f1', 'AUC', 'VALID_LOSS', 'model_id']

    """
    all_data, names = exec_sql(conn, """
    select md.timestamp,md.trialJobId,md.parameterId,md.type,md.sequence,md.data as metrics,TJE.data as hyperparameter,TJE.logPath as logpath, TJE.event as event,
       ep.params as expinfo,ep.id as expid from MetricData as md
    left join TrialJobEvent TJE on md.trialJobId = TJE.trialJobId left join ExperimentProfile as ep
where md.type='FINAL' and TJE.event='WAITING' and ep.revision =1""")
    ret_array = []
    if all_data is None:
        return None
    else:
        for record in all_data:
            res = dict(zip(names, record))
            metrics_ = json.loads(json.loads(res["metrics"]))

            for key_, val_ in metrics_.items():
                res[f"metric_{key_}".upper()] = val_
            for key_, val_ in json.loads(res["expinfo"]).items():
                res[f"expinfo_{key_}".upper()] = val_
                if key_ == "experimentName":
                    _arr = parse_exp_name_to_dict(val_)
                    res.update(_arr)
            for key_, val_ in parse_cmd_to_dict(res['expinfo_trialCommand'.upper()]).items():
                res[f"trialcmd_{key_}".upper()] = val_
            ret_array.append(res)
        if ret_array:
            return pd.DataFrame(ret_array)
        else:
            return None


def parse_exp_name_to_dict(exp_name: str):
    """
    Parse the experiment name to dict.
    an example expname:
    model_entry_file=main_sk.py&search_space_file=none&exp_name=exp_aiops_if_final_v24&max_trial_number=32&trial_concurrency=12&template_file=temp_baseline.yaml&round_rate=none&seed=1&data_sample_rate=0.125&data_sample_method=random&dry_run=false&dataset=iopscompetition&data_id=6d1114ae-be04-3c46-b5aa-be1a003a57cd&nni_tuner=rs&model=isolationforest&gpus=none&wait_interval_time=3&iter_counter=81&n_experiments=162&n_experiment_restart=3

    Parameters
    ----------
    exp_name :

    Returns
    -------

    """
    ret = {}
    if exp_name is None:
        return ret

    if exp_name.find("&") == -1 and exp_name.find("=") == -1:
        logw("exp name can't contains & or =")
        return ret

    for val in exp_name.split("&"):
        try:
            _key, _val = val.split("=")
            ret[_key.upper()] = _val
        except:
            logw(f"Invalid items: {val}")

    return ret


def export_all_experiment_to_mysql(nni_home, table_name='test'):
    """
    Export all result in `nni_home` to mysql database.

    Parameters
    ----------
    nni_home :

    Returns
    -------

    """
    all_metrics = []
    for exp_id in os.listdir(nni_home):
        if str(exp_id).startswith(".") or str(exp_id).startswith("_"):
            # pass the _latest and start_or_restart with .
            continue
        exp_home = os.path.join(nni_home, exp_id)
        if os.path.isdir(exp_home):
            # is a experiment dir
            try:
                conn = get_nni_connection_by_file_and_exp_id(nni_home, exp_id)
                if conn is not None:
                    metric = get_nni_big_table(conn)
                    if metric is not None:
                        metric['nni_exp_directory_name'] = exp_id
                        all_metrics.append(metric)
                    conn.close()

            except:
                traceback.print_exc()
    if len(all_metrics) > 0:

        df = pd.concat(all_metrics, axis=0)
        for key, val in df.dtypes.items():
            if val in ['object']:
                df[key] = df[key].astype('str')

        # fix: ValueError: inf cannot be used with MySQL
        df = df.replace([np.inf, -np.inf], np.nan)
        save_pandas_to_mysql(df, table_name=table_name)
    else:
        log.warn("No metrics to export!")


EXPORT_TYPE_SQLITE = 'sqlite'
EXPORT_TYPE_CSV = 'csv'
EXPORT_TYPE_MYSQL = 'mysql'


def export_all_experiment(nni_home, experiment_working_directory, export_type=None):
    """
    Export all result in `nni_home` to mysql database.

    Parameters
    ----------
    export_type : list
        What type of database to export, default ['csv', 'sqlite'].
    experiment_working_directory : str
        The subdirectory in nni-experiments.
    nni_home : str
        The absolut directory of nni_experiments

    Returns
    -------

    """
    if export_type is None:
        export_type = [EXPORT_TYPE_SQLITE, EXPORT_TYPE_MYSQL]
    UtilSys.is_debug_mode() and log.info(f"{Emjoi.CAR}Starts to export {nni_home}")
    all_metrics = []
    for exp_id in os.listdir(nni_home):
        if str(exp_id).startswith(".") or str(exp_id).startswith("_"):
            # pass the _latest and start_or_restart with .
            UtilSys.is_debug_mode() and log.info(f"Pass for {exp_id}")
            continue

        exp_home = os.path.join(nni_home, exp_id)
        if os.path.isdir(exp_home):
            # is a experiment dir
            conn = get_nni_connection_by_file_and_exp_id(nni_home, exp_id)
            if conn is not None:
                metric = get_nni_big_table(conn)
                if metric is not None:
                    metric['nni_exp_directory_name'] = exp_id
                    all_metrics.append(metric)
                else:
                    logw(f"{exp_id} has no metric.")
                conn.close()
            else:
                log.warning(f"Cannot get connection for {exp_id}, because the connection is {conn}")

    if len(all_metrics) > 0:
        df = pd.concat(all_metrics, axis=0)
        for key, val in df.dtypes.items():
            if val in ['object']:
                df[key] = df[key].astype('str')

        # fix: ValueError: inf cannot be used with MySQL
        df = df.replace([np.inf, -np.inf], np.nan)

        for _et in export_type:

            try:
                exp_name_arr = df[EI.EXP_NAME].unique()
                UtilSys.is_debug_mode() and log.info(exp_name_arr)
                exp_name = exp_name_arr[0]
            except:
                loge(f"Duplicate experiment name in a experiment: {experiment_working_directory}")
                continue

            if _et == EXPORT_TYPE_SQLITE:
                save_pandas_to_sqlite(df,
                                      table_name=exp_name,
                                      file_name=os.path.basename(experiment_working_directory))
            elif _et == EXPORT_TYPE_MYSQL:
                save_pandas_to_mysql(df, table_name=exp_name)
            elif _et == EXPORT_TYPE_CSV:
                save_pandas_to_csv(df, file_name=os.path.basename(experiment_working_directory))
            else:
                raise RuntimeError(f"Unknown export type {_et} in [{export_type}]")

        # save_pandas_to_mysql(df, table_name=table_name)
    else:
        log.warning(f"Experiment {nni_home} has no metrics to export!")


def get_available_nni_experiment_port(port_offset=50000):
    for port in np.arange(start=port_offset, stop=65535, step=1):
        if not is_port_listing(port):
            UtilSys.is_debug_mode() and log.info(f"{Emjoi.SETTING} Port {port} is available")
            return port

    raise RuntimeError("There is no available port to used.")


class NNIResults:
    def __init__(self, home, ext):
        self.home = home
        self.ext = ext

    def load_all_metrics(self):
        files = gaf(self.home, self.ext)
        outs = []
        for _f in files:
            if str(_f).find("_latest") > -1:
                print("Pass for file: {}".format(str(_f)))
                continue

            try:
                conn = get_nni_connection_by_file(_f)
                metrics = get_all_metric_of_nni_db(conn)
                if metrics is None:
                    print("OptMetricsType is None.")
                    continue
                outs = outs + metrics
            except Exception as e:
                raise e
        return outs

    def save(self):
        outs = self.load_all_metrics()
        PDUtil.save_to_excel(pd.DataFrame(outs), "overall")


class NNIExpConf:
    def __init__(self, config_file):
        self.config_file = config_file

    def set_gpu_index(self, gpu_index):
        import yaml

        # 读取 YAML 文件
        with open(self.config_file, 'r') as file:
            data = yaml.safe_load(file)

        # todo: set
        # 修改 trial_command 字段
        data['trial_command'] = f'export CUDA_VISIBLE_DEVICES={gpu_index};' + data['trial_command']

        # 写入 YAML 文件
        with open(self.config_file, 'w') as file:
            yaml.dump(data, file)
