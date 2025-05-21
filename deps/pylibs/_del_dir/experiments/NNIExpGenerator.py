"""
Limit nni experiment instance by environment MAX_N_EXPERIMENTS, default 5.
"""

import argparse
import os
import re
import sys
import time
from copy import deepcopy

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from nni.tools.nnictl.launcher import create_experiment
from pylibs.common import ConstTuner, Config, ConstNNI, Emjoi, SampleType
from pylibs.experiments.NNIExpResult import NNIExperimentUtils
from pylibs.util_dataset import DatasetName
from pylibs.utils.util_base64 import base64_encode_str
from pylibs.utils.util_datatime import get_datatime
from pylibs.utils.util_feishu import send_msg_to_feishu
from pylibs.utils.util_hash import get_str_hash
from pylibs.utils.util_json import is_json
from pylibs.utils.util_log import get_logger
from pylibs.utils.util_message import log_warn_msg, logi
from pylibs.utils.util_nni import get_available_nni_experiment_port, get_experiments_by_name
from pylibs.utils.util_nni_dispatcherlog import NNIDispatcherLog
from pylibs.utils.util_nni_exp_info import NNIExperimentInfo
from pylibs.utils.util_nni_managerlog import NNIManagerLog
from pylibs.utils.util_nni_sqlite import NNISqlite
from pylibs.utils.util_nnictl import NNICTL

log = get_logger()


def _get_exp_id_by_name(exp_name):
    cur_exp = get_experiments_by_name(exp_name)
    return cur_exp[ConstNNI.KEY_CMD_ID].values[0]


class Args:
    pass


class NNIExpGenerator:
    """
    Generator of NNI experiment.
    NNI: https://nni.readthedocs.io/en/stable/

    xxx_home: indicate a directory, e.g., /root/nni-experiments
    xxx_file: indicate a file, e.g., /root/home/a.csv


    **Model parameter** starts with mp_,
    e.g., `mp_dataset=1` will become `python coca.py --m_dataset 1`, where `coca.py`
    is a model implementation.

    **Experiment parameter(generating experiments for nni) starts np_**,
    e.g., `np_model_entry_file=2` will become `python experiment.py --n_model_entry_file 2`, where `coca.py`


    If you want to add a new parameter (--hpy 2) to the target model_entry_file, e.g., adding --hpy for main_dask.py, you can
    NNIExpGenerator(_mp_hpy=2)


    iter_counter indicates how many experiments you have run, which is started with 1.
    n_experiments indicates the number of ✅All experiments.

    """
    PATTERN_MODEL_PARAMETERS = re.compile("^_mp_(.*)")
    CONFIG_SEARCH_SPACE_BASE = "search-spaces"

    # The available port in docker of GPU is `-p 50000-50100`
    NNI_PORT_OFFSET = 50000

    # if the number of experiments is greater than 20, then stop ✅All experiments
    MAX_N_EXPERIMENTS = int(os.environ.get("MAX_N_EXPERIMENTS")) if os.environ.get(
        "MAX_N_EXPERIMENTS") is not None else 5

    EXPERIMENT_ID_QUEUE = []

    def __init__(self,

                 home: str = os.path.abspath(os.path.dirname(__file__)),
                 model_entry_file: str = 'main_sk.py',
                 search_space_file: str = None,
                 exp_name: str = 'test_exp_name',
                 max_trial_number: int = 1,
                 trial_concurrency: int = 1,
                 template_file: str = "temp_baseline.yaml",
                 round_rate: float = None,
                 seed: int = 0,
                 data_sample_rate: float = None,
                 data_sample_method: str = None,
                 dry_run: bool = True,
                 dataset: str = DatasetName.DEBUG.value,
                 data_id: str = None,
                 nni_tuner: str = 'RS',
                 model: str = "IsolationForest",
                 gpus: str = None,
                 wait_interval_time: int = 3,
                 iter_counter: int = 0,
                 n_experiments: int = 1,
                 n_experiment_restart: int = 3,
                 **kwargs
                 ):
        """
        The experiment generator.
        Examples
        --------
        If you want to add a new parameter (--hpy 2) to the target model_entry_file,
        e.g., adding --hpy for main_dask.py, you can NNIExpGenerator(_mp_hpy=2)

        Parameters
        ----------
        n_experiment_restart:int
            How many times to restart the experiment if failed. Default 3.
        n_experiments:int
            The number of experiments. It needs you to calculate before instantiating this class. Specially you can get
            the value of n_experiments by `get_number_of_experiments(args)`.
        iter_counter:int
            How many times to initialize this class. Default 0.
        exp_name: str
            The name of the experiment
        nni_tuner: str
            The hyperparameter tuner of NNI, see https://nni.readthedocs.io/en/stable/hpo/tuners.html.
        max_trial_number: int
            The number of trials in an experiment.
        trial_concurrency:int
            The number of threads to parallel running the experiment.
        home: str
            The working directory for the experiment
        entry_file: str
            The entry file for the experiment, e.g., main_dask.py
        search_space_file: str
            The search space file saved as json of NNI, see https://nni.readthedocs.io/en/stable/hpo/search_space.html.
            Relative path to the home.
        template_file: str
            The template for generating search space file. Default temp_baseline.yaml.
        gpus: str
            The exp_index of the GPU to use for training. Multi-GPUs can be used by comma separated.
            -1  means to use CPU for working.
            e.g., gpus="0,1" means to use the first and second GPU to train the model.
        wait_interval_time: int
            The scan interval for exec NNI results, i.e. the command: nnictl experiments list --all

        """
        # How many times you call self._create_nni_experiment_and_wait_v2.

        assert isinstance(dataset, str)
        assert isinstance(model, str)

        self._mp_dataset = dataset
        self._mp_seed = seed
        self._mp_model = model
        self._mp_data_id = data_id
        self._mp_data_sample_rate = data_sample_rate
        self._mp_gpus = gpus
        self._n_experiments = n_experiments

        self._experiment_recreate_counter = 0
        self._n_experiment_restart = n_experiment_restart
        self._iter_counter = iter_counter
        self._set_model_parameters(kwargs)
        self._max_trial_number = max_trial_number
        self._exp_name = exp_name
        self._nni_tuner = nni_tuner
        self._avaliable_nni_tuners = NNICTL().get_all_tuners()
        if self._nni_tuner not in self._avaliable_nni_tuners:
            raise RuntimeError(f"NNI tuner [{self._nni_tuner}] is not available. "
                               f"Please registry by nnictl algo register, e.g.,\n"
                               f"nnictl algo register -m tuner/nni_sample_base_config/scbol_tuner.yaml.\n"
                               f"Available tuners: {self._avaliable_nni_tuners}")
        self._class_args = self._prepare_class_args(round_rate)
        self._dry_run = dry_run
        self._kwargs = kwargs
        self._all_args = self._get_all_parameters(locals(), kwargs)
        # The exp id in running.
        self._exp_info: NNIExperimentInfo = NNIExperimentInfo()  # type:
        self._home = os.path.abspath(home)
        if not os.path.exists(self._home):
            raise NotADirectoryError(f"[{self._home}] must be a dir!")
        self._model_entry_file = model_entry_file
        self._runtime = os.path.join(self._home, "runtime")
        self._trail_concurrency = trial_concurrency
        if not os.path.exists(self._runtime):
            os.makedirs(self._runtime)

        if data_sample_method is not None:
            self._mp_data_sample_method = data_sample_method
        else:
            self._mp_data_sample_method = SampleType.RANDOM

        if data_sample_rate is not None:
            self._mp_data_sample_rate = data_sample_rate
        else:
            self._mp_data_sample_rate = 1

        if seed is None:
            self._mp_seed = 0
        else:
            self._mp_seed = seed

        self._wait_interval_time = wait_interval_time

        self._search_space_file = self._prepare_search_space_file()
        # Parse template
        env = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "templates")))  # 创建一个包加载器对象
        self.template = env.get_template(template_file)

        self.commands = []

        self._nni_experiments_home = self.get_nni_experiments_home()
        if not os.path.exists(self._nni_experiments_home):
            os.makedirs(self._nni_experiments_home)
        assert os.path.exists(self._nni_experiments_home)

        UtilSys.is_debug_mode() and log.info(
            f"{Emjoi.CAR} ({self._get_finished_rate()}) Start experiment [{exp_name}], "
            f"hash of parameters: {get_str_hash(self._generated_experiment_name())}")
        UtilSys.is_debug_mode() and log.info(
            f"{Emjoi.SETTING} Home for nni-experiments : [{self._nni_experiments_home}]")

        self._exp_name_generated = self._generated_experiment_name()

    def _increase_n_create_experiment(self):
        """
        To record how many times we recreate the current experiment.

        Returns
        -------

        """
        self._experiment_recreate_counter = self._experiment_recreate_counter + 1

    def _can_able_recreate_experiment(self):
        """
        True if  we can  delete the current  experiment, and then recreate it.
        False otherwise.

        Returns
        -------

        """
        return self._experiment_recreate_counter < self._n_experiment_restart

    def _generate_model_args(self):
        """
        Generate the parameters of the model, e.g., main_torch.py --args

        Returns
        -------

        """

        __base64_encoded = ["--hps"]
        models_args = ""
        for _k, _v in self.__dict__.items():
            m_key = self._get_model_parameter_key(_k)
            if m_key is not None:
                if str(_v).find("\"") > -1:
                    raise RuntimeError(f"The value for key [{_k}] of the parameter cant contain [\"]")

                if str(_v).find("--") > -1:
                    raise RuntimeError(f"Error pass parameter value, got {_v}. \n If your parameter value contains ', "
                                       f"like JSON string you must enclose it by \" ")

                # If value is json, base64 is to exchange json information in yaml
                if m_key.strip() in __base64_encoded and is_json(_v):
                    _v = base64_encode_str(_v)

                # _v = str(_v).replace(":", "::")
                models_args = f"{models_args} {m_key} {_v}"

        UtilSys.is_debug_mode() and log.info(f"Generating model args: \n {models_args}")

        return models_args

    def get_nni_experiments_home(self):
        """
        Get the absolut path of nni-experiments of nni.

        Returns
        -------
        str
            The absolut path of  nni-experiments
        """
        assert self._exp_name is not None
        return os.path.abspath(os.path.join(self._home, "../../", "nni-experiments", self._exp_name))

    def _get_class_args(self):
        return self._class_args

    def _init_config_files(self):
        """
        Generated Experiment files for NNI

        Parameters
        -----------

        """
        UtilSys.is_debug_mode() and log.info(f"{Emjoi.SETTING} Generate experiment configuration ...")
        assert type(self._nni_tuner) == str
        UtilSys.is_debug_mode() and log.info(f"App home {os.path.abspath(self._get_config_home())}")

        # Generate experiment configuration
        self._generate_config_for_nni_exp_config()

        # Generate bash file
        # self._generate_nni_running_bash()

    def _generate_config_for_nni_exp_config(self):
        assert os.path.exists(self._home)
        assert os.path.exists(os.path.join(self._home, self._model_entry_file))
        template_result = self.template.render(
            entry_file=self._model_entry_file,
            entry_file_args=self._generate_model_args(),
            exp_name=self._generated_experiment_name(),
            search_space_file=self._get_search_space_file(),
            code_directory=self._get_code_directory(),
            max_trial_number=self._get_max_trial_number(),
            trial_concurrency=self._trail_concurrency,
            exp_working_directory=self._nni_experiments_home,
            trial_code_directory=self._home,
            nni_tuner=self._nni_tuner,
            class_args=self._get_class_args()
        )  # 渲染

        assert os.path.exists(self._get_config_home())
        assert self._generated_experiment_name() is not None
        # 配置 nni_sample_base_config 的路径
        config_file_abspath = self._get_config_file_path()

        with open(config_file_abspath, "w") as f:
            f.writelines(template_result)

        UtilSys.is_debug_mode() and log.info(f"\n\n Configuration: \n{template_result}\n\n")

        # run_cmd = self._get_nni_create_command()
        # self.commands.append(run_cmd)

    def _get_config_file_path(self):
        return os.path.join(self._get_config_home(), f"{get_str_hash(self._generated_experiment_name())}.yaml")

    def _get_nni_create_command(self):
        """
        Returns the command for nnictl create --port xxxx --config xxxx

        Returns
        -------
        str
            The command for create nni experiment:  nnictl create  --port xx --config xx
        """
        return f"nnictl create --port {self.get_config_running_port()} --config {self._get_config_file_path()} --debug"
        # return f"bash {self._prepare_local_running_bash_file_name()}"

    def _get_config_save_home(self):
        """
        Configure the :
        1. search space file
        2. config save home.

        Returns
        -------

        """
        config_save_home = os.path.join(self._home, Config.EXP_DIRECTORY_SAVE_HOME)

        if not os.path.exists(config_save_home):
            os.makedirs(config_save_home)

        if not os.path.isdir(config_save_home):
            raise IsADirectoryError(f"Expect a director for [{config_save_home}] ")

        _entry_file = os.path.join(self._home, self._model_entry_file)
        if not os.path.exists(_entry_file):
            raise FileNotFoundError(f"Cant find entry file named [{os.path.basename(_entry_file)}]"
                                    f"\nMove [{os.path.basename(_entry_file)}] to directory "
                                    f"{os.path.abspath(os.path.dirname(_entry_file))}")

        return config_save_home

    def _prepare_search_space_file(self):
        """
        Get the absolute path of search file, e.g.,
        '/Users/sunwu/SyncResearch/p1_ch05/experiments/search_spaces/deepsvdd_search_space.json'
        Returns
        -------
        str

        """
        search_space_file = os.path.abspath(
            os.path.join(self._home, self.CONFIG_SEARCH_SPACE_BASE, self._mp_model.lower() + ".json"))

        _search_space_home = os.path.abspath(os.path.dirname(search_space_file))
        if not os.path.exists(_search_space_home):
            os.makedirs(_search_space_home)

        if not os.path.exists(search_space_file):
            errmsg = f"Cant find search space file named [{os.path.basename(search_space_file)}]" \
                     f"\nMove [{os.path.abspath(search_space_file)}] to directory:\n" \
                     f"{os.path.abspath(search_space_file)}"
            raise FileNotFoundError(errmsg)
        assert os.path.exists(search_space_file)
        return os.path.abspath(search_space_file)

    def _get_search_space_file(self):
        return self._search_space_file

    def _get_config_home(self):
        """
        Get the config home of NNI experiments, e.g., '/Users/sunwu/SyncResearch/p1_ch05/experiments/configs'

        Returns
        -------
        str

        """
        return self._get_config_save_home()

    def _generate_nni_running_bash(self):
        """
        生成NNI实现的执行文件
        Returns
        -------

        """
        cmds = deepcopy(self.commands)
        cmds.insert(0, f"cd {self._home}\n")
        with open(self._prepare_local_running_bash_file_name(), "w") as f:
            f.writelines(cmds)

        UtilSys.is_debug_mode() and log.info(f"Generate bash file:\n{self._prepare_local_running_bash_file_name()}\n")
        assert os.path.exists(self._prepare_local_running_bash_file_name())

    def get_config_running_port(self):
        #
        port = get_available_nni_experiment_port(self.NNI_PORT_OFFSET)
        # if the number of experiments is greater than 25, then stop ✅All experiments

        if len(NNIExpGenerator.EXPERIMENT_ID_QUEUE) >= self.MAX_N_EXPERIMENTS:
            _stop_exp_id = NNIExpGenerator.EXPERIMENT_ID_QUEUE.pop()
            UtilSys.is_debug_mode() and log.info(
                f"NNI instance is greater than {self.MAX_N_EXPERIMENTS}, stop previous experiment {_stop_exp_id}")
            NNICTL.stop_experiment(_stop_exp_id)
        # if port >= self.NNI_PORT_OFFSET + self.MAX_N_EXPERIMENTS:
        #     log.info(f"NNI instance is greater than {self.MAX_N_EXPERIMENTS}, stop previous experiments.")
        #     NNICTL.stop_all()
        return port

    def start_or_restart(self):
        """
        Training the model one the search space.

        Returns
        -------

        """

        self._exp_info: NNIExperimentInfo = self._update_experiment_info()

        # If the experiment is running, wait for it to finish
        if self._exp_info.is_running_alive():
            logi(f"Experiment is running.")
            self._wait_experiment_done()

        # If the experiment is not exists, create a new one.
        elif self._exp_info.get_exp_id() is None:
            self._init_config_files()
            self._create_nni_experiment_and_wait_v2()


        else:
            # If stopped or view, Check whether all the trails is successful.
            pass
            # if self._check_all_trials_is_finished_and_report_progress():
            #     # self._exp_info.stop_experiment()
            #     finished_msg = f"Experiment [{self._exp_info.get_exp_id()}] is finished with status " \
            #                    f"{self._exp_info.get_exp_status()} Review it by " \
            #                    f"nnictl view --port {self.get_config_running_port()} {self._exp_info.exp_id}\n\n\n"
            #     log_success_msg(finished_msg)
            # else:
            #     finished_msg = f"Experiment [{self._exp_info.get_exp_id()}] is finished with status " \
            #                    f"{self._exp_info.get_exp_status()} Review it by " \
            #                    f"nnictl view --port {self.get_config_running_port()} {self._exp_info.exp_id}\n\n\n"
            #     log_error_msg(finished_msg)

        self._check_all_trials_is_finished_and_report_progress()

    def _update_experiment_info(self):
        """
        Update experiment info for this experiment.

        Returns
        -------

        """
        return NNICTL.get_experiment_by_name_from_shell(self._generated_experiment_name())

    def _check_all_trials_is_finished_and_report_progress(self):
        ni = NNISqlite(self._nni_experiments_home, self._exp_info.get_exp_id())
        if ni.is_all_trials_success():
            # Dot not changed the text, the message is used for testing
            report_metric = f"{Emjoi.SUCCESS}({self._get_finished_rate()}) " \
                            f"All trials in [{self._exp_name}, {self._exp_info.get_exp_id()}] are success. " \
                            f"Mean default metric is {ni.get_mean_default_metrics()}\n\n\n\n"
            UtilSys.is_debug_mode() and log.info(report_metric)
            send_msg_to_feishu(report_metric)
            return True

        else:
            msg = f"Trials {ni.get_failed_trials()} are failed, so re-create this experiment " \
                  f"{self._generated_experiment_name()} "
            log_warn_msg(msg)
            self._recreate_experiment()
            return False

    def _recreate_experiment(self):
        if self._can_able_recreate_experiment():
            error_mess = f"{Emjoi.WARNING} Recreate experiment {self._exp_name}({self._exp_info.get_exp_id()}) " \
                         f"because finding failed trials in {self.get_nni_experiments_home()}"
            UtilSys.is_debug_mode() and log.info(error_mess)
            self._exp_info.delete()

            # fix: to prevent run in a circle: delete a not existing experiment.
            self._init_status()
            self.start()
            send_msg_to_feishu(error_mess)
        else:
            error_mess = f"{Emjoi.ERROR} Experiment {self._exp_name} is failed " \
                         f"after retry {self._n_experiment_restart}  in [{self.get_nni_experiments_home()}].\n" \
                         f"Please checking your program!!!"
            log.error(error_mess)
            send_msg_to_feishu(error_mess)
            raise RuntimeError(error_mess)

    def __wait_exp_done_v2(self):
        if self._exp_info.get_exp_id() is None:
            msg = "Experiment is not found before waiting experiment mark_as_finished."
            send_msg_to_feishu(msg)
            raise RuntimeError(msg)
        if not self._dry_run:
            while True:
                if self._exp_info.is_done_or_error():
                    break
                else:
                    self._print_runing_info()
                    self._wait()

    def __wait_exp_done_v1(self):
        if self._exp_info.get_exp_id() is None:
            msg = "Experiment is not found before waiting experiment mark_as_finished."
            send_msg_to_feishu(msg)
            raise RuntimeError(msg)
        if not self._dry_run:
            while True:

                # log.info(f"Environment: {exec_cmd_return_stdout_and_stderr('echo $PATH;conda env list')}")
                # log.info(f"===>\nUpdated experiment status: {pprint.pformat(self._exp_info.__dict__)}\n<===")

                nml = NNIManagerLog(self._nni_experiments_home, self._exp_info.get_exp_id())
                if nml.is_experiment_done():
                    break
                else:
                    self._print_runing_info()

                if nml.is_experiment_error():
                    content_ = f"Found error in experiment {self._exp_info.exp_id}. " \
                               f"\nThe nnimanager.log is located at {nml.get_nni_namager_file_name()}." \
                               f"\nThe contents of nnimanager.log is " \
                               f"\n{nml.get_nni_namager_file_content()}"
                    send_msg_to_feishu(content_)
                    raise RuntimeError(content_)
                elif nml.is_experiment_stopped():
                    UtilSys.is_debug_mode() and log.info(f"Experiment {self._exp_info.exp_id} is stopped. "
                                                         f"\nThe nnimanager.log is located at {nml.get_nni_namager_file_name()}."
                                                         f"\nThe contents of nnimanager.log is \n{nml.get_nni_namager_file_content()} ")

                # Fix: issue_yv37rfbh
                ndp = NNIDispatcherLog(self._nni_experiments_home, self._exp_info.get_exp_id())
                if ndp.has_error():
                    self._recreate_experiment()
                    break
                self._wait()

    def _print_runing_info(self):
        UtilSys.is_debug_mode() and log.info(
            f"{Emjoi.WAITING} [{get_datatime()}] ({self._get_finished_rate()}) "
            f"Experiment (name={self._exp_name},"
            f"exp_id={self._exp_info.get_exp_id()},"
            f"retry={self._experiment_recreate_counter}/{self._n_experiment_restart}) is running at "
            f"[{self._exp_info.get_view_url()}], "
            f"waiting {self._wait_interval_time}s ...")

    def _wait_experiment_done(self):
        # self.__wait_exp_done_v1()
        self.__wait_exp_done_v2()

    def get_best_result(self):
        """
        获取所有实现的做好的结果，根据 default 字段排序。

        Call this after get_exp_result

        Returns
        -------
        pd.DataFrame
            所有实现的最好结果

        """
        if self._dry_run:
            return None
        exp_best_results = get_experiments_by_name(self._generated_experiment_name())
        if exp_best_results is not None:
            exp_ids = exp_best_results[ConstNNI.KEY_CMD_ID].to_list()
            if len(exp_ids) > 0:
                nni_exp = NNIExperimentUtils(
                    home_for_nni_experiments=self._nni_experiments_home,
                    exp_name=self._generated_experiment_name(),
                    exp_ids=exp_ids)
                best_result = nni_exp.get_best_results()
                return best_result

        else:
            log.warning("Exp_best_results is None")
            return None

    def _create_nni_experiment_and_wait_v2(self):
        # create_experiment

        self._increase_n_create_experiment()

        if not self._dry_run:

            # Create experiment
            args = argparse.ArgumentParser()
            args.port = self.get_config_running_port()
            args.config = self._get_config_file_path()
            args.debug = True
            args.url_prefix = ""
            args.foreground = False
            create_experiment(args)

            try:
                self._exp_info: NNIExperimentInfo = self._update_experiment_info()
            except RuntimeError as e:
                send_msg_to_feishu("Recreate experiment failed, retrying!!!")
                NNICTL.stop_all()
                self._wait()
                self.start()

            msg = f"{Emjoi.START} ({self._get_finished_rate()}) Experiment [{self._exp_name}] with id " \
                  f"[{self._exp_info.get_exp_id()}] is running at {self._exp_info.get_view_url()}"
            send_msg_to_feishu(msg)
            if self._exp_info is None:
                send_msg_to_feishu("Running experiment failed.")
                sys.exit(-1)
            self.EXPERIMENT_ID_QUEUE.insert(0, self._exp_info.get_exp_id())
            self._wait_experiment_done()

    def _prepare_local_running_bash_file_name(self):
        return os.path.abspath(os.path.join(self._get_config_home(), get_str_hash(self.get_exp_name()) + ".sh"))

    @staticmethod
    def _get_exp_status_by_name(exp_name):
        cur_exp = get_experiments_by_name(exp_name)
        if cur_exp is None:
            return None
        return cur_exp['status'].values[0]

    def _prepare_class_args(self, round_rate):
        """
        Get round_rate for nni nni_tuner.

        Returns
        -------

        """
        if round_rate is None:
            round_rate = [0.8, 0.2]

        if self._nni_tuner in [ConstTuner.RS_RNS, ConstTuner.MLHS_RNS, ConstTuner.LHS_RNS, ConstTuner.RS_RRS,
                               ConstTuner.LHS_RRS]:
            return f"round_rate: {round_rate}"
        else:
            return ""

    def _get_max_trial_number(self):
        return self._max_trial_number

    def _get_code_directory(self):
        return os.path.abspath(os.path.join(self._home, self._model_entry_file))

    def get_exp_name(self):
        """
        Get experiment user of current exp.

        Returns
        -------

        """
        return self._generated_experiment_name()

    def _get_exp_home(self):
        return self._home

    def _get_dispatcher_log_file(self):
        """
        Return the absolute path dispatcher.log.
        e.g., '/Users/sunwu/nni-experiments/jvg6t9e2/log/dispatcher.log'

        Returns
        -------

        """
        return os.path.abspath(
            os.path.join(self._nni_experiments_home, _get_exp_id_by_name(self._generated_experiment_name()),
                         f"log/dispatcher.log"))

    def _wait(self):
        time.sleep(self._wait_interval_time)

    def _generated_experiment_name(self):
        """
        Generate experiment name by parameters.
        Returns
        -------

        """

        _k_str = ""
        for _k, _v in self._all_args.items():
            if not re.search("^[\w.\-_]+$", str(_v)):
                UtilSys.is_debug_mode() and log.info(f"Pass for {_v} when generating experiment name")
                continue
            # _k_arr = re.split("[_ -]", _k)
            # _key = "".join([w[0] for w in _k_arr])
            # _k = re.sub("[_\-]", "", _k)
            _k_str = f"{_k_str}&{_k}={_v}"

        _k_str = _k_str.lower()[1:]
        # log.info(f"{Emjoi.SETTING} Init experiment name as [{_k_str}]")
        return _k_str

    @staticmethod
    def _get_all_parameters(param: dict, kwargs: dict):
        """
        Append all parameters of kwargs to locals

        Parameters
        ----------
        param :
        kwargs :

        Returns
        -------

        """
        p = kwargs.copy()
        for _k, _v in param.items():
            if type(_v) is dict:
                continue
            p[_k] = _v
        return p

    def _set_model_parameters(self, kwargs):
        """
        Set the parameters in the kwargs starting with `_mp_` to the model.

        Parameters
        ----------
        kwargs :

        Returns
        -------

        """
        for _k, _v in kwargs.items():
            key = self._get_model_parameter_key(_k)
            if key is not None:
                setattr(self, _k, _v)

    def _get_model_parameter_key(self, _k):
        """
        If the _k is start_or_restart with `_mp_`, then it's a model parameter, and return the key name. For example,
        `_mp_abc` will return "--abc".

        None if _k is not start_or_restart with `_mp_`.


        Returns
        -------

        """
        _s = self.PATTERN_MODEL_PARAMETERS.search(_k)
        if _s:
            m_key = " --" + _s.group(1)
            return m_key
        else:
            return None

    def _get_finished_rate(self):
        percent = np.round(self._iter_counter * 100 / self._n_experiments, 2)
        return f"{self._iter_counter} / {self._n_experiments} = {percent}%"

    def _init_status(self):
        NNICTL.reset()
