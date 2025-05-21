import json
import numpy as np
import os
import pandas as pd
import random
import sqlite3
from numpy.testing import assert_almost_equal

from nni.tuner import Tuner

from pylibs.utils.util_log import get_logger
from tuner.sampling.base_sampling import BaseSampling

log = get_logger()


class BaseTuner(Tuner):

    def __init__(self, args, kwargs, seed=0):
        self.round_sample_size: list = []
        self.opt_round_samples: dict = {}
        self.trial_results: list = []
        self.n_trails: int = 0
        self.format_search_space = None
        self.bound_beta = None
        self.seed = seed

        # 原始输入的 搜索空间， 不能改变
        self.format_search_space = None
        self.init_samples = None
        self.opt_round_samples = {}

        # 记录历史结果, 搜索空间为实际的搜索空间
        self.trial_results = []
        self.search_space: dict = {}

        # The sample size for the first repeat 1
        np.random.seed(self.seed)
        random.seed(self.seed)

        # kwargs['n_trails'] for debug
        self.n_trails = self.get_n_trials_from_experiment_profile() or kwargs['n_trails']

        # Assert for prod
        assert self.n_trails > 0

        if "opt_top_n" in kwargs.keys():
            self.opt_top_n = kwargs['opt_top_n']
        else:
            self.opt_top_n = int(self.n_trails / 10) if int(self.n_trails / 10) > 0 else 1

        assert self.opt_top_n > 0, f"The top of the points should be larger than 0, but received {self.opt_top_n}"

        # 递归绑定中绑定范围的系数
        if "bound_beta" in kwargs.keys():
            self.bound_beta = kwargs['bound_beta']
        else:
            self.bound_beta = 1.

        if "seed" in kwargs.keys():
            self.seed = kwargs['seed']
        else:
            self.seed = 0

        # 每轮采样的比例
        if "round_rate" in kwargs.keys():
            self.round_rate = kwargs['round_rate']
        else:
            self.round_rate = [0.5, 0.5]
        assert_almost_equal(np.sum(self.round_rate), 1)

        # 每轮样本的大小
        # 如3轮抽取10个样本，比例是6:2:2，那么结果是 [6,8,10], 6 是第一轮的，8 是第二轮的累加和， 10 是前面的累加和
        # 第一轮的样本是6，第二轮的样本是8-6=2，第三轮的样本是10-8=2
        _rate_sum = 0
        for i, _rate in enumerate(self.round_rate):
            _rate_sum = _rate_sum + _rate
            round_size = int(np.round(self.n_trails * _rate_sum))
            if round_size > 0:
                self.round_sample_size.append(round_size)

        # 处理奇数情况
        if self.round_sample_size[-1] - 1 == self.n_trails:
            self.round_sample_size[-1] = self.round_sample_size[-1] - 1

        assert_almost_equal(self.round_sample_size[-1], self.n_trails)

    def check_receive_metric(self, parameters, value):
        for val in self.format_search_space.values():
            if val.type == "choice":
                parameters[val.name] = self.get_choice_value_index(val.name, parameters[val.name])
        self.save_metric_and_parameters(parameters, value)

        if type(value) == dict:
            metric = value['default']
        else:
            metric = value
        return metric

    def get_round_constraint(self, _round, sample_method: BaseSampling):
        current_round_sample_size = self.round_sample_size[_round] - self.round_sample_size[_round - 1]
        if self.opt_round_samples.get(_round) is not None:
            return self.opt_round_samples.get(_round)

        if len(self.trial_results) == 0:
            log.error("Cant get result of trails, maybe n_tails is so small")
            return None
        data = pd.DataFrame(self.trial_results)
        sort_data = data.sort_values(by="metric", ascending=False)

        top_n_sample_size = np.zeros((self.opt_top_n,), dtype=int)

        for _top_n in range(self.opt_top_n):
            top_n_sample_size[_top_n] = int(current_round_sample_size / self.opt_top_n)

        if np.sum(top_n_sample_size) + 1 == current_round_sample_size:
            top_n_sample_size[0] = top_n_sample_size[0] + 1

        assert_almost_equal(np.sum(top_n_sample_size), current_round_sample_size)

        for _top_n, _sample_size in enumerate(top_n_sample_size):
            if _sample_size > 0:
                # 拿到 TOP i 中性能最好的点，i=0,1,...,opt_top_n(int)
                # 处理数据量小的时候
                _top_n_index = _top_n if len(sort_data) > _top_n else 0
                top = sort_data.iloc[_top_n_index, :]

                # 缩小参数范围
                # names, constraints = self.get_new_parameters(top, self.trial_results)

                # 重新生成样本
                new_search_space = self.get_new_search_space(top)
                sampler = sample_method(_sample_size, new_search_space)
                current_round_samples = sampler.get_samples()
                if self.opt_round_samples.get(_round) is not None:
                    self.opt_round_samples[_round] = pd.concat(
                        [self.opt_round_samples[_round], current_round_samples])
                else:
                    self.opt_round_samples[_round] = current_round_samples

                print(
                    f"\nTop {_top_n} ( metric =[{top['metric']}] ) with search space {new_search_space},\n"
                    f"samples: \n{current_round_samples}\n")

        return self.opt_round_samples[_round]

    def get_new_search_space(self, best_point) -> dict:
        """

        Parameters
        ----------
        best_point: pd.DataFrame


        Returns
        -------
        dict
            A search space define in nni, seeing detail for https://nni.readthedocs.io/en/stable/hpo/search_space.html

        """
        best_parameters = best_point['parameters']
        _search_space = pd.DataFrame(pd.DataFrame(self.trial_results).iloc[:, -1].to_list())
        constrants = {}
        for key in best_parameters.keys():
            value = best_parameters[key]
            if type(value) == str:
                value = self.get_choice_value_index(key, value)
            assert type(value) == float

            sort_values = _search_space.sort_values(by=key, ascending=False)
            index = np.where(sort_values[key] == value)[0][0]

            _lower_index = index + 1 if index < len(sort_values) - 1 else index
            _upper_index = index - 1 if index > 0 else index

            lower = sort_values.iloc[_lower_index, :][key]
            upper = sort_values.iloc[_upper_index, :][key]
            constrants[key] = {}
            constrants[key]['_type'] = self.format_search_space.get(("latent_dim",)).type
            _value = np.copy(self.format_search_space.get(("latent_dim",)).values)
            _value[0] = lower * self.bound_beta
            _value[1] = upper * self.bound_beta
            constrants[key]['_value'] = _value
        return constrants

    def get_choice_value_index(self, key, value):
        """

        Parameters
        ----------
        key
        value:str
            The value of the opinion

        Returns
        -------

        """
        choice_option: list = self.format_search_space[(key,)].values
        value = choice_option.index(value) * 1.
        return value

    def get_round(self, parameter_id):
        for round, sample_size in enumerate(self.round_sample_size):
            if parameter_id < sample_size:
                return round

    def get_experiment_config(self, db_file_name, key="maxTrialNumber"):

        """
        在 nni_tuner 中获取当前实验的配置，如最大执行次数 max_trial_number，其 key 要转换为驼峰命名法


        配置样例：
        trial_code_directory: ../../
        experimentWorkingDirectory: ../../../nni-experiments
        max_trial_number: 50
        trial_concurrency: 2
        nni_tuner:
          user: LHSRNS
          class_args:
            optimize_mode: maximize
            n_trails: 50

        training_service:
          platform: local
        Parameters
        ----------
        db_file_name

        key:str
            配置的值

        Returns
        -------
        int
            max_trial_number
        """
        try:
            conn = sqlite3.connect(db_file_name)
            c = conn.cursor()
            exec = c.execute("SELECT * from ExperimentProfile ORDER BY revision DESC")
            recored = exec.fetchone()
            assert recored is not None
            profile = json.loads(recored[0])
            return profile.get(key)
        except Exception as e:
            UtilSys.is_debug_mode() and log.info("Error for getting experiment profile")
            UtilSys.is_debug_mode() and log.info(e)
            return None

    def get_n_trials_from_experiment_profile(self):
        exp_log_directory = self.get_exp_home_directory()
        db_file = os.path.join(exp_log_directory, "db", "nni.sqlite")
        n_trails = self.get_experiment_config(db_file, key="maxTrialNumber")
        # 获取实验数量
        if n_trails is None:
            UtilSys.is_debug_mode() and log.info(
                "Error when getting maxTrialNumber from experiment nni_sample_base_config."
                "\nExcept > 1, received None")
            return False
        else:
            UtilSys.is_debug_mode() and log.info(f"N trials= [{n_trails}] from {db_file}")
            return n_trails

    def get_exp_home_directory(self):
        exp_log_directory = os.environ.get("NNI_LOG_DIRECTORY") or "."
        home = os.path.abspath(os.path.join(exp_log_directory, '..'))
        UtilSys.is_debug_mode() and log.info(f"Experiment home is {home}")
        return home

    def save_metric_and_parameters(self, parameters: dict, value):
        """
        保存参数和性能值到 当前实验文件夹的 parameters_and_value.csv 文件中
        Parameters
        ----------
        parameters
        value

        Returns
        -------

        """
        metric = {}
        if type(value) != dict:
            metric['default'] = value
        else:
            metric = value
        home = self.get_exp_home_directory()
        mp_file = os.path.join(home, "parameters_and_value.csv")
        UtilSys.is_debug_mode() and log.info(f"Metric file is {mp_file}")
        names = np.concatenate([list(parameters.keys()), list(metric.keys())])
        values = np.concatenate([list(parameters.values()), list(metric.values())]).tolist()

        header = False if os.path.exists(mp_file) is True else True
        pd.DataFrame([values], columns=names).to_csv(mp_file, mode="a", header=header, index=False)

    def generate_parameters_by_parameter_id(self, parameter_id):
        UtilSys.is_debug_mode() and log.info(
            f"call generate_parameters_by_parameter_id!\ntype(self.init_samples)={type(self.init_samples)},"
            f"\nparameter_id={parameter_id}"
            f"\nself.init_samples.shape={self.init_samples.shape}")
        # fix bug when trialConcurrency = 2 and maxTrialNumber=1
        parameter_index = self.init_samples.shape[0] - 1 if parameter_id > self.init_samples.shape[
            0] - 1 else parameter_id

        generated_parameters = self.init_samples.iloc[parameter_index, :].to_dict()
        UtilSys.is_debug_mode() and log.info(
            f"\n\ngenerate_parameters:\nparameter_id:{parameter_id}\ngenerated_parameters:{generated_parameters}")

        if generated_parameters is None:
            raise ValueError("generated_parameters is None")
        return generated_parameters
