#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/10/27 10:23
# @Author  : gsunwu@163.com
# @File    : RS.py
# @Description:
from dataclasses import dataclass
from bayes_opt import UtilityFunction
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from smac.runhistory import TrialInfo, TrialValue
from tshpo.hpo.api import HPO


@dataclass
class BO(HPO):
    """
    最小化任务， 不是最大化哦
    """
    # the hyperparameters of GP
    surrogate_model: callable = None
    kappa: float = 2.576
    size_of_candidate_samples: int = 10000  # 随机抽样时抽取多少个来评估呢？

    # 用来预热的超参数优化的数量
    n_warmup_sample: int = 30

    def __post_init__(self):
        if self.surrogate_model is None:
            self.surrogate_model = GaussianProcessRegressor(
                kernel=Matern(nu=2.5),
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=self.history.seed,
            )

        # 初始化15个超参数
        self.init_configs = self.cs.sample_configuration(size=self.n_warmup_sample)

    def optimize(self):

        """
        贝叶斯优化，转为最小化问题了.
        历史数据存在th中了，相当于tell在history中，这里只负责ask

        Parameters
        ----------
        th :
        cs :
        surrogate_model :
        random_state :

        Returns
        -------

        """
        # 初始化默认 surrogate model
        # 如果训练历史为空，就随机返回一个配置
        th = self.history
        cs = self.cs
        if th.is_empyt():
            return cs.sample_configuration(size=1)

        train_history_array = th.get_array_history()
        candidate_samples = cs.sample_configuration(size=self.size_of_candidate_samples)

        train_x = train_history_array[:, 0:-1]
        train_y = train_history_array[:, -1]

        candidate_x = th.get_array_of_configurations(candidate_samples)

        # 加个负号，转为最小值问题
        self.surrogate_model.fit(train_x, -train_y)

        acq = UtilityFunction.ucb(
            x=candidate_x,
            kappa=self.kappa,
            gp=self.surrogate_model
        )
        best_candidata_index = acq.argmax()
        next_config = candidate_samples[best_candidata_index]
        # 是否选择模型的超参数
        return next_config

    def tell(self, info: TrialInfo, value: TrialValue, save: bool = True) -> None:
        """
        tell 的工作已经迁移到history中，在history 中已经做了tell的工作，这里不需要再重新添加
        Parameters
        ----------
        info :
        value :
        save :

        Returns
        -------

        """
        pass

    def ask(self):
        if self.history.count < self.n_warmup_sample:
            # warm up
            return TrialInfo(config=self.init_configs[self.history.count])
        else:
            # optimization
            _config = self.optimize()
            return TrialInfo(config=_config)
