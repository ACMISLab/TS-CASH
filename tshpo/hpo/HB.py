#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/10/28 16:49
# @Author  : gsunwu@163.com
# @File    : HB.py
# @Description:
# !/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/10/27 10:23
# @Author  : gsunwu@163.com
# @File    : RS.py
# @Description:
from dataclasses import dataclass
from smac import Scenario, HyperbandFacade
from smac.initial_design import RandomInitialDesign
from smac.runhistory import TrialInfo, TrialValue
from tshpo.hpo.api import HPO


@dataclass
class HB(HPO):
    optimizer: HyperbandFacade = None

    def __post_init__(self):
        def _train_placeholder(config=None, seed=None, budget=None):  # 占位符，兼容smac3, 无实际作用
            pass

        scenario = Scenario(self.cs,
                            deterministic=True,
                            n_trials=1,
                            min_budget=10,
                            max_budget=100,
                            seed=self.econf.random_state,
                            )

        initial_design = RandomInitialDesign(scenario,
                                             n_configs=None, n_configs_per_hyperparameter=10,
                                             max_ratio=0.25, additional_configs=None, seed=self.econf.random_state)

        intensifier = HyperbandFacade.get_intensifier(
            scenario,
        )
        self.optimizer = HyperbandFacade(scenario,
                                         _train_placeholder,
                                         intensifier=intensifier,
                                         initial_design=initial_design,
                                         overwrite=True,
                                         logging_level=False
                                         )

    def tell(self, info: TrialInfo, value: TrialValue, save: bool = True) -> None:
        self.optimizer.tell(info, value, False)

    def ask(self):
        return self.optimizer.ask()
