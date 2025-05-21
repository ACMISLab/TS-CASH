#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/3/28 19:54
# @Author  : gsunwu@163.com
# @File    : test_debug.py
# @Description:
from functools import partial
from pathlib import Path, PosixPath

from smac import Scenario, HyperparameterOptimizationFacade

from pylibs.smac3.search_pace import GenerateConfigSpace, OptConf, objective_function_object

cs = GenerateConfigSpace().config()

# Init smac Scenario
scenario = Scenario(
    configspace=cs,
)
scenario.load(Path("/Users/sunwu/SW-Research/pylibs/pylibs/smac3/tests/2"))
print(scenario)
conf=OptConf(
    exp_name='effect_n_trials_2024_03_29',
    job_name='autoscaling_data_False',
    out_home=Path('/remote-home/cs_acmis_sunwu/experiment_results/automl_results/debug_False/effect_n_trials_2024_03_29'),
    dataset_name='YAHOO',
    data_id='YahooA4Benchmark-TS8_data.out',
    seed=0,
    is_auto_data_scaling=False,
    client_type='local',
    is_debug=False,
    facade='hyperparameter_optimization',
    out_dir=PosixPath('/Users/sunwu/SW-Research/pylibs/pylibs/smac3/tests/2'),
    index=4,
    total=180,
    test_rate=0.4,
    n_trials=100,
    opt_metric='BEST_F1_SCORE'
)
new_objective_function = partial(objective_function_object,
                                     conf=conf)
smac=HyperparameterOptimizationFacade(
    scenario=scenario,
    target_function=new_objective_function,
)
print(smac.optimize())