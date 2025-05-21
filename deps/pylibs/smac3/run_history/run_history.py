#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/3/26 21:42
# @Author  : gsunwu@163.com
# @File    : run_history.py
# @Description:
import numpy as np
import smac

from exps.search_pace import GenerateConfigSpace, UtilScenario

"""
import smac
file="/Users/sunwu/SW-Research/pylibs/automl/effect_n_trials_all_2024_03_19/hyperparameter_optimization/SMD/machine-1-5.test.csv@23.out/trials_30/test_rate_0.3/opt_metric_BEST_F1_SCORE/autoscaling_data_False/2/runhistory.json"
with open(file,"r") as f:
    data=json.load(f)
print(data)
"""

cs = GenerateConfigSpace().config()
rh = smac.runhistory.RunHistory()
rh.load(
    "/Users/sunwu/Documents/experiment_results/automl_results/debug_True/test_exp/hyperparameter_optimization/DEBUG1/debug11.out/trials_10/test_rate_0.3/opt_metric_VUS_ROC/autoscaling_data_False/0/runhistory.json",
    cs)


us=UtilScenario(output_directory="/Users/sunwu/Documents/experiment_results/automl_results/debug_True/test_exp/hyperparameter_optimization/DEBUG1/debug11.out/trials_10/test_rate_0.3/opt_metric_VUS_ROC/autoscaling_data_False/0",configuration_space=cs)
print(us.is_finished())
print(us.used_walltime())
print(us.used_target_function_walltime())
print(us.best_config())
print(us.minimal_cost())
#
# print("default acc %s" % rh.get_min_cost(cs.get_default_configuration()))
# print(np.min(list(rh._min_cost_per_config.values())))
# print(rh.finished)
# # print("best acc:",rh.get_configs_per_budget())