#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/4/22 10:32
# @Author  : gsunwu@163.com
# @File    : evolution_examples.py
# @Description:
import pandas as pd
from ConfigSpace import ConfigurationSpace, Categorical, Float
from smac import Scenario
from tqdm import tqdm

from exps.e_cash_libs import RandomFacade
from pylibs.config import Env
from pylibs.hpo.black_box_function import black_function_1
from pylibs.utils.util_gnuplot import Gnuplot

cs = ConfigurationSpace(
    name="myspace",
    space={
        "classifier": Categorical("classifier", ['blax_function']),
        "x": Float("x", [-10, 10.0]),
    },
    seed=42
)
scenario = Scenario(
    configspace=cs,
    n_trials=140,
    name="debug",
    output_directory=Env.get_runtime_home(),
    seed=1,
    crash_cost=1,
    use_default_config=True,
    deterministic=True
)
ef = RandomFacade(scenario=scenario)

for i in tqdm(range(140)):
    conf = ef.ask()
    acc = black_function_1(conf['x'])
    print(f"{conf}={acc}")
    ef.tell(conf, fun_cost=acc, time=1)

run_histories = ef.get_history_obj()

costs = []
for run_his in run_histories:
    costs.append(run_his.cost)

gp = Gnuplot()
gp.set_output_pdf("debug.pdf")
gp.add_data(pd.DataFrame(costs))
gp.sets("""
      plot $df using 1 with lp
      """)
gp.show()