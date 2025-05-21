#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/4/22 10:32
# @Author  : gsunwu@163.com
# @File    : evolution_examples.py
# @Description:
import numpy as np
import pandas as pd

from pylibs.hpo.black_box_function import black_function_1
from pylibs.utils.util_gnuplot import Gnuplot

xes=np.linspace(-10,10,1000)
costs = []
for x in xes:
    acc = black_function_1(x)
    costs.append(acc)

gp = Gnuplot()
gp.set_output_pdf("debug.pdf")
gp.add_data(pd.DataFrame(costs))
gp.sets("""
      plot $df using 1 with lp
      """)
gp.show()