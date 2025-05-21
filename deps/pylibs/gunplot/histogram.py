#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/3/8 19:08
# @Author  : gsunwu@163.com
# @File    : histogram.py
# @Description:
from dataclasses import dataclass

import pandas as pd

from pylibs.utils.util_gnuplot import Gnuplot


@dataclass
class Histogram:
    """
    The histogram for gnuplot
    """
    file_name: str = "test.pdf"
    df: pd.DataFrame = None

    def two_comp(self, df):
        """
       'data_id',
       'VUS_ROC_mean_disable_autoscale',
       'VUS_ROC_std_disable_autoscale',
       'VUS_ROC_mean_enable_autoscale',
       'VUS_ROC_std_disable_autoscale'

        Parameters
        ----------
        df :

        Returns
        -------

        """
        gp = Gnuplot()
        gp.set_output_pdf(self.file_name)
        gp.add_data(df, header=False)
        gp.set("set size 1,0.75")
        gp.set("set origin 0,0.2")
        gp.set("set xtics rotate by -30 noenhanced")
        gp.set("set style fill solid")
        gp.set("set key at screen 0.5,screen 0.98 center maxrows 1")
        gp.set('set yr [0:1]')
        gp.unset('unset colorbox')
        gp.set('set style histogram errorbars lw 2')
        gp.plot(
            'plot $df using 2:3:xtic(1) title "Full" w histograms palette frac 0.1, "" using 4:5 title "AutoScaling" w histograms palette frac 0.2')
        gp.write_to_file("gnu")
        gp.show()
