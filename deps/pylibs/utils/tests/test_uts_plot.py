#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/5/1 10:02
# @Author  : gsunwu@163.com
# @File    : test_uts_plot.py
# @Description:
import os.path
import unittest
import logging

import pandas as pd
from matplotlib import pyplot as plt, spines
from matplotlib.ticker import MultipleLocator

from pylibs.config import Env
from pylibs.utils.util_bash import BashUtil
from pylibs.utils.util_matplotlib import AxHelper
from pylibs.utils.util_univariate_time_series_view import TSPlot

log = logging.getLogger(__name__)


class TestDatasetLoader(unittest.TestCase):

    def test_demo(self):
        fig,ax=TSPlot.plot_uts_timeseries(pd.DataFrame([
            [0.2, 1],
            [0.1, 0],
            [0.3, 1],
            [0.4, 0],
        ], columns=['value', 'label']),return_fig_and_ax=True)
        AxHelper.disable_grid(ax)
        AxHelper.enable_grid(ax)
        fig.savefig("ts.png")
        BashUtil.exe_cmd("open ts.png")
    def test_demo2(self):
        from pylibs.utils.util_matplotlib import AxHelper
        from exps.e_cash_libs import SELECTED_DATASETS_IDS
        from pylibs.uts_dataset.dataset_loader import KFDL
        from pylibs.utils.util_bash import BashUtil
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt
        from pylibs.utils.util_univariate_time_series_view import TSPlot
        pdf_file = os.path.join(Env.get_runtime_home(),'selected_uts_view.pdf')
        with PdfPages(pdf_file) as pdf:
            for (data_name, data_id), view_name in SELECTED_DATASETS_IDS.items():
                # print(data_name,data_id,view_name)
                dl = KFDL(dataset_name=data_name, data_id=data_id)
                original_data = dl._source_df()

                fig, ax = TSPlot.plot_uts_timeseries(original_data, return_fig_and_ax=True)
                AxHelper.disable_grid(ax)
                ax.set_title("View ID = "+view_name)
                # 保存当前图形到 PDF 中
                pdf.savefig(fig)

                # 关闭当前图形，释放内存
                plt.close(fig)
                # break
        BashUtil.exe_cmd(f"open {pdf_file}")