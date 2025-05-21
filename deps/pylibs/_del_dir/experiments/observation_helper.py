#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/9/12 22:03
# @Author  : gsunwu@163.com
# @File    : observation_lib.py
# @Description:
# JLUtil.clear_all_calche()

"""
目标:
统计训练数据量与模型性能的关系.

训练数据量为x轴.
模型性能为y轴. 模型性能选择 VUS_ROC, VUS_PR, F1, RF1, Precision, RPrecision

包含的实验
ml baseline:   V404_baseline_ml_VUS_ROC_0.001_random
ml baseline 0: V404_baseline_ml_0_VUS_ROC_0.001_random
ml fastuts:    V404_fastuts_ml_VUS_ROC_0.001_lhs

dl 把ml 换为dl即可

"""
import os.path
import threading

import pandas as pd

from pylibs.config import GCFV2, ServerName, ExpServerConf
from pylibs.exp_ana_common.ExcelOutKeys import EK, PaperFormat
from pylibs.exp_ana_common.uts_results_tools import merge_kfold_data
from pylibs.experiments.exp_helper import JobConfV1
from pylibs.utils.util_bash import BashUtil
from pylibs.utils.util_common import UC
from pylibs.utils.util_directory import make_dirs
from pylibs.utils.util_exp_result import ER
from pylibs.utils.util_gnuplot import Gnuplot


def plot_exp_perf(metric_file="./metrics.csv"):
    # @cache_
    def load_data(file):
        df = pd.read_csv(file)
        df = df[df[EK.VUS_ROC] != -1]
        _, _df = merge_kfold_data(df)

        return _df

    df = load_data(metric_file)
    df.to_excel(os.path.basename((metric_file)))
    # df[EK.DATA_SAMPLE_RATE].replace(-1, 99999, inplace=True)
    target_arr = []
    headers = None
    for (_model_name, _data_sample_rate), _data in df.groupby(by=[EK.MODEL_NAME, EK.DATA_SAMPLE_RATE]):

        # 将-1替换为真实数据量
        if _data_sample_rate == -1:
            # 五折交叉,测试集*5=训练集
            _data_sample_rate = _data[(EK.TEST_LEN, EK.MEAN)].mean() * 4
        if 1 > _data_sample_rate > 0:
            _data_sample_rate = str(int(_data_sample_rate * 100))
        else:
            _data_sample_rate = str(int(_data_sample_rate))
        _perf_vus_roc = _data[(EK.MEAN, EK.VUS_ROC)].mean()
        _perf_pr = _data[(EK.MEAN, EK.VUS_PR)].mean()
        _perf_rf = _data[(EK.RPRECISION, EK.MEAN)].mean()
        _time = _data[(EK.MEAN, EK.ELAPSED_TRAIN)].sum() / 60

        if headers is None:
            headers = [EK.MODEL_NAME, EK.DATA_SAMPLE_RATE, EK.VUS_ROC, EK.VUS_PR, EK.RF, EK.ORI_TRAIN_TIME_MEAN]
        target_arr.append([_model_name, _data_sample_rate, _perf_vus_roc, _perf_pr, _perf_rf, _time])

    plot_pdf = pd.DataFrame(target_arr, columns=headers)
    plot_pdf['sort_index'] = plot_pdf[EK.DATA_SAMPLE_RATE].astype('float')
    plot_pdf = plot_pdf.sort_values(by="sort_index", ascending=True)
    gp = Gnuplot(home=PaperFormat.HOME_LATEX_HOME_IET)
    gp.set_output_pdf(PaperFormat.get_label_name(), 12, 7)
    gp.set_multiplot(3, 3)

    # MODEL_SKLEARN = ["hbos", "pca", "lof", "iforest", 'random_forest', 'ocsvm']
    # MODEL_OBSERVATION_ML = ["hbos", "pca", "lof", "iforest", 'random_forest', 'ocsvm']
    _model_names = JobConfV1.MODEL_OBSERVATION_DL + JobConfV1.MODEL_OBSERVATION_ML
    # _model_names = ['lstm', 'coca', 'iforest', 'random_forest', 'ocsvm', 'hbos', 'lof', 'tadgan', 'knn']

    for _index, _model_name in enumerate(_model_names):
        # for _model_name, _data in plot_pdf.groupby(by=EK.MODEL_NAME):
        # gp = Gnuplot(home=PaperFormat.HOME_LATEX_HOME)
        # gp.set_output_pdf(PaperFormat.get_label_name() + f"_{_model_name}", 6, 3)
        _data = plot_pdf[plot_pdf[EK.MODEL_NAME] == _model_name]

        if _data.shape[0] == 0:
            continue
        # _data = _data.iloc[1:, :]
        _min_time = _data[EK.ORI_TRAIN_TIME_MEAN].min()
        _max_time = _data[EK.ORI_TRAIN_TIME_MEAN].max()

        gp.set(f'set title "({_index + 1})  {PaperFormat.maps(_model_name)}" noenhanced')
        if _index not in [1, 2, 4, 5, 7, 8]:
            gp.set('set ylabel "Model accuracy" ')
        else:
            gp.set('set ylabel "" ')

        if _index not in [0, 1, 3, 4, 6, 7]:
            gp.set('set y2label "Training time (min)"')
        else:
            gp.set('set y2label ""')

        # gp.set("set key outside bottom center horizontal")

        if _index == 0:
            gp.set("set key at screen 0.5,screen 0.98 center maxrows 1 ")
        else:
            gp.set("set key off ")

        gp.set('set xtics rotate by -45')
        gp.set('set y2tics format "%.1f"')
        # gp.set('set yr [0.5:]')
        if _index <= 5:
            gp.set('set xlabel ""')
        else:
            gp.set('set xlabel "The number of the training data s_t"')

        # gp.set('set y2tics ')
        gp.add_data(_data)
        gp.plot(
            'plot $df using 0:3:xticlabels(2) axis x1y1 with lp title "VUS ROC", '
            '$df using 0:4 axis x1y1 with lp title "VUS PR",'
            '$df using 0:5 axis x1y1 with lp title "Range Precision",'
            '$df using 0:6 axis x1y2 with line lc 20 title "Training time"'
        )

    gp.show()
    gp.write_to_file()
    # PDUtil.save_to_excel(df, "ob_servation", home=UC.get_entrance_directory())


from pylibs.exp_ana_common.ExcelOutKeys import EK, PaperFormat
from pylibs.exp_ana_common.uts_results_tools import merge_kfold_data
from pylibs.experiments.exp_helper import JobConfV1
from pylibs.utils.util_gnuplot import Gnuplot


def plot_overview_v2(filename):
    _data = pd.read_pickle(filename)
    df = _data[_data[EK.VUS_ROC] != -1]
    _, df = merge_kfold_data(df)
    # df[EK.DATA_SAMPLE_RATE].replace(-1, 99999, inplace=True)
    target_arr = []
    headers = None
    for (_model_name, _data_sample_rate), _data in df.groupby(by=[EK.MODEL_NAME, EK.DATA_SAMPLE_RATE]):
        # print("model name:", _model_name)
        # 将-1替换为真实数据量
        if _data_sample_rate == -1:
            # 五折交叉,测试集*5=训练集
            _data_sample_rate = _data[(EK.TEST_LEN, EK.MEAN)].mean() * 4

        _data_sample_rate = str(int(_data_sample_rate))
        _perf_vus_roc = _data[(EK.MEAN, EK.VUS_ROC)].mean()
        _perf_pr = _data[(EK.MEAN, EK.VUS_PR)].mean()
        _perf_rf = _data[(EK.RPRECISION, EK.MEAN)].mean()
        _time = _data[(EK.MEAN, EK.ELAPSED_TRAIN)].sum() / 60

        if headers is None:
            headers = [EK.MODEL_NAME, EK.DATA_SAMPLE_RATE, EK.VUS_ROC, EK.VUS_PR, EK.RF, EK.ORI_TRAIN_TIME_MEAN]
        target_arr.append([_model_name, _data_sample_rate, _perf_vus_roc, _perf_pr, _perf_rf, _time])

    plot_pdf = pd.DataFrame(target_arr, columns=headers)
    plot_pdf['sort_index'] = plot_pdf[EK.DATA_SAMPLE_RATE].astype('float')
    plot_pdf = plot_pdf.sort_values(by="sort_index", ascending=True)
    gp = Gnuplot(home=PaperFormat.HOME_LATEX_HOME_IET)
    gp.set_output_pdf(PaperFormat.get_label_name(), 12, 7)
    gp.set_multiplot(3, 3)

    # MODEL_SKLEARN = ["hbos", "pca", "lof", "iforest", 'random_forest', 'ocsvm']
    # MODEL_OBSERVATION_ML = ["hbos", "pca", "lof", "iforest", 'random_forest', 'ocsvm']
    _model_names = JobConfV1.MODEL_OBSERVATION_DL + JobConfV1.MODEL_OBSERVATION_ML
    # _model_names = ['lstm', 'coca', 'iforest', 'random_forest', 'ocsvm', 'hbos', 'lof', 'tadgan', 'knn']

    for _index, (_model_name, _data) in enumerate(plot_pdf.groupby(by=EK.MODEL_NAME)):
        # gp = Gnuplot(home=PaperFormat.HOME_LATEX_HOME)
        # gp.set_output_pdf(PaperFormat.get_label_name() + f"_{_model_name}", 6, 3)
        # _data = plot_pdf[plot_pdf[EK.MODEL_NAME] == _model_name]
        # _data = _data.iloc[1:, :]
        _min_time = _data[EK.ORI_TRAIN_TIME_MEAN].min()
        _max_time = _data[EK.ORI_TRAIN_TIME_MEAN].max()

        gp.set(f'set title "({_index + 1})  {PaperFormat.maps(_model_name)}" noenhanced')
        if _index not in [1, 2, 4, 5, 7, 8]:
            gp.set('set ylabel "Accuracy" ')
        else:
            gp.set('set ylabel "" ')

        if _index not in [0, 1, 3, 4, 6, 7]:
            gp.set('set y2label "Training time (min)"')
        else:
            gp.set('set y2label ""')

        # gp.set("set key outside bottom center horizontal")
        # 外放legend(key)
        if _index == 0:
            gp.set("set key at screen 0.5,screen 0.98 center maxrows 1 ")
        else:
            gp.set("set key off ")

        gp.set('set xtics rotate by -45')
        gp.set('set y2tics format "%.1f"')
        # gp.set('set yr [0.5:]')
        if _index <= 5:
            gp.set('set xlabel ""')
        else:
            gp.set('set xlabel "The number of the training data s_t"')

        # gp.set('set y2tics ')
        gp.add_data(_data)
        gp.plot(
            'plot $df using 0:3:xticlabels(2) axis x1y1 with lp title "VUS ROC", '
            # '$df using 0:4 axis x1y1 with lp title "VUS PR",'
            '$df using 0:5 axis x1y1 with lp title "Range Precision",'
            '$df using 0:6 axis x1y2 with line lc 20 title "Training time"'
        )

    gp.show()
    gp.write_to_file()
    # PDUtil.save_to_excel(df, "ob_servation", home=UC.get_entrance_directory())


def plot_overview_v3(filename):
    df = _load_data(filename)
    # df[EK.DATA_SAMPLE_RATE].replace(-1, 99999, inplace=True)
    target_arr = []
    headers = None
    for (_model_name, _data_sample_rate), _data in df.groupby(by=[EK.MODEL_NAME, EK.DATA_SAMPLE_RATE]):
        # print("model name:", _model_name)
        # 将-1替换为真实数据量
        if _data_sample_rate < 8:
            continue
        if _data_sample_rate == -1:
            # 五折交叉,测试集*5=训练集
            _data_sample_rate = _data[(EK.TEST_LEN, EK.MEAN)].mean() * 4

        _data_sample_rate = str(int(_data_sample_rate))
        _perf_vus_roc = _data[(EK.MEAN, EK.VUS_ROC)].mean()
        _perf_pr = _data[(EK.MEAN, EK.VUS_PR)].mean()
        _perf_rf = _data[(EK.MEAN, EK.RF)].mean()
        _time = _data[(EK.MEAN, EK.ELAPSED_TRAIN)].sum() / 60

        if headers is None:
            headers = [EK.MODEL_NAME, EK.DATA_SAMPLE_RATE, EK.VUS_ROC, EK.VUS_PR, EK.RF, EK.ORI_TRAIN_TIME_MEAN]
        target_arr.append([_model_name, _data_sample_rate, _perf_vus_roc, _perf_pr, _perf_rf, _time])

    plot_pdf = pd.DataFrame(target_arr, columns=headers)
    plot_pdf['sort_index'] = plot_pdf[EK.DATA_SAMPLE_RATE].astype('float')
    plot_pdf = plot_pdf.sort_values(by="sort_index", ascending=True)
    gp = Gnuplot(home=PaperFormat.HOME_LATEX_HOME_IET)
    gp.set_output_pdf(PaperFormat.get_label_name(), 12, 7)
    gp.set_multiplot(3, 3)

    # MODEL_SKLEARN = ["hbos", "pca", "lof", "iforest", 'random_forest', 'ocsvm']
    # MODEL_OBSERVATION_ML = ["hbos", "pca", "lof", "iforest", 'random_forest', 'ocsvm']
    _model_names = JobConfV1.MODEL_OBSERVATION_DL + JobConfV1.MODEL_OBSERVATION_ML
    # _model_names = ['lstm', 'coca', 'iforest', 'random_forest', 'ocsvm', 'hbos', 'lof', 'tadgan', 'knn']

    for _index, (_model_name, _data) in enumerate(plot_pdf.groupby(by=EK.MODEL_NAME)):
        # gp = Gnuplot(home=PaperFormat.HOME_LATEX_HOME)
        # gp.set_output_pdf(PaperFormat.get_label_name() + f"_{_model_name}", 6, 3)
        # _data = plot_pdf[plot_pdf[EK.MODEL_NAME] == _model_name]
        # _data = _data.iloc[1:, :]
        _min_time = _data[EK.ORI_TRAIN_TIME_MEAN].min()
        _max_time = _data[EK.ORI_TRAIN_TIME_MEAN].max()

        gp.set(f'set title "({_index + 1})  {PaperFormat.maps(_model_name)}" noenhanced')
        if _index not in [1, 2, 4, 5, 7, 8]:
            gp.set('set ylabel "Accuracy" ')
        else:
            gp.set('set ylabel "" ')

        if _index not in [0, 1, 3, 4, 6, 7]:
            gp.set('set y2label "Training time (min)"')
        else:
            gp.set('set y2label ""')

        # gp.set("set key outside bottom center horizontal")
        # 外放legend(key)
        if _index == 0:
            gp.set("set key at screen 0.5,screen 0.98 center maxrows 1 ")
        else:
            gp.set("set key off ")

        gp.set('set xtics rotate by -45')
        gp.set('set y2tics format "%.1f"')
        # gp.set('set yr [0.5:]')
        if _index <= 5:
            gp.set('set xlabel ""')
        else:
            gp.set('set xlabel "The number of the training data s_i"')

        # gp.set('set y2tics ')
        gp.add_data(_data)
        gp.plot(
            'plot $df using 0:3:xticlabels(2) axis x1y1 with lp lc 14 lt 14 title "VUS ROC", '
            '$df using 0:4 axis x1y1 with lp lc 15 lt 15  title "VUS PR",'
            '$df using 0:5 axis x1y1 with lp lc 16  lt 16 title "RF",'
            '$df using 0:6 axis x1y2 with line lc 17 lt 17 title "Training time"'
        )

    gp.show()
    gp.write_to_file()
    # PDUtil.save_to_excel(df, "ob_servation", home=UC.get_entrance_directory())


def plot_overview_v4(filename):
    df = _load_metrics_bz2(filename)
    # df[EK.DATA_SAMPLE_RATE].replace(-1, 99999, inplace=True)
    target_arr = []
    headers = None
    for (_model_name, _data_sample_rate), _data in df.groupby(by=[EK.MODEL_NAME, EK.DATA_SAMPLE_RATE]):
        # print("model name:", _model_name)
        # 将-1替换为真实数据量
        if _data_sample_rate == -1:
            # 更新测试集: 五折交叉,测试集*5=训练集
            _data_sample_rate = _data[(EK.TEST_LEN, EK.MEAN)].mean() * 4
        else:
            _data_sample_rate = str(_data_sample_rate)
            pass

        _perf_vus_roc = _data[(EK.MEAN, EK.VUS_ROC)].mean()
        _perf_pr = _data[(EK.MEAN, EK.VUS_PR)].mean()
        _perf_rf = _data[(EK.MEAN, EK.RF)].mean()
        _time = _data[(EK.MEAN, EK.ELAPSED_TRAIN)].sum() / 60

        if headers is None:
            headers = [EK.MODEL_NAME, EK.DATA_SAMPLE_RATE, EK.VUS_ROC, EK.VUS_PR, EK.RF, EK.ORI_TRAIN_TIME_MEAN]
        target_arr.append([_model_name, _data_sample_rate, _perf_vus_roc, _perf_pr, _perf_rf, _time])

    plot_pdf = pd.DataFrame(target_arr, columns=headers)
    plot_pdf['sort_index'] = plot_pdf[EK.DATA_SAMPLE_RATE].astype('float')
    plot_pdf = plot_pdf.sort_values(by="sort_index", ascending=True)
    gp = Gnuplot(home=PaperFormat.HOME_LATEX_HOME_IET)
    gp.set_output_pdf(PaperFormat.get_label_name(), 12, 7)
    gp.set_multiplot(3, 3)

    # MODEL_SKLEARN = ["hbos", "pca", "lof", "iforest", 'random_forest', 'ocsvm']
    # MODEL_OBSERVATION_ML = ["hbos", "pca", "lof", "iforest", 'random_forest', 'ocsvm']
    _model_names = JobConfV1.MODEL_OBSERVATION_DL + JobConfV1.MODEL_OBSERVATION_ML
    # _model_names = ['lstm', 'coca', 'iforest', 'random_forest', 'ocsvm', 'hbos', 'lof', 'tadgan', 'knn']

    for _index, (_model_name, _data) in enumerate(plot_pdf.groupby(by=EK.MODEL_NAME)):
        # gp = Gnuplot(home=PaperFormat.HOME_LATEX_HOME)
        # gp.set_output_pdf(PaperFormat.get_label_name() + f"_{_model_name}", 6, 3)
        # _data = plot_pdf[plot_pdf[EK.MODEL_NAME] == _model_name]
        # _data = _data.iloc[1:, :]
        _min_time = _data[EK.ORI_TRAIN_TIME_MEAN].min()
        _max_time = _data[EK.ORI_TRAIN_TIME_MEAN].max()

        gp.set(f'set title "({_index + 1})  {PaperFormat.maps(_model_name)}" noenhanced')
        if _index not in [1, 2, 4, 5, 7, 8]:
            gp.set('set ylabel "Accuracy" ')
        else:
            gp.set('set ylabel "" ')

        if _index not in [0, 1, 3, 4, 6, 7]:
            gp.set('set y2label "Training time (min)"')
        else:
            gp.set('set y2label ""')

        # gp.set("set key outside bottom center horizontal")
        # 外放legend(key)
        if _index == 0:
            gp.set("set key at screen 0.5,screen 0.98 center maxrows 1 ")
        else:
            gp.set("set key off ")

        gp.set('set xtics rotate by -45')
        gp.set('set y2tics format "%.1f"')
        # gp.set('set yr [0.5:]')
        if _index <= 5:
            gp.set('set xlabel ""')
        else:
            gp.set('set xlabel "Training samples (%)"')

        # gp.set('set y2tics ')
        gp.add_data(_data)
        gp.plot(
            'plot $df using 0:3:xticlabels(2) axis x1y1 with lp lc 14 lt 14 title "VUS ROC", '
            '$df using 0:4 axis x1y1 with lp lc 15 lt 15  title "VUS PR",'
            '$df using 0:5 axis x1y1 with lp lc 16  lt 16 title "RF",'
            '$df using 0:6 axis x1y2 with line lc 17 lt 17 title "Training time"'
        )

    gp.show()
    gp.write_to_file()
    # PDUtil.save_to_excel(df, "ob_servation", home=UC.get_entrance_directory())


def plot_overview_v5(filename):
    df = ER.load_merge_metrics_bz2(filename)
    # df[EK.DATA_SAMPLE_RATE].replace(-1, 99999, inplace=True)
    target_arr = []
    headers = None
    for (_model_name, _data_sample_rate), _data in df.groupby(by=[EK.MODEL_NAME, EK.DATA_SAMPLE_RATE]):
        # print("model name:", _model_name)
        # 将-1替换为真实数据量
        if _data_sample_rate == -1:
            # 更新测试集: 五折交叉,测试集*5=训练集
            _data_sample_rate = _data[(EK.TEST_LEN, EK.MEAN)].mean() * 4
        else:
            _data_sample_rate = str(int(_data_sample_rate))
            pass

        _perf_vus_roc = _data[(EK.MEAN, EK.VUS_ROC)].mean() * 100
        _perf_pr = _data[(EK.MEAN, EK.VUS_PR)].mean() * 100
        _perf_rf = _data[(EK.MEAN, EK.RF)].mean() * 100
        _time = _data[(EK.MEAN, EK.ELAPSED_TRAIN)].sum() / 60

        if headers is None:
            headers = [EK.MODEL_NAME, EK.DATA_SAMPLE_RATE, EK.VUS_ROC, EK.VUS_PR, EK.RF, EK.ORI_TRAIN_TIME_MEAN]
        target_arr.append([_model_name, _data_sample_rate, _perf_vus_roc, _perf_pr, _perf_rf, _time])

    plot_pdf = pd.DataFrame(target_arr, columns=headers)
    plot_pdf['sort_index'] = plot_pdf[EK.DATA_SAMPLE_RATE].astype('float')
    plot_pdf = plot_pdf.sort_values(by="sort_index", ascending=True)
    gp = Gnuplot(home=UC.get_entrance_directory())
    gp.set_output_pdf(PaperFormat.get_label_name(), 12, 7)
    gp.set_multiplot(3, 3)

    # MODEL_SKLEARN = ["hbos", "pca", "lof", "iforest", 'random_forest', 'ocsvm']
    # MODEL_OBSERVATION_ML = ["hbos", "pca", "lof", "iforest", 'random_forest', 'ocsvm']
    _model_names = JobConfV1.MODEL_OBSERVATION_DL + JobConfV1.MODEL_OBSERVATION_ML
    # _model_names = ['lstm', 'coca', 'iforest', 'random_forest', 'ocsvm', 'hbos', 'lof', 'tadgan', 'knn']

    # for _index, (_model_name, _data) in enumerate(plot_pdf.groupby(by=EK.MODEL_NAME)):

    for _index, (_model_name) in enumerate(_model_names):
        print(f"🚕 starting {_model_name}")
        # gp = Gnuplot(home=PaperFormat.HOME_LATEX_HOME)
        # gp.set_output_pdf(PaperFormat.get_label_name() + f"_{_model_name}", 6, 3)
        _data = plot_pdf[plot_pdf[EK.MODEL_NAME] == _model_name]

        # _data = _data.iloc[1:, :]

        _min_time = _data[EK.ORI_TRAIN_TIME_MEAN].min()
        _max_time = _data[EK.ORI_TRAIN_TIME_MEAN].max()

        gp.set(f'set title "({_index + 1})  {PaperFormat.maps(_model_name)}" noenhanced')
        if _index not in [1, 2, 4, 5, 7, 8]:
            gp.set('set ylabel "Accuracy" ')
        else:
            gp.set('set ylabel "" ')

        if _index not in [0, 1, 3, 4, 6, 7]:
            gp.set('set y2label "Training time (min)"')
        else:
            gp.set('set y2label ""')

        # gp.set("set key outside bottom center horizontal")
        # 外放legend(key)
        if _index == 0:
            gp.set("set key at screen 0.5,screen 0.98 center maxrows 1 ")
        else:
            gp.set("set key off ")

        gp.set('set xtics rotate by -45')
        gp.set('set y2tics format "%0.0f"')
        gp.set('set ytics format "%0.0f"')
        # gp.set('set xtics format "%d"')
        # gp.set('set yr [0.5:]')
        if _index <= 5:
            gp.set('set xlabel ""')
        else:
            gp.set('set xlabel "The number of training sliding windows"')

        # gp.set('set y2tics ')
        gp.add_data(_data)
        gp.plot(
            'plot $df using 0:3:xticlabels(2) axis x1y1 with lp lc 14 lt 14 title "VUS ROC", '
            '$df using 0:4 axis x1y1 with lp lc 15 lt 15  title "VUS PR",'
            '$df using 0:5 axis x1y1 with lp lc 16  lt 16 title "RF",'
            '$df using 0:6 axis x1y2 with line lc 17 lt 17 title "Training time"'
        )
    gp.show()
    gp.write_to_file()
    # PDUtil.save_to_excel(df, "ob_servation", home=UC.get_entrance_directory())


from pylibs.utils.util_joblib import cache_


@cache_
def _load_data(filename):
    _data = pd.read_pickle(filename)
    df = _data[_data[EK.VUS_ROC] != -1]
    _, df = merge_kfold_data(df)
    return df


def download_all_to_one_folder_lan(version_prefix="v40"):
    all = []

    for obj in [GCFV2.get_server_conf(ServerName.GPU_153_LAN),
                GCFV2.get_server_conf(ServerName.GPU_100_9_LAN),
                GCFV2.get_server_conf(ServerName.GPU_219_LAN)]:
        obj: ExpServerConf = obj
        local = f"/Users/sunwu/Downloads/download_metrics/"
        make_dirs(local)
        # filter = JobConfV1.get_exp_version()[:-1]
        # --debug=FILTER
        # --debug=FILTER
        cmd = f'sshpass -p "{obj.password}" rsync    -e "ssh -p {obj.port} -o PubkeyAuthentication=yes   -o stricthostkeychecking=no" -avi  -f"- *score.npz"  -f"+ {version_prefix}*/***"  ' \
              f' -f"- *"   {obj.username}@{obj.ip}:{obj.runtime}  {local}'
        t = threading.Thread(target=BashUtil.run_command_print_progress, args=(cmd,))
        t.start()
        all.append(t)

    for t in all:
        t.join()


def download_all_to_one_folder_wan(version_prefix="v40"):
    all = []

    for obj in [GCFV2.get_server_conf(ServerName.GPU_153_WAN),
                GCFV2.get_server_conf(ServerName.GPU_100_9_WAN)]:
        obj: ExpServerConf = obj
        local = f"/Users/sunwu/Downloads/download_metrics/"
        make_dirs(local)
        # filter = JobConfV1.get_exp_version()[:-1]
        # --debug=FILTER
        # --debug=FILTER
        cmd = f'sshpass -p "{obj.password}" rsync    -e "ssh -p {obj.port} -o PubkeyAuthentication=yes   -o stricthostkeychecking=no" -avi  -f"- *score.npz"  -f"+ {version_prefix}*/***"  ' \
              f' -f"- *"   {obj.username}@{obj.ip}:{obj.runtime}  {local}'
        t = threading.Thread(target=BashUtil.run_command_print_progress, args=(cmd,))
        t.start()
        all.append(t)

    for t in all:
        t.join()
