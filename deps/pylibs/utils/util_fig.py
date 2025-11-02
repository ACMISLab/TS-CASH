import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt


def plot_x(x, color, figsize=(8, 8), n_rows=1, n_cols=1, index_plot=1):
    # 创建图
    fig: Figure = plt.figure(figsize=figsize)

    # 添加子图
    # add_subplot(n_rows, n_cols, exp_index)
    ax: Axes = fig.add_subplot(n_rows, n_cols, index_plot)
    ax.scatter(np.arange(len(x)), c=color)
    return fig, ax
