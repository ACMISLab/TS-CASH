from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt


def get_fig(fig_size=(4, 3)) -> Figure:
    fig = plt.figure(figsize=fig_size, dpi=300, tight_layout=True)
    # ax: Axes = fig.add_subplot(1, 1, 1)
    # ax.scatter(range(len(data)), data)
    return fig


if __name__ == '__main__':
    fig = get_fig()
    ax: Axes = fig.add_subplot(1, 1, 1)
    ax.plot([1, 2, 3])
    plt.show()
