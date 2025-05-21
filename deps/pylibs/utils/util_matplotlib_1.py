# This is a modified version of merlion removing plotly.
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Module for visualizing model predictions.
"""
from typing import Dict, Union
from copy import copy
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylibs.utils.util_log import get_logger

logger = get_logger()


def plot_time_series_for_ndarray(value: np.ndarray,
                                 ground_truth: np.ndarray,
                                 predict_score: np.ndarray,
                                 title: str = "My Figure",
                                 time_step=2):
    """
    The blue curves are the original data value. The pink areas represent the ground-truth anomalies including point
    and subsequence anomalies. The red curves are the anomaly scores predicted by COCA.

    Ref: [1] R. Wang et al., “Deep Contrastive One-Class Time Series Anomaly Detection.” arXiv, Oct. 08, 2022. Accessed:
    Dec. 09, 2022. [Online]. Available: http://arxiv.org/abs/2207.01472

    Parameters
    ----------
    title :
        The title of the figure
    value :
        The value of the original data
    ground_truth :
        A set, which values are one of 0 or 1
    predict_score :
        A float set predicted by a model
    time_step :
        The step for generating sliding window

    Returns
    -------

    """
    assert isinstance(value, np.ndarray), f"value must be a ndarray, received {type(value)}"
    assert isinstance(ground_truth, np.ndarray), f"ground_truth must be a ndarray, received {type(ground_truth)}"
    assert isinstance(predict_score, np.ndarray), f"predict_score must be a ndarray, received {type(predict_score)}"

    test_score_origin = predict_score
    fig = plt.figure(facecolor="w", figsize=(10, 6))

    # plot time-series value. axis x is a set {0,1,...,len(data)}
    ax = fig.add_subplot(111)
    t = np.arange(0, len(value), 1)
    ax.plot(t, value, linewidth=1)
    ax.set_ylabel('KPI Value', fontsize=10)

    # plot ground-truth anomaly in pink
    splits = np.where(ground_truth[1:] != ground_truth[:-1])[0] + 1
    splits = np.concatenate(([0], splits, [len(ground_truth) - 1]))

    for k in range(len(splits) - 1):
        if ground_truth[splits[k]]:  # If splits[k] is anomalous
            ax.axvspan(t[splits[k]], t[splits[k + 1]], color="#e07070", alpha=0.5)

    # plot predict anomaly score
    predict = np.tile(test_score_origin.reshape(-1, 1), time_step).flatten()
    # predict = np.tile(test_score_origin.reshape(-1, 1), time_step).flatten()
    t_pred = np.arange(0, len(predict), 1)
    ax2 = ax.twinx()
    ax2.set_ylabel('Anomaly Score', fontsize=10)
    ax2.plot(t_pred, predict, linewidth=1, color='r')
    plt.title(title)
    plt.show()


def plot_time_series_for_merlion(
        title,
        value: Union[UnivariateTimeSeries, TimeSeries],
        ground_truth: Union[UnivariateTimeSeries, TimeSeries],
        predict_score: Union[UnivariateTimeSeries, TimeSeries],
        is_show=True,
):
    """
    The blue curves are the original data value. The pink areas represent the ground-truth anomalies including point
    and subsequence anomalies. The red curves are the anomaly scores predicted by COCA.
    
    Ref: [1] R. Wang et al., “Deep Contrastive One-Class Time Series Anomaly Detection.” arXiv, Oct. 08, 2022. Accessed:
    Dec. 09, 2022. [Online]. Available: http://arxiv.org/abs/2207.01472

    Parameters
    ----------
    is_show : bool
        Whether to show the figure.
    title : str
        The title of the figure
    value : UnivariateTimeSeries
        The value of the time series
    ground_truth : UnivariateTimeSeries
        The label for the original data point
    predict_score : UnivariateTimeSeries
        The predicted score by a model, a set of float values.

    Returns
    -------
    fig,ax

    """
    if isinstance(value, TimeSeries):
        assert value.dim == 1
        value = value.univariates[value.names[0]]

    if isinstance(ground_truth, TimeSeries):
        assert ground_truth.dim == 1
        ground_truth = ground_truth.univariates[ground_truth.names[0]]
    if isinstance(predict_score, TimeSeries):
        assert predict_score.dim == 1
        predict_score = predict_score.univariates[predict_score.names[0]]

    assert isinstance(value, UnivariateTimeSeries), "Value must be a TimeSeries or UnivariateTimeSeries"
    assert isinstance(ground_truth, UnivariateTimeSeries), "ground_truth must be a TimeSeries or UnivariateTimeSeries"
    assert isinstance(predict_score, UnivariateTimeSeries), "predict_score must be a TimeSeries or UnivariateTimeSeries"
    tfg = UTSFigure(y=value, anom=predict_score)
    fig, ax = tfg.plot(title=title, ax=None)
    plot_anoms(ax=ax, anomaly_labels=ground_truth)
    if is_show:
        plt.show()
    return fig, ax


def plot_anoms(ax: plt.Axes, anomaly_labels: UnivariateTimeSeries, color: str = "#e07070", alpha: float = 0.5):
    """
    Plots anomalies as pink windows on the matplotlib ``Axes`` object ``ax``.
    """
    if anomaly_labels is None:
        return ax
    anomaly_labels = anomaly_labels.to_pd()
    t, y = anomaly_labels.exp_index, anomaly_labels.values
    splits = np.where(y[1:] != y[:-1])[0] + 1
    splits = np.concatenate(([0], splits, [len(y) - 1]))
    for k in range(len(splits) - 1):
        if y[splits[k]]:  # If splits[k] is anomalous
            ax.axvspan(t[splits[k]], t[splits[k + 1]], color=color, alpha=alpha)
    return ax


class UTSFigure:
    """
    Class for visualizing predictions of univariate anomaly detection & forecasting models.
    """

    _default_label_alias = dict(yhat="Forecast", anom="Anomaly Score")

    def __init__(
            self,
            y: UnivariateTimeSeries = None,
            anom: UnivariateTimeSeries = None,
            yhat: UnivariateTimeSeries = None,
            yhat_lb: UnivariateTimeSeries = None,
            yhat_ub: UnivariateTimeSeries = None,
            y_prev: UnivariateTimeSeries = None,
            yhat_prev: UnivariateTimeSeries = None,
            yhat_prev_lb: UnivariateTimeSeries = None,
            yhat_prev_ub: UnivariateTimeSeries = None,
            yhat_color: str = None,
    ):
        """
        :param y: the true value of the time series
        :param anom: anomaly scores returned by a model
        :param yhat: forecast returned by a model
        :param yhat_lb: lower bound on ``yhat`` (if model supports uncertainty estimation)
        :param yhat_ub: upper bound on ``yhat`` (if model supports uncertainty estimation)
        :param y_prev: portion of time series preceding ``y``
        :param yhat_prev: model's forecast of ``y_prev``
        :param yhat_prev_lb: lower bound on ``yhat_prev`` (if model supports uncertainty estimation)
        :param yhat_prev_ub: upper bound on ``yhat_prev`` (if model supports uncertainty estimation)
        :param yhat_color: the color in which to plot the forecast
        """
        assert not (anom is not None and y is None), "If `anom` is given, `y` must also be given"

        if yhat is None:
            assert yhat_lb is None and yhat_ub is None, "Can only give `yhat_lb` and `yhat_ub` if `yhat` is given"
        else:
            assert (yhat_lb is None and yhat_ub is None) or (
                    yhat_lb is not None and yhat_ub is not None
            ), "Must give both or neither of `yhat_lb` and `yhat_ub`"

        if yhat_prev is None:
            assert (
                    yhat_prev_lb is None and yhat_prev_ub is None
            ), "Can only give `yhat_prev_lb` and `yhat_prev_ub` if `yhat_prev` is given"
        else:
            assert (yhat_prev_lb is None and yhat_prev_ub is None) or (
                    yhat_prev_lb is not None and yhat_prev_ub is not None
            ), "Must give both or neither of `yhat_prev_lb` and `yhat_prev_ub`"

        self.y = y
        self.anom = anom
        self.yhat = yhat
        if yhat_lb is not None and yhat_ub is not None:
            self.yhat_iqr = TimeSeries({"lb": yhat_lb, "ub": yhat_ub}).align()
        else:
            self.yhat_iqr = None

        self.y_prev = y_prev
        self.yhat_prev = yhat_prev
        if yhat_prev_lb is not None and yhat_prev_ub is not None:
            self.yhat_prev_iqr = TimeSeries({"lb": yhat_prev_lb, "ub": yhat_prev_ub}).align()
        else:
            self.yhat_prev_iqr = None

        self.yhat_color = yhat_color if isinstance(yhat_color, str) else "#0072B2"

    @property
    def t0(self):
        """
        :return: First time being plotted.
        """
        ys = [self.anom, self.y, self.yhat, self.y_prev, self.yhat_prev]
        return min(y.index[0] for y in ys if y is not None and len(y) > 0)

    @property
    def tf(self):
        """
        :return: Final time being plotted.
        """
        ys = [self.anom, self.y, self.yhat, self.y_prev, self.yhat_prev]
        return max(y.index[-1] for y in ys if y is not None and len(y) > 0)

    @property
    def t_split(self):
        """
        :return: Time splitting train from test.
        """
        if self.y_prev is not None:
            return self.y_prev.index[-1]
        if self.yhat_prev is not None:
            return self.yhat_prev.index[-1]
        return None

    def get_y(self):
        """Get all y's (actual values)"""
        if self.y is not None and self.y_prev is not None:
            return self.y_prev.concat(self.y)
        elif self.y_prev is not None:
            return self.y_prev
        elif self.y is not None:
            return self.y
        else:
            return None

    def get_yhat(self):
        """Get all yhat's (predicted values)."""
        if self.yhat is not None and self.yhat_prev is not None:
            return self.yhat_prev.concat(self.yhat)
        elif self.yhat_prev is not None:
            return self.yhat_prev
        elif self.yhat is not None:
            return self.yhat
        else:
            return None

    def get_yhat_iqr(self):
        """Get IQR of predicted values."""
        if self.yhat_iqr is not None and self.yhat_prev_iqr is not None:
            return self.yhat_prev_iqr + self.yhat_iqr
        elif self.yhat_prev_iqr is not None:
            return self.yhat_prev_iqr
        elif self.yhat_iqr is not None:
            return self.yhat_iqr
        else:
            return None

    def plot(self, title=None, metric_name=None, figsize=(1000, 600), ax=None, label_alias: Dict[str, str] = None):
        """
        Plots the figure in matplotlib.

        :param title: title of the plot.
        :param metric_name: name of the metric (y axis)
        :param figsize: figure size in pixels
        :param ax: matplotlib axes to add the figure to.
        :param label_alias: dict which maps entities in the figure,
            specifically ``y_hat`` and ``anom`` to their label names.

        :return: (fig, ax): matplotlib figure & matplotlib axes
        """
        # determine full label alias
        label_alias = {} if label_alias is None else label_alias
        full_label_alias = copy(self._default_label_alias)
        full_label_alias.update(label_alias)

        # Get the figure
        figsize = (figsize[0] / 100, figsize[1] / 100)
        if ax is None:
            fig = plt.figure(facecolor="w", figsize=figsize)
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()
        ax.set_facecolor((0.9, 0.9, 0.9))

        # Get & plot the actual value (if applicable)
        lines = []
        y = self.get_y()
        if y is not None:
            metric_name = y.name if metric_name is None else metric_name
            ln = ax.plot(y.index, y.np_values, c="blue", alpha=0.8, lw=1, zorder=1, label=metric_name)
            lines.extend(ln)

        # Dotted line to cordon off previous times from current ones
        t_split = self.t_split
        if t_split is not None:
            ax.axvline(t_split, ls="--", lw=2, c="k")

        # Get & plot the prediction (if applicable)
        yhat = self.get_yhat()
        if yhat is not None:
            metric_name = yhat.name if metric_name is None else metric_name
            yhat_label = full_label_alias.get("yhat")
            ln = ax.plot(yhat.index, yhat.np_values, ls="-", c=self.yhat_color, zorder=0, label=yhat_label)
            lines.extend(ln)

        # Get & plot the uncertainty of the prediction (if applicable)
        iqr = self.get_yhat_iqr()
        if iqr is not None:
            lb, ub = iqr.univariates["lb"], iqr.univariates["ub"]
            ax.UnivariateTimeSerieseen(lb.exp_index, lb.values, ub.values, color=self.yhat_color, alpha=0.2, zorder=2)

        # Plot anomaly scores if desired
        if self.anom is not None and self.y is not None:
            ax2 = ax.twinx()
            anom_vals = self.anom.np_values
            anom_label = full_label_alias.get("anom")
            ln = ax2.plot(self.anom.index, anom_vals, color="r", label=anom_label)
            ax2.set_ylabel(anom_label)
            minval, maxval = min(anom_vals), max(anom_vals)
            delta = maxval - minval
            if delta > 0:
                ax2.set_ylim(minval - delta / 8, maxval + 2 * delta)
            else:
                ax2.set_ylim(minval - 1 / 30, maxval + 1)
            lines.extend(ln)

        # Format the axes before returning the figure
        locator = AutoDateLocator(interval_multiples=False)
        formatter = AutoDateFormatter(locator)
        ax.set_xlim(self.t0 - (self.tf - self.t0) / 20, self.tf + (self.tf - self.t0) / 20)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
        ax.set_xlabel("Time")
        ax.set_ylabel(metric_name)
        ax.set_title(title if title else metric_name)
        fig.legend(lines, [l.get_label() for l in lines])
        fig.tight_layout()
        return fig, ax


class MTSFigure:
    def __init__(
            self,
            y: TimeSeries = None,
            anom: TimeSeries = None,
            yhat: TimeSeries = None,
            yhat_lb: TimeSeries = None,
            yhat_ub: TimeSeries = None,
            y_prev: TimeSeries = None,
            yhat_prev: TimeSeries = None,
            yhat_prev_lb: TimeSeries = None,
            yhat_prev_ub: TimeSeries = None,
            yhat_color: str = None,
    ):
        assert y is not None, "`y` must be given"

        if yhat is None:
            assert yhat_lb is None and yhat_ub is None, "Can only give `yhat_lb` and `yhat_ub` if `yhat` is given"
        else:
            assert (yhat_lb is None and yhat_ub is None) or (
                    yhat_lb is not None and yhat_ub is not None
            ), "Must give both or neither of `yhat_lb` and `yhat_ub`"

        if yhat_prev is None:
            assert (
                    yhat_prev_lb is None and yhat_prev_ub is None
            ), "Can only give `yhat_prev_lb` and `yhat_prev_ub` if `yhat_prev` is given"
        else:
            assert (yhat_prev_lb is None and yhat_prev_ub is None) or (
                    yhat_prev_lb is not None and yhat_prev_ub is not None
            ), "Must give both or neither of `yhat_prev_lb` and `yhat_prev_ub`"

        self.y = y
        self.anom = anom
        self.yhat = yhat
        self.yhat_lb = yhat_lb
        self.yhat_ub = yhat_ub

        self.y_prev = y_prev
        self.yhat_prev = yhat_prev
        self.yhat_prev_lb = yhat_prev_lb
        self.yhat_prev_ub = yhat_prev_ub
        self.yhat_color = yhat_color if isinstance(yhat_color, str) else "#0072B2"

    @property
    def t0(self):
        ys = [self.anom, self.y, self.yhat, self.y_prev, self.yhat_prev]
        return min(y.t0 for y in ys if y is not None and len(y) > 0)

    @property
    def tf(self):
        ys = [self.anom, self.y, self.yhat, self.y_prev, self.yhat_prev]
        return max(y.tf for y in ys if y is not None and len(y) > 0)

    @property
    def t_split(self):
        if self.y_prev is not None:
            return pd.to_datetime(self.y_prev.tf, unit="s")
        if self.yhat_prev is not None:
            return pd.to_datetime(self.yhat_prev.tf, unit="s")
        return None

    @staticmethod
    def _combine_prev(x, x_prev):
        if x is not None and x_prev is not None:
            return x_prev + x
        elif x_prev is not None:
            return x_prev
        elif x is not None:
            return x
        else:
            return None

    def get_y(self):
        """Get all y's (actual values)"""
        return self._combine_prev(self.y, self.y_prev)

    def get_yhat(self):
        """Get all yhat's (predicted values)."""
        return self._combine_prev(self.yhat, self.yhat_prev)

    def get_yhat_iqr(self):
        """Get IQR of predicted values."""
        return self._combine_prev(self.yhat_lb, self.yhat_prev_lb), self._combine_prev(self.yhat_ub, self.yhat_prev_ub)

    @staticmethod
    def _get_layout(title, figsize):
        layout = dict(
            showlegend=True,
            xaxis=dict(
                title="Time",
                type="date",
                rangeselector=dict(
                    buttons=list(
                        [
                            dict(count=7, label="1w", step="day", stepmode="backward"),
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all"),
                        ]
                    )
                ),
                rangeslider=dict(visible=True),
            ),
        )
        layout["title"] = title if title else "Untitled"
        if figsize is not None:
            assert len(figsize) == 2, "figsize should be (width, height)."
            layout["width"] = figsize[0]
            layout["height"] = figsize[1]
        return layout
