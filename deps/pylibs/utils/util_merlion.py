import numpy as np
import pandas as pd

from merlion.utils import TimeSeries, UnivariateTimeSeries


def convert_timeseries_to_list(time_series: TimeSeries):
    """
    Convert a TimeSeries to a list.

    Parameters
    ----------
    time_series :

    Returns
    -------

    """
    assert time_series.dim == 1, "Only support one UnivariateTimeSeries in TimeSeries"
    return time_series.univariates[time_series.names[0]].values


def convert_timeseries_to_ndarray(timeseries: TimeSeries):
    """
    Convert a TimeSeries to a ndarray.

    Parameters
    ----------
    timeseries :

    Returns
    -------

    """
    return np.asarray(convert_timeseries_to_list(timeseries))


def generate_timeseries_from_list(data: list,
                                  start_datetime='1/1/2022'):
    kpi = generate_univariate_timeseries_from_list(data, start_datetime)
    return TimeSeries(univariates=[kpi])


def generate_univariate_timeseries_from_list(data: list,
                                             start_datetime='1/1/2022'):
    kpi = UnivariateTimeSeries(
        time_stamps=pd.date_range(start=start_datetime, freq='1min', periods=len(data)),
        # timestamps in units of seconds
        values=data,
        name="kpi"  # optional: a name for this univariate
    )
    return kpi


def generate_random_univariate_timeseries(n_points=10, start_datetime='1/1/2022', freq="1min") -> UnivariateTimeSeries:
    """
    Get a UnivariateTimeSeries of merlion. E.g.,

    .. code-block::

                                  kpi
        time
        2022-01-01 00:00:00   1.0
        2022-01-04 10:40:00   2.0
        2022-01-07 21:20:00   3.0
        2022-01-11 08:00:00   4.0
        2022-01-14 18:40:00   5.0
        2022-01-18 05:20:00   6.0
        2022-01-21 16:00:00   7.0
        2022-01-25 02:40:00   8.0
        2022-01-28 13:20:00   9.0
        2022-02-01 00:00:00  10.0


    Parameters
    ----------
    freq : str
        '1min1`
    start_datetime : str
        The end datetime  to generate
    n_points : int
        The number of points to generate.

    Returns
    -------
    UnivariateTimeSeries


    """
    kpi = UnivariateTimeSeries(
        time_stamps=pd.date_range(start=start_datetime, periods=n_points, freq=freq),
        # timestamps in units of seconds
        values=np.random.random_sample((n_points,)),  # time series values
        name="kpi"  # optional: a name for this univariate
    )
    return kpi


def generate_random_timeseries(n_points=10,
                               start_datetime='1/1/2022', freq='1min') -> TimeSeries:
    """
    Get a TimeSeries of merlion.


    Parameters
    ----------
    lower : int
        The lower bound of the values to generate
    upper :int
        The upper bound of the values to generate
    end_datetime : str
        The start_or_restart datetime  to generate
    start_datetime : str
        The end datetime  to generate
    n_points : int
        The number of points to generate.

    Returns
    -------

    """
    kpi = generate_random_univariate_timeseries(n_points, start_datetime, freq=freq)
    return TimeSeries(univariates=[kpi])


class UtilMerlion:

    @staticmethod
    def uts_to_numpy(uts_data):
        try:
            ret = uts_data.univariates['anomaly'].values
        except:
            ret = uts_data.univariates['anom_score'].values
        return np.asarray(ret)
