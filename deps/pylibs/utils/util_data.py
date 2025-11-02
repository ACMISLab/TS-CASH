import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def load_cv_data(file_path, n_split=5, normalization=False):
    """
    Split m_dataset into k-fold.
    The return data is standardized.

    Usage
    ----------
    for  x, train_y, train_missing, test_x, test_y, test_missing,mean,std in load_cv_data(self.data):
         ...

    Parameters
    ----------
    file_path:
        abs path of the file to read, e.g., datasets/cpu4.csv. The file must contain three
        columns named timestamp,value,label, where timestamp means the Unix Timestamp, e.g., 1469376300;
        value is the observation of that data point; label is a indicator whether a data point is anomaly;
        if label is equal to 1, it is a anomaly; If you hove no label on your data, you can set
        label to 0. for all data point.

    n_split
        How many folds you want to split.

    normalization
        The value will be standardized ((x-x_mean)/std) if normalization=False.
        Otherwise,the value will be normalized ((x-min)/(max-min))。
    Returns
    -------
    list of  1-D array filled missing data
        x, train_y, train_missing, test_x, test_y, test_missing,mean,std
    """

    X, Y, missing, mean, std = load_data(file_path, normalization=normalization)
    kf = KFold(n_splits=n_split)
    for train_index, test_index in kf.split(X):
        yield X[train_index], Y[train_index], missing[train_index], X[test_index], Y[test_index], missing[
            test_index], mean, std


def load_data(file, normalization=False):
    """
    Load data from xxx.csv, which must contain three columns ['timestamp', 'value', 'label']
    A function wrapper the donut function
    Parameters
    ----------
    file

    Returns
    -------
    train_values,labels,missing,mean,std

    """
    data = pd.read_csv(file)
    timestamps, values, labels = data['timestamp'], data['value'], data['label']
    timestamps, missing, (values, labels) = complete_timestamp_donut(timestamps, (values, labels))
    excludes = np.logical_or(labels, missing)
    if normalization is False:
        train_values, mean, std = standardize_kpi_donut(values, excludes=excludes)
    else:
        train_values, mean, std = normalize_kpi(values, excludes=excludes)
    return train_values, labels, missing, mean, std


def complete_timestamp_donut(timestamp, arrays=None):
    """
    Complete `timestamp` such that the time interval is homogeneous.

    Zeros will be inserted into each array in `arrays`, at missing points.
    Also, an indicator array will be returned to indicate whether each
    point is missing or not.

    Args:
        timestamp (np.ndarray): 1-D int64 array, the timestamp values.
            It can be unsorted.
        arrays (Iterable[np.ndarray]): The 1-D arrays to be filled with zeros
            according to `timestamp`.

    Returns:
        np.ndarray: A 1-D int64 array, the completed timestamp.
        np.ndarray: A 1-D int32 array, indicating whether each point is missing.
        list[np.ndarray]: The arrays, missing points filled with zeros.
            (optional, return only if `arrays` is specified)
    """
    timestamp = np.asarray(timestamp, np.int64)
    if len(timestamp.shape) != 1:
        raise ValueError('`timestamp` must be a 1-D array')

    has_arrays = arrays is not None
    arrays = [np.asarray(array) for array in (arrays or ())]
    for i, array in enumerate(arrays):
        if array.shape != timestamp.shape:
            raise ValueError('The shape of ``arrays[{}]`` does not agree with '
                             'the shape of `timestamp` ({} vs {})'.
                             format(i, array.shape, timestamp.shape))

    # sort the timestamp, and check the intervals
    src_index = np.argsort(timestamp)
    timestamp_sorted = timestamp[src_index]
    intervals = np.unique(np.diff(timestamp_sorted))
    interval = np.min(intervals)
    if interval == 0:
        raise ValueError('Duplicated values in `timestamp`')
    for itv in intervals:
        if itv % interval != 0:
            raise ValueError('Not all intervals in `timestamp` are multiples '
                             'of the minimum interval')

    # prepare for the return arrays
    length = (timestamp_sorted[-1] - timestamp_sorted[0]) // interval + 1
    ret_timestamp = np.arange(timestamp_sorted[0],
                              timestamp_sorted[-1] + interval,
                              interval,
                              dtype=np.int64)
    ret_missing = np.ones([length], dtype=np.int32)
    ret_arrays = [np.zeros([length], dtype=array.dtype) for array in arrays]

    # copy values to the return arrays
    dst_index = np.asarray((timestamp_sorted - timestamp_sorted[0]) // interval,
                           dtype=np.int)
    ret_missing[dst_index] = 0
    for ret_array, array in zip(ret_arrays, arrays):
        ret_array[dst_index] = array[src_index]

    if has_arrays:
        return ret_timestamp, ret_missing, ret_arrays
    else:
        return ret_timestamp, ret_missing


def standardize_kpi_donut(values, mean=None, std=None, excludes=None):
    """
    Standardize a
    Args:
        values (np.ndarray): 1-D `float32` array, the KPI observations.
        mean (float): If not :obj:`None`, will use this `mean` to standardize
            `values`. If :obj:`None`, `mean` will be computed from `values`.
            Note `mean` and `std` must be both :obj:`None` or not :obj:`None`.
            (default :obj:`None`)
        std (float): If not :obj:`None`, will use this `std` to standardize
            `values`. If :obj:`None`, `std` will be computed from `values`.
            Note `mean` and `std` must be both :obj:`None` or not :obj:`None`.
            (default :obj:`None`)
        excludes (np.ndarray): Optional, 1-D `int32` or `bool` array, the
            indicators of whether each point should be excluded for computing
            `mean` and `std`. Ignored if `mean` and `std` are not :obj:`None`.
            (default :obj:`None`)

    Returns:
        np.ndarray: The standardized `values`.
        float: The computed `mean` or the given `mean`.
        float: The computed `std` or the given `std`.
    """
    values = np.asarray(values, dtype=np.float32)
    if len(values.shape) != 1:
        raise ValueError('`values` must be a 1-D array')
    if (mean is None) != (std is None):
        raise ValueError('`mean` and `std` must be both None or not None')
    if excludes is not None:
        excludes = np.asarray(excludes, dtype=np.bool)
        if excludes.shape != values.shape:
            raise ValueError('The shape of `excludes` does not agree with '
                             'the shape of `values` ({} vs {})'.
                             format(excludes.shape, values.shape))

    if mean is None:
        if excludes is not None:
            val = values[np.logical_not(excludes)]
        else:
            val = values
        mean = val.mean()
        std = val.std()
    return (values - mean) / std, mean, std


def normalize_kpi(values, excludes=None):
    """
    Standardize a
    Args:
        values (np.ndarray): 1-D `float32` array, the KPI observations.
        excludes (np.ndarray): Optional, 1-D `int32` or `bool` array, the
            indicators of whether each point should be excluded for computing
            `mean` and `std`. Ignored if `mean` and `std` are not :obj:`None`.
            (default :obj:`None`)

    Returns:
        np.ndarray: The standardized `values`.
        float: The computed `mean` or the given `mean`.
        float: The computed `std` or the given `std`.
    """
    values = np.asarray(values, dtype=np.float32)
    if len(values.shape) != 1:
        raise ValueError('`values` must be a 1-D array')
    if excludes is not None:
        excludes = np.asarray(excludes, dtype=np.bool)
        if excludes.shape != values.shape:
            raise ValueError('The shape of `excludes` does not agree with '
                             'the shape of `values` ({} vs {})'.
                             format(excludes.shape, values.shape))

    if excludes is not None:
        val: np.ndarray = values[np.logical_not(excludes)]
    else:
        val = values
    max = val.max()
    min = val.min()

    return (values - min) / (max - min), val.mean(), val.std()


def load_data_by_donut_function(file, test_portion=0.2, no_label=False):
    """
    A wrapper function for loading m_dataset of Donut.

    Return:
    train_values,train_labels,train_missing,test_values,test_labels,test_missing,train_mean,train_std,test_mean, test_std

    Parameters
    ----------
    file :
    test_portion :

    Returns
    -------

    """
    # Read the raw data.
    df = pd.read_csv(file)
    timestamp, values, labels = df['timestamp'], df['value'], df['label']

    # If there is no label, simply use all zeros.
    if no_label:
        labels = np.zeros_like(values, dtype=np.int32)

    # Complete the timestamp, and obtain the missing point indicators.
    timestamp, missing, (values, labels) = \
        complete_timestamp_donut(timestamp, (values, labels))

    # Split the training and testing data.
    test_n = int(len(values) * test_portion)
    train_values, test_values = values[:-test_n], values[-test_n:]
    train_labels, test_labels = labels[:-test_n], labels[-test_n:]
    train_missing, test_missing = missing[:-test_n], missing[-test_n:]

    # Standardize the training and testing data.
    train_values, mean, std = standardize_kpi_donut(
        train_values, excludes=np.logical_or(train_labels, train_missing))
    test_values, _, _ = standardize_kpi_donut(test_values, mean=mean, std=std)

    return train_values, train_labels, \
        train_missing, test_values, test_labels, \
        test_missing, mean, std, mean, std


def load_data_by_donut_function_from_two_file(training_file, test_file):
    """
    A wrapper function for loading m_dataset of Donut.

    Return:
    train_values,train_labels,train_missing,\
           test_values,test_labels,test_missing,\
           train_mean, train_std, test_mean, test_std


    Parameters
    ----------
    training_file :
    test_file:


    Returns
    -------

    """
    # Read the raw data.
    train_df = pd.read_csv(training_file)
    train_timestamp, train_values, train_labels = train_df['timestamp'], train_df['value'], train_df['label']

    test_df = pd.read_csv(test_file)
    test_timestamp, test_values, test_labels = test_df['timestamp'], test_df['value'], test_df['label']

    # Complete the timestamp, and obtain the missing point indicators.
    train_timestamp, train_missing, (train_values, train_labels) = \
        complete_timestamp_donut(train_timestamp, (train_values, train_labels))

    test_timestamp, test_missing, (test_values, test_labels) = \
        complete_timestamp_donut(test_timestamp, (test_values, test_labels))

    # Standardize the training and testing data.
    train_values, train_mean, train_std = standardize_kpi_donut(
        train_values, excludes=np.logical_or(train_labels, train_missing))

    test_values, test_mean, test_std = standardize_kpi_donut(test_values,
                                                             excludes=np.logical_or(test_labels, test_missing))

    return train_values, train_labels, train_missing, \
        test_values, test_labels, test_missing, \
        train_mean, train_std, test_mean, test_std
