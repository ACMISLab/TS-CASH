import os.path
from enum import Enum
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from numpy.testing import assert_almost_equal
from sklearn.preprocessing import MinMaxScaler

from pylibs.common import ConstDataset
from pylibs.utils.util_log import get_logger
from pylibs.utils.utils import convert_str_to_float

from pylibs.utils.util_numpy import enable_numpy_reproduce


def get_test_time_series():
    """
    Generate timeseries for 1D

    Returns
    -------

    """
    enable_numpy_reproduce(1)
    _n_samples = 30
    x = np.random.standard_normal((_n_samples))
    x = np.concatenate([x, [0, 8, 10, 30]])
    x = np.expand_dims(x, axis=1)
    y = np.concatenate([np.zeros(_n_samples), [1, 1, 1, 1]])
    return x, y


class DatasetName(Enum):
    KDDCUP_99 = "KDDCUP_99"
    ODDS_SATIMAGE = "SATIMAGE"
    ODDS_SMTP = "SMTP"
    ODDS_THYROID = "THYROID"
    ODDS_SATELLITE = "SATELLITE"
    ODDS_FORESTCOVER = "FORESTCOVER"
    ODDS_SHUTTLE = "SHUTTLE"
    MUSK = "MUSK"
    DEBUG = "DEBUG"
    TEST = "TEST"
    AIOPS = ""


class DatasetType:
    IOpsCompetition = "IOpsCompetition"
    NAB = "NAB"


log = get_logger()


def _get_file_by_dataset_type(dataset, data_home, data_split=2, repo="odds"):
    """
    Return the file by m_dataset type and repo.
    the directory is named: data_home + repo + m_dataset+_type(train, test, all).e.g.
    - datasets/odds/letter/train.npz
    - datasets/odds/letter/all.npz
    - datasets/odds/letter/test.npz

    Parameters
    ----------
    dataset : str
        The m_dataset name

    data_split : int
        0: training set, 1: test set, 2: all m_dataset.
        default 2

    data_home : str
        The home directory for the m_dataset

    repo : str
        The data repo, default odds

    Returns
    -------

    """
    if data_home is None:
        raise ValueError(f"data_home expect a directory, but received {data_home}/")
    data_home = os.path.abspath(data_home)
    data_split = int(data_split)
    _type = 0
    if data_split == 0:
        _type = "train"
    elif data_split == 1:
        _type = "test"
    elif data_split == 2:
        _type = "all"
    else:
        raise TypeError(f"Unsupported data_split={data_split}")

    lower_dataset = str(dataset).lower()
    file = os.path.join(data_home, repo, lower_dataset, _type)
    if not file.endswith(".npz"):
        file = file + ".npz"
    if file is None:
        raise ValueError(f"Cant find m_dataset={dataset}")

    if not os.path.exists(file):
        raise FileNotFoundError(f"[{os.path.abspath(file)}] is not found")
    return file


def load_data(name,
              dataset_type=0,
              test_size=0.2,
              seed=1111,
              include_anomaly=False,
              data_home=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../datasets")):
    """
    Load training set and testing set.

    Return  x_train,y_train,x_valid,y_valid

    Examples
    --------
    # Load all test set
    x_train,y_train=load_csv("data",dataset_type=1,test_size=0)

    Parameters
    ----------
    include_anomaly : bool
        Whether to include anomaly point.
    data_home : str
        The data home for m_dataset
    dataset_type : int
        Data type.
        0: train.npz (the training m_dataset, 80%),
        1: test.npz (the test set,20),
        2: all.npz (all the m_dataset)
    seed : int
        The seed value.
    name : DatasetName
        the data set m_dataset, one of DatasetName. Default DatasetName.KDDCUP_99.value
    test_size : float
        the test size, a float between 0. and 1.

    Returns
    -------
    list
        x_train,y_train,x_valid,y_valid
        or
        x_train,y_train if test_size=True
    """
    if dataset_type == 1:
        test_size = 0.

    if not os.path.exists(data_home):
        raise NotADirectoryError(f"data_home [{data_home}] is not found. ")

    if isinstance(name, DatasetName):
        name = name.value

    if name not in [v.value for v in DatasetName]:
        raise ValueError(f"Dataset with user [{name}] is not existed.")
    file = _get_file_by_dataset_type(name, dataset_type)

    file = os.path.join(os.path.dirname(__file__), file)
    if not os.path.exists(file):
        raise FileNotFoundError(f"File {os.path.abspath(file)} is not existed")

    with np.load(file, allow_pickle=True) as data:
        X = data['X']
        y = np.reshape(data['y'], -1)

    _n_train = int((1 - test_size) * X.shape[0])

    if test_size == 0:
        return X, y
    else:
        x_train_, x_test, y_train_, y_test = X[:_n_train], X[_n_train:], y[:_n_train], y[_n_train:]

        if include_anomaly is True:
            UtilSys.is_debug_mode() and log.info(f"x_train (include anomaly): {x_train_.shape},x_test:{x_test.shape}")
            return x_train_, y_train_, x_test, y_test
        else:
            x_train, y_train = x_train_[y_train_ == 0], y_train_[y_train_ == 0]
            UtilSys.is_debug_mode() and log.info(f"x_train (include anomaly): {x_train_.shape},x_test:{x_test.shape}")
            UtilSys.is_debug_mode() and log.info(
                f"x_train (not include anomaly): {x_train.shape},x_test:{x_test.shape}")
            return x_train, y_train, x_test, y_test


def load_odds_data_two_splits_with_sampling(
        data,
        test_size=None,
        include_train_anomaly=False,
        data_home=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../datasets"),
        train_data_sample_method=ConstDataset.SAMPLING_RANDOM.value,
        train_data_sample_rate=1
):
    x_train, y_train, x_test, y_test = load_odds_data_two_splits(dataset_name=data,
                                                                 test_size=test_size,
                                                                 include_train_anomaly=include_train_anomaly,
                                                                 data_home=data_home
                                                                 )

    train_data_sample_rate = convert_str_to_float(train_data_sample_rate)

    assert isinstance(train_data_sample_method, str), "Data sample method must be str"
    n_sample_train = int(train_data_sample_rate * x_train.shape[0])
    if ConstDataset.SAMPLING_RANDOM.value == train_data_sample_method:
        # 随机抽取
        random_index = np.sort(np.random.choice(x_train.shape[0], size=n_sample_train, replace=False))
        x_train_sample = x_train[random_index]
        y_train_sample = y_train[random_index]
        UtilSys.is_debug_mode() and log.info(
            f"Sampling result for training set: x_train_sample: {x_train_sample.shape},y_train_sample={y_train_sample.shape}")
        UtilSys.is_debug_mode() and log.info(
            f"Sampling result for training set: x_valid: {x_test.shape},y_valid={y_test.shape}")
        return x_train_sample, y_train_sample, x_test, y_test

    else:
        raise TypeError(f"Unsupported sample method {train_data_sample_method}")


def load_aiops_data_two_splits_with_sampling(
        data,
        test_size=None,
        include_train_anomaly=False,
        data_home=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../datasets"),
        train_data_sample_method=ConstDataset.SAMPLING_RANDOM.value,
        train_data_sample_rate=1
):
    x_train, y_train, x_test, y_test = load_odds_data_two_splits(dataset_name=data,
                                                                 test_size=test_size,
                                                                 include_train_anomaly=include_train_anomaly,
                                                                 data_home=data_home
                                                                 )

    train_data_sample_rate = convert_str_to_float(train_data_sample_rate)

    assert isinstance(train_data_sample_method, str), "Data sample method must be str"
    n_sample_train = int(train_data_sample_rate * x_train.shape[0])
    if ConstDataset.SAMPLING_RANDOM.value == train_data_sample_method:
        # 随机抽取
        random_index = np.sort(np.random.choice(x_train.shape[0], size=n_sample_train, replace=False))
        x_train_sample = x_train[random_index]
        y_train_sample = y_train[random_index]
        UtilSys.is_debug_mode() and log.info(
            f"Sampling result for training set: x_train_sample: {x_train_sample.shape},y_train_sample={y_train_sample.shape}")
        UtilSys.is_debug_mode() and log.info(
            f"Sampling result for training set: x_valid: {x_test.shape},y_valid={y_test.shape}")
        return x_train_sample, y_train_sample, x_test, y_test

    else:
        raise TypeError(f"Unsupported sample method {train_data_sample_method}")


def load_odds_data_two_splits(dataset_name,
                              test_size=None,
                              include_train_anomaly=False,
                              data_home=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../datasets")):
    """
    Load training set and testing set.

    Return  x_train,y_train,x_test,y_test

    Examples
    --------
    # Load all test set
    x_train,y_train=load_csv("dataset_name",dataset_type=1,test_size=0)

    Parameters
    ----------
    include_train_anomaly : bool
        Whether to include anomaly point.
    data_home : str
        The data home for m_dataset
    dataset_name : str or DatasetName
        `DatasetName` means to load data by m_dataset name in DatasetName. `str` means load data from *.npz
    test_size : float,int,str, None
        the test size, a float between 0. and 1. or can be converted to [0,1].
        None means use the default value 0.4.
        Available values: x in [0,1], None (means 0.4), `1/2`

    Returns
    -------
    list
        x_train,y_train,x_valid,y_valid
        or
        x_train,y_train if test_size=True
    """

    test_size = convert_str_to_float(test_size)
    if isinstance(dataset_name, DatasetName):
        dataset_name = dataset_name.value

    if os.path.splitext(str(dataset_name))[-1] != '.npz':
        # load the m_dataset by m_dataset name from the database
        X, y = load_outlier_detection_dataset_from_db(dataset_name)
    else:
        # Load the m_dataset from the file
        UtilSys.is_debug_mode() and log.info(f"load the m_dataset from file {dataset_name}")
        file = dataset_name
        with np.load(file, allow_pickle=True) as data:
            X = data['X']
            y = np.reshape(data['y'], -1)

    _n_train = int((1 - test_size) * X.shape[0])
    x_train_all, x_test_all, y_train_all, y_test_all = X[:_n_train], X[_n_train:], y[:_n_train], y[_n_train:]
    assert x_train_all.shape[0] + x_test_all.shape[0] == X.shape[0]

    if include_train_anomaly is True:
        UtilSys.is_debug_mode() and log.info(f"x_train: {x_train_all.shape},x_test:{x_test_all.shape}")
        return x_train_all.astype(np.float16), y_train_all.astype(np.float16), x_test_all.astype(
            np.float16), y_test_all.astype(np.float16)
    else:
        x_train_normal, y_train_normal = x_train_all[y_train_all == 0], y_train_all[y_train_all == 0]
        UtilSys.is_debug_mode() and log.info(
            f"x_train(normal/all): {x_train_normal.shape[0]}/{x_train_all.shape[0]}, x_test: {x_test_all.shape}")
        return x_train_normal.astype(np.float16), y_train_normal.astype(np.float16), x_test_all.astype(
            np.float16), y_test_all.astype(np.float16)


def load_aiops_data_two_splits(dataset_name,
                               test_size=None,
                               include_train_anomaly=False,
                               data_home=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../datasets")):
    """
    Load training set and testing set.

    Return  x_train,y_train,x_test,y_test

    Examples
    --------
    # Load all test set
    x_train,y_train=load_csv("dataset_name",dataset_type=1,test_size=0)

    Parameters
    ----------
    include_train_anomaly : bool
        Whether to include anomaly point.
    data_home : str
        The data home for m_dataset
    dataset_name : str or DatasetName
        `DatasetName` means to load data by m_dataset name in DatasetName. `str` means load data from *.npz
    test_size : float,int,str, None
        the test size, a float between 0. and 1. or can be converted to [0,1].
        None means use the default value 0.4.
        Available values: x in [0,1], None (means 0.4), `1/2`

    Returns
    -------
    list
        x_train,y_train,x_valid,y_valid
        or
        x_train,y_train if test_size=True
    """

    test_size = convert_str_to_float(test_size)
    if isinstance(dataset_name, DatasetName):
        dataset_name = dataset_name.value

    if os.path.splitext(str(dataset_name))[-1] != '.npz':
        # load the m_dataset by m_dataset name from the database
        X, y = load_dataset_aiops_2018(dataset_name)
    else:
        # Load the m_dataset from the file
        UtilSys.is_debug_mode() and log.info(f"load the m_dataset from file {dataset_name}")
        file = dataset_name
        with np.load(file, allow_pickle=True) as data:
            X = data['X']
            y = np.reshape(data['y'], -1)

    _n_train = int((1 - test_size) * X.shape[0])
    x_train_all, x_test_all, y_train_all, y_test_all = X[:_n_train], X[_n_train:], y[:_n_train], y[_n_train:]
    assert x_train_all.shape[0] + x_test_all.shape[0] == X.shape[0]

    if include_train_anomaly is True:
        UtilSys.is_debug_mode() and log.info(f"x_train: {x_train_all.shape},x_test:{x_test_all.shape}")
        return x_train_all.astype(np.float16), y_train_all.astype(np.float16), x_test_all.astype(
            np.float16), y_test_all.astype(np.float16)
    else:
        x_train_normal, y_train_normal = x_train_all[y_train_all == 0], y_train_all[y_train_all == 0]
        UtilSys.is_debug_mode() and log.info(
            f"x_train(normal/all): {x_train_normal.shape[0]}/{x_train_all.shape[0]}, x_test: {x_test_all.shape}")
        return x_train_normal.astype(np.float16), y_train_normal.astype(np.float16), x_test_all.astype(
            np.float16), y_test_all.astype(np.float16)


def load_data_sampling_train(name, data_sample_method, data_sample_rate, dataset_type=0, test_size=0.2, seed=0):
    """
    Sampling the data from training set. The test set is not changed.

    Return x_train_sample, y_train_sample, x_valid, y_valid or raise Error

    Parameters
    ----------
    seed :
    test_size :
    dataset_type :
    name : str or DatasetName
         Data set m_dataset is in DatasetName
    data_sample_method : str
        Sampling method, which is in ConstDataset, for training set
    data_sample_rate : float
        Sampling size

    Returns
    -------
    x_train_sample, y_train_sample, x_valid, y_valid or None

    """
    if isinstance(name, DatasetName):
        name = name.value

    x_train, y_train, x_test, y_test = load_data(name, dataset_type=dataset_type, test_size=test_size, seed=seed)
    if data_sample_method not in [v.value for v in ConstDataset]:
        raise ValueError(f"data_sample_method [{data_sample_method}] is not implemented")

    assert isinstance(data_sample_rate, float)

    n_train = x_train.shape[0]
    n_sample_train = int(data_sample_rate * n_train)
    if ConstDataset.SAMPLING_RANDOM.value == data_sample_method:
        # 随机抽取
        random_index = np.random.randint(low=0, high=n_sample_train, size=n_sample_train)
        x_train_sample = x_train[random_index]
        y_train_sample = y_train[random_index]
        UtilSys.is_debug_mode() and log.info(
            f"Sampling result for training set: x_train_sample: {x_train_sample.shape},y_train_sample={y_train_sample.shape}")
        UtilSys.is_debug_mode() and log.info(
            f"Sampling result for training set: x_valid: {x_test.shape},y_valid={y_test.shape}")
        return x_train_sample, y_train_sample, x_test, y_test

    else:
        raise TypeError("Unsupported sample method")


def load_data_sampling_train_two_split(dataset_name,
                                       data_sample_method,
                                       data_sample_rate,
                                       include_train_anomaly=True,
                                       test_size=0.2,
                                       data_home=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                              "../../datasets")):
    """
    Sampling the data from training set. The test set is not changed.

    Return x_train_sample, y_train_sample, x_valid, y_valid or raise Error

    Parameters
    ----------
    data_home :
    include_train_anomaly :
    test_size :
    dataset_name : str or DatasetName
         Data set m_dataset is in DatasetName
    data_sample_method : str
        Sampling method, which is in ConstDataset, for training set
    data_sample_rate : float
        Sampling size

    Returns
    -------
    x_train_sample, y_train_sample, x_valid, y_valid or None

    """
    if isinstance(dataset_name, DatasetName):
        dataset_name = dataset_name.value

    x_train, y_train, x_test, y_test = load_odds_data_two_splits(
        dataset_name,
        include_train_anomaly=include_train_anomaly,
        test_size=test_size,
        data_home=data_home
    )

    if data_sample_method not in [v.value for v in ConstDataset]:
        raise ValueError(f"data_sample_method [{data_sample_method}] is not implemented")

    if isinstance(data_sample_rate, int) or isinstance(data_sample_rate, str):
        data_sample_rate = float(data_sample_rate)

    assert isinstance(data_sample_rate, float)

    n_train = x_train.shape[0]
    n_sample_train = int(data_sample_rate * n_train)
    if ConstDataset.SAMPLING_RANDOM.value == data_sample_method:
        # 随机抽取
        random_index = np.random.randint(low=0, high=n_sample_train, size=n_sample_train)
        x_train_sample = x_train[random_index]
        y_train_sample = y_train[random_index]
        UtilSys.is_debug_mode() and log.info(
            f"Sampling result for training set: x_train_sample: {x_train_sample.shape},y_train_sample={y_train_sample.shape}")
        UtilSys.is_debug_mode() and log.info(
            f"Sampling result for training set: x_valid: {x_test.shape},y_valid={y_test.shape}")
        return x_train_sample, y_train_sample, x_test, y_test

    else:
        raise TypeError("Unsupported sample method")


def save_data_to_file_normalized(x, y, file, test_size=0.4):
    """
    Save x,y to three file: all.npz,train.npz,test.npz, the split rate is 8:2 of training set and test set.

    Parameters
    ----------
    test_size : float
        The size of the test set
    x : np.ndarray
        (n_samples, n_features)
    y : np.ndarray
        (n_samples, 1), label corresponds to x.
    file : str
        the file user

    Returns
    -------

    """
    # We normalize the data to [0,1] instead of standardizing the data since if
    # the data does not follow normal distribution then this will make problems,
    # see Ali, Peshawa Jamal Muhammad, et al. "Data normalization and standardization: a
    # technical report." Mach Learn Tech Rep 1.1 (2014): 1-6.
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    n_train = int(x.shape[0] * (1 - test_size))
    x_train, x_test, y_train, y_test = x[:n_train], x[n_train:], y[:n_train], y[n_train:]
    home = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(file)), "../",
                                        os.path.basename(file)[:os.path.basename(file).rfind(".")]))
    UtilSys.is_debug_mode() and log.info(f"Data save home: {home}")
    if test_size < 0.5:
        assert x_train.shape[0] > x_test.shape[0], "Error when split training m_dataset"
        assert y_train.shape[0] > y_test.shape[0], "Error when split test m_dataset"
    if not os.path.exists(home):
        os.makedirs(home)
    assert x_train.shape[0] + x_test.shape[0] == x.shape[0]
    np.savez_compressed(os.path.join(home, "all.npz"), X=x, y=y)
    np.savez_compressed(os.path.join(home, "train.npz"), X=x_train, y=y_train)
    np.savez_compressed(os.path.join(home, "test.npz"), X=x_test, y=y_test)
    return os.path.join(home, "all.npz"), os.path.join(home, "train.npz"), os.path.join(home, "test.npz")


def save_outlier_detection_dataset_to_db(x, y, dataset_name):
    """
    Save x,y to three file: all.npz,train.npz,test.npz, the split rate is 8:2 of training set and test set.

    Parameters
    ----------
    x : np.ndarray
        (n_samples, n_features)
    y : np.ndarray
        (n_samples, 1), label corresponds to x.
    file : str
        the file user

    Returns
    -------

    """
    # We normalize the data to [0,1] instead of standardizing the data since if
    # the data does not follow normal distribution then this will make problems,
    # see Ali, Peshawa Jamal Muhammad, et al. "Data normalization and standardization: a
    # technical report." Mach Learn Tech Rep 1.1 (2014): 1-6.
    # scaler = StandardScaler()

    assert dataset_name is not None, "Dataset name cannot be None"
    scaler = MinMaxScaler()
    normal_x = scaler.fit_transform(x)
    connection = create_pandas_sqlite_connection()
    if len(y.shape) == 1:
        y = np.expand_dims(y, axis=1)

    columns = [f"f_{i}" for i in range(normal_x.shape[1])]
    columns.append("label")
    target_pd = pd.DataFrame(np.concatenate([normal_x, y], axis=1), columns=columns)

    target_pd.to_sql(con=connection, name=dataset_name, if_exists="replace", index=False)


def load_outlier_detection_dataset_from_db(dataset_name, db_file=None):
    """
    Loads a m_dataset from sqlite database (ODDS).

    Parameters
    ----------
    dataset_name : str
        The m_dataset name
    db_file :
        The file to load the m_dataset from
    Returns
    -------

    """
    if db_file is None:
        db_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../datasets/m_dataset.db")

    sql = f"select * from   '{dataset_name}'"
    dataset = pd.read_sql(sql=sql, con=create_pandas_sqlite_connection())
    return dataset.iloc[:, 0:-1].values, dataset.iloc[:, -1].values


def save_dataset_aiops_2018_to_db(data: pd.DataFrame, dataset_name: str):
    """
    Save x,y to three file: all.npz,train.npz,test.npz, the split rate is 8:2 of training set and test set.

    Parameters
    ----------
    dataset_name : str
        The m_dataset name
    data : pd.DataFrame
        DataFrame with columns  ['timestamp', 'value', 'label']

    Returns
    -------

    """
    # We normalize the data to [0,1] instead of standardizing the data since if
    # the data does not follow normal distribution then this will make problems,
    # see Ali, Peshawa Jamal Muhammad, et al. "Data normalization and standardization: a
    # technical report." Mach Learn Tech Rep 1.1 (2014): 1-6.
    assert isinstance(data, pd.DataFrame), "input data must be a DataFrame"
    target_columns = np.intersect1d(data.columns, ['timestamp', 'value', 'label'])
    np.testing.assert_equal(np.sort(['timestamp', 'value', 'label']), np.sort(target_columns),
                            err_msg="data columns must contain ['timestamp', 'value', 'label']")

    assert dataset_name is not None, "Dataset name cannot be None"
    value = data.loc[:, "value"]
    normal_x = (value - value.min()) / (value.max() - value.min())
    source_timestamp = data['timestamp']
    timestamp, missing, (train_x, label) = complete_timestamp(source_timestamp,
                                                              (normal_x, data['label'].values))

    pd.DataFrame({
        "timestamp": timestamp,
        'value': train_x,
        'label': label,
        'missing': missing,
    }).to_sql(
        con=create_pandas_sqlite_connection(),
        name=dataset_name,
        if_exists="replace",
        index=False
    )


def load_dataset_aiops_2018(kpi_id, sliding_window, is_include_anomaly_windows=True):
    """
    load m_dataset of aiops 2018 by kpi_id and return the windows.

    The data has been normalized (the values are between 0 and 1)

    return x,train_label,test_x,test_label,train_missing, test_missing

    Parameters
    ----------
    sliding_window : int or None
        The window size of the m_dataset. If None,return the training set.

    is_include_anomaly_windows : bool
        Determined whether to include or exclude anomaly windows in training data.
        True: return all windows of the m_dataset.
        False: return normal windows (without anomaly window) of the m_dataset

    kpi_id :

    Returns
    -------

    """
    train_x, train_label, train_missing, test_x, test_label, test_missing = get_aiops_data_by_id(kpi_id)
    return get_sliding_window_data(
        train_x, train_label, train_missing, test_x, test_label, test_missing, sliding_window,
        is_include_anomaly_windows
    )


def get_sliding_window_data(train_x, train_label, train_missing,
                            test_x, test_label, test_missing,
                            sliding_window_size, is_include_anomaly_windows=True):
    if sliding_window_size is None:
        return train_x, train_label, test_x, test_label, train_missing, test_missing
    else:
        anomaly_index = np.logical_or(train_label, train_missing)
        sliding_windows = sliding_window_view(train_x, window_shape=sliding_window_size)
        if is_include_anomaly_windows is False:
            anomaly_windows = sliding_window_view(anomaly_index, window_shape=sliding_window_size)
            anomaly_windows_indicator = np.sum(anomaly_windows, axis=1)
            normal_windows_index = np.where(anomaly_windows_indicator == 0, True, False)
            return sliding_windows[normal_windows_index], train_label, \
                sliding_window_view(test_x, window_shape=sliding_window_size), \
                test_label, train_missing, test_missing
        else:

            return sliding_windows, train_label, \
                sliding_window_view(test_x, window_shape=sliding_window_size), test_label, \
                train_missing, test_missing


def get_aiops_data_by_id(kpi_id, ret_timestamp=False):
    """
    Get aiops data by kpi_id from database

    Return  x, train_label, train_missing, test_x, test_label, test_missing
    Parameters
    ----------
    ret_timestamp : bool
        Whether to add timestamp to the returned results.
    kpi_id :

    Returns
    -------
    list
        If ret_timestamp = false (default), returns
            x, train_label, train_missing, test_x, test_label, test_missing
        If ret_timestamp = true,
            returns  x, train_label, train_missing, train_timestamp, test_x, test_label, \
            test_missing, test_timestamp

    """
    train_prefix = "AIOPS18_KPI_TRAIN_"
    test_prefix = "AIOPS18_KPI_TEST_"
    con = create_pandas_sqlite_connection()
    train = pd.read_sql(sql=f"select timestamp,value,label,missing from `{train_prefix}{kpi_id}`", con=con)
    test = pd.read_sql(sql=f"select timestamp,value,label,missing from `{test_prefix}{kpi_id}`", con=con)

    train_x, train_label, train_missing, train_timestamp = train.loc[:, "value"].values, \
        train.loc[:, "label"].values, \
        train.loc[:, "missing"].values, \
        train.loc[:, "timestamp"].values
    test_x, test_label, test_missing, test_timestamp = test.loc[:, "value"].values, \
        test.loc[:, "label"].values, \
        test.loc[:, "missing"].values, \
        test.loc[:, "timestamp"].values
    if ret_timestamp is True:
        return train_x, train_label, train_missing, train_timestamp, test_x, test_label, test_missing, test_timestamp
    else:
        return train_x, train_label, train_missing, test_x, test_label, test_missing


def get_sliding_windows(data, window_size, label=None, missing=None, include_anomaly=True) -> np.ndarray:
    """
    Returns the sliding windows for the data.

    Anomaly are referenced the data with label=1 and missing = 1
    ----------
    Examples


    ----------
    train_data: list
        The data, 1-D array
    window_size: int
        The size of sliding window
    include_nan: bool
        Whether including the nan windows
    Returns
    -------
    np.ndarray
        2-D array, Sliding windows
    """

    if window_size > len(data):
        raise ValueError("window_size can't greater than the len of data")

    if label is not None:
        assert (len(data)) == len(label)
    if missing is not None:
        assert len(label) == len(missing)
    # We set label as 0 for the default value of all data, which means that all data points are normal data point.
    if label is None:
        label = np.zeros((len(data),))

    # Default value 0 for missing indicator, which means that there are not existed missing data in the imput
    if missing is None:
        missing = np.zeros((len(data),))

    data_window = sliding_window_view(data, window_size)

    if include_anomaly:
        return data_window
    else:

        label_window = sliding_window_view(label, window_size)
        missing_window = sliding_window_view(missing, window_size)
        label_true = np.sum(label_window, axis=-1)
        missing_true = np.sum(missing_window, axis=-1)

        anomaly_index = label_true + missing_true
        normal_data_index = np.where(anomaly_index == 0)
        return data_window[normal_data_index]


def complete_timestamp(timestamp, arrays=None):
    """
    Complete `timestamp` such that the time interval is homogeneous.


    Zeros will be inserted into each array in `arrays`, at missing points.
    Also, an indicator array will be returned to indicate whether each
    point is missing or not.

    Examples
    --------
    timestamps, missing, (values, labels) = complete_timestamp(timestamps, (values, labels))

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
    dst_index = np.asarray((timestamp_sorted - timestamp_sorted[0]) // interval)
    ret_missing[dst_index] = 0
    for ret_array, array in zip(ret_arrays, arrays):
        ret_array[dst_index] = array[src_index]

    if has_arrays:
        return ret_timestamp, ret_missing, ret_arrays
    else:
        return ret_timestamp, ret_missing


def standardize_kpi(values, mean=None, std=None, excludes=None):
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
