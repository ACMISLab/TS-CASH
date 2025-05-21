import datetime
import os
import sys
import time
import typing
from pathlib import Path
from typing import Union
import pandas
import pandas as pd
from numpy.testing import assert_equal

from pylibs.config import Env
from pylibs.exp_ana_common.ExcelOutKeys import PF, EK, PaperFormat
from pylibs.utils.util_common import UtilComm, UC
from pylibs.utils.util_datatime import DateTimeFormat
from pylibs.utils.util_directory import make_dirs
from pylibs.utils.util_exp_result import ER
from pylibs.utils.util_file import grf
from pylibs.utils.util_network import get_host_ip
from pylibs.utils.util_number import is_number
from pylibs.utils.util_system import UtilSys
import numpy as np

import logging

log = logging.getLogger(__name__)


def butify_print_options():
    pd.options.display.float_format = "{:.6f}".format


def show_all_pandas():
    """
    显示 pandas 所有的行和列
    Returns
    -------

    """
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    # 设置value的显示长度为100，默认为50
    pd.set_option('max_colwidth', 30)


def create_pandas_sqlite_connection(db_home=os.path.join(os.path.dirname(__file__), '../../', "datasets"),
                                    db_name='m_dataset.sqlite'):
    """

    Parameters
    ----------
    db_home :
    db_name :

    Returns
    -------

    """
    from sqlalchemy import create_engine
    db_file = os.path.join(db_home, db_name)
    engine = create_engine(f"sqlite:////{db_file}")
    UtilSys.is_debug_mode() and log.info(f"Created pandas sqlite connection url: {engine.url}")
    return engine


def get_pandas_sqlite_connection(file):
    from sqlalchemy import create_engine
    abs_file = os.path.abspath(file)
    engine = create_engine(f"sqlite:////{abs_file}")
    UtilSys.is_debug_mode() and log.info(f"Url: {engine.url}")
    return engine


def get_pandas_mysql_connection():
    """
    Get the connection of mysql.

    Returns
    -------

    """
    return get_mysql_connection()


def save_pandas_to_sqlite(df: pd.DataFrame,
                          table_name="nni_results",
                          ext=".sqlite",
                          file_name=""):
    entry_file_home = os.path.dirname(os.path.abspath(sys.argv[0]))
    home = os.path.join(entry_file_home, "export_db")
    from pylibs.utils.util_file import make_dirs
    make_dirs(home)
    UtilSys.is_debug_mode() and log.info(f"Database save home: {home}")
    dt = datetime.datetime.now()
    # https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior
    conn = create_pandas_sqlite_connection(db_home=home,
                                           db_name=f"{dt.strftime(DateTimeFormat.YEAR_DAY_INDEX)}__name_{file_name}__ip_{get_host_ip()}{ext}")
    effected_rows = df.to_sql(name=table_name, con=conn, if_exists="replace", index=False, dtype=None)
    UtilSys.is_debug_mode() and log.info(f"Effect {effected_rows} record(s) for  {conn.url} ")


def save_pandas_to_csv(df: pd.DataFrame,
                       file_name="",
                       ext=".csv"
                       ):
    entry_file_home = os.path.dirname(os.path.abspath(sys.argv[0]))
    home = os.path.join(entry_file_home, "export_db")
    from pylibs.utils.util_file import make_dirs
    make_dirs(home)

    dt = datetime.datetime.now()
    # https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior
    file_name = os.path.join(home,
                             f"{dt.strftime(DateTimeFormat.YEAR_DAY_INDEX)}__name_{file_name}__ip_{get_host_ip()}{ext}")
    file_name = os.path.abspath(file_name)
    UtilSys.is_debug_mode() and log.info(f"CSV path: {file_name}")
    df.to_csv(file_name)


def cache_df(df: pd.DataFrame, cache_id: str):
    """

    Cache pd.DataFrame to file.
    Used to speed up for read data from remote MySQL.

    Used with read_from_cache

    Parameters
    ----------
    df :
        The data to cache
    cache_id:str
        The cache id.

    Returns
    -------

    """
    file_name = grf(ext=".csv", name=cache_id)
    df.to_csv(file_name)


def read_from_cache(cache_id: str) -> Union[pd.DataFrame, None]:
    """
    Used with cache_df

    Parameters
    ----------
    cache_id :

    Returns
    -------

    """
    file_name = grf(ext=".csv", name=cache_id)
    if os.path.exists(file_name):
        UtilSys.is_debug_mode() and log.info(f"Loading data from cache {file_name}")
        return pd.read_csv(file_name)
    else:
        return None


def print_pretty_table(df: pandas.DataFrame, name=""):
    """
    Print a pretty table to console, for example:

    | str_sample_rate   |   sample_rate |   elapsed_time_seconds |   elapsed_time_nanosecond | kpi_id                               |   seed |
    |-------------------+---------------+------------------------+---------------------------+--------------------------------------+--------|
    | 1/1               |     1         |                   2355 |             2355358658283 | f0932edd-6400-3e63-9559-0a9860a1baa9 |      3 |
    | 1/4               |     0.25      |                    291 |              291599653085 | f0932edd-6400-3e63-9559-0a9860a1baa9 |      3 |
    | 1/8               |     0.125     |                    148 |              148642564198 | f0932edd-6400-3e63-9559-0a9860a1baa9 |      3 |
    | 1/32              |     0.03125   |                     37 |               37014433898 | f0932edd-6400-3e63-9559-0a9860a1baa9 |      3 |
    | 1/64              |     0.015625  |                     18 |               18284627423 | f0932edd-6400-3e63-9559-0a9860a1baa9 |      3 |
    | 1/128             |     0.0078125 |                     12 |               12195186832 | f0932edd-6400-3e63-9559-0a9860a1baa9 |      3 |
    +-------------------+---------------+------------------------+---------------------------+--------------------------------------+--------+

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------

    """
    from tabulate import tabulate
    # https://pypi.org/project/tabulate/
    jobs = tabulate(df,
                    tablefmt='psql',
                    showindex=True,
                    maxcolwidths=8,
                    floatfmt='.2f',
                    headers=df.columns
                    )
    print(jobs)


def log_pretty_table(df: pandas.DataFrame):
    print_pretty_table(df)


def calculate_mean(df):
    # 求所有数字列的均值
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns
    numeric_mean = df[numeric_cols].mean()
    return numeric_mean


def split_mean_and_std_list(values):
    """
    将 ["66.67(±34.66)"] 转为 means 和 std 两个数组

    means,stds=split_mean_and_std_list(["66.67(±34.66)"])
    Parameters
    ----------
    values :

    Returns
    -------

    """
    means = []
    stds = []
    for _val in values:
        assert _val.find("±") > -1, "input values is not fit"
        _val = _val.replace("(", "")
        _val = _val.replace(")", "")
        arr = _val.split("±")
        mean = float(arr[0])
        std = float(arr[1])
        means.append(mean)
        stds.append(std)
    return means, stds

    pass


def _process_ext(name, ext):
    if not name.endswith(ext):
        name = name + ext
    return name


class PDUtil:
    @staticmethod
    def get_number_columns(df):
        """
        返回df中所有的数字列

        Returns
        -------

        """
        return df.select_dtypes(include=['int', 'float']).columns

    @staticmethod
    def get_not_number_columns(df):
        """
        返回df中所有的数字列

        Returns
        -------

        """
        return df.select_dtypes(exclude=['int', 'float']).columns

    @staticmethod
    def save_to_excel(df: pd.DataFrame,
                      name: str = UC.get_filename_entry(),
                      index=True, index_label='exp_index',
                      append_entry=False):
        """
        保存为文件

        Returns
        -------

        """
        file_name = Path(name)
        if not file_name.parent.exists():
            file_name.parent.mkdir(parents=True)

        log.info(f"Saved to [{file_name.as_posix()}]")
        df.to_excel(file_name, index_label=index_label, index=index)
        return file_name.as_posix()

    @staticmethod
    def get_df_from_list(data: list):
        res = []
        for v in data:
            if v is not None:
                res.append(v)
            else:
                print("Skip none")
                continue
        return pd.DataFrame(res)

    @staticmethod
    def save_list_to_csv(data_list: list, name=f"test_csv_{time.time()}",
                         home: Path = Path(Env.get_runtime_home())) -> typing.Union[str, None]:
        if not home.parent.exists():
            home.mkdir(parents=True)
        res = []
        for l in data_list:
            if l is not None:
                res.append(l)
            else:
                print("Skip none")
                continue
        if len(res) == 0:
            return None
        else:
            return PDUtil.save_to_csv(pd.DataFrame(res),
                                      name=name,
                                      home=home,
                                      index=False)

    @staticmethod
    def save_list_to_excel(data_list: list, name=f"test_csv_{time.time()}",
                           home: Path = Path(Env.get_runtime_home())) -> typing.Union[str, None]:
        if not home.parent.exists():
            home.mkdir(parents=True)
        res = []
        for l in data_list:
            if l is not None:
                res.append(l)
            else:
                print("Skip none")
                continue
        if len(res) == 0:
            return None
        else:
            return PDUtil.save_to_excel(pd.DataFrame(res),
                                        name=name,
                                        index=False)

    @staticmethod
    def save_to_csv(data: typing.Union[pd.DataFrame, list],
                    name: str = "test.csv",
                    home=Env.get_runtime_home(),
                    index=False,
                    index_label='exp_index',
                    append_entry=False):
        """
        保存为文件

        Returns
        -------

        """
        if type(data) == list:
            df = pd.DataFrame(data)
        elif type(data) == pd.DataFrame:
            df = data
        else:
            raise RuntimeError(f"Unsupported data type {type(data)}")
        make_dirs(home)
        if name.endswith('.csv'):
            name = name[:-4]
        if append_entry:
            prefix = os.path.basename(sys.argv[0]).split(".")[0]
            file_name = os.path.join(home, f"{prefix}_{name}.csv")
        else:
            file_name = os.path.join(home, f"{name}.csv")

        print(f"CSV file saved to {os.path.abspath(file_name)}")
        df.to_csv(file_name, index_label=index_label, index=index)
        return os.path.abspath(file_name)

    @staticmethod
    def save_to_bz2(df: pd.DataFrame, name, home=UtilComm.get_runtime_directory(),
                    append_entry=False):
        """
        保存为文件

        Returns
        -------

        """
        make_dirs(home)
        file_name = os.path.join(home, _process_ext(name, ".bz2"))
        # data.to_pickle("aa.bz2", compression="bz2")
        log.info(f"✅ Saved {os.path.basename(file_name)} to {os.path.abspath(file_name)}")
        df.to_pickle(file_name, compression="bz2")
        return os.path.abspath(file_name)

    @staticmethod
    def print_pretty_table(data):
        if isinstance(data, dict):
            data = pd.DataFrame({
                "Name": data.keys(),
                "Value": data.values()
            })
        print_pretty_table(data)

    @classmethod
    def save_to_dat(cls, df, name, home="./"):
        """
        Save the xxx.dat

        Parameters
        ----------
        param :
        param1 :

        Returns
        -------

        """
        make_dirs(home)
        file_name = f"{home}/{name}.dat"
        UtilSys.is_debug_mode() and log.info(f"Saved metrics to {os.path.abspath(file_name)}")
        df.to_csv(file_name, sep='', index=False, header=False)

    @classmethod
    def save_to_latex(cls, df, file_path, caption, label, table_start=False, home=UC.get_runtime_directory(), maps={}):
        """

        Parameters
        ----------
        file_path :
        df :
        columns :
        caption :
        label :
        wide : 是否是table*

        Returns
        -------

        """
        if not file_path.endswith(".tex"):
            file_path = file_path + ".tex"
        latex = df.to_latex(index=False, label=label, caption=caption)

        for key, val in maps.items():
            latex = latex.replace(key, val)

        # 格式化模型名称
        for _model_name_ori in PaperFormat.model_name_maps.keys():
            latex = latex.replace(_model_name_ori, PaperFormat.model_name_maps.get(_model_name_ori))
        latex = latex.replace(EK.MODEL_NAME, "Model")

        latex = str(latex).replace("%", "\%")
        if table_start:
            latex = latex.replace("table}", "table*}")
        file_path = os.path.join(home, file_path)
        log.info(f"File saved to {os.path.abspath(file_path)}")
        with open(file_path, "w") as f:
            f.write(latex)

    @classmethod
    def append_avg_at_the_bottom(cls, _td: pd.DataFrame, fun=np.mean, col_name="Avg."):
        """
        Append a summary cols at the end of the pd.DataFrame (_td)

        Parameters
        ----------
        _td :
        fun : np.function
            np.mean, np.std or other function you want to calculate the characteristics.

        Returns
        -------

        """
        avg_arr = []
        for key in _td.columns:
            try:
                _values = _td[key].values
                if str(_values[0]).find("±") > -1:
                    # 如果是 66.67(±34.66) 这种数据, 就转为 mean 和 std 两个数组
                    _means, _stds = split_mean_and_std_list(_values)
                    avg_arr.append(ER.format_perf_mean_and_std(fun(_means), np.mean(_stds)))
                else:
                    if is_number(_values[0]):
                        avg_arr.append(fun(_values))
                    else:
                        raise ValueError
            except:
                avg_arr.append(f"\midrule \n{col_name}")

        return pd.concat([_td, pd.DataFrame([avg_arr], columns=_td.columns)])

    @classmethod
    def append_median_at_the_bottom(cls, _td: pd.DataFrame):
        """
        Append a summary cols at the end of the pd.DataFrame (_td)

        Parameters
        ----------
        _td :
        fun : np.function
            np.mean, np.std or other function you want to calculate the characteristics.

        Returns
        -------

        """
        return PDUtil.append_avg_at_the_bottom(_td, fun=np.median, col_name="Median")

    @classmethod
    def append_avg_on_the_right(cls, _td: pd.DataFrame, fun=np.mean, start_index=1, col_name="Avg."):
        """
        Append a summary cols on the right side of the DataFrame.

        start_index and end_index denote the range where you want to calculate the range.


        Parameters
        ----------
        _td : pd.DataFrame
            The original DataFrame
        fun : np.mean
            What function you want to use.
        start_index :

        Returns
        -------

        """
        avg_arr = []
        for key, _val in _td.iterrows():
            try:
                _values = _val[start_index:].values
                if str(_values[0]).find("±") > -1:
                    # 如果是 66.67(±34.66) 这种数据, 就转为 mean 和 std 两个数组
                    _means, _stds = split_mean_and_std_list(_values)

                    avg_arr.append(ER.format_perf_mean_and_std(fun(_means), np.mean(_stds)))
                else:
                    avg_arr.append(fun(_values))
            except:
                avg_arr.append(col_name)
        _td[col_name] = avg_arr
        return _td

    @DeprecationWarning
    @staticmethod
    def merge_mean_and_std(df: pd.DataFrame, time_scale=1 / 3600):
        _out = []
        for _key, _data in df.iterrows():
            _out.append({
                # EK.TARGET_PERF: _data[EK.TARGET_PERF],
                PF.MODEL: _data[EK.MODEL_NAME],
                PF.P_VALUE: _data[EK.P_VALUE],
                PF.BEST_SR: ER.format_perf_mean_and_std(_data[EK.FAST_BEST_SR_MEAN], _data[EK.FAST_BEST_SR_STD],
                                                        scale=100,
                                                        decimal=2),
                PF.ORI_PERF: ER.format_perf_mean_and_std(_data[EK.ORI_PERF_MEAN], _data[EK.ORI_PERF_STD],
                                                         scale=100,
                                                         decimal=2),
                PF.FAST_PERF: ER.format_perf_mean_and_std(_data[EK.FAST_PERF_MEAN], _data[EK.FAST_PERF_STD],
                                                          scale=100,
                                                          decimal=2),
                PF.ORI_TIME: ER.format_perf_mean_and_std(_data[EK.ORI_TRAIN_TIME_MEAN], _data[EK.ORI_TRAIN_TIME_STD],
                                                         scale=time_scale,
                                                         decimal=2),
                PF.FAST_TIME: ER.format_perf_mean_and_std(_data[EK.FAST_TRAIN_TIME_MEAN], _data[EK.FAST_TRAIN_TIME_STD],
                                                          scale=time_scale,
                                                          decimal=2),
                PF.DATA_PROC_TIME: ER.format_perf_mean_and_std(_data[EK.FAST_DATA_PROCESSING_TIME_MEAN],
                                                               _data[EK.FAST_DATA_PROCESSING_TIME_STD],
                                                               scale=time_scale,
                                                               decimal=2
                                                               ),
                PF.SPEED_UP: _data[EK.SPEED_UP],
                EK.SORT_KEY: _data[EK.SORT_KEY]
            })
        return pd.DataFrame(_out, columns=_out[0].keys())

    @classmethod
    def append_sum_on_the_right(cls, _data):
        return PDUtil.append_avg_on_the_right(_data, fun=np.sum, col_name="Sum")

    @classmethod
    def append_sum_at_the_bottom(cls, _data):
        return PDUtil.append_avg_at_the_bottom(_data, fun=np.sum, col_name="Sum")

    @classmethod
    def get_file_name(cls, param, home):
        pass

    @classmethod
    def remove_largest_and_smallest(cls, item: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        去掉 column_name 中值最大和最小的一行，共去掉两行
        Parameters
        ----------
        item :
        column_name :

        Returns
        -------

        """
        assert item.shape[0] >= 3, f"excepted input.shape[0] >=3, but received {item.shape[0]}"
        assert type(item) == pd.DataFrame
        max_rows = item.nlargest(1, column_name)
        min_rows = item.nsmallest(1, column_name)
        result = item[~item.index.isin(max_rows.index.union(min_rows.index))]
        return result


class UtilPD(PDUtil):
    pass


class DataSortUtil:
    @staticmethod
    def sort_by_category(df, rank_name, order_arr, pre_sort_key=None, suf_sort_keys=None):
        """
        Sort the df by rank_name, where rank_name is string, and the order is according to order_arr

        The final sort key is: pre_sort_key+['rank']+suf_sort_keys

        Example:

            order the pdf by "name", where the order is based on order_arr
            df = pd.DataFrame([
                ["decision_tree", 1],
                ["pca", 2],
                ["random_forest", 3],
            ], columns=['name', "val"])

            dsu = DataSortUtil.sort_by_category(df, "name",
                                                order_arr=["decision_tree", "random_forest", "pca"])
            print(dsu)
                         name  val
            0  decision_tree    1
            2  random_forest    3
            1            pca    2
        """
        if suf_sort_keys is None:
            suf_sort_keys = []
        if pre_sort_key is None:
            pre_sort_key = []

        df['rank'] = pd.Categorical(df[rank_name], categories=order_arr, ordered=True)
        df = df.sort_values(by=pre_sort_key + ['rank'] + suf_sort_keys)
        df = df.drop('rank', axis=1)
        return df


if __name__ == '__main__':
    arr = ["66.67(±34.66)"]
    means, stds = split_mean_and_std_list(arr)
    assert_equal(means, [66.67])
    assert_equal(stds, [34.66])
    df = pd.DataFrame([
        ["decision_tree", 1],
        ["pca", 2],
        ["random_forest", 3],
    ], columns=['name', "val"])

    dsu = DataSortUtil.sort_by_category(df, "name",
                                        order_arr=["decision_tree", "random_forest", "pca"])
    print(dsu)
    """
                     name  val
        0  decision_tree    1
        2  random_forest    3
        1            pca    2
    """
