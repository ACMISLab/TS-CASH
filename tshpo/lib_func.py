#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/10/12 19:37
# @Author  : gsunwu@163.com
# @File    : libs.py
# @Description:
import logging
import os.path
import sys
from joblib import Memory, memory, Parallel, delayed

log_ = None


def is_macos():
    """
    是否是 macos 系统
    Returns
    -------

    """
    return sys.platform == 'darwin'


def get_memory():
    # 创建一个缓存目录
    memory = Memory(os.path.join(os.path.dirname(__file__), "cache"), verbose=0)
    return memory


def get_log():
    global log_
    if log_ is None:
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.INFO if is_macos() else logging.ERROR
        )
        log_ = logging.getLogger(__name__)
    return log_


def clear_cache():
    get_memory().clear()


_client = None


def run_jobs_on_dask(train_model_smac, jobs):
    """

    Parameters
    ----------
    train_model_smac :  要运行的函数
    jobs : 函数的参数

    Returns
    -------

    """
    global _client
    if _client is None:
        from distributed import LocalCluster
        from distributed import Client
        cluster = LocalCluster(n_workers=int(os.cpu_count() * 0.7), threads_per_worker=1, dashboard_address=":8888",
                               ip="your_server_ip")
        _client = Client(cluster)
    _features = []
    for _econf, _run_job in jobs:
        _features.append([_econf, _client.submit(train_model_smac, run_job=_run_job)])
    results = _client.gather(_features)
    return results


def run_jobs_on_joblib(train_model_smac, jobs):
    """

    Parameters
    ----------
    train_model_smac : 要运行的函数
    jobs : 函数的参数

    Returns
    -------

    """
    results = Parallel(n_jobs=int(os.cpu_count() * 0.7), verbose=10)(
        delayed(train_model_smac)(_run_job) for (_econf, _run_job) in jobs)
    return results
