#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/9/22 21:24
# @Author  : gsunwu@163.com
# @File    : util_parallel.py
# @Description:
import threading
import multiprocessing
from helper import start_experiments
from pylibs.utils.util_system import UtilSys
def _parallel_run_jobs(array, gpu_index):
    while len(array) > 0:
        cf,counter= array.pop(0)
        start_experiments(cf,counter)


def _progress_bar(jobs, num_jobs):
    with tqdm(total=num_jobs) as pbar:
        _pre = num_jobs - len(jobs)
        set_percent(pbar, _pre, num_jobs)
        while len(jobs) > 0:
            _cur = num_jobs - len(jobs)
            if _cur == _pre:
                time.sleep(3)
                continue
            else:
                _pre = _cur
                conf: ExpConf = jobs[0]
                # update the progress
                pbar.set_postfix_str(
                    f"{conf.model_name}|{conf.data_sample_rate}|{conf.anomaly_window_type}|{conf.dataset_name}|{conf.exp_name}")
                set_percent(pbar, _cur, num_jobs)
                time.sleep(3)

        # While is break, so update the progress to full (all)
        set_percent(pbar, num_jobs)


class ParallelJobs:
    """
    Parallel run jobs on multi-GPUs and multi-CPUs
    """

    def __init__(self, n_par=UtilSys.get_cpu_num(0.9)):
        """

        Parameters
        ----------
        """
        # the number of parallel jobs
        self._n_par=n_par

    def run(self, original_jobs):
        n_jobs = len(original_jobs)
        with multiprocessing.Manager() as manager:
            target_jobs_array = manager.list()
            for _job in original_jobs:
                target_jobs_array.append(_job)

            processes = []
            # 支持多显卡
            # 每个显卡上多并发
            for _ in range(self._n_par):
                mp = multiprocessing.Process(target=_parallel_run_jobs, args=(target_jobs_array,None))
                mp.start()
                processes.append(mp)

            # process bar
            progress_bar = multiprocessing.Process(target=_progress_bar, args=(target_jobs_array, n_jobs))
            progress_bar.start()

            processes.append(progress_bar)
            for mp in processes:
                mp.join()