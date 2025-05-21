#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/2/25 17:12
# @Author  : gsunwu@163.com
# @File    : util_job.py
# @Description:
import os

def pr(fun, jobs: list):
    return parallel_run(fun, jobs)

def parallel_run(fun, jobs: list):
    """
Parallely exec fun for the jobs.
    fun: the function to run
    jobs: a list, which elements is a json.



Examples:

from pylibs.utils.util_job import parallel_run

jobs = [
    {"dataset_name": "d1", "data_id": "id1"},
    {"dataset_name": "d2", "data_id": "id2"},
    {"dataset_name": "d3", "data_id": "id3"}
]
def fun(dataset_name, data_id):
    print(f"run on {dataset_name}>>{data_id}")
    return dataset_name+data_id
res=parallel_run(fun, jobs)
print(res)


    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    executor = ThreadPoolExecutor(max_workers=int(os.cpu_count() * 0.8))
    all_task = [executor.submit(fun, **job) for job in jobs]

    ret_arr = []
    for future in as_completed(all_task):
        data = future.result()
        ret_arr.append(data)
    return ret_arr

