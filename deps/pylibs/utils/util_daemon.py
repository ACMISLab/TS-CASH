#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/3/18 15:07
# @Author  : gsunwu@163.com
# @File    : util_daemon.py
# @Description:
import multiprocessing
import os
import sys
import time
import traceback
from functools import partial

from pylibs.utils.util_feishu import send_msg_to_feishu
from pylibs.utils.util_log_server import zs

log=zs.get_logger("dask")

def fun():
    for i in range(3):
        send_msg_to_feishu(f"pid-{os.getpid()}-{i}")
        time.sleep(3)


def daemon_run(target_function):
    """
    让程序在后台运行

    Parameters
    ----------
    function :

    Returns
    -------

    """
    # 第一次fork
    if os.fork() > 0:
        return True

    # 从父进程环境脱离
    os.chdir('/')  # 改变工作目录
    os.setsid()  # 创建新的会话
    os.umask(0)  # 重新设置文件创建掩码

    # 第二次fork
    if os.fork() > 0:
        sys.exit(0)  # 第一子进程退出

    sys.stdout.flush()
    sys.stderr.flush()

    with open('/dev/null', 'r') as f:
        os.dup2(f.fileno(), sys.stdin.fileno())
    with open('/dev/null', 'a+') as f:
        os.dup2(f.fileno(), sys.stdout.fileno())
        os.dup2(f.fileno(), sys.stderr.fileno())

    # 在这里执行守护进程的工作
    # send_msg_to_feishu(os.getpid())
    try:
        target_function()
    except Exception as e:
        log.error(traceback.format_exc())


if __name__ == "__main__":
    print(1)
    daemon_run(fun)
    print(2)
    daemon_run(fun)
