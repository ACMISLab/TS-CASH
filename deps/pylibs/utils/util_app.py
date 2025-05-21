#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/3/2 12:39
# @Author  : gsunwu@163.com
# @File    : util_app.py
# @Description:
import os
import sys
import traceback

import fasteners


def single_application(fun):

    """
    保证一个进程只有一个实例运行

    Examples:

    from pylibs.utils.util_app import single_application
    def run():
        time.sleep(30)
    single_application(run)

    Parameters
    ----------
    fun :

    Returns
    -------

    """
    # 定义一个锁文件，用于实现跨进程的锁定
    lockfile_path = f"/tmp/{os.path.basename(sys.argv[0])}.lock"
    lock = fasteners.InterProcessLock(lockfile_path)
    # print("尝试启动应用...")
    have_lock = False
    try:
        # 尝试获取锁
        have_lock = lock.acquire(blocking=False)
        if have_lock:
            # print("没有其他实例在运行，当前应用已启动。")
            # 你的应用逻辑
            print("App is running...")
            fun()
        else:
            print("Another app has been running.")
            sys.exit(0)

    except IOError as e:
        print("Error, cant get file lock", e)
        traceback.print_exc()
    finally:
        if have_lock:
            # 释放锁
            lock.release()
            print("App has been down.")