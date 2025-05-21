import os
import sys
import tempfile
from dataclasses import dataclass

import psutil
from psutil import NoSuchProcess


def single_process_check(pid_file=None):
    """
    Check if this process has been running. Exit if it has been running, otherwise start_or_restart to run it.


    Returns
    -------

    """
    if has_process_running(pid_file=pid_file):
        print(f"Process has been running at {os.getpid()} ")
        sys.exit(-1)


def has_process_running(pid_file=None):
    """
    Determines whether this process is running.

    Ensure that only one instance is running

    The principle is to record process id to a file (/tmp/sys.argv[0]).

    Returns
    -------

    """
    pid = get_process_id_from_file(pid_file)
    if pid == -1:
        flag = False
    elif pid > 0:

        try:
            pid = psutil.Process(int(pid))
            if pid.is_running():
                flag = True
            else:
                flag = False
        except NoSuchProcess:
            flag = False

    else:
        flag = False

    if not flag:
        write_process_id_to_file()

    return flag


def get_process_id_from_file(pid_file=None):
    """
    -1 means not found

    Returns
    -------

    """
    if pid_file is None:
        p_file = get_process_pid_file()
    else:
        p_file = pid_file
    if not os.path.exists(p_file):
        return -1
    with open(get_process_pid_file(), "r") as f:
        pid = f.read()
    return int(pid)


def write_process_id_to_file():
    pid = os.getpid()
    with open(get_process_pid_file(), "w") as f:
        f.write(str(pid))
    return pid


def get_process_pid_file():
    process_name = os.path.basename(sys.argv[0])
    process_file = os.path.join("/tmp", process_name + ".pid")
    return process_file


@dataclass
class ProcessStats:
    """
    process = ProcessStats(pid_file="/tmp/dask_scheduler.pid")
    process.update_status()
    print(process.is_running())

    """
    pid_file: str = "/tmp/dask_scheduler.pid"

    def __post_init__(self):
        if self.pid_file is None:
            self.pid_file = os.path.join(
                tempfile.gettempdir(),
                os.path.basename(sys.argv[0]) + ".pid"
            )

    def update_status(self):
        pid = os.getpid()
        with open(self.pid_file, "w") as f:
            f.write(str(pid))
        return pid

    def is_running(self):
        pid = self._get_pid_from_file()
        if pid == -1:
            flag = False
        elif pid > 0:

            try:
                pid = psutil.Process(int(pid))
                if pid.is_running():
                    flag = True
                else:
                    flag = False
            except NoSuchProcess:
                flag = False

        else:
            flag = False
        return flag

    def _get_pid_from_file(self):
        try:
            with open(self.pid_file, "r") as f:
                pid = f.read()
            return int(pid)
        except:
            return -1

    def get_pid(self):
        return self._get_pid_from_file()


import subprocess

def kill_process_by_name(process_name):
    # kill_process_by_name("/usr/local/bin/frpc")
    try:
        # 查找进程PID
        pids = subprocess.check_output(["pgrep", "-f", process_name]).decode().strip().split('\n')
        for pid in pids:
            # 杀死进程
            subprocess.run(["kill", "-9", pid])
            print(f"Process {pid} has been killed.")
    except subprocess.CalledProcessError as e:
        print(f"No process with name {process_name} found.")

