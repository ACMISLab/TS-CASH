import os
import sys
import time

import psutil

from pylibs.utils.util_system import UtilSys


def _write_pid(pidfile):
    with open(pidfile, 'w') as f:
        UtilSys.is_debug_mode() and log.info(f"Saving  {os.getpid()} to {pidfile}")
        f.write(str(os.getpid()))


from pylibs.utils.util_log import get_logger

log = get_logger()


def single_program():
    """
    只允许一个程序的示例运行。
    Returns
    -------

    """
    pidfile = f"/tmp/{str(os.path.basename(sys.argv[0]))}.pid"
    if os.path.isfile(pidfile):
        with open(pidfile, 'r') as f:
            content = f.read()
        # 转为整数
        number = int(content)
        if _check_pid(number):
            print(f"Process is running {number}")
            sys.exit(-1)
        else:
            _write_pid(pidfile)
    else:
        _write_pid(pidfile)
    UtilSys.is_debug_mode() and log.info("Process has dont been running!")


def _check_pid(pid):
    try:
        process = psutil.Process(pid)
        if process.status() == psutil.STATUS_ZOMBIE:
            return False
        return process.is_running()
    except psutil.NoSuchProcess:
        return False


if __name__ == '__main__':
    single_program()
    time.sleep(1000)
