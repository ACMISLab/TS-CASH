import os

import psutil


def is_running(name="process.pid"):
    if not os.path.exists(name):
        return False
    with open(name) as f:
        cur_pid = f.read()
    print(cur_pid)
    pid = os.getpid()
    print(psutil.Process(pid))


if __name__ == '__main__':
    print(is_running("aaa.pid"))
