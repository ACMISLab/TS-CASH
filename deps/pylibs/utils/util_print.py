import time

from pylibs.utils.util_log import get_logger

log = get_logger()


def sprint(msg):
    print("\n")
    print("=" * 64)
    print("=" * 10 + "  " + msg)
    print("=" * 64)


def print_progress_info(msg):
    """
    只显示一行信息. 用于打印进度信息
    Returns
    -------

    """
    print(msg, end="\r", flush=True)


if __name__ == '__main__':
    for i in range(10):
        print_progress_info(f"kljsdf {i}")
        time.sleep(1)
