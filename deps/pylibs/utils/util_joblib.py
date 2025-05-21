import numpy as np

from pylibs.utils.util_common import UtilComm
from pylibs.utils.util_sys import get_user_cache_home

from joblib import Memory

from pylibs.utils.util_system import UtilSys


class JLUtil:
    memory = None

    @staticmethod
    def get_memory(home=UtilComm.get_joblib_cache_dir(), verbose=0):
        if JLUtil.memory is None:
            cache_dir = get_user_cache_home() if home is None else home
            if UtilSys.is_debug_mode():
                from pylibs.utils.util_log import get_logger
                log = get_logger()
                log.debug(f"Joblib cache dir: {home}")
                verbose = 0

            JLUtil.memory = Memory(cache_dir, verbose=verbose)
        return JLUtil.memory

    @staticmethod
    def clear_all_calche():
        JLUtil.memory.clear()


def cache_(func):
    mem = JLUtil.get_memory()
    return mem.cache(func=func)


if __name__ == '__main__':
    memory = JLUtil.get_memory()


    @memory.cache
    def f(a):
        print(f"calcu {a}")
        return a ** 1024


    f(1)
    f(1)
    f(3)


    @memory.cache
    def two(a, b, c):
        print(f"calcu {a}")
        return len(a) + len(b) + len(c)


    two(np.zeros((10,)), np.zeros((10,)), np.zeros((10,)))
    two(np.zeros((10,)), np.zeros((10,)), np.zeros((10,)))
    two(np.zeros((10,)), np.zeros((10,)), np.zeros((10,)))


    @cache_
    def aaa(a=1):
        print("calling aaa")


    aaa(1)
