import os

from pylibs.utils.util_bash import BashUtil
from pylibs.utils.util_directory import make_dirs
def get_gpu_device():
    import torch.cuda
    if torch.cuda.is_available():
        return 'cuda'

    else:
        return 'cpu'

def get_user_home():
    """
    Returns the user's home directory: /Users/sunwu

    """
    return os.path.expandvars('$HOME')


def get_user_cache_home():
    """
    Returns the user's cache directory:  /Users/sunwu/.cache

    """
    cache_path = os.path.join(get_user_home(), ".cache")
    make_dirs(cache_path)
    return cache_path


def get_num_cpus():
    return os.cpu_count()

def get_cpu_count():
    return get_num_cpus()

def get_gpu_count():
    return int(BashUtil.exe_cmd("nvidia-smi --exec-gpu=gpu_name --format=csv,noheader | wc -l"))


if __name__ == '__main__':
    print(get_user_home())
    print(get_user_cache_home())
    print(os.cpu_count())
    print(get_gpu_count())
