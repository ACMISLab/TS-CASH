import os
import traceback

from pylibs.utils.util_log import get_logger

log = get_logger()


def make_dirs(path_dir):
    """
    Create a directory if it not exists. otherwise does nothing.

    Parameters
    ----------
    path_dir :

    Returns
    -------

    """

    try:
        if not os.path.exists(path_dir):
            oldmask = os.umask(000)
            os.makedirs(path_dir, 0o777)
            os.umask(oldmask)
    except FileExistsError as e:
        pass
        traceback.print_exc()
    except Exception as e:
        log.error(traceback.format_stack())
        traceback.print_exc()


class DUtil:
    @staticmethod
    def mkdir(dir_name):
        make_dirs(dir_name)
        assert os.path.exists(dir_name)
