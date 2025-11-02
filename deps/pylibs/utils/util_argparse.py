import argparse

from pylibs.common import Emjoi
from pylibs.utils.util_log import get_logger
from pylibs.utils.util_system import UtilSys

log = get_logger()


def get_number_of_experiments(parsed_args: argparse.Namespace) -> int:
    """
    Get the combination of experiment args from argparse.parse_and_run()

    e.g., `--model OCSVM IsolationForest --seed 1 2 3`  generates 6 experiments.
    `--model OCSVM  --seed 1 2 3` generates 4 experiments.

    Parameters
    ----------
    parsed_args : argparse.Namespace

    Returns
    -------


    """
    count = 1
    for _key in vars(parsed_args):
        _val = getattr(parsed_args, _key)
        UtilSys.is_debug_mode() and log.info(f"key:{_key},val: {_val},type:{type(_val)}")
        if isinstance(_val, (list, tuple)):
            count = count * len(_val)
    UtilSys.is_debug_mode() and log.info(f"{Emjoi.SETTING} The number of experiments is {count}")
    return count
