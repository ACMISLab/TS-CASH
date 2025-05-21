import io
import os.path

import requests

from pylibs.utils.util_directory import make_dirs


def lowercase_keys(dictionary):
    """
    将 Python 字典（dict）中的所有键设为小写
    Parameters
    ----------
    dictionary :

    Returns
    -------

    """
    return {key.lower(): value for key, value in dictionary.items()}


if __name__ == '__main__':
    my_dict = {"Name": "John", "Age": 30, "City": "New York"}
    new_dict = lowercase_keys(my_dict)
    print(new_dict)
