#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/7/20 17:39
# @Author  : gsunwu@163.com
# @File    : util_cache.py
# @Description:
import dataclasses
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import typing


@dataclasses.dataclass
class NumpyCache:
    name: typing.Union[str, int]

    def get_cache_file_name(self) -> str:
        file = Path("cache", f"{os.path.basename(sys.argv[0])}_{self.name}.npz")
        if file.is_dir():
            shutil.rmtree(file.as_posix())
        if not file.parent.exists():
            file.parent.mkdir(parents=True)
        return file.as_posix()

    def is_cached(self):
        return os.path.exists(self.get_cache_file_name())

    def cache(self, **kwargs):
        cache_prefix = self.get_cache_file_name()
        np.savez(cache_prefix, **kwargs)

    def get_cache(self):
        data = np.load(self.get_cache_file_name())
        return data


if __name__ == '__main__':
    cache = NumpyCache(11)
    print(cache.is_cached())
    cache.cache(a=1, b=2, c=3)
    print(cache.get_cache())
