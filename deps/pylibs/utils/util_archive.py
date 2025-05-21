#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/3/18 10:21
# @Author  : gsunwu@163.com
# @File    : util_archive.py
# @Description:
import os.path
import shutil
from pylibs.config import Env


class UtilZip:
    @staticmethod
    def create_zip(target_dir, output_name="./aaa.zip", archive_format='zip'):
        if output_name.endswith(".zip"):
            output_name = os.path.join(Env.get_runtime_home(), output_name[0:-4])
        # 创建压缩文件
        return shutil.make_archive(output_name, archive_format, target_dir)


if __name__ == '__main__':
    res = UtilZip.create_zip("/Users/sunwu/SW-Research/runtime/plot_uts_and_score", "./del.zip")
    print(res)
    assert os.path.exists(res) == True
