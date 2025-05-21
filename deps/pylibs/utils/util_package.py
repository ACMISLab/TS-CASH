#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/2/28 20:36
# @Author  : gsunwu@163.com
# @File    : util_package.py
# @Description:
import pkg_resources
from pathlib import Path
def get_installed_packages():
    installed_packages = pkg_resources.working_set
    installed_packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])
    return installed_packages_list

def  save_installed_packages(filename:Path):
    print(f"Saving package to {filename.absolute()}")
    packages=get_installed_packages()
    packages=[item+"\n" for item in packages]
    with open(filename, "w") as f:
        f.writelines(packages)