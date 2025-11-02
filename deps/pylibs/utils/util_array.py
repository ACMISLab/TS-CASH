from os import makedirs
import sys
import numpy as np
import os.path
import pandas as pd
from pylibs.utils.util_datatime import get_str_datetime
from pylibs.utils.util_pandas import PDUtil
from pylibs.utils.util_common import UtilComm


class ArrSaver:
    """
    接受数组并保存为文件
    """

    def __init__(self, home=UtilComm.get_entrance_directory()):
        self.array = []

    def append(self, *args):
        print("add element:", args)
        self.array.append(args)

    def save_to_excel(self, columns=None):
        PDUtil.save_to_excel(pd.DataFrame(self.array, columns=columns), "save_arr",home=UtilComm.get_runtime_directory())

def arr_to_csv(arr,filename=None):
    if filename is None:
        filename = os.path.basename(sys.argv[0]).replace(".py","")+f"_{get_str_datetime()}.csv"
        
    home=os.path.join(os.path.dirname(sys.argv[0]),"runtime")
    makedirs(home,exist_ok=True)
    target_file_name=os.path.abspath(os.path.join(home,filename))
    print("Save file to {}".format(target_file_name))
    pd.DataFrame(arr).to_csv(target_file_name)


if __name__ == '__main__':
    ar = ArrSaver()
    ar.append(1, 2, 3, "aa")
    ar.save_to_excel()
