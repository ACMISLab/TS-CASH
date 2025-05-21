import logging
import os
import shutil
import sys
import time

import pandas as pd

from pylibs.utils.util_common import UtilComm
from pylibs.utils.util_directory import make_dirs
from pylibs.utils.util_system import UtilSys

log = logging.getLogger(__name__)
from pylibs.utils.util_joblib import cache_
import yaml


def get_all_files(home, ext):
    files = FileUtils().get_all_files(home=home, ext=ext)
    return files


def gaf(home, ext):
    return get_all_files(home, ext)


def generate_random_file(ext=".csv", home=os.path.join(os.path.dirname(sys.argv[0]), 'runtime'), prefix="", name=None):
    """
    生成随机文件

    Parameters
    ----------
    name : str
        The file user. Default `str(time.time_ns())`.
    ext : str
        The file ext.
    home :str
        The home to save the file
    prefix: str
        The prefix of the file, i.e., .pdf
    Returns
    -------
    str
        The file path

    """
    assert type(ext) == str

    if not ext.startswith("."):
        ext = "." + ext

    if ext == ".":
        ext = ""

    make_dirs(home)

    if name is None:
        name = str(time.time_ns())
    file = os.path.abspath(os.path.join(home, prefix + name + ext))
    UtilSys.is_debug_mode() and log.info(f"File is located: {os.path.abspath(file)}")
    return file


def grf(ext=".csv", home=os.path.join(os.path.dirname(sys.argv[0]), 'runtime'), prefix="", name=None):
    return generate_random_file(ext=ext, home=home, prefix=prefix, name=name)


class FileUtils:
    @classmethod
    def get_file_size_mb(cls, f):
        return round(os.path.getsize(f) / (1024 * 1024), 2)

    @classmethod
    def save_str_to_file(cls, filename, contents):
        with open(filename, "w+") as f:
            f.writelines(contents)

    def __init__(self):
        self.files = []

    def _list_all_files(self, home, ext):
        # UtilSys.is_debug_mode() and log.info(f"Loading file from [{os.path.basename(home)}] with ext [{ext}]")
        for file in os.listdir(home):
            file = os.path.join(home, file)
            if os.path.isdir(file):
                self._list_all_files(file, ext)
            else:
                if ext is None:
                    self.files.append(os.path.abspath(file))
                elif str(file).endswith(ext):
                    self.files.append(os.path.abspath(file))
                else:
                    log.debug(f"Pass file: {os.path.basename(file)}")

    def _list_all_dir(self, home):
        # UtilSys.is_debug_mode() and log.info(f"Loading file from [{os.path.basename(home)}] with ext [{ext}]")
        for file in os.listdir(home):
            file = os.path.join(home, file)
            if os.path.isdir(file):
                self._list_all_dir(file)
            else:
                self.files.append(os.path.dirname(os.path.abspath(file)))

    def get_all_files(self, home, ext=".out"):
        """
        Get all file (including subdir) with ext
        :param home:
        :param ext:
        :return:
        """
        if len(self.files) == 0:
            self._list_all_files(home, ext)

        return self.files

    @staticmethod
    def save_df_to_excel(df: pd.DataFrame, name, home=os.path.dirname(sys.argv[0])):
        if not name.endswith(".xlsx"):
            name = name + ".xlsx"

        out_file = os.path.join(home, name)
        make_dirs(os.path.dirname(out_file))
        UtilSys.is_debug_mode() and log.info(f"Saving df to {os.path.abspath(out_file)}")

        df.to_excel(out_file, index=False)

    @staticmethod
    def save_txt_to_gnu(text, name, home=UtilComm.get_runtime_directory()):
        if not name.endswith(".gnu"):
            name = name + ".gnu"
        path = os.path.join(home, name)
        UtilSys.is_debug_mode() and log.info(f"Saving [{name}] to {os.path.abspath(path)}")
        with open(path, "w") as f:
            f.write(text)


class FileUtil(FileUtils):
    def __init__(self):
        super().__init__()

    @classmethod
    def save_to_yaml(cls, data: dict, file_path=None):
        if file_path is None:
            file_path = os.path.join(UtilComm.get_runtime_directory(), "examples.yaml")
        make_dirs(os.path.dirname(file_path))
        abs_path = os.path.abspath(file_path)
        log.info(f"Write data to {abs_path}")
        with open(abs_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)


def delete_file(file_name):
    os.remove(file_name)


def remove_dir(dir_name):
    """
    Recursively delete a directory tree.

    Parameters
    ----------
    dir_name :

    Returns
    -------

    """
    shutil.rmtree(dir_name)


if __name__ == '__main__':
    files = FileUtils().get_all_files("./", ext=".py")
    print("\n" * 3)
    print(f"Number of KPI: {len(files)}")
    plot = """
    set term pdfcairo lw 2 font "Times New Roman,20" enhanced size 16,6
    set output "test.pdf"
    set multiplot layout 2,2

    #set style data histogram
    #set style histogram errorbars gap 1 lw 1
    set ylabel "Avg. Training Time (sec)"
    set xlabel "Training Data Ratio"
    set title "Decision Tree"
    set key reverse
    set key samplen 2
    set key right bottom
    set grid

    list=system('ls -1B *.dat')
    print list


    do for [filename in list] {
        set title  filename[1:(strlen(filename)-4)] noenhanced
        plot filename u 0:4:xtic(3) with lp lt 8 title "Perf."
    }


    unset multiplot
    set output
                """

    FileUtil.save_txt_to_gnu(plot, "aaa")
