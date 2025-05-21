#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/3/17 17:25
# @Author  : gsunwu@163.com
# @File    : util_zfile.py
# @Description:
import os.path
import sys
from dataclasses import dataclass
from pathlib import Path

from pylibs.utils.util_bash import BashUtil
from pylibs.utils.util_servers import Server, Servers
from pylibs.utils.util_datetime import get_current_date

@dataclass
class UtilZfile:
    ZFILE_DATA_HOME = "/docker/appdata/zfile/data"

    @staticmethod
    def upload_file(source_file, remote_file=None):
        """
        上传文件到zfile.

        本质上是传输到文件夹下面，然后使用 zfile 读取文件列表
        Parameters
        ----------
        source_file :
        remote_file : str
            /a/b.txt

        Returns
        -------

        """
        server = Servers.SALI114

        if remote_file is None:
            remote_home = Path(UtilZfile.ZFILE_DATA_HOME, get_current_date(),os.path.basename(source_file))
        else:
            remote_home = Path(UtilZfile.ZFILE_DATA_HOME, remote_file)
            if remote_home.suffix == "":
                raise ValueError("File extension is not specified.")
        server.get_ssh().mkdir(os.path.dirname(remote_home));
        cmd = f'sshpass -p \'{server.password}\' rsync  -h -e "ssh -p {server.port} -o PubkeyAuthentication=yes -o stricthostkeychecking=no" -avi -m  {source_file} {server.username}@{server.ip}:{remote_home}'
        print(f"Upload {source_file} to {remote_home}")
        BashUtil.run_command_print_progress(cmd)


if __name__ == '__main__':
    # UtilZfile.upload_file(sys.argv[0], 'exp_data.py')
    UtilZfile.upload_file(sys.argv[0])
