# 关闭连接
import os
import re
import sys
import time
import traceback

import paramiko
from pylibs.common import Emjoi
from pylibs.utils.util_bash import BashUtil
from pylibs.utils.util_directory import make_dirs
from pylibs.config import ExpServerConf

import logging

log = logging.getLogger(__name__)
logging.getLogger("paramiko").setLevel(logging.DEBUG)
from rich import print

"""
文件夹不存在，就执行命令，否者跳过
test ! -d "/remote-home/cs_acmis_sunwu/2024/p3" && echo 1
"""

_PATTERN_FLAG = r'\[__S\d+W__\]'


def post_process_data(data):
    return data
    # _first_success_line="[__S2W__]"
    # # return re.sub(_PATTERN_FLAG, '', data)
    # _index=data.find(_first_success_line)
    # if _index==-1:
    #     return ""
    # else:
    #     return re.sub(_PATTERN_FLAG, '', data)


def remove_color_codes(text) -> str:
    # 去除invoke_shell 返回的颜色控制码:
    # 正则表达式用于匹配 ANSI 颜色控制码
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)


def post_process_invoke_shell_data(data):
    return data.replace(";echo [EOF]", "").replace("[EOF]", "")


def filter_print(msg):
    _first_success_line = "Last login"
    _index = msg.find(_first_success_line)
    if _index > -1:
        out_msg = msg[_index:]
    else:
        out_msg = msg

    print(re.sub(_PATTERN_FLAG, '', out_msg))


class SSHClient:
    def __init__(self,
                 hostname="your_server_ip",
                 port=22,
                 username="root",
                 password="your_server_password"):
        """
        gpu_index: str
            split by , . e.g., 1 or 1,2
        """
        self._counter = None
        self.username = username
        self.hostname = hostname
        self.port = port
        self.password = password
        print(f"Connecting to {hostname}:{port}...")
        self.ssh = paramiko.SSHClient()
        # self.ssh.load_system_host_keys()
        # 允许将信任的主机自动加入到 host_allow 列表，此方法必须放在connect方法的前面
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        # 调用connect方法连接服务器
        try:
            self.ssh.connect(hostname=hostname, port=port, username=username, password=password,
                             timeout=10, allow_agent=False, look_for_keys=False)
        except Exception as e:
            print(f"Connecting to {hostname}:{port} failed since {e}")
            sys.exit(-1)
        else:
            print(f"Connecting to {hostname}:{port} successfully")

        # log.info(f"Connecting to {hostname}:{port} successfully")
        self.shell = None
        self.ftp = None

    def kill_program_by_name(self, name):
        """
        kill a program by name. e.g.,

        pgrep supervisord|xargs  -r  kill  -9
        Parameters
        ----------
        name :

        Returns
        -------

        """
        self.exec(f"pgrep {name}|xargs -r kill -9")

    def kill_program_by_substr(self, name):
        """
        kill a program by any substring contained in the program argument.
        e.g., kill the following program by ssh_tunel_from_S164_to_S153.conf
        /root/miniconda3/bin/python /root/miniconda3/bin/supervisord -c /remote-home/cs_acmis_sunwu/ssh_tunel_from_S164_to_S153.conf

        pgrep supervisord|xargs  -r  kill  -9
        Parameters
        ----------
        name :

        Returns
        -------

        """
        self.exec(f"pgrep -f {name}|xargs -r kill -9")

    def cpu_count(self):
        """
        Returns the number of CPUs cores

        Returns
        -------

        """
        # res=self.send("nproc")
        # print(">>"*30)
        # print(res)
        # print("<<"*30)
        # return int(res.split("nproc")[1].split("\r\n")[1].split("\r")[1])
        stdin, stdout, stderr = self.ssh.exec_command('nproc')
        cpu_cores = stdout.read().strip()
        return int(cpu_cores)

    def execs(self, cmds):
        """
        执行多行命令
        :param cmds:
        :type cmds:
        :return:
        :rtype:
        """
        lines = cmds.split("\n")
        for line in lines:
            exe_line = line.strip()
            if exe_line == "":
                pass
            else:
                self.exec(exe_line)

    def exec(self, cmd):
        return self.send(cmd)

    def send(self, cmd: str, print_result=True) -> str:
        """
        允许顺序连续的执行命令, 就像 Item2 一样.

        执行完之后,才会返回. 这里使用了tricks: 在命令后面加添加";echo [EOF]", 然后判断"\r\n[EOF]\r\n" 是否存在, 如果存在,证明执行完成了


        Parameters
        ----------
        print_result :
        cmd :

        Returns
        -------

        """
        END_FLAG = r"[__S\#W__]"
        if self.shell is None:
            self.shell = self.ssh.invoke_shell(environment={
                "force_color_prompt": "yes",
            }
            )
            self._counter = 0
            self._send(f"export PS1='{END_FLAG}'")

        # 等命令执行完成之后再返回. 使用了tricks: 在命令后面加添加";echo [EOF]", 然后判断"\r\n[EOF]\r\n" 是否存在, 如果存在,证明执行完成了

        cmd = cmd.strip()
        self._send(cmd)
        time.sleep(0.1)
        received_data = b""

        while True:
            try:
                if not self.shell.recv_ready():
                    time.sleep(0.05)

                data = self.shell.recv(8192)
                if print_result:
                    try:
                        filter_print(data.decode('utf-8'))
                    except:
                        continue

                received_data += data

                # test the result is finished, using the PS1=[\#], \# show how
                # many the  command executed
                flags = "[__S{}W__]".format(self._counter + 1).encode("utf-8")
                if received_data.find(flags) > -1:
                    return post_process_data(received_data.decode('utf-8', 'ignore'))
                time.sleep(0.05)
            except Exception as e:
                traceback.print_exc()
                return ""

    # sshpass -p  'acmis' rsync  -h -e "ssh -p 60013 -o PubkeyAuthentication=yes -o stricthostkeychecking=no" -avi -m   -f"+ */" -f"- ***/.git/***" -f"+ *.py" -f"+ *.sh" -f"+ *.csv" -f"+ **/Makefile"  -f"- *"  --delete /remote-home/cs_acmis_sunwu/2024/p2/pylibs/ root@your_server_ip:/remote-home/cs_acmis_sunwu/2024/pylibs/
    def __decodeCommandRes(self, std):
        return std.read().decode()

    # @DeprecationWarning
    # def exec(self, cmd, end="\n", env=None):
    #     log.info(f"{cmd}")
    #     stdin, stdout, stderr = self.ssh.exec_command(cmd, get_pty=True, environment=env)
    #     while True:
    #         line = stdout.readline()
    #         if not line:
    #             break
    #         print(line.strip(), end=end)
    #     return self

    def wraper_cmd(self, cmd):
        return cmd

    def upload_dir(self, local_upload_dir, remote_upload_dir, recursive=True):
        """
        上传本地文件夹 local_upload_dir 中的文件到远程路径 remote_upload_dir

        如果想要递归上传，指定
        Parameters
        ----------
        recursive:bool
            是否递归上传子文件夹
        local_upload_dir
        remote_upload_dir

        Returns
        -------

        """
        self._init_ftp()
        files = os.listdir(local_upload_dir)

        self._check_directory(remote_upload_dir)
        for file in files:
            if file.startswith("."):
                continue

            if os.path.isdir(os.path.join(local_upload_dir, file)):
                if recursive:
                    self.upload_dir(os.path.join(local_upload_dir, file),
                                    os.path.join(remote_upload_dir, file), recursive=recursive)
            else:

                upload_local_file = os.path.join(local_upload_dir, file)
                upload_remote_file = os.path.join(remote_upload_dir, file)
                self.upload_file(upload_local_file, upload_remote_file)
                try:
                    self.ftp.lstat(upload_remote_file)
                    log.info(
                        f"{Emjoi.SUCCESS} Upload file success, from {upload_local_file} -> {upload_remote_file}")
                except:
                    log.error(f"{Emjoi.FAILED} Upload file error, from {upload_local_file} -> {upload_remote_file}")

    def download_dir(self, remote_dir, local_dir, recursive=True, filename_pattern=None):
        ftp = self.ssh.open_sftp()
        files = ftp.listdir(remote_dir)
        files_attr = ftp.listdir_attr(remote_dir)
        for file, file_attr in zip(files, files_attr):
            if file.startswith("."):
                continue

            if file_attr.longname.startswith('d'):
                if recursive:
                    self.download_dir(os.path.join(remote_dir, file),
                                      os.path.join(local_dir, file), recursive=recursive,
                                      filename_pattern=filename_pattern)
            else:
                if filename_pattern is None:
                    self.download_one_file(file, ftp, local_dir, remote_dir)
                else:
                    if re.search(filename_pattern, file):
                        self.download_one_file(file, ftp, local_dir, remote_dir)
                    else:
                        log.info(f"Pass [{file}] with pattern [{filename_pattern}]")

    # def download_one_file(self, file, ftp, local_dir, remote_dir):
    #     remote_file = os.path.join(remote_dir, file)
    #     local_file = os.path.join(local_dir, file)
    #     if not os.path.exists(local_dir):
    #         os.makedirs(local_dir)
    #     ftp.get(remote_file, local_file)
    #
    #     if os.path.exists(local_file):
    #         log.info(
    #             f"{Emjoi.SUCCESS} Download  success！from {os.path.basename(remote_file)} from {remote_file} to {local_file}")
    #     else:
    #         log.error(
    #             f"{Emjoi.FAILED} Download  error！from {os.path.basename(remote_file)} from {remote_file} to {local_file}",
    #             file=sys.stderr)

    def _check_directory(self, target_dir):
        try:
            self.ftp.stat(target_dir)
        except:
            self.ftp.mkdir(target_dir)

    def __del__(self):
        self.ssh.close()

    def close(self):
        self.ssh.close()
        self.ftp.close()
        self.shell.close()

    def send_file(self):
        pass

    def upload_file_with_ftp(self, upload_local_file, upload_remote_file):
        print(f"\nUpload \n{upload_local_file} \n== to ==>\n{upload_remote_file}\n")
        self._init_ftp()
        return self.ftp.put(upload_local_file, upload_remote_file)

    def download_file(self, remote_path, local_path):
        print(f"Downloading {local_path} from {remote_path}")
        # sshpass -p Fdd19950518* rsync -avi  -e "ssh -p 20045 -o PubkeyAuthentication=yes   -o stricthostkeychecking=no" root@your_server_ip:/remote-home/acmis_fdd/sw_research_code/runtime/ ./
        make_dirs(os.path.dirname(local_path))
        BashUtil.run_command_print_progress(
            f'sshpass -p {self.password} rsync -ah --stats  -e "ssh -p {self.port} -o PubkeyAuthentication=yes   -o '
            f'stricthostkeychecking=no" {self.username}@{self.hostname}:{remote_path}  {local_path}'
        )
        # self._init_ftp()
        # self.ftp.get(remote_path, local_path)

    def _init_ftp(self):
        if self.ftp is None:
            log.info("Init ftp connection")
            self.ftp = self.ssh.open_sftp()

    @staticmethod
    def get_ssh_from_conf(conf: ExpServerConf):
        ssh = SSHClient(hostname=conf.ip, port=conf.port, username=conf.username, password=conf.password)
        return ssh

    def _send(self, param):

        param = param + "\n"
        self.shell.send(param.encode("utf-8"))
        self._counter += 1

    def mkdir(self, target_dir):
        return self.exec(f"test ! -d \"{target_dir}\" && mkdir -p {target_dir}")


if __name__ == '__main__':
    pass
    # ssh = SSHClient(conf.ip, conf.port, conf.username, conf.password)
    # ssh.send("cd /remote-home")
    # ssh.upload_dir("/Users/sunwu/SW-Research/sw-research-code/A01_result_analysis/A01_data_stastics", "/root/")
    # ssh.send("ls /root/")
    # ssh.prepare_env("/Users/sunwu/Downloads/download_metrics/V900_01_observation_VUS_ROC_0.001_random/metrics.csv",
    #                 "/root/metrics.csv")
    # ssh.send("ls /root/")

    # 32 = {str} '\x1b[31m2.清理系统盘请参考：https://www.autodl.com/docs/qa/\x1b[0m'
    # 36 = {str} '\x1b[34;42mV900_02_fastuts_VUS_ROC_0.001_random\x1b[0m'
    # 33 = {str} 'root@79bc33a413bc:~# ls /remote-home/cs_acmis_sunwu/sw_research_code/runtime/'
    # 34 = {str} '\x1b[0m\x1b[34;42mV900_01_observation_VUS_ROC_0.001_random\x1b[0m'
    # 31 = {str} '\x1b[31m1.系统盘较小请将大的数据存放于数据盘或网盘中，重置系统时数据盘和网盘中的数据不受影响\x1b[0m'
    # 38 = {str} '\x1b[34;42mV901_01_observation_VUS_ROC_0.001_random\x1b[0m'
    # 30 = {str} '\x1b[31m*注意: \x1b[0m'
    # 37 = {str} '\x1b[34;42mV900_03_fastuts_sup1_VUS_ROC_0.01_lhs\x1b[0m'
    # 35 = {str} '\x1b[34;42mV900_02_fastuts_VUS_ROC_0.001_dist1\x1b[0m'
    # print(remove_color_codes('\x1b[34;42mV900_02_fastuts_VUS_ROC_0.001_dist1\x1b[0m'))
